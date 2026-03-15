"""
EvalLLMClient - 适配 llm.py（data_eval / 混元）API 到 swarm 的 LLM 接口

用法：创建 Agent 时传入 llm_client=EvalLLMClient(model_name="kimi-k2.5")，
即可用你们现有的 get_model_answer(model_name, prompt, history) 驱动 swarm。

注意：当前 data_eval/混元 接口不支持 tools（function calling），
因此只能跑「单智能体、无 create_subagent/assign_task」的对话；
若传入 tools，模型不会返回 tool_calls，会直接以文本回复并结束。
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# 确保能 import 到上级目录的 llm（Agent_swarm/llm.py）
_AGENT_SWARM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _AGENT_SWARM_DIR not in sys.path:
    sys.path.insert(0, _AGENT_SWARM_DIR)

try:
    import llm as eval_llm
except Exception as e:
    eval_llm = None
    logger.warning("eval_llm_client: 无法 import llm，请从 Agent_swarm 目录运行或设置 PYTHONPATH: %s", e)

MAX_RETRIES = 3
RETRY_DELAY = 2


def _to_text(content: Any) -> str:
    """从 message content 取出纯文本"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("value", "") or ""
        return ""
    return str(content)


def _messages_to_prompt_and_history(messages: List[Dict[str, Any]]) -> tuple:
    """
    将 swarm 的 messages 转为 llm.py 的 (prompt, history)。
    history 使用 CommonLLM 格式（user/system 的 content 为 [{"type":"text","value": ...}]），
    混元侧会在 get_model_answer 里把 list 转成 string。
    """
    if not messages:
        return "请回复。", []

    # 先转成统一的 role + 文本 content 列表，并把 tool 转成“用户看到的工具结果”
    flat: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = _to_text(m.get("content"))
        if role == "system":
            flat.append({"role": "system", "content": content})
        elif role == "user":
            flat.append({"role": "user", "content": content})
        elif role == "assistant":
            flat.append({"role": "assistant", "content": content or "(无文本)"})
        elif role == "tool":
            flat.append({"role": "user", "content": f"Tool result: {content}"})

    if not flat:
        return "请回复。", []

    last = flat[-1]
    prompt = last["content"] or "请回复。"
    history_items = flat[:-1]

    # 转为 CommonLLM 的 history 格式（user/system 用 [{"type":"text","value": ...}]）
    history = []
    for item in history_items:
        r, c = item["role"], item["content"]
        if r in ("system", "user"):
            history.append({"role": r, "content": [{"type": "text", "value": c}]})
        else:
            history.append({"role": r, "content": c})

    return prompt, history


class EvalLLMClient:
    """
    使用 llm.py 的 get_model_answer(model_name, prompt, history) 实现
    swarm 需要的 async chat(messages, tools=..., temperature=..., max_tokens=...) 接口。
    """

    def __init__(self, model_name: str):
        """
        Args:
            model_name: llm.py 中的模型名，如 "kimi-k2.5", "gpt4o", "hunyuan-t1-online-latest"
        """
        self.model_name = model_name
        if eval_llm is None:
            raise RuntimeError("无法导入 llm 模块，请确保从 Agent_swarm 目录运行或设置 PYTHONPATH")
        if "hunyuan" in model_name:
            if model_name not in getattr(eval_llm, "HUNYUAN_MODEL_MARKERS", {}):
                raise ValueError(f"未知的混元模型: {model_name}")
        else:
            if model_name not in getattr(eval_llm, "COMMON_MODEL_MARKERS", {}):
                raise ValueError(f"未知的通用模型: {model_name}")

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        与 swarm 的 LLMClient.chat 兼容：用 llm.get_model_answer 同步调用，
        在 executor 中执行，返回 { role, content, finish_reason }。
        当前后端不支持 tools，故永不返回 tool_calls。
        """
        prompt, history = _messages_to_prompt_and_history(messages)
        loop = asyncio.get_event_loop()

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                # 同步 get_model_answer 在线程池中执行，避免阻塞事件循环
                answer = await loop.run_in_executor(
                    None,
                    lambda: eval_llm.get_model_answer(self.model_name, prompt, history, use_cache=True),
                )
                if not answer or answer == "none":
                    raise RuntimeError("get_model_answer 返回空或 'none'")
                return {
                    "role": "assistant",
                    "content": answer.strip(),
                    "finish_reason": "stop",
                }
            except Exception as e:
                last_error = e
                logger.warning("EvalLLMClient attempt %s/%s failed: %s", attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        logger.error("EvalLLMClient failed after %s attempts: %s", MAX_RETRIES, last_error)
        return {
            "role": "assistant",
            "content": f"Error: {str(last_error)}",
            "finish_reason": "error",
        }
