"""
TrpcOpenAIClient - 使用 OpenAI 客户端对接 trpc-gpt-eval 网关

用法与 OpenAI 兼容接口一致，api_key 格式示例：
  api_key = f"{APP_ID}:{APP_KEY}?provider=moonshot&timeout=600&model={model_id}"
  base_url = "http://trpc-gpt-eval.production.polaris:8080/v1"

若网关支持 function calling，传 tools 后会返回 tool_calls，多智能体可正常跑；
若不支持，则只返回 content，退化为单轮文本。
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional

from openai import AsyncOpenAI
import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2
DEFAULT_TIMEOUT = 600  # 与 api_key 里 timeout 一致时可调


def build_trpc_api_key(
    app_id: str,
    app_key: str,
    provider: str = "moonshot",
    timeout: int = 600,
    model: Optional[str] = None,
) -> str:
    """拼装 trpc 网关的 api_key 字符串。"""
    q = f"provider={provider}&timeout={timeout}"
    if model:
        q += f"&model={model}"
    return f"{app_id}:{app_key}?{q}"


class TrpcOpenAIClient:
    """
    使用 OpenAI SDK 调用 trpc-gpt-eval 网关，与 swarm 的 LLMClient 接口一致。
    支持 tools：若网关返回 tool_calls 则多智能体可正常执行。
    """

    def __init__(
        self,
        model_id: str = "kimi-k2-0905-preview",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        app_id: Optional[str] = None,
        app_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Args:
            model_id: 模型名，如 kimi-k2-0905-preview
            api_key: 直接传完整 api_key（格式见模块说明）；若不传则用 app_id + app_key 拼
            base_url: 网关地址，默认 trpc-gpt-eval
            app_id: 与 app_key 二选一与 api_key 搭配，用于拼 api_key
            app_key: 同上
            timeout: 请求超时（秒），拼进 api_key 时用
        """
        self.model_id = model_id
        self.base_url = base_url or os.getenv(
            "TRPC_OPENAI_BASE_URL",
            "http://trpc-gpt-eval.production.polaris:8080/v1",
        )
        if api_key:
            self.api_key = api_key
        elif app_id and app_key:
            self.api_key = build_trpc_api_key(
                app_id, app_key, timeout=timeout, model=model_id
            )
        else:
            self.api_key = os.getenv("TRPC_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TrpcOpenAIClient 需要 api_key，或 (app_id, app_key)，或环境变量 TRPC_OPENAI_API_KEY / OPENAI_API_KEY"
            )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(float(timeout), connect=30.0),
            max_retries=0,
        )

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        与 LLMClient.chat 一致。若网关支持 tools，会返回 tool_calls。
        """
        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message

                result = {
                    "role": "assistant",
                    "content": message.content or "",
                    "finish_reason": getattr(choice, "finish_reason", "stop"),
                }
                if hasattr(message, "reasoning_content") and message.reasoning_content:
                    result["reasoning_content"] = message.reasoning_content

                if getattr(message, "tool_calls", None):
                    result["tool_calls"] = [
                        {
                            "id": getattr(tc, "id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in message.tool_calls
                    ]

                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "TrpcOpenAIClient attempt %s/%s failed: %s",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        logger.error("TrpcOpenAIClient failed after %s attempts: %s", MAX_RETRIES, last_error)
        return {
            "role": "assistant",
            "content": f"Error: {str(last_error)}",
            "finish_reason": "error",
        }
