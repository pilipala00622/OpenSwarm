"""
Example: 用 trpc 网关 + CodeRunner 工具跑「编程判断素数」示例

请求格式与示例一致：
  model: kimi-k2.5（参考 llm.py COMMON_MODEL_MARKERS）
  messages: [{"role": "user", "content": "编程判断 3214567 是否是素数。"}]
  tools: [CodeRunner]

鉴权优先顺序：
  1) 环境变量 TRPC_OPENAI_API_KEY 或 TRPC_APP_ID + TRPC_APP_KEY
  2) 未设置时参考 llm.py：使用 llm.APP_ID、llm.APP_KEY，base_url 同 llm（trpc-gpt-eval:8080/v1）

Usage:
    cd Agent_swarm && python swarm_example/run_examples/run_trpc_code_runner.py
"""

import asyncio
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_swarm_example = os.path.dirname(_script_dir)
_agent_swarm = os.path.dirname(_swarm_example)
if _agent_swarm not in sys.path:
    sys.path.insert(0, _agent_swarm)
if _swarm_example not in sys.path:
    sys.path.insert(0, _swarm_example)

from swarm_example.agent.agent import Agent, AgentConfig
from swarm_example.rollout.main_rollout import MainRollout
from swarm_example.rollout.base import RolloutConfig
from swarm_example.utils.trpc_openai_client import TrpcOpenAIClient, build_trpc_api_key
from swarm_example.utils.tools_flow import get_tools_full_logic, print_tools_full_logic
from swarm_example.tool.code_runner import CodeRunnerTool

MODEL_ID = os.environ.get("TRPC_MODEL", "kimi-k2.5")
QUERY = "编程判断 3214567 是否是素数。"

# 与 llm.py 一致：trpc 网关地址（OpenAI 风格用 /v1）
TRPC_BASE_URL_DEFAULT = "http://trpc-gpt-eval.production.polaris:8080/v1"


def _get_api_key_and_base():
    """优先用环境变量，否则参考 llm.py 的 APP_ID / APP_KEY"""
    api_key = os.environ.get("TRPC_OPENAI_API_KEY")
    base_url = os.environ.get("TRPC_OPENAI_BASE_URL") or TRPC_BASE_URL_DEFAULT
    if api_key:
        return api_key, base_url
    app_id = os.environ.get("TRPC_APP_ID")
    app_key = os.environ.get("TRPC_APP_KEY")
    if app_id and app_key:
        return build_trpc_api_key(app_id, app_key, timeout=600, model=MODEL_ID), base_url
    try:
        import llm as _llm
        app_id = getattr(_llm, "APP_ID", None)
        app_key = getattr(_llm, "APP_KEY", None)
        if app_id and app_key:
            return build_trpc_api_key(app_id, app_key, timeout=600, model=MODEL_ID), base_url
    except Exception:
        pass
    return None, base_url


async def main():
    api_key, base_url = _get_api_key_and_base()
    if not api_key:
        print("Error: 请设置 TRPC_OPENAI_API_KEY 或 TRPC_APP_ID + TRPC_APP_KEY，或在 Agent_swarm 下运行以使用 llm.py 的 APP_ID/APP_KEY")
        sys.exit(1)

    client = TrpcOpenAIClient(
        model_id=MODEL_ID,
        api_key=api_key,
        base_url=base_url,
        timeout=600,
    )

    # kimi-k2.5 网关要求 temperature=1
    temperature = 1.0 if "kimi-k2.5" in MODEL_ID else 0.3
    config = AgentConfig(
        name="trpc_code_agent",
        system_prompt="你是一个编程助手。请使用 CodeRunner 工具运行代码完成任务。",
        model_id=MODEL_ID,
        temperature=temperature,
    )

    agent = Agent(
        config=config,
        tools=[CodeRunnerTool()],
        llm_client=client,
    )
    rollout = MainRollout(RolloutConfig(max_steps=20, terminal_mode=True))

    # 输出 tools 全部逻辑（schemas + 流程 + 代码索引）
    print_tools_full_logic(agent)
    print("=" * 60)
    print("Request 格式与示例一致：")
    print(f"  model: {MODEL_ID}")
    print(f"  messages: [{{role: user, content: \"{QUERY}\"}}]")
    print("  tools: [CodeRunner]")
    print("=" * 60)

    result = await rollout.run(agent, QUERY)

    print("=" * 60)
    print(f"Status: {result.status.value}, Steps: {result.steps}")
    if result.final_response:
        print("Response:", (result.final_response or "")[:800])


if __name__ == "__main__":
    asyncio.run(main())
