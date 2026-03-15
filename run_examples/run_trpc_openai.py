"""
Example: 用 OpenAI 客户端对接 trpc-gpt-eval 网关跑 swarm

api_key 格式：APP_ID:APP_KEY?provider=moonshot&timeout=600&model=模型名
base_url：http://trpc-gpt-eval.production.polaris:8080/v1

若网关支持 tools（function calling），可跑多智能体；若不支持则退化为单轮文本。

环境变量二选一：
  A) TRPC_OPENAI_API_KEY = "APP_ID:APP_KEY?provider=moonshot&timeout=600&model=kimi-k2-0905-preview"
  B) TRPC_APP_ID + TRPC_APP_KEY（脚本会拼成上述格式）

可选：TRPC_OPENAI_BASE_URL（默认即 trpc 网关）

Usage:
    export TRPC_APP_ID="your-app-id"
    export TRPC_APP_KEY="your-app-key"
    cd Agent_swarm && python swarm_example/run_examples/run_trpc_openai.py
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

MODEL_ID = os.environ.get("TRPC_MODEL", "kimi-k2-0905-preview")


async def main():
    api_key = os.environ.get("TRPC_OPENAI_API_KEY")
    if not api_key:
        app_id = os.environ.get("TRPC_APP_ID")
        app_key = os.environ.get("TRPC_APP_KEY")
        if not app_id or not app_key:
            print("Error: 请设置 TRPC_OPENAI_API_KEY 或 TRPC_APP_ID + TRPC_APP_KEY")
            sys.exit(1)
        api_key = build_trpc_api_key(app_id, app_key, timeout=600, model=MODEL_ID)

    client = TrpcOpenAIClient(
        model_id=MODEL_ID,
        api_key=api_key,
        base_url=os.environ.get("TRPC_OPENAI_BASE_URL"),
        timeout=600,
    )

    config = AgentConfig(
        name="trpc_agent",
        system_prompt="You are a helpful assistant.",
        model_id=MODEL_ID,
        temperature=0.7,
    )

    # 先跑单轮对话验证连通性
    agent = Agent(config=config, tools=[], llm_client=client)
    rollout = MainRollout(RolloutConfig(max_steps=10, terminal_mode=True))

    query = "用一两句话介绍你能做什么。"
    print("Backend: OpenAI client -> trpc-gpt-eval")
    print(f"Model: {MODEL_ID}")
    print("=" * 60)
    result = await rollout.run(agent, query)
    print("=" * 60)
    print(f"Status: {result.status.value}, Steps: {result.steps}")
    if result.final_response:
        print(f"Response: {(result.final_response or '')[:400]}...")


if __name__ == "__main__":
    asyncio.run(main())
