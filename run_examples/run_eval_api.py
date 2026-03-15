"""
Example: 用 llm.py（data_eval / 混元）API 跑 swarm 单智能体

不设 KIMI_API_KEY，改用 Agent_swarm/llm.py 的 get_model_answer(model_name, prompt, history)。
需在 Agent_swarm 目录或设置 PYTHONPATH 能 import 到 llm。

当前 data_eval/混元 不支持 tools，因此只跑「单智能体、无工具」对话；
模型名用 llm.py 里已有的，如 kimi-k2.5、gpt4o、hunyuan-t1-online-latest 等。

Usage:
    cd Agent_swarm && python swarm_example/run_examples/run_eval_api.py
    或
    cd Agent_swarm/swarm_example && pip install -e . && python run_examples/run_eval_api.py
"""

import asyncio
import os
import sys

# 把 Agent_swarm 加入 path，便于 import llm 和 swarm_example 子包
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
from swarm_example.utils.eval_llm_client import EvalLLMClient

# 仅测试 kimi-k2.5（llm.py COMMON_MODEL_MARKERS）
EVAL_MODEL_NAME = "kimi-k2.5"


async def main():
    # 使用 EvalLLMClient 接入 llm.get_model_answer
    llm_client = EvalLLMClient(model_name=EVAL_MODEL_NAME)

    config = AgentConfig(
        name="eval_agent",
        system_prompt="你是一个有用的助手。请简洁回答。",
        model_id=EVAL_MODEL_NAME,  # 仅用于显示，实际请求走 llm_client
        temperature=0.7,
    )

    # 不传 tools：当前 eval API 不支持 tool_calls，只做纯对话
    agent = Agent(config=config, tools=[], llm_client=llm_client)

    rollout = MainRollout(RolloutConfig(max_steps=10, terminal_mode=True))

    query = "用一两句话介绍一下你自己能做什么。"
    print(f"Query: {query}")
    print("=" * 60)
    print(f"Backend: llm.get_model_answer (model={EVAL_MODEL_NAME})")
    print("=" * 60)

    result = await rollout.run(agent, query)

    print("\n" + "=" * 60)
    print("===== RESULT =====")
    print(f"Status: {result.status.value}")
    print(f"Steps: {result.steps}")
    if result.final_response:
        print(f"Response: {result.final_response[:500]}{'...' if len(result.final_response or '') > 500 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
