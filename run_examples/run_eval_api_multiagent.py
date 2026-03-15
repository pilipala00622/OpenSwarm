"""
Example: 用 llm.py（eval API）跑「多智能体编排」代码路径（退化为单轮文本）

与 run_kimi_kimi 相同的编排：主智能体带 create_subagent、assign_task、search。
但当前 data_eval/混元 接口不支持 tools，API 不会返回 tool_calls，
因此主智能体会直接以一段文本回复并结束，不会真正创建/调用子智能体。

用途：验证多智能体流程在 eval API 下的降级行为；真正多智能体需用
支持 function calling 的接口（如 run_kimi_kimi.py + KIMI_API_KEY）。

Usage:
    cd Agent_swarm && python swarm_example/run_examples/run_eval_api_multiagent.py
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
from swarm_example.utils.eval_llm_client import EvalLLMClient
from swarm_example.tool.search import SearchTool
from swarm_example.swarm_tool.create_subagent import CreateSubagentTool
from swarm_example.swarm_tool.task import TaskTool

EVAL_MODEL_NAME = "kimi-k2.5"

SYSTEM_PROMPT = """你是一个编排智能体，可以创建子智能体并分配任务。

可用工具：search（搜索）、create_subagent（创建子智能体）、assign_task（给子智能体派任务）。
遇到可拆分的任务时，先 create_subagent 再 assign_task，最后汇总结果。"""


async def main():
    llm_client = EvalLLMClient(model_name=EVAL_MODEL_NAME)

    config = AgentConfig(
        name="main_agent",
        system_prompt=SYSTEM_PROMPT,
        model_id=EVAL_MODEL_NAME,
        temperature=0.7,
    )

    search_tool = SearchTool()
    agent_registry = {}
    create_subagent_tool = CreateSubagentTool(agent_registry)
    task_tool = TaskTool(agent_registry, max_steps=20)
    tools = [search_tool, create_subagent_tool, task_tool]

    agent = Agent(config=config, tools=tools, llm_client=llm_client)
    task_tool.set_parent_agent(agent)
    task_tool.set_parent_tools(tools)

    rollout = MainRollout(RolloutConfig(max_steps=50, terminal_mode=True))

    query = "请先创建一个专门做调研的子智能体，再让它总结一下 2024 年 AI 领域的三件大事，最后把结果告诉我。"
    print("[说明] 当前 eval API 不支持 tool_calls，主智能体不会真正调用 create_subagent/assign_task，")
    print("       会直接以一段文本回复并结束（退化为单轮）。真正多智能体需用支持 tools 的接口（如 KIMI_API_KEY）。")
    print()
    print(f"Query: {query}")
    print("=" * 60)
    print(f"Backend: llm.get_model_answer (model={EVAL_MODEL_NAME})")
    print("=" * 60)

    result = await rollout.run(agent, query)

    print("\n" + "=" * 60)
    print("===== RESULT =====")
    print(f"Status: {result.status.value}")
    print(f"Steps: {result.steps}")
    print(f"Sub-agents used: {len(result.subs)}  (预期为 0，因 API 未返回 tool_calls)")
    if result.final_response:
        r = (result.final_response or "")[:600]
        print(f"Response: {r}{'...' if len(result.final_response or '') > 600 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
