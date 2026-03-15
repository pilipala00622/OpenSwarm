"""
Example: Run with kimi-k2.5 as main agent and qwen as sub-agent

Required environment variables:
- KIMI_API_KEY: API key for kimi (api.moonshot.cn)
- OPENAI_API_KEY: API key for qwen (openai.app.msh.team)
- SERPER_API_KEY: API key for Serper search (optional)

Usage:
    export KIMI_API_KEY="your-kimi-api-key"
    export OPENAI_API_KEY="your-openai-compatible-api-key"
    export SERPER_API_KEY="your-serper-api-key"  # optional
    python run_kimi_qwen.py
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import Agent, AgentConfig
from rollout.main_rollout import MainRollout
from rollout.base import RolloutConfig
from tool.search import SearchTool
from swarm_tool.create_subagent import CreateSubagentTool
from swarm_tool.task import TaskTool

SYSTEM_PROMPT = """You are an autonomous AI agent designed for complex, multi-step tasks.

## Core Principles
1. **Think step-by-step**: Break down complex tasks into manageable steps
2. **Use tools wisely**: Choose the most appropriate tool for each step
3. **Verify results**: Check outputs before proceeding
4. **Iterate as needed**: Refine approach based on results
5. **Report findings**: Summarize key findings and actions taken

## Available Tools
- search: Search the web for information
- create_subagent: Create specialized sub-agents for specific tasks
- assign_task: Delegate tasks to created sub-agents

## Sub-Agent Delegation Guidelines
When a task can be parallelized or requires specialized focus:
1. Use create_subagent to create agents with specific expertise
2. Use assign_task to delegate work to these agents
3. Synthesize results from multiple sub-agents

## Response Format
- Start with understanding the task
- Execute required steps using appropriate tools
- End with a clear summary of results and findings"""


async def main():
    # Get API keys from environment
    kimi_api_key = os.environ.get("KIMI_API_KEY")
    qwen_api_key = os.environ.get("OPENAI_API_KEY")

    if not kimi_api_key:
        print("Error: KIMI_API_KEY environment variable is required")
        print("Usage: export KIMI_API_KEY='your-api-key'")
        sys.exit(1)

    if not qwen_api_key:
        print("Error: OPENAI_API_KEY environment variable is required for sub-agent")
        print("Usage: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # kimi + qwen config
    config = AgentConfig(
        name="main_agent",
        system_prompt=SYSTEM_PROMPT,
        model_id="kimi-k2.5",
        api_key=kimi_api_key,
        api_base_url="https://api.moonshot.cn/v1",
        subagent_model_id="qwen2.5-72b-instruct",
        subagent_api_key=qwen_api_key,
        subagent_api_base_url="https://openai.app.msh.team/v1",
        temperature=1.0,  # kimi-k2.5 requires temperature=1.0
    )

    # Create tools
    search_tool = SearchTool()
    agent_registry = {}
    create_subagent_tool = CreateSubagentTool(agent_registry)
    task_tool = TaskTool(agent_registry, max_steps=20)

    tools = [search_tool, create_subagent_tool, task_tool]

    # Create agent
    agent = Agent(config=config, tools=tools)

    # Set parent references
    task_tool.set_parent_agent(agent)
    task_tool.set_parent_tools(tools)

    # Create rollout with storage
    storage_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "result",
        "kimi_qwen_result.jsonl"
    )

    rollout_config = RolloutConfig(
        max_steps=50,
        terminal_mode=True,
        storage_path=storage_path,
        print_tool_calls=True,
        print_tool_results=True,
    )

    rollout = MainRollout(rollout_config)

    # Example query
    query = "请帮我搜集并整理一些专家级别的 AI 基准测试（benchmarks），涵盖物理、化学、生物、医学和数学等领域，重点关注大学本科、研究生或科研级难度的评测; 多次调用subagent给我一个全面的结果"

    print(f"Query: {query}")
    print("=" * 60)
    print(f"Config: main=kimi-k2.5, sub=qwen2.5-72b-instruct")
    print(f"Storage: {storage_path}")
    print("=" * 60)

    result = await rollout.run(agent, query)

    print("\n" + "=" * 60)
    print("===== FINAL RESULT =====")
    print(f"Status: {result.status.value}")
    print(f"Steps: {result.steps}")
    print(f"Sub-agents used: {len(result.subs)}")
    print(f"Result saved to: {storage_path}")


if __name__ == "__main__":
    asyncio.run(main())
