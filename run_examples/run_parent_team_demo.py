"""
Demo: compare parent mode vs team mode without external LLM APIs.

This script uses a tiny in-memory demo LLM client so the behavior is deterministic:
- parent mode: sub-agent returns directly to the lead
- team mode: sub-agents broadcast updates to the lead inbox, and the lead can
  inspect inbox state and clean up team resources

Usage:
    python3 run_examples/run_parent_team_demo.py
"""

import asyncio
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional


def _load_open_swarm_package():
    """Load the repo root as a package named open_swarm."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    init_path = os.path.join(root_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "open_swarm",
        init_path,
        submodule_search_locations=[root_dir],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["open_swarm"] = module
    assert spec.loader is not None
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module


open_swarm = _load_open_swarm_package()

Agent = open_swarm.Agent
AgentConfig = open_swarm.AgentConfig
BackgroundOutputTool = open_swarm.BackgroundOutputTool
TaskStore = open_swarm.TaskStore
TaskTool = open_swarm.TaskTool
TeamCleanupTool = open_swarm.TeamCleanupTool
TeamLeadInboxTool = open_swarm.TeamLeadInboxTool
TeamMembersTool = open_swarm.TeamMembersTool
TeamStatusTool = open_swarm.TeamStatusTool
TeamMailbox = open_swarm.TeamMailbox


class DemoLLMClient:
    """Small deterministic client used to demonstrate mode differences."""

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Dict[str, Any]:
        del temperature, max_tokens, kwargs

        tool_names = [
            tool["function"]["name"]
            for tool in (tools or [])
            if tool.get("type") == "function"
        ]
        user_messages = [m for m in messages if m.get("role") == "user"]
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        prompt = user_messages[-1]["content"] if user_messages else "unknown task"
        prompt = prompt.replace("=== Parent Context Start ===", "").replace("=== Parent Context End ===", "").strip()

        if "team_message" in tool_names and not tool_messages:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_team_message",
                        "type": "function",
                        "function": {
                            "name": "team_message",
                            "arguments": json.dumps({
                                "content": f"Working on: {prompt}",
                                "broadcast": True,
                            }),
                        },
                    }
                ],
                "finish_reason": "tool_calls",
            }

        if "team_message" in tool_names:
            await asyncio.sleep(0.35)
            return {
                "content": f"TEAM MODE: finished '{prompt}' after sharing progress with the team.",
                "finish_reason": "stop",
            }

        return {
            "content": f"PARENT MODE: finished '{prompt}' and returned directly to the lead.",
            "finish_reason": "stop",
        }


async def run_parent_demo():
    print("=" * 72)
    print("PARENT MODE DEMO")
    print("=" * 72)

    llm_client = DemoLLMClient()
    lead_config = AgentConfig(
        name="lead_parent",
        system_prompt="You are the lead agent.",
        model_id="demo-llm",
        subagent_mode="parent",
    )
    task_tool = TaskTool(agent_registry={}, max_steps=4)
    lead_tools = [task_tool]
    lead_agent = Agent(config=lead_config, tools=lead_tools, llm_client=llm_client)
    task_tool.set_parent_agent(lead_agent)
    task_tool.set_parent_tools(lead_tools)

    result = await task_tool.execute(
        prompt="Review parser edge cases",
        subagent_mode="parent",
    )
    print(result.to_str())
    print()


async def run_team_demo():
    print("=" * 72)
    print("TEAM MODE DEMO")
    print("=" * 72)

    llm_client = DemoLLMClient()
    shutil.rmtree(".demo_agent_tasks", ignore_errors=True)
    shutil.rmtree(".demo_agent_team", ignore_errors=True)
    task_store = TaskStore(store_dir=".demo_agent_tasks")
    mailbox = TeamMailbox(base_dir=".demo_agent_team")

    lead_config = AgentConfig(
        name="lead_team",
        system_prompt="You are the lead agent.",
        model_id="demo-llm",
        subagent_mode="team",
    )
    task_tool = TaskTool(
        agent_registry={},
        task_store=task_store,
        max_steps=4,
        team_mailbox=mailbox,
    )
    lead_inbox_tool = TeamLeadInboxTool(task_tool)
    team_members_tool = TeamMembersTool(task_tool)
    team_status_tool = TeamStatusTool(task_tool)
    team_cleanup_tool = TeamCleanupTool(task_tool)
    background_output_tool = BackgroundOutputTool(task_tool)
    lead_tools = [
        task_tool,
        background_output_tool,
        lead_inbox_tool,
        team_members_tool,
        team_status_tool,
        team_cleanup_tool,
    ]
    lead_agent = Agent(config=lead_config, tools=lead_tools, llm_client=llm_client)
    task_tool.set_parent_agent(lead_agent)
    task_tool.set_parent_tools(lead_tools)

    task_a = task_store.create("Implement parser", description="Teammate A handles parser implementation")
    task_b = task_store.create("Write tests", description="Teammate B handles regression tests")

    launch_a = await task_tool.execute(
        prompt="Implement parser task",
        task_id=task_a.id,
        subagent_mode="team",
        run_in_background=True,
    )
    launch_b = await task_tool.execute(
        prompt="Write regression tests task",
        task_id=task_b.id,
        subagent_mode="team",
        run_in_background=True,
    )

    print(launch_a.to_str())
    print(launch_b.to_str())
    print()

    await asyncio.sleep(0.1)

    print("Registered team members:")
    members = await team_members_tool.execute()
    print(members.to_str())
    print()

    print("Team status while teammates are running:")
    status_running = await team_status_tool.execute()
    print(status_running.to_str())
    print()

    print("Attempt cleanup while teammates are still running:")
    premature_cleanup = await team_cleanup_tool.execute()
    print(premature_cleanup.to_str())
    print()

    for task_id in ("subagent_1", "subagent_2"):
        while task_tool.is_background_running(task_id):
            await asyncio.sleep(0.1)
        result = await background_output_tool.execute(task_id=task_id)
        print(result.to_str())
        print()

    print("Lead inbox after teammates broadcast updates:")
    lead_messages = await lead_inbox_tool.execute()
    print(lead_messages.to_str())
    print()

    print("Team status after teammates finish:")
    status_finished = await team_status_tool.execute()
    print(status_finished.to_str())
    print()

    print("Cleanup after teammates finish:")
    cleanup_result = await team_cleanup_tool.execute()
    print(cleanup_result.to_str())
    print()


async def main():
    await run_parent_demo()
    await run_team_demo()


if __name__ == "__main__":
    asyncio.run(main())
