"""
Live demo: run team-lite orchestration against a real function-calling model.

Requirements:
- OPENAI_API_KEY must be set for your OpenAI-compatible provider
- Optional OPENAI_BASE_URL for non-default providers
- Optional OPEN_SWARM_MODEL to override the default model

Usage:
    export OPENAI_API_KEY="sk-..."
    export OPEN_SWARM_MODEL="gpt-4o-mini"   # or another function-calling model
    python3 run_examples/run_team_live_demo.py
"""

import asyncio
import contextlib
import importlib.util
import io
import os
import shutil
import sys


def _load_open_swarm_package():
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
BackgroundCancelTool = open_swarm.BackgroundCancelTool
LLMClient = open_swarm.LLMClient
MainRollout = open_swarm.MainRollout
RolloutConfig = open_swarm.RolloutConfig
TaskCreateTool = open_swarm.TaskCreateTool
TaskGetTool = open_swarm.TaskGetTool
TaskListTool = open_swarm.TaskListTool
TaskStore = open_swarm.TaskStore
TaskTool = open_swarm.TaskTool
TaskUpdateTool = open_swarm.TaskUpdateTool
TeamCleanupTool = open_swarm.TeamCleanupTool
TeamLeadInboxTool = open_swarm.TeamLeadInboxTool
TeamLeadMessageTool = open_swarm.TeamLeadMessageTool
TeamMailbox = open_swarm.TeamMailbox


MODEL_ID = os.getenv("OPEN_SWARM_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are the lead agent for a team-lite orchestration workflow.

Your job is to coordinate teammates using tools, not to solve everything alone.

Available workflow:
1. Create managed tasks with task_create when the work can be split.
2. Launch teammates with assign_task using subagent_mode='team' and run_in_background=true.
3. After teammates are launched, use team_lead_message to broadcast expectations or redirect work.
4. Use background_output to wait for teammate results.
5. Use team_lead_inbox to read teammate messages sent during execution.
6. Synthesize the final result for the user.
7. Use team_cleanup after the work is finished.

Important rules:
- Prefer 2 teammates for this demo.
- Use team mode, not parent mode.
- Do not call search; rely on reasoning and the local task tools.
- Keep the final answer concise and structured.
"""

QUERY = """Plan a small refactor for a TODO-tracking CLI.

Please:
1. Create two managed tasks:
   - one teammate proposes parser/refactor changes
   - one teammate proposes test strategy and regression coverage
2. Spawn both teammates in team mode in the background.
3. After they start, send a short broadcast using team_lead_message asking them to be concise and actionable.
4. Wait for their results using background_output.
5. Read team_lead_inbox to capture their intermediate messages.
6. Produce a combined recommendation.
7. Clean up the team with team_cleanup.
"""


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for run_team_live_demo.py")

    shutil.rmtree(".live_demo_tasks", ignore_errors=True)
    shutil.rmtree(".live_demo_team", ignore_errors=True)

    llm_client = LLMClient(model_id=MODEL_ID)
    task_store = TaskStore(store_dir=".live_demo_tasks")
    team_mailbox = TeamMailbox(base_dir=".live_demo_team")
    agent_registry = {}

    task_tool = TaskTool(
        agent_registry=agent_registry,
        task_store=task_store,
        max_steps=8,
        team_mailbox=team_mailbox,
    )
    background_output = BackgroundOutputTool(task_tool)
    background_cancel = BackgroundCancelTool(task_tool)
    task_create = TaskCreateTool(task_store)
    task_get = TaskGetTool(task_store)
    task_list = TaskListTool(task_store)
    task_update = TaskUpdateTool(task_store)
    team_lead_inbox = TeamLeadInboxTool(task_tool)
    team_lead_message = TeamLeadMessageTool(task_tool)
    team_cleanup = TeamCleanupTool(task_tool)

    tools = [
        task_tool,
        background_output,
        background_cancel,
        task_create,
        task_get,
        task_list,
        task_update,
        team_lead_inbox,
        team_lead_message,
        team_cleanup,
    ]

    agent = Agent(
        config=AgentConfig(
            name="live_lead",
            system_prompt=SYSTEM_PROMPT,
            model_id=MODEL_ID,
            temperature=0.2,
            subagent_mode="team",
        ),
        tools=tools,
        llm_client=llm_client,
    )
    task_tool.set_parent_agent(agent)
    task_tool.set_parent_tools(tools)

    rollout = MainRollout(RolloutConfig(
        max_steps=20,
        terminal_mode=True,
        enable_memory=False,
        enable_tracing=True,
    ))

    print(f"Model: {MODEL_ID}")
    print("=" * 72)
    result = await rollout.run(agent, QUERY)
    print("\n" + "=" * 72)
    print("FINAL STATUS:", result.status.value)
    print("STEPS:", result.steps)
    if result.final_response:
        print("\nFINAL RESPONSE:\n")
        print(result.final_response)
    if result.subs:
        print(f"\nSUB-AGENTS EXECUTED: {len(result.subs)}")


if __name__ == "__main__":
    asyncio.run(main())
