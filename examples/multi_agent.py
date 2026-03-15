"""Multi-Agent Example - Full feature showcase (v0.3.0)

Demonstrates all features aligned with Oh-My-OpenCode:
- Category system (visual-engineering, deep, quick, etc.)
- Tool permission control (blacklist / whitelist)
- Background task execution and retrieval
- Task dependency tracking with automatic parallel scheduling
- Handoff mechanism for cross-session continuity
- Parallel tool execution (asyncio.gather)
- Error recovery with checkpoints
- External memory / context compression (Anthropic-style)
- Dynamic scaling rules
- fork_context support
- Structured execution tracing
"""

import asyncio
import logging
import sys
import os

# Add parent directory to path for imports (when running from examples/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_swarm import (
    Agent,
    AgentConfig,
    MainRollout,
    RolloutConfig,
    SearchTool,
    VerifyTool,
    CreateSubagentTool,
    TaskTool,
    BackgroundOutputTool,
    BackgroundCancelTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    HandoffTool,
    CategoryRegistry,
    TaskStore,
    HandoffManager,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    """Run a multi-agent example with all v0.3.0 features"""

    # --- Infrastructure setup ---

    # Shared agent registry (for create_subagent / assign_task)
    agent_registry = {}

    # Category registry (with optional custom category)
    category_registry = CategoryRegistry(custom_categories={
        "research": {
            "description": "In-depth research with broad search strategy",
            "temperature": 0.5,
            "prompt_append": (
                "You are a thorough researcher. "
                "Use broad-to-narrow search strategy. "
                "Cross-reference multiple sources before concluding."
            ),
        }
    })

    # Storage directories
    storage_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "result"
    )
    os.makedirs(storage_dir, exist_ok=True)

    # Task store for dependency tracking
    task_store = TaskStore(store_dir=os.path.join(storage_dir, "tasks"))

    # Handoff manager for cross-session continuity
    handoff_manager = HandoffManager(handoff_dir=os.path.join(storage_dir, "handoffs"))

    # --- Create tools ---

    search_tool = SearchTool()
    verify_tool = VerifyTool()
    create_subagent = CreateSubagentTool(agent_registry)

    # Task tool with category support
    task_tool = TaskTool(
        agent_registry=agent_registry,
        max_steps=15,
        category_registry=category_registry,
        task_store=task_store,
    )

    # Background tools
    bg_output = BackgroundOutputTool(task_tool)
    bg_cancel = BackgroundCancelTool(task_tool)

    # Task management tools
    task_create = TaskCreateTool(task_store)
    task_get = TaskGetTool(task_store)
    task_list = TaskListTool(task_store)
    task_update = TaskUpdateTool(task_store)

    # Handoff tool
    handoff_tool = HandoffTool(
        handoff_manager=handoff_manager,
        task_store=task_store,
    )

    # --- Create main agent ---

    config = AgentConfig(
        name="orchestrator",
        system_prompt=(
            "You are an orchestrator agent that delegates tasks to specialized sub-agents.\n"
            "\n"
            "Your workflow:\n"
            "1. Assess query complexity to decide how many sub-agents to use\n"
            "2. Use task_create to plan work items with dependencies\n"
            "3. Use create_subagent to define specialized agents, OR use category parameter\n"
            "4. Use assign_task to delegate work (supports category, background execution)\n"
            "5. Use verify_result to validate important findings\n"
            "6. Use handoff to save state for cross-session continuity\n"
            "7. Synthesize results from sub-agents to answer the user\n"
            "\n"
            "Available categories for delegation:\n"
            "- quick: fast, trivial tasks\n"
            "- deep: thorough, autonomous problem-solving\n"
            "- research: in-depth research with cross-referencing (custom)\n"
            "- visual-engineering: frontend/UI/UX work\n"
            "- writing: documentation and prose\n"
            "- ultrabrain: deep reasoning and analysis\n"
            "\n"
            "You can launch background tasks with run_in_background=true\n"
            "and retrieve results later with background_output.\n"
            "\n"
            "Be strategic about when to delegate vs handle directly."
        ),
        model_id="kimi-k2-0711-preview",
        temperature=0.7,
        can_delegate=True,  # Orchestrator CAN delegate
    )

    agent = Agent(
        config=config,
        tools=[
            search_tool, verify_tool, create_subagent, task_tool,
            bg_output, bg_cancel,
            task_create, task_get, task_list, task_update,
            handoff_tool,
        ]
    )

    # Set parent references
    task_tool.set_parent_agent(agent)
    task_tool.set_parent_tools([search_tool, verify_tool])

    # Set handoff messages reference (will be updated after rollout starts)
    # This is done by MainRollout internally

    # --- Create rollout ---

    rollout_config = RolloutConfig(
        max_steps=30,
        terminal_mode=True,
        # Error recovery
        max_consecutive_errors=3,
        checkpoint_interval=10,
        # Memory / context compression (Anthropic-style)
        enable_memory=True,
        max_context_messages=50,
        memory_keep_recent=20,
        memory_dir=os.path.join(storage_dir, "memory"),
        enable_llm_summarise=True,
        # Dynamic scaling
        enable_scaling_rules=True,
        # Tracing
        enable_tracing=True,
        trace_output_path=os.path.join(storage_dir, "trace.jsonl"),
    )
    rollout = MainRollout(rollout_config)

    # --- Run ---

    print("\n" + "="*60)
    print("Starting Multi-Agent Example (v0.3.0 — Oh-My-OpenCode aligned)")
    print("="*60)

    user_message = (
        "I want to learn about the latest developments in AI agents. "
        "Please create a research agent to find information about this topic, "
        "and then summarize the key findings for me."
    )

    result = await rollout.run(
        agent=agent,
        initial_message=user_message
    )

    print("\n" + "="*60)
    print("Final Result:")
    print("="*60)
    print(f"Status: {result.status.value}")
    print(f"Steps: {result.steps}")
    print(f"Agents created: {list(agent_registry.keys())}")
    print(f"Categories available: {list(category_registry.list_categories().keys())}")
    print(f"Tasks tracked: {len(task_store.list_tasks())}")
    if result.final_response:
        print(f"\nResponse:\n{result.final_response}")


if __name__ == "__main__":
    asyncio.run(main())
