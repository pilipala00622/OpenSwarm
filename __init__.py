"""Open Swarm - Multi-Agent Rollout Framework

A lightweight, extensible framework for orchestrating multiple AI agents
to collaboratively solve complex tasks.

Features (v0.4.1):
- Category system for task-specific agent profiles
- Tool permission control (blacklist / whitelist)
- Background task execution and retrieval
- Task dependency tracking with automatic parallel scheduling
- Handoff mechanism for cross-session continuity
- External memory with LLM summarisation
- Structured execution tracing

Example:
    from open_swarm import Agent, AgentConfig, MainRollout, RolloutConfig
    from open_swarm import CategoryRegistry, TaskStore, HandoffManager
    from open_swarm.tool import SearchTool, VerifyTool
    from open_swarm.swarm_tool import (
        CreateSubagentTool, TaskTool,
        BackgroundOutputTool, BackgroundCancelTool,
        TaskClaimTool, TaskCreateTool, TaskListTool, TaskUpdateTool, TaskGetTool,
        HandoffTool,
        TeamCleanupTool, TeamInboxTool, TeamLeadInboxTool, TeamLeadMessageTool,
        TeamMembersTool, TeamMessageTool, TeamStatusTool,
    )

    # Setup
    agent_registry = {}
    category_registry = CategoryRegistry()
    task_store = TaskStore()

    # Create tools
    task_tool = TaskTool(agent_registry, category_registry=category_registry, task_store=task_store)
    bg_output = BackgroundOutputTool(task_tool)
    bg_cancel = BackgroundCancelTool(task_tool)

    # Create agent with category-based delegation
    config = AgentConfig(name="main", system_prompt="You are an orchestrator.")
    agent = Agent(config, tools=[search, task_tool, bg_output, bg_cancel])

    # Run
    rollout = MainRollout(RolloutConfig(max_steps=30))
    result = await rollout.run(agent, "Research AI safety with parallel sub-agents")
"""

from .agent import Agent, AgentConfig
from .rollout import (
    BaseRollout,
    MainRollout,
    SubRollout,
    SubRolloutConfig,
    RolloutConfig,
    RolloutResult,
    RolloutStatus,
)
from .tool import BaseTool, ToolResult, RetrieveContextTool, SearchTool, VerifyTool
from .swarm_tool import (
    CreateSubagentTool,
    TaskTool,
    BackgroundOutputTool,
    BackgroundCancelTool,
    TaskClaimTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    HandoffTool,
    TeamCleanupTool,
    TeamInboxTool,
    TeamLeadInboxTool,
    TeamLeadMessageTool,
    TeamMembersTool,
    TeamMessageTool,
    TeamStatusTool,
)
from .utils import (
    LLMClient,
    AgentMemory,
    RolloutTracer,
    CategoryRegistry,
    CategoryConfig,
    BUILTIN_CATEGORIES,
    KnowledgeEngine,
    TaskStore,
    Task,
    HandoffManager,
    HandoffDocument,
    TeamMailbox,
)

__version__ = "0.4.1"
__author__ = "Open Swarm Contributors"

__all__ = [
    # Agent
    "Agent",
    "AgentConfig",
    # Rollout
    "BaseRollout",
    "MainRollout",
    "SubRollout",
    "SubRolloutConfig",
    "RolloutConfig",
    "RolloutResult",
    "RolloutStatus",
    # Tool
    "BaseTool",
    "ToolResult",
    "RetrieveContextTool",
    "SearchTool",
    "VerifyTool",
    # Swarm Tools
    "CreateSubagentTool",
    "TaskTool",
    "BackgroundOutputTool",
    "BackgroundCancelTool",
    "TaskClaimTool",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskUpdateTool",
    "HandoffTool",
    "TeamCleanupTool",
    "TeamInboxTool",
    "TeamLeadInboxTool",
    "TeamLeadMessageTool",
    "TeamMembersTool",
    "TeamMessageTool",
    "TeamStatusTool",
    # Utils
    "LLMClient",
    "AgentMemory",
    "RolloutTracer",
    "CategoryRegistry",
    "CategoryConfig",
    "BUILTIN_CATEGORIES",
    "KnowledgeEngine",
    "TaskStore",
    "Task",
    "HandoffManager",
    "HandoffDocument",
    "TeamMailbox",
]
