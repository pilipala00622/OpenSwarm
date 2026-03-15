"""CreateSubagent Tool - Create new sub-agent configurations"""

import logging
from typing import Dict, Any

from ..tool.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class CreateSubagentTool(BaseTool):
    """Tool for creating new sub-agent configurations

    This tool allows the main agent to define specialized sub-agents
    with custom system prompts. Created agents can then be used
    with the Task tool.

    TODO (Arch #24): The agent_registry is currently in-memory only.
    For cross-session persistence (e.g. Handoff resumption), the registry
    should be persisted to disk (JSON file) alongside the handoff data.
    """

    def __init__(self, agent_registry: Dict[str, Dict[str, Any]]):
        """Initialize CreateSubagent tool

        Args:
            agent_registry: Shared registry to store agent configurations
        """
        self.agent_registry = agent_registry

    @property
    def name(self) -> str:
        return "create_subagent"

    @property
    def description(self) -> str:
        return (
            "Create a custom subagent with specific system prompt and name for reuse.\n"
            "Guidelines:\n"
            "- Only create sub-agents when the task genuinely benefits from delegation.\n"
            "- For simple queries, answer directly without creating sub-agents.\n"
            "- Give each sub-agent a focused, specific role (e.g., 'physics_researcher', 'code_reviewer').\n"
            "- The system prompt should clearly define the agent's expertise and boundaries."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique name for this agent configuration (e.g., 'researcher', 'code_reviewer')"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "System prompt defining the agent's role, capabilities, and boundaries"
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "If true, overwrite existing agent with the same name (default: false)"
                }
            },
            "required": ["name", "system_prompt"]
        }

    async def execute(self, name: str, system_prompt: str, overwrite: bool = False) -> ToolResult:
        """Create a new sub-agent configuration

        Args:
            name: Unique identifier for the agent
            system_prompt: System prompt for the agent
            overwrite: If True, update existing agent (Fix #13)

        Returns:
            ToolResult indicating success or failure
        """
        try:
            # Check if name already exists
            if name in self.agent_registry and not overwrite:
                return ToolResult(
                    content=f"Agent '{name}' already exists. Set overwrite=true to update it, or choose a different name.",
                    success=True
                )

            action = "Updated" if name in self.agent_registry else "Created"

            # Store configuration
            self.agent_registry[name] = {
                "system_prompt": system_prompt
            }

            logger.info(f"{action} sub-agent: {name}")

            return ToolResult(
                content=f"Successfully {action.lower()} agent '{name}'. You can now use it with the task tool by specifying agent='{name}'.",
                success=True
            )

        except Exception as e:
            logger.error(f"Failed to create sub-agent: {e}")
            return ToolResult(
                content="",
                success=False,
                error=str(e)
            )
