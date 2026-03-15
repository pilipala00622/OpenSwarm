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
                }
            },
            "required": ["name", "system_prompt"]
        }

    async def execute(self, name: str, system_prompt: str) -> ToolResult:
        """Create a new sub-agent configuration

        Args:
            name: Unique identifier for the agent
            system_prompt: System prompt for the agent

        Returns:
            ToolResult indicating success or failure
        """
        try:
            # Check if name already exists
            if name in self.agent_registry:
                return ToolResult(
                    content=f"Agent '{name}' already exists. Use it directly with the task tool, or choose a different name.",
                    success=True
                )

            # Store configuration
            self.agent_registry[name] = {
                "system_prompt": system_prompt
            }

            logger.info(f"Created sub-agent: {name}")

            return ToolResult(
                content=f"Successfully created agent '{name}'. You can now use it with the task tool by specifying agent='{name}'.",
                success=True
            )

        except Exception as e:
            logger.error(f"Failed to create sub-agent: {e}")
            return ToolResult(
                content="",
                success=False,
                error=str(e)
            )
