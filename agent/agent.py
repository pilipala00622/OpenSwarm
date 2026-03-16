"""Agent - Core agent implementation"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set

from ..utils.llm_client import LLMClient
from ..tool.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent.

    Attributes:
        name: Agent identifier.
        system_prompt: System-level instructions for the agent.
        model_id: LLM model to use.
        api_key: API key (defaults to OPENAI_API_KEY env var).
        api_base_url: API base URL (defaults to OPENAI_BASE_URL env var).
        subagent_model_id: Model for sub-agents (defaults to model_id).
        subagent_api_key: API key for sub-agents.
        subagent_api_base_url: Base URL for sub-agents.
        max_tokens: Maximum response tokens.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling parameter.
        reasoning_effort: Reasoning effort level ("low", "medium", "high").
        thinking_budget: Extended thinking token budget (0 = disabled).
        blocked_tools: Tool names to block (blacklist).
        allowed_tools_only: If non-empty, ONLY these tools are allowed (whitelist).
        can_delegate: Whether this agent can delegate to sub-agents.
        subagent_mode: Coordination style for spawned sub-agents ("parent" or "team").
    """
    name: str
    system_prompt: str = "You are a helpful assistant."
    model_id: str = "kimi-k2.5"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    subagent_model_id: Optional[str] = None
    subagent_api_key: Optional[str] = None
    subagent_api_base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    reasoning_effort: str = "medium"
    thinking_budget: int = 0
    blocked_tools: List[str] = field(default_factory=list)
    allowed_tools_only: List[str] = field(default_factory=list)
    can_delegate: bool = True
    subagent_mode: str = "parent"


class Agent:
    """Agent that processes messages and executes tools.

    The Agent is responsible for:
    - Managing conversation with the LLM
    - Executing tool calls (with permission enforcement)
    - Formatting responses

    Tool access control:
    - `blocked_tools`: Blacklist — these tools are hidden from the agent.
    - `allowed_tools_only`: Whitelist — if non-empty, ONLY these tools are available.
    - Blacklist is applied first, then whitelist filters the remainder.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List[BaseTool]] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """Initialize agent.

        Args:
            config: Agent configuration.
            tools: List of tools available to the agent.
            llm_client: Optional custom LLM client (will create one if not provided).
        """
        self.config = config
        # Store all tools (before permission filtering)
        self._all_tools = {tool.name: tool for tool in (tools or [])}
        # Effective tool set (after permission filtering)
        self.tools = self._apply_tool_permissions(self._all_tools)

        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = LLMClient(
                model_id=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base_url,
            )

    def _apply_tool_permissions(
        self, tools: Dict[str, BaseTool]
    ) -> Dict[str, BaseTool]:
        """Apply tool permission rules from config.

        Args:
            tools: Full tool dict.

        Returns:
            Filtered tool dict respecting blocked/allowed rules.
        """
        blocked: Set[str] = set(self.config.blocked_tools)
        allowed: Set[str] = set(self.config.allowed_tools_only)

        result = {}
        for name, tool in tools.items():
            # Blacklist check
            if name in blocked:
                logger.debug(f"Agent '{self.config.name}': tool '{name}' blocked")
                continue
            # Whitelist check (only if whitelist is non-empty)
            if allowed and name not in allowed:
                logger.debug(f"Agent '{self.config.name}': tool '{name}' not in allowed list")
                continue
            result[name] = tool

        # Enforce can_delegate: block delegation tools if disabled
        if not self.config.can_delegate:
            for delegate_tool in ("create_subagent", "assign_task"):
                if delegate_tool in result:
                    result.pop(delegate_tool)
                    logger.debug(
                        f"Agent '{self.config.name}': delegation tool '{delegate_tool}' "
                        f"removed (can_delegate=False)"
                    )

        return result

    def add_tool(self, tool: BaseTool):
        """Add a tool to the agent (subject to permission rules)."""
        self._all_tools[tool.name] = tool
        self.tools = self._apply_tool_permissions(self._all_tools)

    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent."""
        self._all_tools.pop(tool_name, None)
        self.tools.pop(tool_name, None)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-format schemas for all accessible tools."""
        return [tool.to_openai_schema() for tool in self.tools.values()]

    def get_system_message(self) -> Dict[str, str]:
        """Get the system message."""
        return {
            "role": "system",
            "content": self.config.system_prompt
        }

    async def process_message(
        self,
        messages: List[Dict[str, Any]],
        include_system: bool = True,
    ) -> Dict[str, Any]:
        """Process messages and return response.

        Args:
            messages: Conversation messages.
            include_system: Whether to prepend system message.

        Returns:
            Response dict with content and optional tool_calls.
        """
        # Prepare messages (always copy to avoid mutating caller's list)
        if include_system:
            if not messages or messages[0].get("role") != "system":
                messages = [self.get_system_message()] + messages
            else:
                messages = list(messages)  # Fix #4: consistent copy behaviour
        else:
            messages = list(messages)

        # Get tool schemas (only accessible tools)
        tool_schemas = self.get_tool_schemas() if self.tools else None

        # Build optional kwargs for config fields (Fix #8)
        extra_kwargs = {}
        if self.config.top_p != 1.0:
            extra_kwargs["top_p"] = self.config.top_p
        if self.config.reasoning_effort != "medium":
            extra_kwargs["reasoning_effort"] = self.config.reasoning_effort
        if self.config.thinking_budget > 0:
            extra_kwargs["thinking_budget"] = self.config.thinking_budget

        # Call LLM
        response = await self.llm_client.chat(
            messages=messages,
            tools=tool_schemas,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **extra_kwargs,
        )

        return response

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ToolResult:
        """Execute a single tool.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            ToolResult from tool execution.
        """
        tool = self.tools.get(tool_name)
        if not tool:
            # Check if it exists but is blocked
            if tool_name in self._all_tools:
                return ToolResult(
                    content="",
                    success=False,
                    error=f"Tool '{tool_name}' is blocked for this agent"
                )
            return ToolResult(
                content="",
                success=False,
                error=f"Tool '{tool_name}' not found"
            )

        try:
            logger.info(f"Executing tool: {tool_name} with args: {arguments}")
            result = await tool.execute(**arguments)
            logger.info(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return ToolResult(
                content="",
                success=False,
                error=str(e)
            )

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls in parallel using asyncio.gather.

        All tool calls from a single LLM response are executed concurrently,
        which dramatically speeds up multi-subagent scenarios where the lead
        agent launches several sub-agents at once.

        Args:
            tool_calls: List of tool call dicts from LLM response.

        Returns:
            List of tool results formatted for conversation (order preserved).
        """

        async def _execute_single(tc: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a single tool call and format the result."""
            tool_name = tc["function"]["name"]
            args_str = tc["function"]["arguments"]

            # Parse arguments (Fix #12: report parse failures instead of silent {})
            try:
                arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse arguments for tool '{tool_name}': {e}")
                return {
                    "tool_call_id": tc.get("id", tool_name),
                    "role": "tool",
                    "content": f"Error: Could not parse tool arguments (invalid JSON). Raw: {str(args_str)[:200]}",
                }

            # Execute tool
            result = await self.execute_tool(tool_name, arguments)

            # Format for conversation
            return {
                "tool_call_id": tc.get("id", tool_name),
                "role": "tool",
                "content": result.to_str(),
            }

        # Execute all tool calls concurrently
        gathered = await asyncio.gather(
            *[_execute_single(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        # Handle any exceptions that occurred during parallel execution
        results = []
        for i, item in enumerate(gathered):
            if isinstance(item, Exception):
                logger.error(f"Parallel tool execution failed for call {i}: {item}")
                results.append({
                    "tool_call_id": tool_calls[i].get("id", "error"),
                    "role": "tool",
                    "content": f"Error executing tool: {str(item)}",
                })
            else:
                results.append(item)

        return results
