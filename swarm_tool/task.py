"""Task Tool - Launch sub-agents to execute tasks with category and background support.

Enhanced features (aligned with Oh-My-OpenCode):
- Category-based agent configuration (visual-engineering, deep, quick, etc.)
- Background execution with result retrieval
- Task dependency tracking via TaskStore
- Tool permission enforcement per category
"""

import asyncio
import itertools
import logging
from typing import Dict, List, Any, Optional

from ..tool.base import BaseTool, ToolResult
from ..agent.agent import Agent, AgentConfig
from ..rollout.sub_rollout import SubRollout, SubRolloutConfig
from ..utils.category import CategoryRegistry, CategoryConfig
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Maximum number of parent context messages to fork to a sub-agent
FORK_CONTEXT_LIMIT = 10


class TaskTool(BaseTool):
    """Tool for launching sub-agents to execute tasks.

    This tool spawns a sub-agent to handle a specific subtask,
    allowing for hierarchical task decomposition and parallel execution.

    New features:
    - `category`: Select a pre-configured agent profile (e.g. "quick", "deep").
    - `run_in_background`: Launch agent in background, get result later.
    - `load_skills`: List of skill names to inject (reserved for future use).
    """

    def __init__(
        self,
        agent_registry: Dict[str, Dict[str, Any]],
        parent_agent: Optional[Agent] = None,
        parent_tools: Optional[List[BaseTool]] = None,
        max_steps: int = 20,
        category_registry: Optional[CategoryRegistry] = None,
        task_store: Optional[Any] = None,
        max_concurrent_subagents: int = 10,
        subtask_timeout: Optional[float] = None,
    ):
        """Initialize Task tool.

        Args:
            agent_registry: Registry of available agent configurations.
            parent_agent: Reference to parent agent (for context forking).
            parent_tools: Tools to pass to sub-agents.
            max_steps: Maximum steps for sub-agent execution.
            category_registry: Category registry for category-based configs.
            task_store: Optional TaskStore for dependency tracking.
            max_concurrent_subagents: Max number of sub-agents running at once (Arch #20).
            subtask_timeout: Max seconds for a single subtask (Arch #18). None = no limit.
        """
        self.agent_registry = agent_registry
        self.parent_agent = parent_agent
        self.parent_tools = parent_tools or []
        self.max_steps = max_steps
        self.category_registry = category_registry or CategoryRegistry()
        self.task_store = task_store
        # Fix #4: atomic counter using itertools.count
        self._counter = itertools.count(1)
        self.sub_results: List[Dict[str, Any]] = []
        self._parent_messages_ref: Optional[List[Dict[str, Any]]] = None
        # Fix #9: snapshot of parent messages at fork time
        self._parent_messages_snapshot: Optional[List[Dict[str, Any]]] = None
        # Background task tracking
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._background_results: Dict[str, Dict[str, Any]] = {}
        # Fix #4: lock for shared mutable state
        self._lock = asyncio.Lock()
        # Arch #20: concurrency limiter
        self._semaphore = asyncio.Semaphore(max_concurrent_subagents)
        # Arch #18: subtask timeout
        self._subtask_timeout = subtask_timeout

    def set_parent_agent(self, agent: Agent):
        """Set the parent agent reference."""
        self.parent_agent = agent

    def set_parent_tools(self, tools: List[BaseTool]):
        """Set tools available to sub-agents."""
        self.parent_tools = tools

    def set_parent_messages(self, messages: List[Dict[str, Any]]):
        """Set reference to parent rollout's message list for fork_context.

        Fix #9: Also provides snapshot_parent_messages() for deterministic forking.
        """
        self._parent_messages_ref = messages

    def snapshot_parent_messages(self):
        """Take a snapshot of parent messages for deterministic fork_context.

        Should be called by the rollout BEFORE executing parallel tool calls,
        so all concurrent sub-agents see the same context (Fix #9).
        """
        if self._parent_messages_ref:
            self._parent_messages_snapshot = list(self._parent_messages_ref)

    @property
    def name(self) -> str:
        return "assign_task"

    @property
    def description(self) -> str:
        categories = self.category_registry.list_categories()
        cat_list = ", ".join(f"`{k}`" for k in categories.keys())
        return (
            "Launch a new agent to execute a specific subtask.\n"
            "Usage notes:\n"
            "1. You can launch multiple agents concurrently for maximum performance.\n"
            "2. When the agent is done, it returns a single message back to you.\n"
            "3. Set fork_context=true to share the current conversation context.\n"
            "4. Use `category` to select a specialized agent profile "
            f"(available: {cat_list}).\n"
            "5. Set `run_in_background=true` to launch in background; "
            "use `background_output` tool to retrieve results later.\n"
            "6. Specify `task_id` to track this as a managed task with dependencies."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": (
                        "Name of the agent to use (must be created first with create_subagent), "
                        "OR set category instead to use a pre-configured profile."
                    )
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed task description for the sub-agent"
                },
                "fork_context": {
                    "type": "boolean",
                    "description": "Whether to include parent conversation context (default: false)"
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Category preset to use (e.g. 'quick', 'deep', 'visual-engineering'). "
                        "Overrides model/temperature/prompt for the sub-agent."
                    )
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Launch in background and continue working (default: false)"
                },
                "load_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of skill names to load for this task (reserved for future)"
                },
                "task_id": {
                    "type": "string",
                    "description": "Managed task ID to track this execution against"
                },
            },
            "required": ["prompt"]
        }

    def _build_forked_context(self) -> Optional[List[Dict[str, Any]]]:
        """Extract recent parent conversation messages for context forking.

        Fix #9: Prefers snapshot (taken before parallel tool calls) over live reference.
        """
        # Use snapshot if available (deterministic), fall back to live reference
        source = self._parent_messages_snapshot or self._parent_messages_ref
        if not source:
            return None

        non_system = [
            m for m in source
            if m.get("role") != "system"
        ]
        if not non_system:
            return None

        return non_system[-FORK_CONTEXT_LIMIT:]

    def _resolve_agent_config(
        self,
        agent_name: Optional[str],
        category: Optional[str],
        prompt: str,
    ) -> tuple:
        """Resolve agent configuration from registry + category.

        Returns:
            (system_prompt, model_id, api_key, base_url, temperature, max_tokens,
             blocked_tools, allowed_tools_only, can_delegate)
        """
        # Start with defaults from parent agent
        if self.parent_agent:
            default_model = self.parent_agent.config.subagent_model_id or self.parent_agent.config.model_id
            default_api_key = self.parent_agent.config.subagent_api_key or self.parent_agent.config.api_key
            default_base_url = self.parent_agent.config.subagent_api_base_url or self.parent_agent.config.api_base_url
            default_temperature = self.parent_agent.config.temperature
            default_max_tokens = self.parent_agent.config.max_tokens
        else:
            default_model = "kimi-k2.5"
            default_api_key = None
            default_base_url = None
            default_temperature = 0.7
            default_max_tokens = 4096

        # Get system prompt from agent registry (or generate from category)
        if agent_name and agent_name in self.agent_registry:
            system_prompt = self.agent_registry[agent_name]["system_prompt"]
        else:
            system_prompt = "You are a specialized sub-agent. Complete the assigned task thoroughly."

        # Apply category overrides
        blocked_tools = []
        allowed_tools_only = []
        can_delegate = False  # Sub-agents can't delegate by default

        if category:
            cat_config = self.category_registry.resolve_for_task(
                category=category,
                default_model=default_model,
                default_temperature=default_temperature,
                default_max_tokens=default_max_tokens,
            )
            # Override with category values
            default_model = cat_config.model or default_model
            default_temperature = cat_config.temperature
            default_max_tokens = cat_config.max_tokens
            blocked_tools = cat_config.blocked_tools
            allowed_tools_only = cat_config.allowed_tools_only
            can_delegate = cat_config.can_delegate

            # Append category prompt
            if cat_config.prompt_append:
                system_prompt += f"\n\n{cat_config.prompt_append}"

            if cat_config.description:
                system_prompt += f"\n\n[Category: {category} — {cat_config.description}]"

        # Add step limit warning
        system_prompt += f"\n\nIMPORTANT:"
        system_prompt += f"\n- You have a MAXIMUM of {self.max_steps} steps to complete your task."
        system_prompt += f"\n- Return results AS SOON AS you have sufficient information."
        system_prompt += f"\n- If approaching step limit, summarize findings and return."

        return (
            system_prompt, default_model, default_api_key, default_base_url,
            default_temperature, default_max_tokens,
            blocked_tools, allowed_tools_only, can_delegate,
        )

    async def _run_subtask(
        self,
        subagent_id: str,
        agent_name: Optional[str],
        prompt: str,
        category: Optional[str],
        fork_context: bool,
        task_id: Optional[str],
    ) -> Dict[str, Any]:
        """Internal: run a subtask and return result dict.

        Fixes applied:
        - Critical #1: Creates independent LLM client when model differs from parent.
        - Arch #18: Wraps execution with timeout.
        - Arch #20: Respects concurrency semaphore.
        """
        async with self._semaphore:  # Arch #20: concurrency limiter
            (
                system_prompt, model_id, api_key, base_url,
                temperature, max_tokens,
                blocked_tools, allowed_tools_only, can_delegate,
            ) = self._resolve_agent_config(agent_name, category, prompt)

            subagent_config = AgentConfig(
                name=subagent_id,
                system_prompt=system_prompt,
                model_id=model_id,
                api_key=api_key,
                api_base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                blocked_tools=blocked_tools,
                allowed_tools_only=allowed_tools_only,
                can_delegate=can_delegate,
            )

            # Critical Fix #1: Create independent LLM client when model/key/url differs
            llm_client = None
            if self.parent_agent:
                parent_cfg = self.parent_agent.config
                parent_model = parent_cfg.subagent_model_id or parent_cfg.model_id
                parent_key = parent_cfg.subagent_api_key or parent_cfg.api_key
                parent_url = parent_cfg.subagent_api_base_url or parent_cfg.api_base_url

                if (model_id != parent_model or
                    api_key != parent_key or
                    base_url != parent_url):
                    # Different model/credentials → independent LLM client
                    llm_client = LLMClient(
                        model_id=model_id,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    logger.info(
                        f"{subagent_id}: using independent LLM client "
                        f"(model={model_id}, parent_model={parent_model})"
                    )
                else:
                    llm_client = self.parent_agent.llm_client

            # Create sub-agent with tools (permission filtering handled by Agent)
            subagent_tools = list(self.parent_tools)
            # Also include delegation tools if can_delegate is True
            if not can_delegate:
                subagent_tools = [
                    tool for tool in subagent_tools
                    if tool.name not in ("create_subagent", "assign_task")
                ]

            subagent = Agent(
                config=subagent_config,
                tools=subagent_tools,
                llm_client=llm_client,
            )

            # Build forked context
            context_messages = None
            if fork_context:
                context_messages = self._build_forked_context()
                if context_messages:
                    logger.info(f"{subagent_id}: forking {len(context_messages)} messages")

            # Update task status if tracked
            if task_id and self.task_store:
                self.task_store.update(task_id, status="in_progress", active_form=f"Executing: {prompt[:50]}")

            # Run sub-rollout
            rollout_config = SubRolloutConfig(
                max_steps=self.max_steps,
                step_hint=True,
                terminal_mode=False,
            )

            sub_rollout = SubRollout(rollout_config)

            # Arch #18: wrap with timeout if configured
            coro = sub_rollout.run(
                agent=subagent,
                initial_message=prompt,
                context_messages=context_messages,
            )
            if self._subtask_timeout:
                try:
                    result = await asyncio.wait_for(coro, timeout=self._subtask_timeout)
                except asyncio.TimeoutError:
                    logger.error(f"{subagent_id}: timed out after {self._subtask_timeout}s")
                    from ..rollout.base import RolloutResult, RolloutStatus
                    result = RolloutResult(
                        status=RolloutStatus.ERROR,
                        messages=[],
                        steps=0,
                        error=f"Subtask timed out after {self._subtask_timeout} seconds",
                    )
            else:
                result = await coro

            # Format result
            if result.status.value == "completed":
                content = result.final_response or "Task completed but no response generated."
            elif result.status.value == "max_steps_reached":
                content = f"[WARNING: Sub-agent reached step limit]\n\nLast response:\n{result.final_response or 'No response'}"
            elif result.status.value == "error":
                content = f"Sub-agent encountered an error: {result.error}"
            else:
                content = f"Sub-agent finished with status: {result.status.value}"

            logger.info(f"{subagent_id} completed with status: {result.status.value}")

            # Update task status if tracked
            if task_id and self.task_store:
                final_status = "completed" if result.status.value in ("completed", "max_steps_reached") else "pending"
                self.task_store.update(task_id, status=final_status, result=content[:2000])

            result_dict = {
                "agent": agent_name or f"category:{category}",
                "agent_id": subagent_id,
                "category": category,
                "prompt": prompt,
                "messages": result.messages,
                "status": result.status.value,
                "steps": result.steps,
                "content": content,
                "task_id": task_id,
            }

            return result_dict

    async def execute(
        self,
        prompt: str,
        agent: Optional[str] = None,
        fork_context: bool = False,
        category: Optional[str] = None,
        run_in_background: bool = False,
        load_skills: Optional[List[str]] = None,
        task_id: Optional[str] = None,
    ) -> ToolResult:
        """Launch a sub-agent to execute a task.

        Args:
            prompt: Task description for the sub-agent.
            agent: Name of the agent config (from create_subagent). Optional if category provided.
            fork_context: Whether to include parent context.
            category: Category preset to use.
            run_in_background: Launch in background.
            load_skills: Skills to load (reserved for future).
            task_id: Managed task ID.

        Returns:
            ToolResult with sub-agent's response (or background task ID).
        """
        try:
            # Validate: need either agent or category
            if agent and agent not in self.agent_registry and not category:
                available = list(self.agent_registry.keys())
                return ToolResult(
                    content=f"Agent '{agent}' not found. Available: {available}. "
                            f"Create one with create_subagent or specify a category.",
                    success=False,
                    error=f"Agent '{agent}' not found"
                )

            # Fix #4: atomic counter using itertools.count
            subagent_id = f"subagent_{next(self._counter)}"

            logger.info(
                f"Launching {subagent_id} "
                f"(agent={agent}, category={category}, background={run_in_background})"
            )

            if run_in_background:
                # Launch as background task
                bg_task = asyncio.create_task(
                    self._run_subtask(
                        subagent_id, agent, prompt, category,
                        fork_context, task_id,
                    )
                )
                async with self._lock:
                    self._background_tasks[subagent_id] = bg_task

                # Fix #10: handle CancelledError (BaseException in Python 3.9+)
                def _on_complete(future, sid=subagent_id):
                    try:
                        result_dict = future.result()
                        self._background_results[sid] = result_dict
                        self.sub_results.append(result_dict)
                        logger.info(f"Background task {sid} completed")
                    except asyncio.CancelledError:
                        self._background_results[sid] = {
                            "agent_id": sid,
                            "status": "cancelled",
                            "content": f"Background task was cancelled.",
                        }
                        logger.info(f"Background task {sid} was cancelled")
                    except Exception as e:
                        self._background_results[sid] = {
                            "agent_id": sid,
                            "status": "error",
                            "content": f"Background task failed: {e}",
                        }
                        logger.error(f"Background task {sid} failed: {e}")

                bg_task.add_done_callback(_on_complete)

                return ToolResult(
                    content=(
                        f"Background task launched: {subagent_id}\n"
                        f"Use `background_output(task_id='{subagent_id}')` to retrieve results when ready."
                    ),
                    success=True,
                )

            # Synchronous execution
            result_dict = await self._run_subtask(
                subagent_id, agent, prompt, category,
                fork_context, task_id,
            )

            self.sub_results.append(result_dict)

            return ToolResult(
                content=result_dict["content"],
                success=result_dict["status"] in ("completed", "max_steps_reached"),
            )

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            import traceback
            traceback.print_exc()
            return ToolResult(
                content="",
                success=False,
                error=str(e)
            )

    def get_background_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a background task.

        Args:
            task_id: The subagent_id returned when launching.

        Returns:
            Result dict, or None if still running / not found.
        """
        return self._background_results.get(task_id)

    def is_background_running(self, task_id: str) -> bool:
        """Check if a background task is still running."""
        bg_task = self._background_tasks.get(task_id)
        return bg_task is not None and not bg_task.done()

    def cancel_background(self, task_id: str) -> bool:
        """Cancel a running background task.

        Returns:
            True if cancelled, False if not found or already done.
        """
        bg_task = self._background_tasks.get(task_id)
        if bg_task and not bg_task.done():
            bg_task.cancel()
            logger.info(f"Background task {task_id} cancelled")
            return True
        return False

    def cancel_all_background(self) -> int:
        """Cancel all running background tasks (Arch #23: graceful shutdown).

        Returns:
            Number of tasks cancelled.
        """
        cancelled = 0
        for task_id, bg_task in self._background_tasks.items():
            if not bg_task.done():
                bg_task.cancel()
                cancelled += 1
                logger.info(f"Background task {task_id} cancelled (shutdown)")
        return cancelled
