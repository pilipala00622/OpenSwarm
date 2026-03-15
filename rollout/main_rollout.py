"""Main Rollout - Rollout for main agent execution"""

import json
import logging
import os
from typing import Dict, List, Any, Optional

from .base import BaseRollout, RolloutConfig, RolloutResult, RolloutStatus

logger = logging.getLogger(__name__)

# Default scaling rules injected into system prompt (from Anthropic's best practices)
DEFAULT_SCALING_RULES = """
## Resource Allocation Rules

Allocate sub-agents based on query complexity:
- **Simple queries** (single fact / direct answer): 0 sub-agents, answer directly
- **Medium queries** (need to search / compare 2-3 sources): 1-2 sub-agents
- **Complex queries** (multi-dimensional research / deep analysis): 3-5 sub-agents
- **Very complex queries** (comprehensive survey / cross-domain): 5-10 sub-agents

Before starting research, assess query complexity first, then decide sub-agent count.
Do NOT launch unnecessary sub-agents for simple queries.
"""


class MainRollout(BaseRollout):
    """Rollout for main agent execution

    MainRollout provides full capabilities including:
    - Complete tool access
    - Sub-agent spawning via Task tool
    - Message storage (optional)
    - Terminal output formatting
    """

    def __init__(self, config: Optional[RolloutConfig] = None):
        """Initialize main rollout

        Args:
            config: Rollout configuration
        """
        super().__init__(config)
        self.sub_results: List[Dict[str, Any]] = []  # Collect sub-agent results

    async def run(
        self,
        agent,
        initial_message: str,
        context_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> RolloutResult:
        """Execute the main rollout loop

        Args:
            agent: The agent to run
            initial_message: Initial user message
            context_messages: Optional previous context to include

        Returns:
            RolloutResult with execution results
        """
        # Reset state
        self.reset()

        # Initialize messages with system prompt
        system_msg = agent.get_system_message()

        # Inject dynamic scaling rules into system prompt
        if self.config.enable_scaling_rules:
            scaling_text = self.config.scaling_rules or DEFAULT_SCALING_RULES
            system_msg = {
                **system_msg,
                "content": system_msg["content"] + "\n" + scaling_text,
            }

        self.messages = [system_msg]

        # Add context if provided
        if context_messages:
            self.messages.extend(context_messages)

        # Add initial user message
        self.messages.append({
            "role": "user",
            "content": initial_message
        })

        # Pass message reference to TaskTool for fork_context support
        self._setup_fork_context(agent)

        # Pass message reference to HandoffTool for context capture
        self._setup_handoff_ref(agent)

        # Inject agent's LLM client into memory for LLM-based summarisation
        if self.memory and self.config.enable_llm_summarise:
            self.memory._llm_client = agent.llm_client
            logger.info("Memory: LLM summarisation enabled via agent's LLM client")

        result = RolloutResult(
            status=RolloutStatus.RUNNING,
            messages=self.messages,
            steps=0,
        )

        # Main execution loop
        while self.current_step < self.config.max_steps and not self.interrupted:
            try:
                # Save checkpoint periodically
                if self._should_save_checkpoint():
                    self._save_checkpoint()
                    if self.tracer:
                        self.tracer.log("checkpoint", agent_id=agent.config.name,
                                        step=self.current_step)

                # Compress context if memory is enabled and messages are too long
                if self.memory and self.memory.should_compress(self.messages):
                    compressed_count = len(self.messages)
                    self.messages = await self.memory.compress(self.messages)
                    logger.info(
                        f"Context compressed at step {self.current_step} "
                        f"(total compressed: {self.memory.get_total_compressed()} messages)"
                    )
                    if self.tracer:
                        self.tracer.log("compression", agent_id=agent.config.name,
                                        step=self.current_step,
                                        before=compressed_count, after=len(self.messages))

                # Get agent response
                if self.tracer:
                    self.tracer.log("llm_call", agent_id=agent.config.name,
                                    step=self.current_step,
                                    model=agent.config.model_id,
                                    message_count=len(self.messages))

                response = await agent.process_message(
                    messages=self.messages,
                    include_system=False,  # System already in messages
                )

                # Reset consecutive error counter on successful LLM call
                self.consecutive_errors = 0

                # Print step info
                self._print_step(self.current_step, response)

                # Add assistant message
                assistant_msg = self._format_assistant_message(response)
                self.messages.append(assistant_msg)

                # Handle tool calls
                if response.get("tool_calls"):
                    # Trace tool executions
                    if self.tracer:
                        for tc in response["tool_calls"]:
                            tool_name = tc.get("function", {}).get("name", "unknown")
                            self.tracer.log("tool_exec", agent_id=agent.config.name,
                                            step=self.current_step, tool=tool_name)
                            if tool_name == "assign_task":
                                self.tracer.log("subagent_spawn", agent_id=agent.config.name,
                                                step=self.current_step,
                                                subagent_id=f"subagent_{len(self.sub_results)+1}")

                    tool_results = await agent.execute_tool_calls(response["tool_calls"])

                    # Print and add tool results
                    for tr in tool_results:
                        self._print_tool_result(
                            tr.get("tool_call_id", "unknown"),
                            tr.get("content", "")
                        )
                        self.messages.append(tr)

                    self.current_step += 1
                    continue

                # Check completion
                if self._is_complete(response):
                    result = RolloutResult(
                        status=RolloutStatus.COMPLETED,
                        messages=self.messages,
                        steps=self.current_step,
                        final_response=response.get("content"),
                    )
                    break

                self.current_step += 1

            except Exception as e:
                self.consecutive_errors += 1
                logger.error(
                    f"Error in rollout step {self.current_step} "
                    f"(consecutive: {self.consecutive_errors}/{self.config.max_consecutive_errors}): {e}"
                )
                if self.tracer:
                    self.tracer.log("error", agent_id=agent.config.name,
                                    step=self.current_step,
                                    error=str(e),
                                    consecutive=self.consecutive_errors)

                if self.consecutive_errors >= self.config.max_consecutive_errors:
                    # Too many consecutive errors - try restoring checkpoint first
                    if self.checkpoints and self.consecutive_errors == self.config.max_consecutive_errors:
                        logger.info("Max consecutive errors reached, attempting checkpoint restore...")
                        if self._restore_last_checkpoint():
                            if self.tracer:
                                self.tracer.log("recovery", agent_id=agent.config.name,
                                                step=self.current_step, action="checkpoint_restore")
                            # Add a hint so the agent knows something went wrong
                            self.messages.append({
                                "role": "user",
                                "content": (
                                    "[System] Multiple consecutive errors occurred. "
                                    "State has been restored to an earlier checkpoint. "
                                    "Please try a different approach to complete the task."
                                )
                            })
                            self.consecutive_errors = 0  # Reset after restore
                            continue

                    # No checkpoint or restore didn't help - abort
                    result = RolloutResult(
                        status=RolloutStatus.ERROR,
                        messages=self.messages,
                        steps=self.current_step,
                        error=f"Aborted after {self.consecutive_errors} consecutive errors. Last: {str(e)}",
                    )
                    break

                # Still under the error threshold - notify the agent and continue
                if self.tracer:
                    self.tracer.log("recovery", agent_id=agent.config.name,
                                    step=self.current_step, action="error_notification")
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[System] Error in previous step: {str(e)}. "
                        f"({self.consecutive_errors}/{self.config.max_consecutive_errors} consecutive errors). "
                        f"Please try a different approach to continue."
                    )
                })
                self.current_step += 1
                continue

        # Handle max steps reached
        if self.current_step >= self.config.max_steps:
            result = RolloutResult(
                status=RolloutStatus.MAX_STEPS_REACHED,
                messages=self.messages,
                steps=self.current_step,
                final_response=self._get_last_assistant_content(),
            )

        # Handle interruption
        if self.interrupted:
            result = RolloutResult(
                status=RolloutStatus.INTERRUPTED,
                messages=self.messages,
                steps=self.current_step,
            )

        if self.config.terminal_mode:
            print(f"\n{'='*60}")
            print(f"Rollout completed: {result.status.value}")
            print(f"Total steps: {result.steps}")
            print('='*60)

        # Collect sub-agent results from TaskTool if available
        result.subs = self._collect_sub_results(agent)

        # Save to storage if path configured
        if self.config.storage_path:
            self._save_result(result)

        # Export trace if configured
        if self.tracer:
            if self.config.trace_output_path:
                self.tracer.to_jsonl(self.config.trace_output_path)
            if self.config.terminal_mode:
                trace_summary = self.tracer.summary()
                print(f"\nTrace Summary:")
                print(f"  LLM calls: {trace_summary['llm_calls']}")
                print(f"  Tool executions: {trace_summary['tool_executions']}")
                print(f"  Sub-agents spawned: {trace_summary['subagents_spawned']}")
                print(f"  Errors: {trace_summary['errors']}")
                print(f"  Total time: {trace_summary['total_time_seconds']}s")

        return result

    def _get_last_assistant_content(self) -> Optional[str]:
        """Get content from the last assistant message"""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return None

    def _setup_fork_context(self, agent):
        """Pass message list reference to TaskTool for fork_context support.

        This allows sub-agents to receive parent conversation context
        when fork_context=True is specified in the assign_task call.
        """
        task_tool = agent.tools.get("assign_task")
        if task_tool and hasattr(task_tool, "set_parent_messages"):
            task_tool.set_parent_messages(self.messages)
            logger.info("Fork context enabled: TaskTool now has access to parent messages")

    def _setup_handoff_ref(self, agent):
        """Pass message list reference to HandoffTool for context capture."""
        handoff_tool = agent.tools.get("handoff")
        if handoff_tool and hasattr(handoff_tool, "set_messages_ref"):
            handoff_tool.set_messages_ref(self.messages)
            logger.info("Handoff tool connected to message history")

    def _collect_sub_results(self, agent) -> List[Dict[str, Any]]:
        """Collect sub-agent results from TaskTool"""
        subs = []
        # Find TaskTool in agent's tools (now named assign_task)
        task_tool = agent.tools.get("assign_task")
        if task_tool and hasattr(task_tool, "sub_results"):
            subs = task_tool.sub_results
        return subs

    def _save_result(self, result: RolloutResult):
        """Save result to storage path as jsonl

        storage_path can be:
        - A directory: saves to {dir}/result.jsonl
        - A file path ending with .jsonl: saves directly to that file
        """
        try:
            storage_path = self.config.storage_path

            if storage_path.endswith(".jsonl"):
                # Direct file path
                filepath = storage_path
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
            else:
                # Directory path
                os.makedirs(storage_path, exist_ok=True)
                filepath = os.path.join(storage_path, "result.jsonl")

            # Append to jsonl file
            with open(filepath, "a", encoding="utf-8") as f:
                data = result.to_storage_format()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

            logger.info(f"Saved result to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
