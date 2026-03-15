"""Sub Rollout - Rollout for sub-agent execution"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from .base import BaseRollout, RolloutConfig, RolloutResult, RolloutStatus

logger = logging.getLogger(__name__)


@dataclass
class SubRolloutConfig(RolloutConfig):
    """Configuration for sub-agent rollout"""
    max_steps: int = 20  # Lower default for sub-agents
    step_hint: bool = True  # Show step hints in tool results
    terminal_mode: bool = False  # Quieter by default
    # Sub-agents generally don't need scaling rules or memory for their shorter runs
    enable_scaling_rules: bool = False
    enable_memory: bool = False
    enable_tracing: bool = False
    # Lighter error tolerance for sub-agents
    max_consecutive_errors: int = 2
    checkpoint_interval: int = 0  # Disabled by default for short sub-agent runs


class SubRollout(BaseRollout):
    """Rollout for sub-agent execution

    SubRollout is designed for sub-agents with:
    - Lower step limits
    - Step tracking in tool results
    - Context isolation
    - Error recovery (aligned with MainRollout)
    - Simplified output
    """

    def __init__(self, config: Optional[SubRolloutConfig] = None):
        """Initialize sub rollout

        Args:
            config: Sub-rollout configuration
        """
        super().__init__(config or SubRolloutConfig())
        self.step_hint = getattr(self.config, 'step_hint', True)

    async def run(
        self,
        agent,
        initial_message: str,
        context_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> RolloutResult:
        """Execute the sub-agent rollout loop

        Args:
            agent: The sub-agent to run
            initial_message: Task description from parent
            context_messages: Optional context from parent (if fork_context=True)

        Returns:
            RolloutResult with execution results
        """
        # Reset state
        self.reset()

        # Initialize messages with system prompt
        self.messages = [agent.get_system_message()]

        # Add forked context if provided
        if context_messages:
            # Add context marker
            self.messages.append({
                "role": "user",
                "content": "=== Parent Context Start ==="
            })
            self.messages.extend(context_messages)
            self.messages.append({
                "role": "user",
                "content": "=== Parent Context End ==="
            })

        # Add task message
        self.messages.append({
            "role": "user",
            "content": initial_message
        })

        result = RolloutResult(
            status=RolloutStatus.RUNNING,
            messages=self.messages,
            steps=0,
        )

        # Execution loop
        while self.current_step < self.config.max_steps and not self.interrupted:
            try:
                # Save checkpoint periodically (if enabled)
                if self._should_save_checkpoint():
                    self._save_checkpoint()

                # Get agent response
                response = await agent.process_message(
                    messages=self.messages,
                    include_system=False,
                )

                # Reset consecutive error counter on success
                self.consecutive_errors = 0

                # Fix #3: Guard against empty responses
                if self._is_empty_response(response):
                    self.consecutive_errors += 1
                    logger.warning(
                        f"Sub-agent empty response at step {self.current_step}"
                    )
                    self.messages.append({
                        "role": "user",
                        "content": "[System] Received empty response. Please provide an answer or use a tool."
                    })
                    self.current_step += 1
                    continue

                # Print step info (if terminal mode enabled)
                self._print_step(self.current_step, response)

                # Add assistant message
                assistant_msg = self._format_assistant_message(response)
                self.messages.append(assistant_msg)

                # Handle tool calls
                if response.get("tool_calls"):
                    tool_results = await agent.execute_tool_calls(response["tool_calls"])

                    # Add step hints to tool results
                    for tr in tool_results:
                        if self.step_hint:
                            tr["content"] = self._add_step_hint(tr.get("content", ""))
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
                    f"Error in sub-rollout step {self.current_step} "
                    f"(consecutive: {self.consecutive_errors}/{self.config.max_consecutive_errors}): {e}"
                )

                if self.consecutive_errors >= self.config.max_consecutive_errors:
                    # Try checkpoint restore if available
                    if self.checkpoints and self.consecutive_errors == self.config.max_consecutive_errors:
                        if self._restore_last_checkpoint():
                            self.messages.append({
                                "role": "user",
                                "content": (
                                    "[System] Multiple errors occurred. "
                                    "State restored to checkpoint. Try a different approach."
                                )
                            })
                            self.consecutive_errors = 0
                            continue

                    # Abort
                    result = RolloutResult(
                        status=RolloutStatus.ERROR,
                        messages=self.messages,
                        steps=self.current_step,
                        error=f"Sub-agent aborted after {self.consecutive_errors} consecutive errors. Last: {str(e)}",
                    )
                    break

                # Under threshold - notify and continue
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[System] Error: {str(e)}. "
                        f"({self.consecutive_errors}/{self.config.max_consecutive_errors} errors). "
                        f"Try a different approach."
                    )
                })
                self.current_step += 1
                continue

        # Handle max steps reached
        if self.current_step >= self.config.max_steps:
            logger.warning(f"Sub-agent reached max steps ({self.config.max_steps})")
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

        return result

    def _add_step_hint(self, content: str) -> str:
        """Add step tracking hint to tool result

        Args:
            content: Original tool result content

        Returns:
            Content with step hint prepended
        """
        remaining = self.config.max_steps - self.current_step - 1
        hint = f"[Step {self.current_step + 1}/{self.config.max_steps}]"

        if remaining <= 3:
            hint += f" Warning: Only {remaining} steps remaining. Prepare final answer."

        return f"{hint}\n\n{content}"

    def _get_last_assistant_content(self) -> Optional[str]:
        """Get content from the last assistant message"""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return None
