"""Base Rollout - Abstract base class for rollout implementations"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

from ..utils.memory import AgentMemory
from ..utils.tracer import RolloutTracer

logger = logging.getLogger(__name__)


class RolloutStatus(str, Enum):
    """Status of rollout execution"""
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_STEPS_REACHED = "max_steps_reached"
    ERROR = "error"
    INTERRUPTED = "interrupted"


@dataclass
class RolloutConfig:
    """Configuration for rollout behavior"""
    max_steps: int = 50
    terminal_mode: bool = True
    storage_path: Optional[str] = None  # If set, save results to {path}/result.jsonl
    print_tool_calls: bool = True
    print_tool_results: bool = True
    # Error recovery settings
    max_consecutive_errors: int = 3  # Max consecutive errors before aborting
    checkpoint_interval: int = 10     # Save checkpoint every N steps
    # Memory / context compression settings
    enable_memory: bool = True           # Enable automatic context compression
    max_context_messages: int = 50       # Compress when messages exceed this count
    memory_keep_recent: int = 20         # Keep this many recent messages after compression
    memory_dir: Optional[str] = None     # Directory to persist memory phase files (Anthropic-style)
    enable_llm_summarise: bool = True    # Use LLM for high-quality summarisation (requires agent LLM)
    # Dynamic scaling settings
    enable_scaling_rules: bool = True    # Inject scaling rules into system prompt
    scaling_rules: Optional[str] = None  # Custom scaling rules (uses default if None)
    # Tracing / observability settings
    enable_tracing: bool = True          # Enable structured execution tracing
    trace_output_path: Optional[str] = None  # If set, auto-export trace to this JSONL path


@dataclass
class RolloutResult:
    """Result of a rollout execution"""
    status: RolloutStatus
    messages: List[Dict[str, Any]]
    steps: int
    final_response: Optional[str] = None
    error: Optional[str] = None
    subs: List[Dict[str, Any]] = field(default_factory=list)  # Sub-agent results

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "messages": self.messages,
            "steps": self.steps,
            "final_response": self.final_response,
            "error": self.error,
            "subs": self.subs,
        }

    def to_storage_format(self) -> Dict[str, Any]:
        """Convert to storage format for jsonl"""
        return {
            "main": self.messages,
            "subs": self.subs,
        }


class BaseRollout(ABC):
    """Abstract base class for all rollout implementations

    A rollout manages the execution loop of an agent, handling:
    - Message history management
    - Step counting and limits
    - Tool execution flow
    - Completion detection
    """

    def __init__(self, config: Optional[RolloutConfig] = None):
        """Initialize rollout

        Args:
            config: Rollout configuration
        """
        self.config = config or RolloutConfig()
        self.messages: List[Dict[str, Any]] = []
        self.current_step = 0
        self.interrupted = False
        # Error recovery state
        self.consecutive_errors = 0
        self.checkpoints: List[Dict[str, Any]] = []
        # Memory / context compression (LLM client injected later by subclass)
        self.memory = AgentMemory(
            max_context_messages=self.config.max_context_messages,
            keep_recent=self.config.memory_keep_recent,
            memory_dir=self.config.memory_dir,
        ) if self.config.enable_memory else None
        # Structured execution tracing
        self.tracer = RolloutTracer() if self.config.enable_tracing else None

    @abstractmethod
    async def run(
        self,
        agent,
        initial_message: str,
        context_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> RolloutResult:
        """Execute the rollout loop

        Args:
            agent: The agent to run
            initial_message: Initial user message
            context_messages: Optional previous context to include

        Returns:
            RolloutResult with execution results
        """
        pass

    def _is_complete(self, response: Dict[str, Any]) -> bool:
        """Check if the task is complete

        Args:
            response: LLM response dict

        Returns:
            True if task is complete
        """
        # Complete if no tool calls and has content
        if response.get("content") and not response.get("tool_calls"):
            return True

        # Complete if finish_reason is "stop"
        if response.get("finish_reason") == "stop":
            return True

        return False

    def _format_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format LLM response as assistant message

        Args:
            response: Raw LLM response

        Returns:
            Formatted assistant message
        """
        message = {
            "role": "assistant",
            "content": response.get("content", ""),
        }

        # Preserve reasoning_content for thinking models (like kimi-k2.5)
        if response.get("reasoning_content"):
            message["reasoning_content"] = response["reasoning_content"]

        if response.get("tool_calls"):
            message["tool_calls"] = response["tool_calls"]

        return message

    def _print_step(self, step: int, response: Dict[str, Any]):
        """Print step information to terminal

        Args:
            step: Current step number
            response: LLM response
        """
        if not self.config.terminal_mode:
            return

        print(f"\n{'='*60}")
        print(f"[Step {step}]")
        print('='*60)

        # Print content
        if response.get("content"):
            print(f"\nContent:\n{response['content']}")

        # Print tool calls
        if self.config.print_tool_calls and response.get("tool_calls"):
            print(f"\nTool Calls:")
            for tc in response["tool_calls"]:
                func = tc.get("function", {})
                print(f"  - {func.get('name', 'unknown')}")
                print(f"    Args: {func.get('arguments', '{}')}")

    def _print_tool_result(self, tool_name: str, result: str):
        """Print tool result to terminal

        Args:
            tool_name: Name of the tool
            result: Tool execution result
        """
        if not self.config.terminal_mode or not self.config.print_tool_results:
            return

        print(f"\nTool Result [{tool_name}]:")
        # Truncate long results
        if len(result) > 500:
            print(f"  {result[:500]}... (truncated)")
        else:
            print(f"  {result}")

    def interrupt(self):
        """Signal to interrupt the rollout"""
        self.interrupted = True
        logger.info("Rollout interrupted")

    def reset(self):
        """Reset rollout state"""
        self.messages = []
        self.current_step = 0
        self.interrupted = False
        self.consecutive_errors = 0
        self.checkpoints = []
        if self.memory:
            self.memory.reset()
        if self.tracer:
            self.tracer.reset()

    def _save_checkpoint(self):
        """Save current state as a checkpoint for error recovery.

        Checkpoints are saved periodically (controlled by checkpoint_interval)
        and can be used to restore state after consecutive errors.
        """
        import copy
        self.checkpoints.append({
            "step": self.current_step,
            "messages_count": len(self.messages),
            "messages_snapshot": copy.deepcopy(self.messages),
        })
        logger.info(f"Checkpoint saved at step {self.current_step} ({len(self.messages)} messages)")

    def _should_save_checkpoint(self) -> bool:
        """Check if a checkpoint should be saved at the current step."""
        return (
            self.config.checkpoint_interval > 0
            and self.current_step > 0
            and self.current_step % self.config.checkpoint_interval == 0
        )

    def _restore_last_checkpoint(self) -> bool:
        """Restore state from the most recent checkpoint.

        Returns:
            True if a checkpoint was restored, False if none available.
        """
        if not self.checkpoints:
            logger.warning("No checkpoints available to restore")
            return False
        checkpoint = self.checkpoints.pop()
        self.messages = checkpoint["messages_snapshot"]
        self.current_step = checkpoint["step"]
        logger.info(f"Restored checkpoint from step {checkpoint['step']}")
        return True
