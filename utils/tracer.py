"""Rollout Tracer - Structured execution tracing for observability

Anthropic emphasises monitoring agent *decision patterns* and *interaction
structures* rather than just individual tool calls. This module provides a
lightweight event-based tracer that records:
  - LLM calls (timing, token hints)
  - Tool executions (name, success/failure)
  - Sub-agent lifecycle (spawn, complete)
  - Errors and recovery events
  - Decision points (checkpoints, compressions)

Events are stored in-memory and can be exported to JSONL for post-hoc analysis.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """A single trace event."""
    timestamp: float          # Seconds since tracer start
    event_type: str           # e.g. "llm_call", "tool_exec", "subagent_spawn", "error", etc.
    agent_id: str             # Which agent produced this event
    step: int                 # Rollout step number
    data: Dict[str, Any] = field(default_factory=dict)  # Event-specific payload


class RolloutTracer:
    """Structured execution tracer for rollout observability.

    Usage:
        tracer = RolloutTracer()
        tracer.log("llm_call", agent_id="main", step=0, model="kimi-k2.5", tokens=1500)
        tracer.log("tool_exec", agent_id="main", step=0, tool="search", success=True)
        tracer.log("subagent_spawn", agent_id="main", step=1, subagent_id="subagent_1")
        ...
        print(tracer.summary())
        tracer.to_jsonl("/path/to/trace.jsonl")
    """

    def __init__(self):
        self.events: List[TraceEvent] = []
        self.start_time = time.time()

    def log(self, event_type: str, agent_id: str, step: int, **data):
        """Record a trace event.

        Args:
            event_type: Category of event. Suggested types:
                - "llm_call": An LLM API call was made
                - "tool_exec": A tool was executed
                - "subagent_spawn": A sub-agent was launched
                - "subagent_complete": A sub-agent finished
                - "error": An error occurred
                - "recovery": Error recovery action (checkpoint restore, retry)
                - "checkpoint": A checkpoint was saved
                - "compression": Memory compression occurred
                - "decision": An important agent decision was made
            agent_id: Identifier for the agent that produced this event.
            step: Current rollout step number.
            **data: Arbitrary key-value data specific to the event type.
        """
        event = TraceEvent(
            timestamp=time.time() - self.start_time,
            event_type=event_type,
            agent_id=agent_id,
            step=step,
            data=data,
        )
        self.events.append(event)
        logger.debug(f"Trace: [{event_type}] agent={agent_id} step={step} {data}")

    def summary(self) -> Dict[str, Any]:
        """Generate an execution summary.

        Returns:
            Dictionary with aggregate statistics about the traced execution.
        """
        elapsed = time.time() - self.start_time
        type_counts = {}
        for e in self.events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        # Extract sub-agent IDs
        subagent_ids = set()
        for e in self.events:
            if e.event_type == "subagent_spawn":
                subagent_ids.add(e.data.get("subagent_id", "unknown"))

        # Calculate tool usage breakdown
        tool_usage = {}
        for e in self.events:
            if e.event_type == "tool_exec":
                tool_name = e.data.get("tool", "unknown")
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        # Arch #19: Aggregate token usage
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for e in self.events:
            if e.event_type == "token_usage":
                prompt_tokens += e.data.get("prompt_tokens", 0)
                completion_tokens += e.data.get("completion_tokens", 0)
                total_tokens += e.data.get("total_tokens", 0)

        return {
            "total_events": len(self.events),
            "total_time_seconds": round(elapsed, 2),
            "event_type_counts": type_counts,
            "llm_calls": type_counts.get("llm_call", 0),
            "tool_executions": type_counts.get("tool_exec", 0),
            "subagents_spawned": len(subagent_ids),
            "subagent_ids": list(subagent_ids),
            "errors": type_counts.get("error", 0),
            "recoveries": type_counts.get("recovery", 0),
            "tool_usage_breakdown": tool_usage,
            # Arch #19: token stats
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def to_jsonl(self, path: str):
        """Export all events to a JSONL file.

        Args:
            path: File path to write the JSONL output to.
        """
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                record = {
                    "ts": round(e.timestamp, 3),
                    "type": e.event_type,
                    "agent": e.agent_id,
                    "step": e.step,
                    **e.data,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Trace exported: {len(self.events)} events → {path}")

    def get_events_by_type(self, event_type: str) -> List[TraceEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_by_agent(self, agent_id: str) -> List[TraceEvent]:
        """Filter events by agent ID."""
        return [e for e in self.events if e.agent_id == agent_id]

    def reset(self):
        """Clear all events and restart the timer."""
        self.events.clear()
        self.start_time = time.time()
