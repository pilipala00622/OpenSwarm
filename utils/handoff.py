"""Handoff - Cross-session context transfer mechanism.

Inspired by Oh-My-OpenCode's /handoff command:
- Creates a detailed context summary for continuing work in a new session.
- Captures the current state, what was done, what remains, and relevant file paths.
- Enables seamless continuation in a fresh session / different agent.

Usage:
    handoff = HandoffManager(handoff_dir="/path/to/handoffs")

    # Create handoff from current state
    doc = handoff.create(
        messages=[...],
        tasks=[...],
        memory_entries=[...],
        agent_name="orchestrator",
        notes="User wants to continue the analysis tomorrow",
    )

    # Later, in a new session, load the handoff
    context = handoff.load(doc.id)
    # Use context["summary"] and context["context_messages"] to bootstrap a new agent
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class HandoffDocument:
    """A handoff document capturing session state for transfer.

    Attributes:
        id: Unique handoff identifier.
        created_at: ISO timestamp.
        agent_name: Name of the agent that created the handoff.
        summary: LLM-generated or heuristic summary of work done.
        completed_work: List of completed items/findings.
        remaining_work: List of remaining items/questions.
        key_files: List of relevant file paths.
        key_decisions: Important decisions made during the session.
        context_messages: Compressed message history for context injection.
        task_snapshot: Snapshot of active tasks with their status.
        notes: Free-form notes from the agent or user.
    """
    id: str = ""
    created_at: str = ""
    agent_name: str = ""
    summary: str = ""
    completed_work: List[str] = field(default_factory=list)
    remaining_work: List[str] = field(default_factory=list)
    key_files: List[str] = field(default_factory=list)
    key_decisions: List[str] = field(default_factory=list)
    context_messages: List[Dict[str, Any]] = field(default_factory=list)
    task_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_context_message(self) -> Dict[str, str]:
        """Convert to a user message suitable for injecting into a new session."""
        parts = [f"[Handoff from {self.agent_name} at {self.created_at}]"]
        parts.append(f"\n## Summary\n{self.summary}")

        if self.completed_work:
            parts.append("\n## Completed Work")
            for item in self.completed_work:
                parts.append(f"- {item}")

        if self.remaining_work:
            parts.append("\n## Remaining Work")
            for item in self.remaining_work:
                parts.append(f"- {item}")

        if self.key_decisions:
            parts.append("\n## Key Decisions")
            for item in self.key_decisions:
                parts.append(f"- {item}")

        if self.key_files:
            parts.append("\n## Relevant Files")
            for f in self.key_files:
                parts.append(f"- {f}")

        if self.task_snapshot:
            parts.append("\n## Active Tasks")
            for t in self.task_snapshot:
                status = t.get("status", "?")
                subject = t.get("subject", "?")
                parts.append(f"- [{status}] {t.get('id', '?')}: {subject}")

        if self.notes:
            parts.append(f"\n## Notes\n{self.notes}")

        return {
            "role": "user",
            "content": "\n".join(parts),
        }


class HandoffManager:
    """Manages creation and loading of handoff documents.

    Handoffs are stored as JSON files for cross-session persistence.
    """

    def __init__(self, handoff_dir: Optional[str] = None):
        """Initialize handoff manager.

        Args:
            handoff_dir: Directory for handoff JSON files. Defaults to .agent_handoffs/.
        """
        self.handoff_dir = handoff_dir or os.path.join(os.getcwd(), ".agent_handoffs")
        os.makedirs(self.handoff_dir, exist_ok=True)

    def create(
        self,
        messages: List[Dict[str, Any]],
        agent_name: str = "agent",
        tasks: Optional[List[Dict[str, Any]]] = None,
        memory_entries: Optional[List[Any]] = None,
        notes: str = "",
        llm_summary: Optional[str] = None,
    ) -> HandoffDocument:
        """Create a handoff document from current session state.

        Args:
            messages: Current conversation messages.
            agent_name: Name of the creating agent.
            tasks: Active task snapshots.
            memory_entries: Memory phase entries.
            notes: Free-form notes.
            llm_summary: Pre-generated LLM summary (if available).

        Returns:
            Created HandoffDocument.
        """
        import uuid
        handoff_id = f"H-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now = datetime.utcnow().isoformat()

        # Extract completed work, remaining work, and key files from messages
        completed = []
        remaining = []
        key_files = []
        key_decisions = []

        # Heuristic extraction from messages
        for msg in messages:
            content = msg.get("content", "")
            if not content or msg.get("role") == "system":
                continue

            # Extract file paths mentioned
            for word in content.split():
                if ("/" in word or "\\" in word) and (
                    word.endswith(".py") or word.endswith(".md") or
                    word.endswith(".js") or word.endswith(".ts") or
                    word.endswith(".json") or word.endswith(".yaml") or
                    word.endswith(".yml") or word.endswith(".toml")
                ):
                    clean = word.strip("'\"()[]{},:;")
                    if clean and clean not in key_files:
                        key_files.append(clean)

        # Build summary
        if llm_summary:
            summary = llm_summary
        else:
            summary = self._heuristic_summary(messages)

        # Compress context: keep system + last 20 messages
        context_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                context_messages.append(msg)
                break
        non_system = [m for m in messages if m.get("role") != "system"]
        context_messages.extend(non_system[-20:])

        doc = HandoffDocument(
            id=handoff_id,
            created_at=now,
            agent_name=agent_name,
            summary=summary,
            completed_work=completed,
            remaining_work=remaining,
            key_files=key_files[:20],  # Limit
            key_decisions=key_decisions,
            context_messages=context_messages,
            task_snapshot=tasks or [],
            notes=notes,
        )

        # Save to disk
        filepath = os.path.join(self.handoff_dir, f"{handoff_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Handoff created: {handoff_id} -> {filepath}")
        return doc

    def load(self, handoff_id: str) -> Optional[HandoffDocument]:
        """Load a handoff document by ID.

        Args:
            handoff_id: Handoff identifier.

        Returns:
            HandoffDocument, or None if not found.
        """
        filepath = os.path.join(self.handoff_dir, f"{handoff_id}.json")
        if not os.path.exists(filepath):
            logger.warning(f"Handoff not found: {filepath}")
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc = HandoffDocument(**{
            k: v for k, v in data.items()
            if k in HandoffDocument.__dataclass_fields__
        })
        logger.info(f"Handoff loaded: {handoff_id}")
        return doc

    def load_latest(self) -> Optional[HandoffDocument]:
        """Load the most recent handoff document.

        Returns:
            Most recent HandoffDocument, or None if none exist.
        """
        files = sorted(
            [f for f in os.listdir(self.handoff_dir) if f.endswith(".json")],
            reverse=True,
        )
        if not files:
            return None
        handoff_id = files[0].replace(".json", "")
        return self.load(handoff_id)

    def list_handoffs(self) -> List[str]:
        """List all handoff IDs, most recent first."""
        files = sorted(
            [f for f in os.listdir(self.handoff_dir) if f.endswith(".json")],
            reverse=True,
        )
        return [f.replace(".json", "") for f in files]

    def _heuristic_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a heuristic summary from messages."""
        assistant_messages = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "assistant" and m.get("content")
        ]
        if not assistant_messages:
            return "No work performed."

        # Take the last assistant message as primary summary source
        last = assistant_messages[-1]
        if len(last) > 500:
            last = last[:500] + "..."

        # Count tool calls
        tool_count = sum(
            1 for m in messages
            if m.get("role") == "tool"
        )

        parts = [
            f"Session involved {len(messages)} messages and {tool_count} tool executions.",
            f"Last agent response: {last}",
        ]
        return "\n".join(parts)
