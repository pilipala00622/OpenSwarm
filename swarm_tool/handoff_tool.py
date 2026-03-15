"""Handoff Tool - Create or load context handoff documents.

Allows agents to create handoff documents for cross-session continuity,
and load previous handoffs to resume work.
"""

import logging
from typing import Dict, Any, List, Optional

from ..tool.base import BaseTool, ToolResult
from ..utils.handoff import HandoffManager

logger = logging.getLogger(__name__)


class HandoffTool(BaseTool):
    """Tool for creating and loading handoff documents."""

    def __init__(
        self,
        handoff_manager: HandoffManager,
        messages_ref: Optional[List[Dict[str, Any]]] = None,
        task_store: Optional[Any] = None,
    ):
        """Initialize handoff tool.

        Args:
            handoff_manager: HandoffManager instance.
            messages_ref: Reference to current conversation messages.
            task_store: Optional TaskStore for task snapshots.
        """
        self._manager = handoff_manager
        self._messages_ref = messages_ref
        self._task_store = task_store

    def set_messages_ref(self, messages: List[Dict[str, Any]]):
        """Set reference to current conversation messages."""
        self._messages_ref = messages

    @property
    def name(self) -> str:
        return "handoff"

    @property
    def description(self) -> str:
        return (
            "Create or load a handoff document for cross-session continuity.\n\n"
            "Actions:\n"
            "- `create`: Capture current session state for later continuation.\n"
            "- `load`: Load a previous handoff by ID.\n"
            "- `load_latest`: Load the most recent handoff.\n"
            "- `list`: List all available handoffs."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "load", "load_latest", "list"],
                    "description": "Handoff action to perform"
                },
                "handoff_id": {
                    "type": "string",
                    "description": "Handoff ID to load (for 'load' action)"
                },
                "notes": {
                    "type": "string",
                    "description": "Free-form notes to include in the handoff (for 'create' action)"
                },
                "agent_name": {
                    "type": "string",
                    "description": "Agent name creating the handoff (for 'create' action)"
                }
            },
            "required": ["action"]
        }

    async def execute(
        self,
        action: str,
        handoff_id: Optional[str] = None,
        notes: str = "",
        agent_name: str = "agent",
    ) -> ToolResult:
        """Execute handoff action."""
        try:
            if action == "create":
                return await self._create_handoff(agent_name, notes)
            elif action == "load":
                if not handoff_id:
                    return ToolResult(
                        content="handoff_id is required for 'load' action.",
                        success=False, error="Missing handoff_id"
                    )
                return self._load_handoff(handoff_id)
            elif action == "load_latest":
                return self._load_latest()
            elif action == "list":
                return self._list_handoffs()
            else:
                return ToolResult(
                    content=f"Unknown action: {action}",
                    success=False, error=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error(f"Handoff action '{action}' failed: {e}")
            return ToolResult(content="", success=False, error=str(e))

    async def _create_handoff(self, agent_name: str, notes: str) -> ToolResult:
        """Create a handoff from current state."""
        messages = self._messages_ref or []

        # Get task snapshots if available
        tasks = []
        if self._task_store:
            for t in self._task_store.list_tasks():
                tasks.append(t.to_dict())

        doc = self._manager.create(
            messages=messages,
            agent_name=agent_name,
            tasks=tasks,
            notes=notes,
        )

        return ToolResult(
            content=(
                f"Handoff created: {doc.id}\n"
                f"Summary: {doc.summary[:200]}...\n"
                f"Key files: {doc.key_files[:5]}\n"
                f"Tasks snapshot: {len(doc.task_snapshot)} tasks\n\n"
                f"To resume in a new session, use: handoff(action='load', handoff_id='{doc.id}')"
            ),
            success=True,
        )

    def _load_handoff(self, handoff_id: str) -> ToolResult:
        """Load a specific handoff.

        TODO (Arch #22): Currently returns handoff as flat text in a ToolResult.
        For true cross-session context restoration, the context_messages from
        the handoff document should be injected directly into the rollout's
        message history (not just rendered as text). This requires a callback
        mechanism from the tool to the rollout to inject messages.
        """
        doc = self._manager.load(handoff_id)
        if not doc:
            return ToolResult(
                content=f"Handoff '{handoff_id}' not found.",
                success=False, error="Not found"
            )

        context_msg = doc.to_context_message()
        return ToolResult(
            content=(
                f"Handoff loaded: {doc.id}\n"
                f"From: {doc.agent_name} at {doc.created_at}\n\n"
                f"{context_msg['content']}"
            ),
            success=True,
        )

    def _load_latest(self) -> ToolResult:
        """Load the most recent handoff."""
        doc = self._manager.load_latest()
        if not doc:
            return ToolResult(content="No handoffs available.", success=True)

        context_msg = doc.to_context_message()
        return ToolResult(
            content=(
                f"Latest handoff loaded: {doc.id}\n"
                f"From: {doc.agent_name} at {doc.created_at}\n\n"
                f"{context_msg['content']}"
            ),
            success=True,
        )

    def _list_handoffs(self) -> ToolResult:
        """List all available handoffs."""
        ids = self._manager.list_handoffs()
        if not ids:
            return ToolResult(content="No handoffs available.", success=True)

        return ToolResult(
            content=f"Available handoffs ({len(ids)}):\n" + "\n".join(f"  - {hid}" for hid in ids),
            success=True,
        )
