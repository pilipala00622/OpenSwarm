"""Team collaboration tools for team-mode sub-agents."""

import json
from typing import Any, Dict, Optional

from ..tool.base import BaseTool, ToolResult
from ..utils.team_mailbox import TeamMailbox


class TeamMessageTool(BaseTool):
    """Send a direct or broadcast message to teammates."""

    def __init__(self, mailbox: TeamMailbox, team_id: str, sender_id: str):
        self._mailbox = mailbox
        self._team_id = team_id
        self._sender_id = sender_id

    @property
    def name(self) -> str:
        return "team_message"

    @property
    def description(self) -> str:
        return (
            "Send a message to one teammate or broadcast to the full team.\n"
            "Use this in team mode to share findings, challenge assumptions, "
            "or coordinate dependencies."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send."
                },
                "recipient": {
                    "type": "string",
                    "description": "Teammate agent_id to message directly. Omit when broadcasting."
                },
                "broadcast": {
                    "type": "boolean",
                    "description": "If true, deliver the message to all teammates except the sender."
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        recipient: Optional[str] = None,
        broadcast: bool = False,
    ) -> ToolResult:
        if not broadcast and not recipient:
            return ToolResult(
                content="",
                success=False,
                error="Either recipient or broadcast=true is required",
            )

        delivered = self._mailbox.send_message(
            team_id=self._team_id,
            sender=self._sender_id,
            recipient=recipient,
            content=content,
            broadcast=broadcast,
        )
        target = "team" if broadcast else recipient
        if delivered == 0:
            return ToolResult(
                content="",
                success=False,
                error=(
                    f"No valid recipients found for message to {target}. "
                    "The teammate may not exist or may not be registered yet."
                ),
            )
        return ToolResult(
            content=f"Message sent from '{self._sender_id}' to {target}. Delivered to {delivered} inbox(es).",
            success=True,
        )


class TeamInboxTool(BaseTool):
    """Read pending messages for the current teammate."""

    def __init__(self, mailbox: TeamMailbox, team_id: str, recipient_id: str):
        self._mailbox = mailbox
        self._team_id = team_id
        self._recipient_id = recipient_id

    @property
    def name(self) -> str:
        return "team_inbox"

    @property
    def description(self) -> str:
        return (
            "Read pending inbox messages for the current teammate.\n"
            "Use this periodically in team mode to incorporate teammates' findings."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "clear": {
                    "type": "boolean",
                    "description": "If true, remove messages from the inbox after reading (default: true)."
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional maximum number of messages to return."
                }
            }
        }

    async def execute(
        self,
        clear: bool = True,
        limit: Optional[int] = None,
    ) -> ToolResult:
        messages = self._mailbox.fetch_messages(
            team_id=self._team_id,
            agent_id=self._recipient_id,
            clear=clear,
            limit=limit,
        )
        if not messages:
            return ToolResult(content="No pending team messages.", success=True)

        return ToolResult(
            content=json.dumps(messages, ensure_ascii=False, indent=2),
            success=True,
        )


class TeamLeadInboxTool(BaseTool):
    """Lead-facing inbox reader backed by TaskTool."""

    def __init__(self, task_tool):
        self._task_tool = task_tool

    @property
    def name(self) -> str:
        return "team_lead_inbox"

    @property
    def description(self) -> str:
        return (
            "Read pending team messages addressed to the lead agent.\n"
            "Useful in team mode to review teammate updates before synthesizing results."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "clear": {
                    "type": "boolean",
                    "description": "If true, remove messages after reading (default: true)."
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional maximum number of messages to return."
                }
            }
        }

    async def execute(
        self,
        clear: bool = True,
        limit: Optional[int] = None,
    ) -> ToolResult:
        messages = self._task_tool.get_lead_inbox(clear=clear, limit=limit)
        if not messages:
            return ToolResult(content="No pending lead messages.", success=True)
        return ToolResult(
            content=json.dumps(messages, ensure_ascii=False, indent=2),
            success=True,
        )


class TeamLeadMessageTool(BaseTool):
    """Lead-facing team message sender backed by TaskTool."""

    def __init__(self, task_tool):
        self._task_tool = task_tool

    @property
    def name(self) -> str:
        return "team_lead_message"

    @property
    def description(self) -> str:
        return (
            "Send a message from the lead to one teammate or broadcast to the team.\n"
            "Use this to coordinate next steps, clarify expectations, or redirect work."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send from the lead."
                },
                "recipient": {
                    "type": "string",
                    "description": "Specific teammate agent_id to message directly."
                },
                "broadcast": {
                    "type": "boolean",
                    "description": "If true, send the message to all registered teammates."
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        recipient: Optional[str] = None,
        broadcast: bool = False,
    ) -> ToolResult:
        result = self._task_tool.send_lead_message(
            content=content,
            recipient=recipient,
            broadcast=broadcast,
        )
        return ToolResult(
            content=json.dumps(result, ensure_ascii=False, indent=2),
            success=result.get("success", False),
            error=None if result.get("success", False) else result.get("message"),
        )


class TeamMembersTool(BaseTool):
    """Lead-facing team membership inspector."""

    def __init__(self, task_tool):
        self._task_tool = task_tool

    @property
    def name(self) -> str:
        return "team_members"

    @property
    def description(self) -> str:
        return (
            "List current registered members for the active team.\n"
            "Useful for checking which teammates exist before sending targeted lead messages."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "include_lead": {
                    "type": "boolean",
                    "description": "If true, include the lead agent in the returned member list."
                }
            }
        }

    async def execute(self, include_lead: bool = True) -> ToolResult:
        members = self._task_tool.get_team_members(include_lead=include_lead)
        return ToolResult(
            content=json.dumps(members, ensure_ascii=False, indent=2),
            success=True,
        )


class TeamStatusTool(BaseTool):
    """Lead-facing team status inspector."""

    def __init__(self, task_tool):
        self._task_tool = task_tool

    @property
    def name(self) -> str:
        return "team_status"

    @property
    def description(self) -> str:
        return (
            "Return the current team status, including team id, members, "
            "and any still-running team-mode background tasks."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "include_lead": {
                    "type": "boolean",
                    "description": "If true, include the lead agent in the members list."
                }
            }
        }

    async def execute(self, include_lead: bool = True) -> ToolResult:
        status = self._task_tool.get_team_status(include_lead=include_lead)
        return ToolResult(
            content=json.dumps(status, ensure_ascii=False, indent=2),
            success=True,
        )


class TeamCleanupTool(BaseTool):
    """Lead-facing cleanup tool for team mailbox state."""

    def __init__(self, task_tool):
        self._task_tool = task_tool

    @property
    def name(self) -> str:
        return "team_cleanup"

    @property
    def description(self) -> str:
        return (
            "Clean up persisted team mailbox state for the current lead.\n"
            "Fails if team-mode background tasks are still running unless force=true."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Cancel running team-mode background tasks before cleanup."
                }
            }
        }

    async def execute(self, force: bool = False) -> ToolResult:
        result = self._task_tool.cleanup_team(force=force)
        return ToolResult(
            content=json.dumps(result, ensure_ascii=False, indent=2),
            success=result.get("success", False),
            error=None if result.get("success", False) else result.get("message"),
        )
