"""Task Management Tools - Create, query, and update managed tasks.

Provides structured task management with dependency tracking,
inspired by Oh-My-OpenCode's task system.

Usage:
    task_create: Create a new task with optional dependencies
    task_get: Retrieve a task by ID
    task_list: List all active tasks
    task_update: Update task status/metadata
"""

import json
import logging
from typing import Dict, Any, Optional

from ..tool.base import BaseTool, ToolResult
from ..utils.task_store import TaskStore

logger = logging.getLogger(__name__)


class TaskCreateTool(BaseTool):
    """Create a new managed task with optional dependency tracking."""

    def __init__(self, task_store: TaskStore):
        self._store = task_store

    @property
    def name(self) -> str:
        return "task_create"

    @property
    def description(self) -> str:
        return (
            "Create a new managed task with auto-generated ID.\n"
            "Tasks support dependencies via `blocked_by` — a task won't start "
            "until all blockers are completed.\n"
            "Tasks with no blockers can run in parallel."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Imperative task description (e.g. 'Build frontend')"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed task description"
                },
                "blocked_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of task IDs that must complete before this task can start"
                },
                "owner": {
                    "type": "string",
                    "description": "Agent name that will own this task"
                },
            },
            "required": ["subject"]
        }

    async def execute(
        self,
        subject: str,
        description: str = "",
        blocked_by: Optional[list] = None,
        owner: Optional[str] = None,
    ) -> ToolResult:
        task = self._store.create(
            subject=subject,
            description=description,
            blocked_by=blocked_by,
            owner=owner,
        )
        blocker_info = f" (blocked by: {task.blocked_by})" if task.blocked_by else " (ready to start)"
        return ToolResult(
            content=f"Task created: {task.id} — {subject}{blocker_info}",
            success=True,
        )


class TaskGetTool(BaseTool):
    """Retrieve a task by ID."""

    def __init__(self, task_store: TaskStore):
        self._store = task_store

    @property
    def name(self) -> str:
        return "task_get"

    @property
    def description(self) -> str:
        return "Retrieve a managed task by its ID."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Task ID (e.g. 'T-abc12345')"
                }
            },
            "required": ["id"]
        }

    async def execute(self, id: str) -> ToolResult:
        task = self._store.get(id)
        if not task:
            return ToolResult(content=f"Task '{id}' not found.", success=False, error="Not found")

        info = json.dumps(task.to_dict(), ensure_ascii=False, indent=2)
        return ToolResult(content=info, success=True)


class TaskListTool(BaseTool):
    """List all active tasks."""

    def __init__(self, task_store: TaskStore):
        self._store = task_store

    @property
    def name(self) -> str:
        return "task_list"

    @property
    def description(self) -> str:
        return "List all active managed tasks with their status and dependencies."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by status (pending, in_progress, completed)"
                },
                "owner": {
                    "type": "string",
                    "description": "Filter by owner agent name"
                }
            }
        }

    async def execute(
        self, status: Optional[str] = None, owner: Optional[str] = None
    ) -> ToolResult:
        tasks = self._store.list_tasks(status=status, owner=owner)
        if not tasks:
            return ToolResult(content="No tasks found.", success=True)

        lines = []
        for t in tasks:
            icon = {"pending": "○", "in_progress": "◉", "completed": "●"}.get(t.status, "?")
            blocked = f" blockedBy: {t.blocked_by}" if t.blocked_by else ""
            blocks = f" blocks: {t.blocks}" if t.blocks else ""
            lines.append(f"{icon} {t.id} [{t.status}] {t.subject}{blocked}{blocks}")

        # Also show ready tasks
        ready = self._store.get_ready_tasks()
        if ready:
            lines.append(f"\nReady to start: {', '.join(t.id for t in ready)}")

        return ToolResult(content="\n".join(lines), success=True)


class TaskUpdateTool(BaseTool):
    """Update a task's status or metadata."""

    def __init__(self, task_store: TaskStore):
        self._store = task_store

    @property
    def name(self) -> str:
        return "task_update"

    @property
    def description(self) -> str:
        return (
            "Update a managed task's status. When a task is completed, "
            "dependent tasks are automatically unblocked."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Task ID to update"
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "deleted"],
                    "description": "New status"
                },
                "result": {
                    "type": "string",
                    "description": "Task result or output"
                },
                "active_form": {
                    "type": "string",
                    "description": "Present continuous description (e.g. 'Building frontend')"
                }
            },
            "required": ["id"]
        }

    async def execute(
        self,
        id: str,
        status: Optional[str] = None,
        result: Optional[str] = None,
        active_form: Optional[str] = None,
    ) -> ToolResult:
        task = self._store.update(
            task_id=id,
            status=status,
            result=result,
            active_form=active_form,
        )
        if not task:
            return ToolResult(content=f"Task '{id}' not found.", success=False, error="Not found")

        msg = f"Task {id} updated: [{task.status}] {task.subject}"
        if task.blocked_by:
            msg += f" (still blocked by: {task.blocked_by})"
        return ToolResult(content=msg, success=True)
