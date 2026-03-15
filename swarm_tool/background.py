"""Background Tools - Retrieve and cancel background task results.

These tools work in conjunction with TaskTool's run_in_background feature,
allowing the agent to launch tasks asynchronously and check on them later.

Inspired by Oh-My-OpenCode's background_output / background_cancel tools.
"""

import logging
from typing import Dict, Any, Optional

from ..tool.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class BackgroundOutputTool(BaseTool):
    """Tool to retrieve results from background tasks.

    After launching a task with run_in_background=true,
    use this tool with the returned task_id to check status
    and retrieve results.
    """

    def __init__(self, task_tool):
        """Initialize with reference to TaskTool.

        Args:
            task_tool: The TaskTool instance that manages background tasks.
        """
        self._task_tool = task_tool

    @property
    def name(self) -> str:
        return "background_output"

    @property
    def description(self) -> str:
        return (
            "Retrieve results from a background task.\n"
            "Use the task_id returned by assign_task(run_in_background=true).\n"
            "Returns the task result if completed, or status if still running."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task ID (e.g. 'subagent_1') returned when launching the background task"
                }
            },
            "required": ["task_id"]
        }

    async def execute(self, task_id: str) -> ToolResult:
        """Retrieve background task result.

        Args:
            task_id: Background task identifier.

        Returns:
            ToolResult with task output or status.
        """
        # Check if still running
        if self._task_tool.is_background_running(task_id):
            return ToolResult(
                content=f"Task '{task_id}' is still running. Try again later.",
                success=True,
            )

        # Check for result
        result = self._task_tool.get_background_result(task_id)
        if result is None:
            # Check if it was never launched
            return ToolResult(
                content=f"Task '{task_id}' not found. It may not have been launched as a background task.",
                success=False,
                error=f"Task '{task_id}' not found"
            )

        content = result.get("content", "No content")
        status = result.get("status", "unknown")

        return ToolResult(
            content=f"[Background task {task_id} - {status}]\n\n{content}",
            success=status in ("completed", "max_steps_reached"),
        )


class BackgroundCancelTool(BaseTool):
    """Tool to cancel a running background task."""

    def __init__(self, task_tool):
        """Initialize with reference to TaskTool.

        Args:
            task_tool: The TaskTool instance that manages background tasks.
        """
        self._task_tool = task_tool

    @property
    def name(self) -> str:
        return "background_cancel"

    @property
    def description(self) -> str:
        return (
            "Cancel a running background task.\n"
            "Use the task_id returned by assign_task(run_in_background=true)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task ID to cancel"
                }
            },
            "required": ["task_id"]
        }

    async def execute(self, task_id: str) -> ToolResult:
        """Cancel a background task.

        Args:
            task_id: Background task identifier.

        Returns:
            ToolResult indicating success.
        """
        if self._task_tool.cancel_background(task_id):
            return ToolResult(
                content=f"Background task '{task_id}' has been cancelled.",
                success=True,
            )
        else:
            return ToolResult(
                content=f"Task '{task_id}' not found or already completed.",
                success=False,
                error=f"Cannot cancel '{task_id}'"
            )
