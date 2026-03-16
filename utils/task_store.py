"""Task Store - Persistent task management with dependency tracking.

Inspired by Oh-My-OpenCode's task system:
- Tasks have unique IDs (T-{uuid})
- Support blockedBy/blocks dependency relationships
- Status lifecycle: pending → in_progress → completed / deleted
- Tasks are stored as JSON files for cross-session persistence
- Tasks with empty blockedBy can run in parallel

Example dependency graph:
    [Build Frontend]    ──┐
                          ├──→ [Integration Tests] ──→ [Deploy]
    [Build Backend]     ──┘
"""

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def _utcnow_iso() -> str:
    """Get current UTC time as ISO string (Fix #16: avoid deprecated utcnow())."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Task:
    """A managed task with dependency tracking.

    Attributes:
        id: Unique task identifier (T-{uuid}).
        subject: Imperative description ("Run tests").
        description: Detailed task description.
        status: One of "pending", "in_progress", "completed", "deleted".
        active_form: Present continuous ("Running tests").
        blocks: Task IDs this task blocks.
        blocked_by: Task IDs blocking this task.
        owner: Agent name that owns this task.
        metadata: Arbitrary key-value metadata.
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.
        result: Task output/result when completed.
    """
    id: str = ""
    subject: str = ""
    description: str = ""
    status: str = "pending"
    active_form: str = ""
    blocks: List[str] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    result: Optional[str] = None

    def is_ready(self) -> bool:
        """Check if this task can be started (all blockers completed)."""
        return self.status == "pending" and len(self.blocked_by) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return asdict(self)


class TaskStore:
    """Persistent task store with dependency management.

    Tasks are stored as JSON files in a configurable directory,
    surviving restarts and enabling cross-session workflows.
    """

    def __init__(self, store_dir: Optional[str] = None):
        """Initialize task store.

        Args:
            store_dir: Directory for task JSON files. Defaults to .agent_tasks/.
        """
        self.store_dir = store_dir or os.path.join(os.getcwd(), ".agent_tasks")
        os.makedirs(self.store_dir, exist_ok=True)
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._load_all()

    def _task_path(self, task_id: str) -> str:
        """Get file path for a task."""
        return os.path.join(self.store_dir, f"{task_id}.json")

    def _load_all(self):
        """Load all tasks from disk."""
        if not os.path.exists(self.store_dir):
            return
        for filename in os.listdir(self.store_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.store_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    task = Task(**{
                        k: v for k, v in data.items()
                        if k in Task.__dataclass_fields__
                    })
                    self._tasks[task.id] = task
                except Exception as e:
                    logger.warning(f"Failed to load task from {filepath}: {e}")
        logger.info(f"TaskStore: loaded {len(self._tasks)} tasks from {self.store_dir}")

    def _save_task(self, task: Task):
        """Save a single task to disk.

        Fix #6: Uses atomic write (write to temp file then rename) to prevent
        corruption from crashes or concurrent writes.
        """
        filepath = self._task_path(task.id)
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, filepath)  # atomic on POSIX

    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: Optional[List[str]] = None,
        owner: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Create a new task.

        Args:
            subject: Imperative task description.
            description: Detailed description.
            blocked_by: List of task IDs that must complete first.
            owner: Agent name.
            metadata: Arbitrary metadata.

        Returns:
            The created Task.

        Raises:
            ValueError: If blocked_by would create a dependency cycle (Fix #7).
        """
        with self._lock:
            task_id = f"T-{uuid.uuid4().hex[:8]}"
            now = _utcnow_iso()

            # Fix #7: Validate blocked_by references exist
            resolved_blockers = []
            for bid in (blocked_by or []):
                if bid not in self._tasks:
                    logger.warning(f"Blocker task {bid} not found, ignoring")
                else:
                    resolved_blockers.append(bid)

            # Fix #7: Check for dependency cycles
            if resolved_blockers and self._would_create_cycle(task_id, resolved_blockers):
                raise ValueError(
                    f"Cannot create task {task_id}: blocked_by={resolved_blockers} "
                    f"would create a dependency cycle"
                )

            task = Task(
                id=task_id,
                subject=subject,
                description=description,
                status="pending",
                blocked_by=resolved_blockers,
                owner=owner,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
            )

            # Update reverse dependency: add this task to blockers' blocks list
            for blocker_id in task.blocked_by:
                blocker = self._tasks.get(blocker_id)
                if blocker and task_id not in blocker.blocks:
                    blocker.blocks.append(task_id)
                    blocker.updated_at = now
                    self._save_task(blocker)

            self._tasks[task_id] = task
            self._save_task(task)
            logger.info(f"Task created: {task_id} - {subject}")
            return task

    def _would_create_cycle(self, new_task_id: str, blocked_by: List[str]) -> bool:
        """Check if adding blocked_by edges would create a dependency cycle (Fix #7).

        Uses DFS: starting from each blocker, walk its blocker chain;
        if we ever reach new_task_id, there's a cycle.
        """
        visited = set()
        stack = list(blocked_by)
        while stack:
            current = stack.pop()
            if current == new_task_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            task = self._tasks.get(current)
            if task:
                stack.extend(task.blocked_by)
        return False

    def get(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def update(
        self,
        task_id: str,
        status: Optional[str] = None,
        result: Optional[str] = None,
        active_form: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Task]:
        """Update a task.

        When a task is completed, its ID is automatically removed from
        the blocked_by list of all dependent tasks, potentially unblocking them.

        Args:
            task_id: Task to update.
            status: New status.
            result: Task result (typically set on completion).
            active_form: Present-continuous description.
            metadata: Additional metadata to merge.

        Returns:
            Updated Task, or None if not found.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task not found: {task_id}")
                return None

            now = _utcnow_iso()
            task.updated_at = now

            if status:
                old_status = task.status
                task.status = status

                # When completed, unblock dependent tasks
                if status == "completed" and old_status != "completed":
                    self._unblock_dependents(task_id, now)

            if result is not None:
                task.result = result
            if active_form is not None:
                task.active_form = active_form
            if metadata:
                task.metadata.update(metadata)

            self._save_task(task)
            logger.info(f"Task updated: {task_id} -> {task.status}")
            return task

    def claim(
        self,
        task_id: str,
        owner: str,
        active_form: Optional[str] = None,
    ) -> Optional[Task]:
        """Atomically claim a ready task for an owner.

        Claim succeeds only when the task exists, is pending, has no blockers,
        and is either unowned or already owned by the same agent.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or not task.is_ready():
                return None
            if task.owner and task.owner != owner:
                return None

            task.owner = owner
            task.status = "in_progress"
            task.updated_at = _utcnow_iso()
            if active_form is not None:
                task.active_form = active_form
            self._save_task(task)
            logger.info(f"Task claimed: {task.id} by {owner}")
            return task

    def claim_next_ready(
        self,
        owner: str,
        active_form: Optional[str] = None,
    ) -> Optional[Task]:
        """Claim the next ready task for an owner."""
        with self._lock:
            ready = [
                task for task in self._tasks.values()
                if task.is_ready() and (task.owner is None or task.owner == owner)
            ]
            ready.sort(key=lambda task: (task.created_at, task.id))
            if not ready:
                return None

            task = ready[0]
            task.owner = owner
            task.status = "in_progress"
            task.updated_at = _utcnow_iso()
            if active_form is not None:
                task.active_form = active_form
            self._save_task(task)
            logger.info(f"Task claimed: {task.id} by {owner}")
            return task

    def _unblock_dependents(self, completed_id: str, timestamp: str):
        """Remove completed task from all dependents' blocked_by lists."""
        for task in self._tasks.values():
            if completed_id in task.blocked_by:
                task.blocked_by.remove(completed_id)
                task.updated_at = timestamp
                self._save_task(task)
                logger.info(
                    f"Task {task.id} unblocked (removed {completed_id}). "
                    f"Remaining blockers: {task.blocked_by}"
                )

    def list_tasks(
        self,
        status: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> List[Task]:
        """List tasks with optional filtering.

        Args:
            status: Filter by status.
            owner: Filter by owner.

        Returns:
            List of matching tasks.
        """
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        if owner:
            tasks = [t for t in tasks if t.owner == owner]
        return tasks

    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute (pending + no blockers)."""
        return [t for t in self._tasks.values() if t.is_ready()]

    def delete(self, task_id: str) -> bool:
        """Delete a task (mark as deleted + remove file).

        Args:
            task_id: Task to delete.

        Returns:
            True if deleted, False if not found.
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        task.status = "deleted"
        # Remove from blockers' blocks lists
        for other in self._tasks.values():
            if task_id in other.blocks:
                other.blocks.remove(task_id)
            if task_id in other.blocked_by:
                other.blocked_by.remove(task_id)

        # Remove file
        filepath = self._task_path(task_id)
        if os.path.exists(filepath):
            os.remove(filepath)

        del self._tasks[task_id]
        logger.info(f"Task deleted: {task_id}")
        return True

    def format_task_list(self) -> str:
        """Format all active tasks as a human-readable string."""
        active = [
            t for t in self._tasks.values()
            if t.status not in ("deleted",)
        ]
        if not active:
            return "No tasks."

        lines = []
        for t in sorted(active, key=lambda x: x.created_at):
            status_icon = {
                "pending": "○",
                "in_progress": "◉",
                "completed": "●",
            }.get(t.status, "?")
            blocker_str = f" blockedBy: {t.blocked_by}" if t.blocked_by else ""
            lines.append(f"  {status_icon} {t.id} [{t.status}] {t.subject}{blocker_str}")

        return "\n".join(lines)
