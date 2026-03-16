"""Team mailbox - lightweight file-backed messaging for team-mode agents."""

import json
import os
import shutil
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TeamMailbox:
    """File-backed mailbox for team-mode collaboration.

    Layout:
    - {base_dir}/{team_id}/members.json
    - {base_dir}/{team_id}/mailbox/{agent_id}.jsonl
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), ".agent_team")
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = Lock()

    def _team_dir(self, team_id: str) -> str:
        return os.path.join(self.base_dir, team_id)

    def _members_path(self, team_id: str) -> str:
        return os.path.join(self._team_dir(team_id), "members.json")

    def _mailbox_dir(self, team_id: str) -> str:
        return os.path.join(self._team_dir(team_id), "mailbox")

    def _inbox_path(self, team_id: str, agent_id: str) -> str:
        return os.path.join(self._mailbox_dir(team_id), f"{agent_id}.jsonl")

    def _ensure_team_dirs(self, team_id: str):
        os.makedirs(self._team_dir(team_id), exist_ok=True)
        os.makedirs(self._mailbox_dir(team_id), exist_ok=True)

    def register_member(self, team_id: str, agent_id: str, role: str = "teammate"):
        self._ensure_team_dirs(team_id)
        members_path = self._members_path(team_id)
        with self._lock:
            members = self.list_members(team_id)
            existing = next((m for m in members if m["agent_id"] == agent_id), None)
            if existing:
                existing["role"] = role
                existing["updated_at"] = _utcnow_iso()
            else:
                members.append({
                    "agent_id": agent_id,
                    "role": role,
                    "joined_at": _utcnow_iso(),
                    "updated_at": _utcnow_iso(),
                })
            with open(members_path, "w", encoding="utf-8") as f:
                json.dump(members, f, ensure_ascii=False, indent=2)

    def list_members(self, team_id: str) -> List[Dict[str, str]]:
        members_path = self._members_path(team_id)
        if not os.path.exists(members_path):
            return []
        with open(members_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def send_message(
        self,
        team_id: str,
        sender: str,
        content: str,
        recipient: Optional[str] = None,
        broadcast: bool = False,
    ) -> int:
        self._ensure_team_dirs(team_id)
        delivered = 0
        with self._lock:
            if broadcast:
                recipients = [
                    member["agent_id"]
                    for member in self.list_members(team_id)
                    if member["agent_id"] != sender
                ]
            elif recipient:
                recipients = [recipient]
            else:
                recipients = []

            payload = {
                "team_id": team_id,
                "sender": sender,
                "recipient": recipient if not broadcast else "*",
                "content": content,
                "broadcast": broadcast,
                "created_at": _utcnow_iso(),
            }

            for target in recipients:
                inbox_path = self._inbox_path(team_id, target)
                with open(inbox_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                delivered += 1

        return delivered

    def fetch_messages(
        self,
        team_id: str,
        agent_id: str,
        clear: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        inbox_path = self._inbox_path(team_id, agent_id)
        if not os.path.exists(inbox_path):
            return []

        with self._lock:
            with open(inbox_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if clear:
                os.remove(inbox_path)

        messages = [json.loads(line) for line in lines]
        if limit is not None:
            messages = messages[:limit]
        return messages

    def cleanup_team(self, team_id: str) -> bool:
        """Delete all persisted mailbox state for a team."""
        team_dir = self._team_dir(team_id)
        if not os.path.exists(team_dir):
            return False
        with self._lock:
            if os.path.exists(team_dir):
                shutil.rmtree(team_dir)
        return True
