"""Agent Memory - External memory with file-backed storage and LLM summarisation

Implements Anthropic's approach of saving completed work phases to external
memory (file system), following their core principles:

1. **Write to disk**: Each compressed phase is persisted as an individual
   Markdown file, so memory survives beyond the current context window.
2. **Pass paths, not text**: Internally we track file paths; full content is
   loaded on-demand only when injecting the memory summary into context.
3. **LLM summarisation**: When an LLM client is provided, the agent's own
   model produces a high-quality summary instead of heuristic text truncation.
   Falls back to heuristic extraction when no LLM client is available.

The compressed message list looks like:
    [system_msg, memory_summary_msg, ...recent_messages]
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template used for LLM-based summarisation
# ---------------------------------------------------------------------------
_SUMMARISE_PROMPT = """\
You are a research-memory compressor. Given a block of conversation messages \
between an assistant and a user (including tool calls and results), produce a \
concise but information-dense summary that captures:

1. **What was done**: actions taken, tools called, queries searched.
2. **Key findings**: important facts, data points, conclusions reached.
3. **Open threads**: anything started but not yet resolved.

Rules:
- Be factual; do not invent information not present in the messages.
- Use bullet points for key findings (max 7 bullets).
- Keep the entire summary under 400 words.
- Write in the same language as the majority of the conversation.

Conversation messages to summarise:
---
{conversation}
---

Produce the summary now."""


@dataclass
class MemoryEntry:
    """A single summarised work phase stored in external memory (disk)."""
    phase: str              # Phase identifier (e.g. "phase_1")
    summary: str            # Human-readable summary of completed work
    key_findings: List[str] = field(default_factory=list)
    step_range: tuple = (0, 0)   # (start_step, end_step) this phase covers
    message_count: int = 0       # Number of messages that were compressed
    file_path: Optional[str] = None  # Path to the persisted .md file on disk


class AgentMemory:
    """External memory manager that compresses old messages, persists them to
    disk as Markdown files, and uses LLM summarisation for high-quality recall.

    Design aligned with Anthropic's multi-agent research system:
    - Each compression phase → independent .md file on disk
    - In-memory index maps phase → file path (lightweight)
    - Full content loaded on-demand when building the context message
    - LLM-based summarisation produces much richer summaries than heuristic
      text truncation (falls back to heuristic when no LLM client is set)

    Usage:
        memory = AgentMemory(
            max_context_messages=50,
            memory_dir="/tmp/agent_memory",
            llm_client=my_llm_client,       # optional, enables LLM summarisation
        )

        # Inside the rollout loop, before each LLM call:
        if memory.should_compress(messages):
            messages = await memory.compress(messages)
    """

    def __init__(
        self,
        max_context_messages: int = 50,
        keep_recent: int = 20,
        memory_dir: Optional[str] = None,
        llm_client: Optional["LLMClient"] = None,
        summarise_model: Optional[str] = None,
    ):
        """
        Args:
            max_context_messages: Trigger compression when message count exceeds this.
            keep_recent: Number of most-recent messages to preserve verbatim.
            memory_dir: Directory to persist memory files. If None, uses a
                        temp directory under the current working directory.
            llm_client: Optional LLMClient instance. When provided, summaries
                        are generated via LLM instead of heuristic extraction.
            summarise_model: Optional model override for the summarisation call.
                             Defaults to the llm_client's own model.
        """
        self.max_context_messages = max_context_messages
        self.keep_recent = keep_recent
        self.entries: List[MemoryEntry] = []

        # File-backed persistence
        self.memory_dir = memory_dir or os.path.join(os.getcwd(), ".agent_memory")
        os.makedirs(self.memory_dir, exist_ok=True)

        # LLM summarisation (optional)
        self._llm_client = llm_client
        self._summarise_model = summarise_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_llm_client(self, llm_client: "LLMClient"):
        """Set the LLM client for summarisation (Fix #17: public setter)."""
        self._llm_client = llm_client

    def should_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Check whether the message list is long enough to warrant compression."""
        return len(messages) > self.max_context_messages

    async def compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress older messages into a memory summary.

        Preserves:
          - The system message (messages[0])
          - The most recent ``keep_recent`` messages

        Everything in between is summarised (via LLM or heuristic), persisted
        to disk, and replaced with a compact ``[Memory]`` context message.

        Args:
            messages: Full message list (including system message at index 0).

        Returns:
            Compressed message list: [system, memory_msg, ...recent].
        """
        if not self.should_compress(messages):
            return messages

        if len(messages) <= self.keep_recent + 1:
            return messages

        system_msg = messages[0]
        recent = messages[-self.keep_recent:]
        middle = messages[1:-self.keep_recent]

        if not middle:
            return messages

        # --- Produce summary (LLM or heuristic) ---
        summary = await self._summarise_messages(middle)
        key_findings = self._extract_key_findings_from_summary(summary)
        phase_id = f"phase_{len(self.entries) + 1}"

        # --- Persist to disk ---
        file_path = self._write_phase_file(phase_id, summary, key_findings, middle)

        entry = MemoryEntry(
            phase=phase_id,
            summary=summary,
            key_findings=key_findings,
            step_range=(0, len(middle)),
            message_count=len(middle),
            file_path=file_path,
        )
        self.entries.append(entry)

        logger.info(
            f"Memory compression: {len(middle)} messages → {phase_id} "
            f"(persisted to {file_path}, total phases: {len(self.entries)})"
        )

        # --- Build compressed message list ---
        memory_msg = {
            "role": "user",
            "content": self._format_memory_context(),
        }
        return [system_msg, memory_msg] + recent

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    async def _summarise_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Produce a summary of a message block.

        Uses the LLM client when available for high-quality summarisation.
        Falls back to heuristic text extraction otherwise.
        """
        if self._llm_client is not None:
            return await self._llm_summarise(messages)
        return self._heuristic_summarise(messages)

    async def _llm_summarise(self, messages: List[Dict[str, Any]]) -> str:
        """Use the LLM to generate a high-quality summary of the messages.

        Fix #5: Uses model parameter override instead of mutating shared llm_client.model_id.
        """
        try:
            # Serialise messages into a readable block for the prompt
            conversation_text = self._messages_to_text(messages)

            prompt = _SUMMARISE_PROMPT.format(conversation=conversation_text)

            # Fix #5: Use model= parameter instead of mutating shared client state
            model_override = self._summarise_model or self._llm_client.model_id

            response = await self._llm_client.chat(
                messages=[
                    {"role": "system", "content": "You are a concise summariser."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
                model=model_override,
            )

            summary = (response.get("content") or "").strip()
            if summary:
                logger.info(f"LLM summarisation produced {len(summary)} chars")
                return summary

            # Fall back if LLM returned empty
            logger.warning("LLM summarisation returned empty, falling back to heuristic")
            return self._heuristic_summarise(messages)

        except Exception as e:
            logger.warning(f"LLM summarisation failed ({e}), falling back to heuristic")
            return self._heuristic_summarise(messages)

    def _heuristic_summarise(self, messages: List[Dict[str, Any]]) -> str:
        """Heuristic text-extraction summary (fallback when no LLM available)."""
        assistant_contents = []
        tool_names = []
        user_queries = []

        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "") or ""

            if role == "assistant" and content.strip():
                assistant_contents.append(content[:300])
            elif role == "tool":
                tool_id = m.get("tool_call_id", "unknown")
                tool_names.append(tool_id)
            elif role == "user" and not content.startswith("[System]"):
                if content.strip():
                    user_queries.append(content[:150])

        parts = [f"Processed {len(messages)} messages."]

        if user_queries:
            parts.append(f"User queries covered: {'; '.join(user_queries[:3])}")
        if tool_names:
            unique_tools = list(set(tool_names))[:5]
            parts.append(f"Tools used: {', '.join(unique_tools)}")
        if assistant_contents:
            parts.append(f"Latest findings: {assistant_contents[-1][:200]}")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # File persistence (Anthropic-style: write to disk, pass paths)
    # ------------------------------------------------------------------

    def _write_phase_file(
        self,
        phase_id: str,
        summary: str,
        key_findings: List[str],
        raw_messages: List[Dict[str, Any]],
    ) -> str:
        """Persist a compressed phase to a Markdown file on disk.

        File structure:
            # phase_1
            ## Summary
            <LLM or heuristic summary>
            ## Key Findings
            - finding 1
            - finding 2
            ## Raw Messages (JSON)
            <full serialised messages for potential re-read>

        Returns:
            Absolute path to the written file.
        """
        ts = int(time.time())
        filename = f"{phase_id}_{ts}.md"
        filepath = os.path.join(self.memory_dir, filename)

        lines = [
            f"# {phase_id}",
            f"",
            f"**Compressed at**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Messages compressed**: {len(raw_messages)}",
            f"",
            f"## Summary",
            f"",
            summary,
            f"",
        ]

        if key_findings:
            lines.append("## Key Findings")
            lines.append("")
            for finding in key_findings:
                lines.append(f"- {finding}")
            lines.append("")

        # Persist raw messages as JSON so the phase can be fully reconstructed
        lines.append("## Raw Messages (JSON)")
        lines.append("")
        lines.append("```json")
        try:
            lines.append(json.dumps(raw_messages, ensure_ascii=False, indent=1))
        except (TypeError, ValueError):
            lines.append("[]  # serialisation failed")
        lines.append("```")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Phase {phase_id} persisted to {filepath}")
        return filepath

    def _read_phase_summary(self, entry: MemoryEntry) -> str:
        """Read a phase summary from disk (on-demand).

        If the file is missing or unreadable, returns the in-memory summary.
        """
        if entry.file_path and os.path.exists(entry.file_path):
            try:
                with open(entry.file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Extract just the Summary section to keep context lean
                return self._extract_summary_section(content) or entry.summary
            except Exception as e:
                logger.warning(f"Could not read {entry.file_path}: {e}")
        return entry.summary

    @staticmethod
    def _extract_summary_section(markdown: str) -> Optional[str]:
        """Extract the '## Summary' section from a phase Markdown file."""
        in_summary = False
        lines = []
        for line in markdown.split("\n"):
            if line.strip().startswith("## Summary"):
                in_summary = True
                continue
            if in_summary:
                if line.strip().startswith("## "):
                    break  # Next section
                lines.append(line)
        text = "\n".join(lines).strip()
        return text if text else None

    # ------------------------------------------------------------------
    # Context formatting (on-demand read from disk)
    # ------------------------------------------------------------------

    def _format_memory_context(self) -> str:
        """Format all memory entries into a context message for the LLM.

        Reads summaries from disk on-demand, keeping the in-memory footprint
        minimal between compression events.
        """
        lines = [
            "[Memory] The following is a summary of earlier research phases. "
            "Use this context to avoid repeating work already done.\n"
        ]

        for entry in self.entries:
            # Read from disk (on-demand) rather than using stale in-memory text
            summary_text = self._read_phase_summary(entry)

            lines.append(f"### {entry.phase} ({entry.message_count} messages compressed)")
            lines.append(summary_text)
            if entry.key_findings:
                lines.append("Key findings:")
                for f in entry.key_findings:
                    lines.append(f"  - {f}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_text(messages: List[Dict[str, Any]], max_chars: int = 12000) -> str:
        """Serialise a block of messages into human-readable text for the LLM.

        Truncates to ``max_chars`` to avoid blowing up the summarisation prompt.
        """
        parts = []
        total = 0
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "") or ""

            # For tool-call assistant messages, show the function names
            if role == "assistant" and m.get("tool_calls"):
                tool_names = [
                    tc.get("function", {}).get("name", "?")
                    for tc in m["tool_calls"]
                ]
                entry = f"[assistant] (called tools: {', '.join(tool_names)})"
                if content:
                    entry += f"\n{content}"
            elif role == "tool":
                tool_id = m.get("tool_call_id", "")
                # Truncate long tool results
                truncated = content[:500] + ("..." if len(content) > 500 else "")
                entry = f"[tool:{tool_id}] {truncated}"
            else:
                entry = f"[{role}] {content}"

            if total + len(entry) > max_chars:
                parts.append(f"... ({len(messages) - len(parts)} messages truncated)")
                break
            parts.append(entry)
            total += len(entry)

        return "\n\n".join(parts)

    def _extract_key_findings_from_summary(self, summary: str) -> List[str]:
        """Extract bullet-point findings from a summary string.

        Works for both LLM-generated (bullet lists) and heuristic summaries.
        """
        findings = []
        for line in summary.split("\n"):
            stripped = line.strip()
            # Match markdown bullet points: "- xxx", "* xxx", "• xxx"
            if stripped and (stripped.startswith("- ") or stripped.startswith("* ")
                            or stripped.startswith("• ")):
                finding = stripped.lstrip("-*• ").strip()
                if finding:
                    findings.append(finding[:200])
        return findings[-7:]  # Keep at most 7 key findings

    def get_total_compressed(self) -> int:
        """Return the total number of messages that have been compressed."""
        return sum(e.message_count for e in self.entries)

    def get_phase_files(self) -> List[str]:
        """Return paths to all persisted phase files."""
        return [e.file_path for e in self.entries if e.file_path]

    def reset(self):
        """Clear all memory entries (does NOT delete files on disk)."""
        self.entries.clear()
