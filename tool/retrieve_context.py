"""Retrieve context tool - local codebase-aware retrieval via KnowledgeEngine."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .base import BaseTool, ToolResult
from ..utils.knowledge_engine import KnowledgeEngine


class RetrieveContextTool(BaseTool):
    """Retrieve relevant codebase context for a task or question."""

    def __init__(
        self,
        knowledge_engine: Optional[KnowledgeEngine] = None,
        root_dir: Optional[str] = None,
    ):
        self.root_dir = os.path.abspath(root_dir or os.getcwd())
        self.knowledge_engine = knowledge_engine or KnowledgeEngine(root_dir=self.root_dir)

    @property
    def name(self) -> str:
        return "retrieve_context"

    @property
    def description(self) -> str:
        return (
            "Retrieve relevant local codebase context using a lightweight knowledge engine.\n"
            "Best for architecture questions, symbol discovery, related files, and task preparation.\n"
            "Returns ranked file hits, matching symbols/headings, snippets, and related paths."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question or retrieval intent, e.g. 'task store claim flow' or 'team mailbox cleanup'."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of primary hits to return (default: 5)."
                },
                "include_related": {
                    "type": "boolean",
                    "description": "If true, include structurally related files such as import neighbors."
                },
                "file_hint": {
                    "type": "string",
                    "description": "Optional path substring to narrow retrieval, e.g. 'swarm_tool/' or 'README.md'."
                },
                "refresh_index": {
                    "type": "boolean",
                    "description": "If true, rebuild the local index before searching."
                },
                "output_format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Return either a readable text summary or raw JSON."
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        include_related: bool = True,
        file_hint: Optional[str] = None,
        refresh_index: bool = False,
        output_format: str = "text",
    ) -> ToolResult:
        hits = self.knowledge_engine.search(
            query=query,
            top_k=max(1, min(top_k, 10)),
            include_related=include_related,
            file_hint=file_hint,
            refresh_index=refresh_index,
        )
        if not hits:
            return ToolResult(
                content=f"No local context hits found for query: {query}",
                success=True,
            )

        if output_format == "json":
            return ToolResult(
                content=json.dumps(hits, ensure_ascii=False, indent=2),
                success=True,
            )

        blocks = [f"Local context hits for: {query}"]
        for idx, hit in enumerate(hits, 1):
            blocks.append(
                "\n".join([
                    f"{idx}. {hit['path']} (score={hit['score']})",
                    f"   reasons: {', '.join(hit['reasons']) or 'n/a'}",
                    f"   symbols: {', '.join(hit['symbols']) or 'n/a'}",
                    f"   related: {', '.join(hit['related_paths']) or 'n/a'}",
                    "   snippet:",
                    "\n".join(f"   {line}" for line in str(hit["snippet"]).splitlines()),
                ])
            )

        return ToolResult(content="\n\n".join(blocks), success=True)
