"""Knowledge engine - lightweight local indexing for codebase-aware context retrieval."""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")


@dataclass
class SymbolInfo:
    """Indexed symbol information."""

    name: str
    kind: str
    lineno: int


@dataclass
class FileRecord:
    """Indexed file contents and metadata."""

    path: str
    rel_path: str
    content: str
    mtime: float
    symbols: List[SymbolInfo] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)


class KnowledgeEngine:
    """Repo-local knowledge engine with lightweight lexical + structural retrieval.

    The implementation is intentionally simple:
    - lexical scoring over file path, symbols, headings, and file content
    - Python AST extraction for defs/imports
    - optional related-file expansion based on import/module relationships
    """

    DEFAULT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}
    DEFAULT_IGNORES = {
        ".git",
        ".idea",
        ".cursor",
        ".vscode",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        ".demo_agent_tasks",
        ".demo_agent_team",
        ".live_demo_tasks",
        ".live_demo_team",
        ".agent_team",
        ".agent_tasks",
        "result",
    }

    def __init__(
        self,
        root_dir: Optional[str] = None,
        include_extensions: Optional[Set[str]] = None,
        ignore_dirs: Optional[Set[str]] = None,
        max_file_size: int = 200_000,
    ):
        self.root_dir = os.path.abspath(root_dir or os.getcwd())
        self.include_extensions = include_extensions or set(self.DEFAULT_EXTENSIONS)
        self.ignore_dirs = ignore_dirs or set(self.DEFAULT_IGNORES)
        self.max_file_size = max_file_size
        self._records: Dict[str, FileRecord] = {}
        self._module_to_path: Dict[str, str] = {}
        self._built = False

    def build_index(self, refresh: bool = False):
        """Build or rebuild the local knowledge index."""
        if self._built and not refresh:
            return

        self._records = {}
        self._module_to_path = {}

        for current_root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            for filename in files:
                path = os.path.join(current_root, filename)
                ext = Path(path).suffix.lower()
                if ext not in self.include_extensions:
                    continue
                try:
                    if os.path.getsize(path) > self.max_file_size:
                        continue
                    record = self._index_file(path)
                    if record:
                        self._records[path] = record
                        module_name = self._module_name_for_path(record.rel_path)
                        if module_name:
                            self._module_to_path[module_name] = path
                except OSError:
                    continue

        self._built = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_related: bool = True,
        file_hint: Optional[str] = None,
        refresh_index: bool = False,
    ) -> List[Dict[str, object]]:
        """Search the local codebase and return ranked context hits."""
        self.build_index(refresh=refresh_index)

        hint = file_hint.replace("\\", "/") if file_hint else None
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scored: List[tuple[float, FileRecord, List[str]]] = []
        for record in self._records.values():
            if hint and hint not in record.rel_path:
                continue

            score, reasons = self._score_record(record, tokens)
            if score > 0:
                scored.append((score, record, reasons))

        scored.sort(key=lambda item: (-item[0], item[1].rel_path))
        primary = scored[: max(1, top_k)]

        results = []
        for score, record, reasons in primary:
            related = self._find_related(record) if include_related else []
            results.append({
                "path": record.rel_path,
                "score": round(score, 2),
                "reasons": reasons,
                "symbols": [symbol.name for symbol in record.symbols[:8]],
                "headings": record.headings[:5],
                "snippet": self._build_snippet(record, tokens),
                "related_paths": related,
            })
        return results

    def _index_file(self, path: str) -> Optional[FileRecord]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return None

        rel_path = os.path.relpath(path, self.root_dir).replace("\\", "/")
        record = FileRecord(
            path=path,
            rel_path=rel_path,
            content=content,
            mtime=os.path.getmtime(path),
        )

        ext = Path(path).suffix.lower()
        if ext == ".py":
            self._extract_python_metadata(record)
        elif ext == ".md":
            self._extract_markdown_metadata(record)
        return record

    def _extract_python_metadata(self, record: FileRecord):
        try:
            tree = ast.parse(record.content)
        except SyntaxError:
            return

        imports: List[str] = []
        symbols: List[SymbolInfo] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(SymbolInfo(name=node.name, kind="function", lineno=node.lineno))
            elif isinstance(node, ast.ClassDef):
                symbols.append(SymbolInfo(name=node.name, kind="class", lineno=node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        record.symbols = sorted(symbols, key=lambda symbol: (symbol.lineno, symbol.name))
        record.imports = imports

    def _extract_markdown_metadata(self, record: FileRecord):
        record.headings = [
            line.lstrip("#").strip()
            for line in record.content.splitlines()
            if line.startswith("#")
        ]

    def _score_record(self, record: FileRecord, tokens: List[str]) -> tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        rel_lower = record.rel_path.lower()
        content_lower = record.content.lower()
        symbol_names = [symbol.name.lower() for symbol in record.symbols]
        headings = [heading.lower() for heading in record.headings]

        for token in tokens:
            token_lower = token.lower()
            matched = False

            if token_lower in rel_lower:
                score += 4.0
                reasons.append(f"path:{token}")
                matched = True

            for symbol_name in symbol_names:
                if token_lower in symbol_name:
                    score += 3.0
                    reasons.append(f"symbol:{token}")
                    matched = True
                    break

            for heading in headings:
                if token_lower in heading:
                    score += 2.5
                    reasons.append(f"heading:{token}")
                    matched = True
                    break

            if token_lower in content_lower:
                score += 1.0
                if not matched:
                    reasons.append(f"content:{token}")

        if record.rel_path.endswith("README.md"):
            score += 0.3
        return score, reasons[:6]

    def _find_related(self, record: FileRecord) -> List[str]:
        related: List[str] = []
        rel_module = self._module_name_for_path(record.rel_path)

        for imported in record.imports[:8]:
            target_path = self._module_to_path.get(imported)
            if target_path and target_path != record.path:
                related.append(os.path.relpath(target_path, self.root_dir).replace("\\", "/"))

        if rel_module:
            for other in self._records.values():
                if other.path == record.path:
                    continue
                if rel_module in other.imports:
                    related.append(other.rel_path)

        deduped = []
        seen = set()
        for rel in related:
            if rel not in seen:
                seen.add(rel)
                deduped.append(rel)
        return deduped[:5]

    def _build_snippet(self, record: FileRecord, tokens: List[str], max_lines: int = 18) -> str:
        lines = record.content.splitlines()
        if not lines:
            return ""

        token_lowers = [token.lower() for token in tokens]
        hit_index = 0
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            if any(token in line_lower for token in token_lowers):
                hit_index = idx
                break

        start = max(0, hit_index - 3)
        end = min(len(lines), start + max_lines)
        numbered = [
            f"{i + 1}: {line}"
            for i, line in enumerate(lines[start:end], start=start)
        ]
        return "\n".join(numbered)

    def _tokenize(self, text: str) -> List[str]:
        raw = TOKEN_RE.findall(text)
        tokens = [token.strip().lower() for token in raw if len(token.strip()) >= 2]
        return list(dict.fromkeys(tokens))

    def _module_name_for_path(self, rel_path: str) -> Optional[str]:
        if not rel_path.endswith(".py"):
            return None
        module = rel_path[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        return module
