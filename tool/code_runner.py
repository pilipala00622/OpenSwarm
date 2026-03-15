"""
CodeRunner Tool - 与网关示例格式一致

请求中 tools 格式示例：
  {
    "type": "function",
    "function": {
      "name": "CodeRunner",
      "description": "代码执行器，支持运行 python 和 javascript 代码",
      "parameters": {
        "type": "object",
        "properties": {
          "language": { "type": "string", "enum": ["python", "javascript"] },
          "code": { "type": "string", "description": "代码写在这里" }
        }
      }
    }
  }
"""

import asyncio
import logging
import subprocess
import tempfile
from typing import Dict, Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

RUN_TIMEOUT = 15  # 秒


class CodeRunnerTool(BaseTool):
    """代码执行器，支持运行 python 和 javascript 代码（与网关示例 schema 一致）"""

    @property
    def name(self) -> str:
        return "CodeRunner"

    @property
    def description(self) -> str:
        return "代码执行器，支持运行 python 和 javascript 代码"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript"],
                },
                "code": {
                    "type": "string",
                    "description": "代码写在这里",
                },
            },
        }

    async def execute(self, language: str, code: str) -> ToolResult:
        if not code or not isinstance(code, str):
            return ToolResult(content="", success=False, error="code 不能为空")
        lang = (language or "python").strip().lower()
        if lang not in ("python", "javascript"):
            return ToolResult(content="", success=False, error="language 须为 python 或 javascript")

        try:
            if lang == "python":
                out = await self._run_python(code)
            else:
                out = await self._run_javascript(code)
            return ToolResult(content=out, success=True)
        except Exception as e:
            logger.exception("CodeRunner failed")
            return ToolResult(content="", success=False, error=str(e))

    async def _run_python(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(code.encode("utf-8"))
            path = f.name
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=RUN_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"执行超时（{RUN_TIMEOUT}s）"
            out = (stdout.decode("utf-8", errors="replace") or "").strip()
            err = (stderr.decode("utf-8", errors="replace") or "").strip()
            if err:
                out = f"{out}\nstderr:\n{err}" if out else f"stderr:\n{err}"
            return out or "(无输出)"
        finally:
            try:
                import os
                os.unlink(path)
            except Exception:
                pass

    async def _run_javascript(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as f:
            f.write(code.encode("utf-8"))
            path = f.name
        try:
            proc = await asyncio.create_subprocess_exec(
                "node",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=RUN_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"执行超时（{RUN_TIMEOUT}s）"
            out = (stdout.decode("utf-8", errors="replace") or "").strip()
            err = (stderr.decode("utf-8", errors="replace") or "").strip()
            if err:
                out = f"{out}\nstderr:\n{err}" if out else f"stderr:\n{err}"
            return out or "(无输出)"
        except FileNotFoundError:
            return "未找到 node，请安装 Node.js 或在网关侧执行 javascript"
        finally:
            try:
                import os
                os.unlink(path)
            except Exception:
                pass
