"""Base Tool - Abstract interface for all tools"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ToolResult:
    """Result of tool execution"""
    content: Any
    success: bool = True
    error: Optional[str] = None

    def to_str(self) -> str:
        """Convert result to string for LLM consumption"""
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.content, str):
            return self.content
        return str(self.content)


class BaseTool(ABC):
    """Abstract base class for all tools

    Subclasses must implement:
        - name: Tool identifier
        - description: What the tool does
        - parameters: JSON Schema for parameters
        - execute: Tool execution logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult containing the execution result
        """
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    async def __call__(self, **kwargs) -> ToolResult:
        """Allow calling tool as a function"""
        return await self.execute(**kwargs)
