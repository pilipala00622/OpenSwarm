"""LLM Client - Simple wrapper for OpenAI-compatible APIs"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class LLMClient:
    """Simple LLM client using OpenAI SDK"""

    def __init__(
        self,
        model_id: str = "kimi-k2-0711-preview",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        import httpx
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=30.0),
            max_retries=0,  # We handle retries ourselves
        )

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        thinking_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request with retry logic

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model override (uses self.model_id if None)
            top_p: Nucleus sampling parameter (omitted if None)
            reasoning_effort: Reasoning effort level for supported models
            thinking_budget: Extended thinking token budget for supported models

        Returns:
            Response dict with content and optional tool_calls
        """
        kwargs = {
            "model": model or self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if top_p is not None:
            kwargs["top_p"] = top_p

        # Pass reasoning/thinking params for models that support them
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if thinking_budget is not None and thinking_budget > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(**kwargs)

                # Extract response
                choice = response.choices[0]
                message = choice.message

                result = {
                    "role": "assistant",
                    "content": message.content or "",
                    "finish_reason": choice.finish_reason,
                }

                # Handle reasoning_content for thinking models (like kimi-k2.5)
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    result["reasoning_content"] = message.reasoning_content

                # Handle tool calls
                if message.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in message.tool_calls
                    ]

                # Extract usage info for token tracking (Arch #19)
                if hasattr(response, 'usage') and response.usage:
                    result["usage"] = {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response.usage, 'total_tokens', 0),
                    }

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"LLM request attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        logger.error(f"LLM request failed after {MAX_RETRIES} attempts: {last_error}")
        # Raise exception instead of returning error content (Fix #2)
        raise RuntimeError(f"LLM request failed after {MAX_RETRIES} attempts: {last_error}")
