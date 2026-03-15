"""Search Tool - Using Serper API for web search"""

import os
import logging
import httpx
from typing import Dict, Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class SearchTool(BaseTool):
    """Web search tool using Serper API

    Serper provides Google Search results via API.
    Get your API key at: https://serper.dev

    Environment variable:
        SERPER_API_KEY: Your Serper API key
    """

    def __init__(self, api_key: str = None):
        """Initialize search tool

        Args:
            api_key: Serper API key. If not provided, uses SERPER_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.api_url = "https://google.serper.dev/search"

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Search for information on the web using Google.\n"
            "Returns search results with titles, snippets, and URLs.\n"
            "\n"
            "Search strategy (follow this order):\n"
            "1. Start with SHORT, BROAD queries to get an overview of the topic\n"
            "2. Then use SPECIFIC, FOCUSED queries to dig deeper into promising leads\n"
            "3. If initial results are poor, REPHRASE the query rather than adding more words\n"
            "4. Use different query angles for the same topic (synonyms, related terms)\n"
            "5. Prefer authoritative sources (academic papers, official docs) over SEO content farms"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str) -> ToolResult:
        """Execute search using Serper API

        Args:
            query: Search query string

        Returns:
            ToolResult with search results
        """
        if not self.api_key:
            return ToolResult(
                content="Error: SERPER_API_KEY environment variable not set. "
                        "Get your API key at https://serper.dev",
                success=False,
                error="Missing API key"
            )

        try:
            logger.info(f"Searching for: {query}")

            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }

            payload = {"q": query}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

            # Format results
            results = []

            # Organic search results
            organic = data.get("organic", [])
            for i, item in enumerate(organic[:5], 1):
                title = item.get("title", "No title")
                snippet = item.get("snippet", "No description")
                link = item.get("link", "No URL")
                results.append(f"{i}. {title}\n   {snippet}\n   URL: {link}")

            # Knowledge graph if available
            kg = data.get("knowledgeGraph", {})
            if kg:
                kg_title = kg.get("title", "")
                kg_desc = kg.get("description", "")
                if kg_title and kg_desc:
                    results.insert(0, f"[Knowledge Graph]\n{kg_title}: {kg_desc}\n")

            if not results:
                return ToolResult(
                    content=f"No results found for: {query}",
                    success=True
                )

            return ToolResult(
                content="\n\n".join(results),
                success=True
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Serper API error: {e}")
            return ToolResult(
                content="",
                success=False,
                error=f"Search API error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ToolResult(
                content="",
                success=False,
                error=str(e)
            )
