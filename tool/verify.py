"""Verify Tool - Validate research results and check factual consistency

Inspired by Anthropic's CitationAgent pattern: before presenting final results,
the agent should verify claims against evidence and check for consistency.

This tool prompts the LLM to critically evaluate a claim and its supporting
evidence, helping catch hallucinations and unsupported assertions.
"""

import logging
from typing import Dict, Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class VerifyTool(BaseTool):
    """Tool for verifying research results and checking factual consistency.

    The agent can call this tool to validate claims against evidence before
    including them in the final response. This implements a lightweight version
    of Anthropic's dedicated CitationAgent pattern.
    """

    @property
    def name(self) -> str:
        return "verify_result"

    @property
    def description(self) -> str:
        return (
            "Verify the quality and consistency of research results.\n"
            "Use this tool to:\n"
            "1. Check if a claim is supported by the provided evidence\n"
            "2. Identify contradictions between multiple sources\n"
            "3. Rate the confidence level of findings\n"
            "Call this before presenting important findings to ensure accuracy."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The claim or finding to verify"
                },
                "evidence": {
                    "type": "string",
                    "description": "The evidence or sources supporting this claim"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context about the research task (optional)"
                }
            },
            "required": ["claim", "evidence"]
        }

    async def execute(
        self,
        claim: str,
        evidence: str,
        context: str = "",
    ) -> ToolResult:
        """Verify a claim against its evidence.

        This tool returns a structured verification prompt that guides the LLM
        to critically evaluate the claim. The actual verification happens in the
        LLM's reasoning when it processes the tool result.

        Args:
            claim: The claim or finding to verify
            evidence: Supporting evidence
            context: Optional additional context

        Returns:
            ToolResult with verification instructions
        """
        verification_prompt = [
            "=== VERIFICATION REQUEST ===",
            "",
            f"**Claim**: {claim}",
            "",
            f"**Evidence**: {evidence}",
        ]

        if context:
            verification_prompt.append(f"\n**Context**: {context}")

        verification_prompt.extend([
            "",
            "**Please evaluate the following**:",
            "1. SUPPORT: Does the evidence directly support the claim? (Yes/Partial/No)",
            "2. CONTRADICTIONS: Are there any contradictions between the claim and evidence?",
            "3. GAPS: What information is missing to fully verify this claim?",
            "4. CONFIDENCE: Rate your confidence in this claim (High/Medium/Low)",
            "5. RECOMMENDATION: Should this claim be included as-is, modified, or excluded?",
            "",
            "Provide your assessment below:",
        ])

        logger.info(f"Verification requested for claim: {claim[:100]}...")

        return ToolResult(
            content="\n".join(verification_prompt),
            success=True,
        )
