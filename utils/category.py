"""Category System - Agent configuration presets optimized for specific domains.

Inspired by Oh-My-OpenCode's category system:
- A Category answers "What kind of work is this?"
- It determines the model, temperature, prompt mindset, and tool restrictions.
- Categories can be combined with specialized tools (skills) to create optimal agents.

Built-in categories:
    - visual-engineering: Frontend, UI/UX, design, styling
    - ultrabrain: Deep logical reasoning, complex architecture
    - deep: Goal-oriented autonomous problem-solving
    - artistry: Highly creative/artistic tasks
    - quick: Trivial single-file changes
    - unspecified-low: Generic tasks, low effort
    - unspecified-high: Generic tasks, high effort
    - writing: Documentation, prose, technical writing
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class CategoryConfig:
    """Configuration preset for a task category.

    Attributes:
        description: Human-readable description shown in task prompts.
        model: Preferred model ID (e.g. "claude-opus-4-6").
        temperature: Creativity level (0.0-2.0).
        top_p: Nucleus sampling parameter (0.0-1.0).
        max_tokens: Maximum response token count.
        prompt_append: Extra text appended to system prompt.
        reasoning_effort: Reasoning effort level ("low", "medium", "high").
        thinking_budget: Extended thinking token budget (0 = disabled).
        blocked_tools: Tool names to block for this category.
        allowed_tools_only: If non-empty, ONLY these tools are allowed.
        can_delegate: Whether agents of this category can re-delegate.
    """
    description: str = ""
    model: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    prompt_append: str = ""
    reasoning_effort: str = "medium"
    thinking_budget: int = 0
    blocked_tools: List[str] = field(default_factory=list)
    allowed_tools_only: List[str] = field(default_factory=list)
    can_delegate: bool = False  # Sub-agents cannot re-delegate by default


# Built-in category definitions
BUILTIN_CATEGORIES: Dict[str, CategoryConfig] = {
    "visual-engineering": CategoryConfig(
        description="Frontend, UI/UX, design, styling, animation",
        model="gemini-3.1-pro",
        temperature=0.8,
        max_tokens=8192,
        prompt_append=(
            "You are a visual engineering specialist. "
            "Focus on crafting polished UI/UX with distinctive aesthetics. "
            "Pay close attention to responsive design, typography, colour palettes, and motion."
        ),
        can_delegate=False,
    ),
    "ultrabrain": CategoryConfig(
        description="Deep logical reasoning, complex architecture decisions requiring extensive analysis",
        model="gpt-5.4",
        temperature=0.3,
        max_tokens=16384,
        reasoning_effort="high",
        thinking_budget=32000,
        prompt_append=(
            "You are an ultra-deep reasoning agent. "
            "Analyse every angle before concluding. "
            "Provide rigorous, well-structured reasoning chains."
        ),
        # Read-only — no file edits, no further delegation
        blocked_tools=["create_subagent", "assign_task"],
        can_delegate=False,
    ),
    "deep": CategoryConfig(
        description="Goal-oriented autonomous problem-solving with thorough research before action",
        model="gpt-5.3-codex",
        temperature=0.5,
        max_tokens=8192,
        reasoning_effort="high",
        prompt_append=(
            "You are an autonomous deep worker. "
            "Research thoroughly before acting. "
            "Explore codebase patterns, complete tasks end-to-end without premature stopping."
        ),
        can_delegate=False,
    ),
    "artistry": CategoryConfig(
        description="Highly creative/artistic tasks, novel ideas",
        model="gemini-3.1-pro",
        temperature=1.2,
        max_tokens=8192,
        prompt_append=(
            "You are a creative artistry agent. "
            "Think boldly and unconventionally. Generate novel, surprising ideas."
        ),
        can_delegate=False,
    ),
    "quick": CategoryConfig(
        description="Trivial tasks — single file changes, typo fixes, simple modifications",
        model="claude-haiku-4-5",
        temperature=0.3,
        max_tokens=2048,
        reasoning_effort="low",
        prompt_append="Complete this quickly and concisely. Minimal changes only.",
        can_delegate=False,
    ),
    "unspecified-low": CategoryConfig(
        description="Tasks that don't fit other categories, low effort required",
        model="claude-sonnet-4-6",
        temperature=0.5,
        max_tokens=4096,
        can_delegate=False,
    ),
    "unspecified-high": CategoryConfig(
        description="Tasks that don't fit other categories, high effort required",
        model="claude-opus-4-6",
        temperature=0.5,
        max_tokens=8192,
        reasoning_effort="high",
        thinking_budget=16000,
        can_delegate=False,
    ),
    "writing": CategoryConfig(
        description="Documentation, prose, technical writing",
        model="gemini-3-flash",
        temperature=0.6,
        max_tokens=8192,
        prompt_append=(
            "You are a skilled technical writer. "
            "Produce clear, well-structured, polished prose. "
            "Maintain consistent tone and style throughout."
        ),
        can_delegate=False,
    ),
}


class CategoryRegistry:
    """Registry for managing task categories.

    Holds built-in categories and allows custom overrides.
    Thread-safe for read operations (categories are deepcopied on access).
    """

    def __init__(self, custom_categories: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialise registry with optional custom category overrides.

        Args:
            custom_categories: Dict mapping category names to config dicts.
                Can override built-in categories or define new ones.
        """
        # Start with built-in defaults
        self._categories: Dict[str, CategoryConfig] = deepcopy(BUILTIN_CATEGORIES)

        # Apply custom overrides
        if custom_categories:
            for name, overrides in custom_categories.items():
                self.register(name, overrides)

    def register(self, name: str, config: dict):
        """Register or override a category.

        Args:
            name: Category name (e.g. "korean-writer").
            config: Dict of CategoryConfig fields to set.
        """
        if name in self._categories:
            # Merge with existing
            existing = self._categories[name]
            for key, value in config.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
                else:
                    logger.warning(f"Unknown category field '{key}' for category '{name}'")
        else:
            # Create new
            self._categories[name] = CategoryConfig(**{
                k: v for k, v in config.items()
                if k in CategoryConfig.__dataclass_fields__
            })
        logger.info(f"Category '{name}' registered/updated")

    def get(self, name: str) -> Optional[CategoryConfig]:
        """Get a category by name (returns a deep copy).

        Args:
            name: Category name.

        Returns:
            CategoryConfig copy, or None if not found.
        """
        cat = self._categories.get(name)
        if cat:
            return deepcopy(cat)
        return None

    def list_categories(self) -> Dict[str, str]:
        """List all available categories with their descriptions.

        Returns:
            Dict mapping category name to description.
        """
        return {
            name: cat.description
            for name, cat in self._categories.items()
        }

    def resolve_for_task(
        self,
        category: Optional[str],
        default_model: str,
        default_temperature: float = 0.7,
        default_max_tokens: int = 4096,
    ) -> CategoryConfig:
        """Resolve category config for a task, with fallback defaults.

        If the category is not found or None, returns a default config
        using the provided model/temperature/max_tokens.

        Args:
            category: Category name, or None.
            default_model: Fallback model ID.
            default_temperature: Fallback temperature.
            default_max_tokens: Fallback max tokens.

        Returns:
            Resolved CategoryConfig.
        """
        if category:
            cat = self.get(category)
            if cat:
                # If category doesn't specify model, use default
                if not cat.model:
                    cat.model = default_model
                return cat
            else:
                logger.warning(
                    f"Category '{category}' not found. "
                    f"Available: {list(self._categories.keys())}. "
                    f"Using default config."
                )

        # Return default config
        return CategoryConfig(
            model=default_model,
            temperature=default_temperature,
            max_tokens=default_max_tokens,
        )
