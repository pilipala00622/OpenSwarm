"""Simple Agent Example - Basic usage of Open Swarm"""

import asyncio
import logging
import sys
import os

# Add parent directory to path for imports (when running from examples/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_swarm import (
    Agent,
    AgentConfig,
    MainRollout,
    RolloutConfig,
    SearchTool,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    """Run a simple agent example"""

    # Create agent configuration
    config = AgentConfig(
        name="simple_agent",
        system_prompt=(
            "You are a helpful assistant. "
            "You can use the search tool to find information. "
            "Be concise and helpful in your responses."
        ),
        model_id="kimi-k2-0711-preview",
        temperature=0.7,
    )

    # Create tools
    search_tool = SearchTool()

    # Create agent
    agent = Agent(config, tools=[search_tool])

    # Create rollout
    rollout_config = RolloutConfig(
        max_steps=10,
        terminal_mode=True,
    )
    rollout = MainRollout(rollout_config)

    # Run agent
    print("\n" + "="*60)
    print("Starting Simple Agent Example")
    print("="*60)

    result = await rollout.run(
        agent=agent,
        initial_message="Hello! Can you tell me what you can do?"
    )

    print("\n" + "="*60)
    print("Final Result:")
    print("="*60)
    print(f"Status: {result.status.value}")
    print(f"Steps: {result.steps}")
    if result.final_response:
        print(f"Response: {result.final_response}")


if __name__ == "__main__":
    asyncio.run(main())
