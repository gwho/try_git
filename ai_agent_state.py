"""
Simple AI Agent State using TypedDict

This module demonstrates how to use TypedDict to define the state
structure for a simple AI agent.
"""

from typing import TypedDict


class AIAgentState(TypedDict):
    """Represents the state of a simple AI agent."""

    # The text input received from the user
    user_input: str

    # The current emotional state of the agent (e.g., "neutral", "helpful", "confused")
    mood: str

    # The agent's generated response to the user
    response: str


# Example usage
if __name__ == "__main__":
    # Create an instance of the agent state
    agent_state: AIAgentState = {
        "user_input": "Hello, how are you?",
        "mood": "friendly",
        "response": "Hello! I'm doing well, thank you for asking. How can I help you today?"
    }

    print("AI Agent State Example")
    print("=" * 50)
    print(f"User Input: {agent_state['user_input']}")
    print(f"Agent Mood: {agent_state['mood']}")
    print(f"Agent Response: {agent_state['response']}")
    print("=" * 50)

    # TypedDict provides type checking - editors can catch errors like this:
    # agent_state["user_input"] = 123  # Type checker would warn: expected str, got int
