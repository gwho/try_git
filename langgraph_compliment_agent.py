"""
LangGraph Personalized Compliment Agent Tutorial
================================================
Learn how to build a stateful AI agent using LangGraph with STATE CONCATENATION

Key Concepts Covered:
- TypedDict for type-safe state management
- State concatenation using Annotated and operator.add
- Node functions (the "workers" in your graph)
- Edge connections (how data flows between nodes)
- StateGraph workflow orchestration

Prerequisites: pip install langgraph
"""

# ============================================================================
# IMPORTS - Modern Python Libraries for AI Workflows
# ============================================================================

from typing import TypedDict, Annotated
# TypedDict: Creates a dictionary with fixed keys and type hints (type safety!)
# Annotated: Adds metadata to types (we'll use it to specify HOW to merge state)

import operator
# operator.add: The function that concatenates lists (+ operation)
# We use this to tell LangGraph: "When updating state, ADD to lists, don't replace"

from langgraph.graph import StateGraph, END
# StateGraph: The main class for building our workflow graph
# END: A special marker that tells the graph "we're done, stop here"


# ============================================================================
# STEP 1: Define the State Structure (The "Memory" of Our Agent)
# ============================================================================

class AgentState(TypedDict):
    """
    This is our agent's "brain" - it remembers everything across steps

    Think of it like a dictionary that gets passed from function to function,
    with each function adding more information to it.

    Why TypedDict?
    - Type safety: Python knows what fields exist and their types
    - IDE autocomplete: Your editor can suggest field names
    - Documentation: Anyone reading your code knows the structure
    """

    # User's name - this stays the same throughout (str type)
    user_name: str

    # User's interests - we'll use this to personalize compliments (str type)
    user_interest: str

    # Messages list - THIS IS THE KEY PART!
    # Annotated[list[str], operator.add] means:
    # - "This is a list of strings"
    # - "When updating, use operator.add (concatenate), don't replace!"
    #
    # WITHOUT operator.add: messages = ["new"] would REPLACE old messages
    # WITH operator.add: messages = ["new"] ADDS to existing messages
    messages: Annotated[list[str], operator.add]

    # Agent's mood - changes based on interactions (str type)
    mood: str


# ============================================================================
# STEP 2: Define Node Functions (The "Workers" That Process State)
# ============================================================================

def greet_user(state: AgentState) -> AgentState:
    """
    NODE 1: Greets the user and sets initial mood

    In LangGraph, a "node" is just a Python function that:
    1. Receives the current state (a dictionary)
    2. Does some work (processing, API calls, etc.)
    3. Returns updates to merge into the state

    Modern Programming Concept: Pure Functions
    - We don't modify the input state directly
    - We return a NEW dictionary with updates
    - LangGraph merges this with existing state automatically
    """

    # Extract user's name from current state
    # This demonstrates Python's dictionary access patterns
    name = state["user_name"]

    # Create a personalized greeting message
    # f-strings (f"...") are modern Python's way to format strings
    greeting = f"Hello {name}! I'm your personal compliment agent! 😊"

    # Return updates to state
    # Notice: We return a dict with ONLY the fields we want to update
    # LangGraph will merge this with existing state
    return {
        "messages": [greeting],  # This will be ADDED to messages list (not replaced!)
        "mood": "friendly"        # This will update (replace) the mood field
    }


def analyze_interest(state: AgentState) -> AgentState:
    """
    NODE 2: Analyzes user's interest and updates mood accordingly

    Design Pattern: Single Responsibility Principle
    - Each function does ONE thing well
    - This function only analyzes interest
    - Separation of concerns makes code easier to test and maintain
    """

    interest = state["user_interest"]

    # Simple sentiment analysis based on interest keywords
    # In a real app, you might use an AI model here
    if "coding" in interest.lower() or "python" in interest.lower():
        analysis = f"I see you're interested in {interest}! That's amazing! 💻"
        new_mood = "excited"
    elif "art" in interest.lower() or "music" in interest.lower():
        analysis = f"Wow, {interest}! You must be very creative! 🎨"
        new_mood = "inspired"
    else:
        analysis = f"I love that you're into {interest}! Tell me more! ✨"
        new_mood = "curious"

    # Return state updates
    # messages gets CONCATENATED (added to existing list)
    # mood gets REPLACED (new value overwrites old)
    return {
        "messages": [analysis],
        "mood": new_mood
    }


def generate_compliment(state: AgentState) -> AgentState:
    """
    NODE 3: Generates a personalized compliment based on accumulated state

    Key Concept: Stateful Processing
    - We can access ALL previous state (name, interest, mood, messages)
    - Each node builds on what previous nodes added
    - This is the power of state management!
    """

    name = state["user_name"]
    interest = state["user_interest"]
    mood = state["mood"]

    # Generate compliment based on current mood
    # Dictionary mapping: Modern Python pattern for clean conditionals
    compliments = {
        "excited": f"{name}, your passion for {interest} is truly inspiring! Keep coding! 🚀",
        "inspired": f"{name}, your creativity in {interest} makes the world more beautiful! 🌟",
        "curious": f"{name}, your curiosity about {interest} shows great character! Keep exploring! 🌈",
        "friendly": f"{name}, you seem like an awesome person! Keep being you! ⭐"
    }

    # Get compliment based on mood, with a fallback default
    # .get(key, default) is safer than compliments[mood] (no KeyError)
    compliment = compliments.get(mood, f"{name}, you're wonderful!")

    return {
        "messages": [compliment]
    }


def summarize_conversation(state: AgentState) -> AgentState:
    """
    NODE 4: Creates a summary of all messages

    Demonstrates: List comprehension and string manipulation
    - We access all accumulated messages
    - Join them with newlines for readable output
    """

    # Get all messages that have been accumulated
    all_messages = state["messages"]

    # Create a formatted summary
    # "\n".join() takes a list and joins with newlines
    # Enumerate adds numbering: (0, item), (1, item), etc.
    summary = "\n--- Conversation Summary ---\n"
    summary += f"Total messages: {len(all_messages)}\n\n"

    for i, message in enumerate(all_messages, 1):
        summary += f"{i}. {message}\n"

    return {
        "messages": [summary]
    }


# ============================================================================
# STEP 3: Build the Graph Workflow (Connect the Nodes)
# ============================================================================

def create_compliment_graph():
    """
    Creates and configures the LangGraph workflow

    Modern Architecture Pattern: Dependency Injection & Graph-based Workflows
    - Nodes are independent, reusable functions
    - Graph defines how they connect (orchestration)
    - Easy to modify workflow without changing node logic
    """

    # Initialize the graph with our state structure
    # StateGraph needs to know what shape of data it's working with
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    # Syntax: add_node(name: str, function: callable)
    # "name" is how we reference the node when building edges
    # "function" is the actual Python function to execute
    workflow.add_node("greet", greet_user)
    workflow.add_node("analyze", analyze_interest)
    workflow.add_node("compliment", generate_compliment)
    workflow.add_node("summarize", summarize_conversation)

    # Set the entry point (where the graph starts)
    # Think of this as main() in a traditional program
    workflow.set_entry_point("greet")

    # Add edges (define the flow/sequence)
    # Syntax: add_edge(from_node, to_node)
    # This creates a directed graph: greet → analyze → compliment → summarize → END
    workflow.add_edge("greet", "analyze")
    workflow.add_edge("analyze", "compliment")
    workflow.add_edge("compliment", "summarize")
    workflow.add_edge("summarize", END)

    # Compile the graph into an executable application
    # This validates the graph structure and prepares it for execution
    app = workflow.compile()

    return app


# ============================================================================
# STEP 4: Run the Agent (Entry Point)
# ============================================================================

def run_compliment_agent(user_name: str, user_interest: str):
    """
    Main function to execute the compliment agent

    Demonstrates: Complete workflow execution
    """

    # Create the graph application
    app = create_compliment_graph()

    # Initial state - this is what we start with
    # Notice: messages starts as EMPTY list
    # Each node will ADD to it (concatenation!)
    initial_state = {
        "user_name": user_name,
        "user_interest": user_interest,
        "messages": [],  # Empty list - nodes will concatenate to this
        "mood": "neutral"
    }

    print("🤖 Starting Compliment Agent...\n")
    print("=" * 60)

    # Execute the graph with initial state
    # .invoke() runs through the entire graph from entry point to END
    final_state = app.invoke(initial_state)

    print("=" * 60)
    print("\n✅ Agent finished!\n")

    # Display results
    print("📊 FINAL STATE:")
    print(f"   Name: {final_state['user_name']}")
    print(f"   Interest: {final_state['user_interest']}")
    print(f"   Final Mood: {final_state['mood']}")
    print(f"   Total Messages: {len(final_state['messages'])}")
    print("\n💬 ALL MESSAGES (Concatenated State):")
    print("=" * 60)

    # Print each message that was added to state
    for msg in final_state['messages']:
        print(msg)
        print()

    return final_state


# ============================================================================
# DEMO: Run the Agent
# ============================================================================

if __name__ == "__main__":
    """
    Python convention: Code here only runs when file is executed directly
    (not when imported as a module)
    """

    # Example 1: User interested in coding
    print("\n🎯 EXAMPLE 1: Coding Enthusiast")
    run_compliment_agent("Alice", "Python programming")

    print("\n" + "=" * 80 + "\n")

    # Example 2: User interested in art
    print("🎯 EXAMPLE 2: Artist")
    run_compliment_agent("Bob", "digital art and painting")
