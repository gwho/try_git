"""
=================================================================
LANGGRAPH EXAMPLE: Memory-Enabled Agent
=================================================================
This demonstrates how to integrate the MemoryManager into LangGraph nodes

WHY THIS EXAMPLE EXISTS:
- Shows complete agent workflow with memory
- Demonstrates when to save vs search memories
- Real-world pattern for memory-augmented agents
"""

from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from memory_manager import memory_manager


# =================================================================
# STATE DEFINITION
# =================================================================
# Defines what data flows through the agent

class AgentState(TypedDict):
    """
    State that flows through LangGraph nodes

    WHY THESE FIELDS:
    - messages: Conversation history
    - user_id: Links memories to specific user
    - relevant_memories: Context from past conversations
    - should_save: Whether current interaction is worth remembering
    """
    messages: List[BaseMessage]
    user_id: str
    relevant_memories: List[dict]
    should_save: bool
    memory_content: str | None


# =================================================================
# NODE 1: Retrieve Relevant Memories
# =================================================================

def retrieve_memories_node(state: AgentState) -> AgentState:
    """
    Search for relevant memories based on current conversation

    WHY THIS NODE:
    - Provides context from past conversations
    - Enables personalized responses
    - Agent "remembers" user preferences

    WHEN THIS RUNS:
    - At the START of every conversation turn
    - Before generating response

    EXAMPLE FLOW:
    User: "I'm hungry, what should I eat?"
    → Search memories: "What food does user like?"
    → Finds: ["User loves pizza", "User is vegetarian"]
    → Response: "How about a veggie pizza?"
    """

    print("\n[NODE] Retrieving relevant memories...")

    # Get the last user message
    last_message = state["messages"][-1]
    query = last_message.content

    # Search for relevant memories
    # WHY threshold 0.7: Good balance of precision vs recall
    relevant_memories = memory_manager.search_memories(
        user_id=state["user_id"],
        query=query,
        match_threshold=0.7,
        match_count=5
    )

    # Update state
    state["relevant_memories"] = relevant_memories

    return state


# =================================================================
# NODE 2: Generate Response with Memory Context
# =================================================================

def generate_response_node(state: AgentState) -> AgentState:
    """
    Generate AI response using retrieved memories as context

    WHY THIS NODE:
    - Combines current input + past memories
    - Creates personalized, context-aware responses
    - Agent appears to "know" the user

    HOW IT WORKS:
    1. Build context from relevant memories
    2. Inject context into system prompt
    3. Generate response with full context
    """

    print("\n[NODE] Generating response with memory context...")

    # Build memory context string
    if state["relevant_memories"]:
        memory_context = "\n".join([
            f"- {mem['content']} (relevance: {mem['similarity']:.0%})"
            for mem in state["relevant_memories"]
        ])
    else:
        memory_context = "No relevant memories found."

    # Create system prompt with memories
    system_message = f"""You are a helpful assistant with access to previous conversations.

Relevant memories from past interactions:
{memory_context}

Use these memories to provide personalized, context-aware responses.
Reference past conversations naturally when relevant."""

    # Build messages for LLM
    messages = [
        {"role": "system", "content": system_message}
    ] + [
        {"role": msg.type, "content": msg.content}
        for msg in state["messages"]
    ]

    # Generate response
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke(messages)

    # Add to conversation
    state["messages"].append(AIMessage(content=response.content))

    return state


# =================================================================
# NODE 3: Decide What to Remember
# =================================================================

def decide_memory_node(state: AgentState) -> AgentState:
    """
    Decide if this interaction should be saved as a long-term memory

    WHY THIS NODE:
    - Not everything is worth remembering
    - Prevents memory database from filling with noise
    - Extracts key facts/preferences from conversation

    DECISION CRITERIA:
    ✅ Save if:
    - User shares preferences ("I love X")
    - User provides personal info ("I live in Y")
    - Important decisions made
    - Novel insights discovered

    ❌ Don't save if:
    - Generic greetings
    - Transactional queries
    - No new information
    """

    print("\n[NODE] Deciding what to remember...")

    # Use LLM to decide
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    decision_prompt = f"""Analyze this conversation and decide if we should save it as a long-term memory.

Conversation:
{state['messages'][-3:]}  # Last 3 messages for context

Instructions:
1. If this contains important information worth remembering (preferences, facts, decisions):
   - Respond with "SAVE: <concise summary of what to remember>"
   - Example: "SAVE: User is vegetarian and allergic to peanuts"

2. If this is generic/transactional with nothing worth remembering:
   - Respond with "SKIP"

Your decision:"""

    response = llm.invoke([{"role": "user", "content": decision_prompt}])
    decision = response.content.strip()

    if decision.startswith("SAVE:"):
        state["should_save"] = True
        state["memory_content"] = decision[5:].strip()  # Remove "SAVE:" prefix
        print(f"✓ Will save: {state['memory_content']}")
    else:
        state["should_save"] = False
        state["memory_content"] = None
        print("✗ Nothing to save")

    return state


# =================================================================
# NODE 4: Save Memory
# =================================================================

def save_memory_node(state: AgentState) -> AgentState:
    """
    Persist the memory to Supabase

    WHY THIS NODE:
    - Actually saves to database
    - Triggers real-time update in UI
    - Becomes searchable for future conversations

    WHAT HAPPENS:
    1. Save to PostgreSQL with embedding
    2. Triggers notify RLS → Supabase Realtime
    3. React UI updates instantly
    4. Next conversation can retrieve this memory
    """

    if state["should_save"] and state["memory_content"]:
        print("\n[NODE] Saving memory to database...")

        # Extract metadata from conversation
        metadata = {
            "source": "langraph_agent",
            "timestamp": "...",  # Add proper timestamp
            "confidence": 0.9,  # Could use LLM to estimate
            "conversation_length": len(state["messages"])
        }

        # Save memory
        # WHY this triggers UI update: Supabase Realtime broadcasts change
        memory_manager.save_memory(
            user_id=state["user_id"],
            content=state["memory_content"],
            metadata=metadata
        )

        print("✓ Memory saved and broadcast to UI!")

    else:
        print("\n[NODE] No memory to save")

    return state


# =================================================================
# BUILD THE GRAPH
# =================================================================

def create_memory_agent() -> StateGraph:
    """
    Assemble the complete agent graph

    WHY THIS STRUCTURE:
    retrieve → generate → decide → save
       ↓          ↓         ↓       ↓
    Get context  Respond  Analyze  Persist

    LINEAR FLOW: Each node feeds the next
    """

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve_memories", retrieve_memories_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("decide_memory", decide_memory_node)
    workflow.add_node("save_memory", save_memory_node)

    # Define the flow
    workflow.set_entry_point("retrieve_memories")
    workflow.add_edge("retrieve_memories", "generate_response")
    workflow.add_edge("generate_response", "decide_memory")
    workflow.add_edge("decide_memory", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow.compile()


# =================================================================
# EXAMPLE USAGE
# =================================================================

if __name__ == "__main__":
    """
    Run a sample conversation to see memory in action
    """

    # Create agent
    agent = create_memory_agent()

    # Initial state
    state = {
        "messages": [
            HumanMessage(content="Hi! I love pizza, especially margherita.")
        ],
        "user_id": "550e8400-e29b-41d4-a716-446655440000",  # Example UUID
        "relevant_memories": [],
        "should_save": False,
        "memory_content": None
    }

    # Run agent
    print("=" * 60)
    print("RUNNING MEMORY-ENABLED AGENT")
    print("=" * 60)

    result = agent.invoke(state)

    print("\n" + "=" * 60)
    print("AGENT RESPONSE:")
    print("=" * 60)
    print(result["messages"][-1].content)

    print("\n" + "=" * 60)
    print("TESTING MEMORY RETRIEVAL:")
    print("=" * 60)

    # Test: Next conversation should remember pizza preference
    state2 = {
        "messages": [
            HumanMessage(content="What should I have for dinner?")
        ],
        "user_id": "550e8400-e29b-41d4-a716-446655440000",
        "relevant_memories": [],
        "should_save": False,
        "memory_content": None
    }

    result2 = agent.invoke(state2)
    print(result2["messages"][-1].content)
    # Should mention pizza based on retrieved memory!
