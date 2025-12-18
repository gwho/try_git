"""
LangGraph Memory Agent Integration with Supabase

This module demonstrates how to integrate LangGraph nodes with Supabase
for persistent memory storage with vector embeddings.

Requirements:
    pip install langgraph langchain-openai supabase httpx

Environment Variables:
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_SERVICE_KEY: Service role key (has RLS bypass)
    OPENAI_API_KEY: OpenAI API key for embeddings
"""

import os
from typing import TypedDict, Annotated, Sequence
from datetime import datetime
import httpx

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from supabase import create_client, Client


# ============================================================
# Configuration
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for agent
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4", temperature=0.7)


# ============================================================
# State Definition
# ============================================================

class AgentState(TypedDict):
    """State for the memory agent"""
    messages: Annotated[Sequence[BaseMessage], "The conversation messages"]
    user_id: str
    relevant_memories: list[dict]
    should_save_memory: bool
    memory_to_save: str | None


# ============================================================
# Memory Operations
# ============================================================

class MemoryManager:
    """Handles all memory operations with Supabase"""

    def __init__(self, supabase_client: Client, embeddings_model: OpenAIEmbeddings):
        self.supabase = supabase_client
        self.embeddings = embeddings_model

    def save_memory(self, user_id: str, content: str, metadata: dict = None) -> dict:
        """
        Save a memory with its vector embedding to Supabase

        Args:
            user_id: The user ID from Supabase auth
            content: The memory content to save
            metadata: Additional metadata (tags, source, etc.)

        Returns:
            The created memory record
        """
        # Generate embedding for the content
        embedding_vector = self.embeddings.embed_query(content)

        # Prepare the memory data
        memory_data = {
            "user_id": user_id,
            "content": content,
            "embedding": embedding_vector,
            "metadata": metadata or {}
        }

        # Insert into Supabase
        result = self.supabase.table("memories").insert(memory_data).execute()

        print(f"✓ Memory saved: {result.data[0]['id']}")
        return result.data[0]

    def search_memories(
        self,
        user_id: str,
        query: str,
        match_threshold: float = 0.7,
        match_count: int = 5
    ) -> list[dict]:
        """
        Search for relevant memories using vector similarity

        Args:
            user_id: The user ID to search memories for
            query: The search query
            match_threshold: Minimum similarity score (0-1)
            match_count: Maximum number of results

        Returns:
            List of matching memories with similarity scores
        """
        # Generate embedding for the query
        query_embedding = self.embeddings.embed_query(query)

        # Call the PostgreSQL function for vector search
        result = self.supabase.rpc(
            "match_memories",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
                "filter_user_id": user_id
            }
        ).execute()

        print(f"✓ Found {len(result.data)} relevant memories")
        return result.data

    def get_recent_memories(self, user_id: str, limit: int = 10) -> list[dict]:
        """Get recent memories for a user"""
        result = self.supabase.rpc(
            "get_recent_memories",
            {"limit_count": limit}
        ).execute()

        return result.data

    def delete_memory(self, memory_id: str) -> None:
        """Delete a specific memory"""
        self.supabase.table("memories").delete().eq("id", memory_id).execute()
        print(f"✓ Memory deleted: {memory_id}")


# Initialize memory manager
memory_manager = MemoryManager(supabase, embeddings)


# ============================================================
# LangGraph Nodes
# ============================================================

def retrieve_memories_node(state: AgentState) -> AgentState:
    """
    Node: Retrieve relevant memories based on the current conversation
    """
    print("\n[Node] Retrieving relevant memories...")

    # Get the last user message
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)

    # Search for relevant memories
    relevant_memories = memory_manager.search_memories(
        user_id=state["user_id"],
        query=query,
        match_threshold=0.7,
        match_count=5
    )

    # Update state with retrieved memories
    state["relevant_memories"] = relevant_memories

    return state


def process_with_memory_node(state: AgentState) -> AgentState:
    """
    Node: Process the user query with context from retrieved memories
    """
    print("\n[Node] Processing with memory context...")

    # Build context from memories
    memory_context = "\n".join([
        f"- {mem['content']} (similarity: {mem['similarity']:.2f})"
        for mem in state["relevant_memories"]
    ]) if state["relevant_memories"] else "No relevant memories found."

    # Create a prompt with memory context
    system_message = f"""You are a helpful assistant with access to previous conversations.

Relevant memories:
{memory_context}

Use these memories to provide personalized and context-aware responses."""

    # Get the conversation messages
    messages = [
        {"role": "system", "content": system_message}
    ] + [
        {"role": msg.type, "content": msg.content}
        for msg in state["messages"]
    ]

    # Generate response
    response = llm.invoke(messages)

    # Add response to messages
    state["messages"] = state["messages"] + [AIMessage(content=response.content)]

    return state


def decide_memory_save_node(state: AgentState) -> AgentState:
    """
    Node: Decide whether the current interaction should be saved as a memory
    """
    print("\n[Node] Deciding whether to save memory...")

    # Use LLM to decide if this is worth remembering
    decision_prompt = f"""Based on this conversation, should we save this as a long-term memory?
Consider saving if:
- User shares personal information or preferences
- Important decisions or conclusions are made
- Novel or valuable insights are discussed

Conversation:
{state["messages"][-2:]}

Respond with:
1. "YES" if should save, followed by what to remember
2. "NO" if not worth saving

Your response:"""

    response = llm.invoke([{"role": "user", "content": decision_prompt}])
    decision = response.content.strip()

    if decision.startswith("YES"):
        state["should_save_memory"] = True
        # Extract what to remember (everything after "YES")
        state["memory_to_save"] = decision[3:].strip()
    else:
        state["should_save_memory"] = False
        state["memory_to_save"] = None

    return state


def save_memory_node(state: AgentState) -> AgentState:
    """
    Node: Save the memory to Supabase with embedding
    """
    if state["should_save_memory"] and state["memory_to_save"]:
        print("\n[Node] Saving memory to Supabase...")

        # Extract metadata from the conversation
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "langgraph_agent",
            "conversation_length": len(state["messages"])
        }

        # Save to Supabase
        memory_manager.save_memory(
            user_id=state["user_id"],
            content=state["memory_to_save"],
            metadata=metadata
        )

        print(f"✓ Memory saved: {state['memory_to_save'][:50]}...")
    else:
        print("\n[Node] No memory to save")

    return state


def webhook_notification_node(state: AgentState) -> AgentState:
    """
    Node: Send webhook notification to Supabase Edge Function
    (Optional - for real-time UI updates)
    """
    print("\n[Node] Sending webhook notification...")

    webhook_url = f"{SUPABASE_URL}/functions/v1/langgraph-webhook"
    webhook_secret = os.getenv("LANGGRAPH_WEBHOOK_SECRET", "")

    payload = {
        "agent_id": state["user_id"],
        "state": {
            "memory": {
                "content": state.get("memory_to_save", ""),
                "should_persist": state.get("should_save_memory", False),
                "metadata": {
                    "user_id": state["user_id"]
                }
            }
        },
        "event_type": "state_update",
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        response = httpx.post(
            webhook_url,
            json=payload,
            headers={
                "x-webhook-secret": webhook_secret,
                "Content-Type": "application/json"
            },
            timeout=10.0
        )

        if response.status_code == 200:
            print("✓ Webhook sent successfully")
        else:
            print(f"⚠ Webhook failed: {response.status_code}")
    except Exception as e:
        print(f"⚠ Webhook error: {str(e)}")

    return state


# ============================================================
# Graph Construction
# ============================================================

def create_memory_agent() -> StateGraph:
    """
    Create the LangGraph workflow for the memory agent
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve_memories", retrieve_memories_node)
    workflow.add_node("process_with_memory", process_with_memory_node)
    workflow.add_node("decide_memory_save", decide_memory_save_node)
    workflow.add_node("save_memory", save_memory_node)
    workflow.add_node("webhook_notification", webhook_notification_node)

    # Define the flow
    workflow.set_entry_point("retrieve_memories")
    workflow.add_edge("retrieve_memories", "process_with_memory")
    workflow.add_edge("process_with_memory", "decide_memory_save")
    workflow.add_edge("decide_memory_save", "save_memory")
    workflow.add_edge("save_memory", "webhook_notification")
    workflow.add_edge("webhook_notification", END)

    return workflow.compile()


# ============================================================
# Example Usage
# ============================================================

def run_example():
    """Example of running the memory agent"""

    # Create the agent
    agent = create_memory_agent()

    # Initialize state with a user query
    initial_state = {
        "messages": [
            HumanMessage(content="Hi! I love pizza, especially margherita.")
        ],
        "user_id": "123e4567-e89b-12d3-a456-426614174000",  # Example UUID
        "relevant_memories": [],
        "should_save_memory": False,
        "memory_to_save": None
    }

    # Run the agent
    print("=" * 60)
    print("Running Memory Agent")
    print("=" * 60)

    result = agent.invoke(initial_state)

    print("\n" + "=" * 60)
    print("Agent Response:")
    print("=" * 60)
    print(result["messages"][-1].content)

    # Example: Search for memories later
    print("\n" + "=" * 60)
    print("Searching for pizza-related memories:")
    print("=" * 60)

    memories = memory_manager.search_memories(
        user_id="123e4567-e89b-12d3-a456-426614174000",
        query="What food does the user like?",
        match_count=3
    )

    for mem in memories:
        print(f"\n- {mem['content']}")
        print(f"  Similarity: {mem['similarity']:.2f}")
        print(f"  Created: {mem['created_at']}")


if __name__ == "__main__":
    run_example()
