"""
=================================================================
MEMORY MANAGER: LangGraph ↔ Supabase Bridge
=================================================================
This module enables LangGraph agents to store and retrieve memories

WHY THIS EXISTS:
- LangGraph nodes need to interact with Supabase
- Encapsulates database operations for agents
- Handles embedding generation automatically
- Provides clean Python API for memory operations

ARCHITECTURE:
    LangGraph Agent Node
           ↓
    MemoryManager (this file)
           ↓
    Supabase (PostgreSQL + pgvector)
"""

import os
from typing import List, Dict, Optional
from datetime import datetime

from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings


# =================================================================
# CONFIGURATION
# =================================================================

# WHY environment variables:
# - Different per environment (dev/staging/prod)
# - Secrets should never be hardcoded
# - Easy to configure in deployment platforms

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Service role for admin access
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate configuration
if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
    raise ValueError(
        "Missing required environment variables: "
        "SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY"
    )


# =================================================================
# MEMORY MANAGER CLASS
# =================================================================
# Encapsulates all memory operations for LangGraph agents

class MemoryManager:
    """
    Manages memory persistence and retrieval for LangGraph agents

    WHY A CLASS:
    - Encapsulates state (supabase client, embeddings model)
    - Reusable across multiple agent nodes
    - Easier to test (can mock in tests)
    - Clear interface for agent developers

    EXAMPLE USAGE:
    ```python
    # In LangGraph node
    from memory_manager import memory_manager

    def process_conversation(state):
        # Save important insight
        memory_manager.save_memory(
            user_id=state['user_id'],
            content="User prefers vegetarian food",
            metadata={"confidence": 0.9, "source": "conversation"}
        )

        # Retrieve relevant memories
        memories = memory_manager.search_memories(
            user_id=state['user_id'],
            query="What are user's food preferences?"
        )

        return state
    ```
    """

    def __init__(self, supabase_client: Client, embeddings_model: OpenAIEmbeddings):
        """
        Initialize the memory manager

        Args:
            supabase_client: Admin client with RLS bypass
            embeddings_model: Model for generating embeddings

        WHY admin client:
        - Agent runs server-side, no user session
        - Needs to create memories for any user
        - Must validate user_id manually (security!)
        """
        self.supabase = supabase_client
        self.embeddings = embeddings_model

    # =================================================================
    # SAVE MEMORY
    # =================================================================

    def save_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Save a memory with automatic embedding generation

        WHY THIS EXISTS:
        - LangGraph nodes shouldn't worry about embedding generation
        - Automatic: Just provide text, get searchable memory
        - Type-safe: Returns the created memory record

        Args:
            user_id: UUID of the user (must validate this!)
            content: The memory text
            metadata: Optional context (tags, source, etc.)

        Returns:
            The created memory record with id

        SECURITY NOTE:
        Since we use service role key, we MUST validate user_id is legitimate!
        In production, add validation:
        - Check user exists: supabase.auth.admin.get_user_by_id(user_id)
        - Verify agent is authorized to create memories for this user

        EXAMPLE:
        ```python
        memory = memory_manager.save_memory(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            content="User mentioned they love hiking in Colorado",
            metadata={
                "tags": ["hobbies", "location"],
                "confidence": 0.95,
                "source": "conversation",
                "extracted_at": datetime.utcnow().isoformat()
            }
        )
        print(f"Saved memory: {memory['id']}")
        ```
        """

        # Step 1: Generate embedding
        # WHY: Enables semantic search later
        # Performance: ~50-100ms for OpenAI API
        embedding_vector = self.embeddings.embed_query(content)

        # Step 2: Prepare memory data
        memory_data = {
            "user_id": user_id,
            "content": content,
            "embedding": embedding_vector,
            "metadata": metadata or {}
        }

        # Step 3: Insert into database
        # WHY .execute(): Supabase Python client requires it
        # WHY [0]: Returns list, we want first (and only) item
        result = (
            self.supabase
            .table("memories")
            .insert(memory_data)
            .execute()
        )

        created_memory = result.data[0]

        print(f"✓ Memory saved: {created_memory['id']}")
        print(f"  Content: {content[:50]}...")
        print(f"  Metadata: {metadata}")

        return created_memory

    # =================================================================
    # SEARCH MEMORIES
    # =================================================================

    def search_memories(
        self,
        user_id: str,
        query: str,
        match_threshold: float = 0.7,
        match_count: int = 5
    ) -> List[Dict]:
        """
        Search memories by meaning using vector similarity

        WHY THIS EXISTS:
        - Core of "intelligent memory" - find by meaning, not keywords
        - LangGraph agents use this to retrieve relevant context
        - Automatic embedding generation for query

        Args:
            user_id: Filter to specific user's memories
            query: Natural language question
            match_threshold: Min similarity (0-1), default 0.7 = 70% similar
            match_count: Max results to return

        Returns:
            List of matching memories with similarity scores

        HOW IT WORKS:
        1. Convert query to embedding: "What food does user like?" → [0.1, 0.4, ...]
        2. PostgreSQL finds similar embeddings using HNSW index
        3. Returns top matches ranked by similarity

        EXAMPLE:
        ```python
        # Agent needs context about user preferences
        memories = memory_manager.search_memories(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            query="What are the user's dietary restrictions?",
            match_threshold=0.75,  # Higher threshold = more precise
            match_count=3  # Only need top 3
        )

        for memory in memories:
            print(f"{memory['content']} (confidence: {memory['similarity']:.2%})")

        # Output:
        # "User is vegetarian" (confidence: 92%)
        # "User allergic to peanuts" (confidence: 85%)
        # "User loves Thai food" (confidence: 78%)
        ```
        """

        # Step 1: Generate embedding for query
        query_embedding = self.embeddings.embed_query(query)

        # Step 2: Call PostgreSQL function
        # WHY .rpc(): Calls stored PostgreSQL function
        # Function name matches what we created in migration
        result = self.supabase.rpc(
            "match_memories",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
                "filter_user_id": user_id
            }
        ).execute()

        memories = result.data

        print(f"✓ Found {len(memories)} relevant memories")
        for i, mem in enumerate(memories, 1):
            print(f"  {i}. {mem['content'][:50]}... (similarity: {mem['similarity']:.2%})")

        return memories

    # =================================================================
    # UTILITY METHODS
    # =================================================================

    def get_recent_memories(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get most recent memories (chronological, not semantic)

        WHY THIS EXISTS:
        - Fast (uses timestamp index, not vector search)
        - Useful for "conversation history" context
        - LangGraph often needs recent context + semantic search

        EXAMPLE:
        ```python
        # Get last 5 memories for conversation context
        recent = memory_manager.get_recent_memories(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            limit=5
        )
        ```
        """
        result = self.supabase.rpc(
            "get_recent_memories",
            {"limit_count": limit}
        ).execute()

        return result.data

    def get_memory_count(self, user_id: str) -> int:
        """
        Count total memories for user

        WHY THIS EXISTS:
        - Agent can gauge how much context it has
        - Decide whether to ask clarifying questions
        - Analytics/debugging

        EXAMPLE:
        ```python
        count = memory_manager.get_memory_count(user_id)
        if count < 5:
            # Not enough context, ask user for more info
            return "I don't know much about you yet. Tell me more!"
        ```
        """
        result = self.supabase.rpc("get_user_memory_count").execute()
        return result.data

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a specific memory

        WHY THIS EXISTS:
        - User requests "forget that"
        - Cleanup of incorrect/outdated memories
        - GDPR compliance (right to be forgotten)
        """
        self.supabase.table("memories").delete().eq("id", memory_id).execute()
        print(f"✓ Memory deleted: {memory_id}")


# =================================================================
# SINGLETON INSTANCE
# =================================================================
# Create single instance to reuse across agent nodes
#
# WHY SINGLETON:
# - Reuses database connection
# - Reuses embeddings model (expensive to initialize)
# - Simpler imports: `from memory_manager import memory_manager`

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

memory_manager = MemoryManager(supabase_client, embeddings_model)
