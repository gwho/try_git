-- =================================================================
-- MEMORY AGENT DATABASE SCHEMA
-- =================================================================
-- This is the foundation layer of our memory system.
-- Every component above this layer depends on this schema.

-- =================================================================
-- EXTENSIONS: Enable PostgreSQL superpowers
-- =================================================================

-- uuid-ossp: Generates cryptographically secure unique IDs
-- WHY: We need reliable, collision-free IDs for all memory records
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- vector: Enables semantic search via embeddings
-- WHY: Allows us to search memories by MEANING, not just keywords
-- Example: "pizza" finds "margherita" even without the word "pizza"
CREATE EXTENSION IF NOT EXISTS "vector";

-- =================================================================
-- TABLE: memories
-- =================================================================
-- This is the SINGLE SOURCE OF TRUTH for all user memories

CREATE TABLE public.memories (
    -- Identity: Who is this memory?
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Ownership: Who does this memory belong to?
    -- WHY: Links to Supabase auth.users for automatic user management
    -- ON DELETE CASCADE: If user is deleted, their memories go too
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Content: What is the actual memory?
    -- WHY: The human-readable text we want to remember
    content TEXT NOT NULL,

    -- Intelligence: The AI representation of this memory
    -- WHY: 1536 dimensions matches OpenAI's text-embedding-3-small
    -- This enables semantic search - finding similar meanings
    embedding vector(1536),

    -- Context: Additional structured data about the memory
    -- WHY: JSONB is flexible - store tags, source, confidence, etc.
    -- Example: {"tags": ["important"], "source": "conversation", "mood": "positive"}
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timeline: When did this happen?
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =================================================================
-- INDEXES: Make queries fast
-- =================================================================

-- Speed up "get all memories for user X"
-- WHY: Without this, PostgreSQL scans EVERY row to find user's memories
CREATE INDEX idx_memories_user_id ON public.memories(user_id);

-- Speed up "get recent memories"
-- WHY: DESC means newest first, which is what users usually want
CREATE INDEX idx_memories_created_at ON public.memories(created_at DESC);

-- Speed up vector similarity search
-- WHY: HNSW (Hierarchical Navigable Small World) is the fastest algorithm
-- for approximate nearest neighbor search. Without this, vector search
-- would be unusably slow with thousands of memories.
CREATE INDEX idx_memories_embedding
ON public.memories USING hnsw (embedding vector_cosine_ops);

-- =================================================================
-- SECURITY: Row Level Security
-- =================================================================
-- This is CRITICAL for multi-tenant applications
-- WHY: Ensures User A can NEVER see User B's memories, even with direct SQL

ALTER TABLE public.memories ENABLE ROW LEVEL SECURITY;

-- Read Policy: "I can only see MY memories"
-- WHY: auth.uid() is Supabase's function that returns the current user
CREATE POLICY "select_own_memories"
    ON public.memories FOR SELECT
    USING (auth.uid() = user_id);

-- Create Policy: "I can only create memories for MYSELF"
-- WHY: Prevents User A from creating fake memories for User B
CREATE POLICY "insert_own_memories"
    ON public.memories FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Update Policy: "I can only modify MY memories"
CREATE POLICY "update_own_memories"
    ON public.memories FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Delete Policy: "I can only delete MY memories"
CREATE POLICY "delete_own_memories"
    ON public.memories FOR DELETE
    USING (auth.uid() = user_id);

-- =================================================================
-- GRANT: Give users permission to use this table
-- =================================================================

GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON public.memories TO authenticated;
