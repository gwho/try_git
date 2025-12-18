-- ================================================
-- Long-Term Memory Agent - Initial Schema Migration
-- ================================================
-- This migration sets up the complete database schema for a LangGraph-powered
-- memory agent with vector embeddings, RLS policies, and search functions.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ================================================
-- TABLES
-- ================================================

-- Memories table: Stores user memories with vector embeddings
CREATE TABLE IF NOT EXISTS public.memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536), -- OpenAI ada-002 embedding size, adjust as needed
    metadata JSONB DEFAULT '{}'::jsonb, -- Store additional context (tags, source, etc.)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON public.memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON public.memories(created_at DESC);

-- Vector similarity index using HNSW (Hierarchical Navigable Small World)
-- This dramatically speeds up vector similarity searches
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON public.memories
USING hnsw (embedding vector_cosine_ops);

-- ================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ================================================

-- Enable RLS on the memories table
ALTER TABLE public.memories ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only view their own memories
CREATE POLICY "Users can view own memories"
    ON public.memories
    FOR SELECT
    USING (auth.uid() = user_id);

-- Policy: Users can insert their own memories
CREATE POLICY "Users can insert own memories"
    ON public.memories
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Policy: Users can update their own memories
CREATE POLICY "Users can update own memories"
    ON public.memories
    FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Policy: Users can delete their own memories
CREATE POLICY "Users can delete own memories"
    ON public.memories
    FOR DELETE
    USING (auth.uid() = user_id);

-- ================================================
-- FUNCTIONS
-- ================================================

-- Function: Search memories by vector similarity
-- Returns memories ranked by cosine similarity to the query embedding
CREATE OR REPLACE FUNCTION match_memories(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10,
    filter_user_id uuid DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    user_id uuid,
    content text,
    metadata jsonb,
    similarity float,
    created_at timestamp with time zone
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.user_id,
        m.content,
        m.metadata,
        1 - (m.embedding <=> query_embedding) as similarity,
        m.created_at
    FROM public.memories m
    WHERE
        (filter_user_id IS NULL OR m.user_id = filter_user_id)
        AND (1 - (m.embedding <=> query_embedding)) > match_threshold
        AND m.user_id = auth.uid() -- Ensure RLS is respected
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function: Update the updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Automatically update updated_at on row updates
CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON public.memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ================================================
-- HELPER FUNCTIONS FOR ANALYTICS (OPTIONAL)
-- ================================================

-- Function: Get memory count for current user
CREATE OR REPLACE FUNCTION get_user_memory_count()
RETURNS bigint
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT COUNT(*)
    FROM public.memories
    WHERE user_id = auth.uid();
$$;

-- Function: Get recent memories for current user
CREATE OR REPLACE FUNCTION get_recent_memories(
    limit_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    created_at timestamp with time zone
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.metadata,
        m.created_at
    FROM public.memories m
    WHERE m.user_id = auth.uid()
    ORDER BY m.created_at DESC
    LIMIT limit_count;
END;
$$;

-- ================================================
-- GRANT PERMISSIONS
-- ================================================

-- Grant access to authenticated users
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON public.memories TO authenticated;

-- Grant execute permissions on functions
GRANT EXECUTE ON FUNCTION match_memories TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_memory_count TO authenticated;
GRANT EXECUTE ON FUNCTION get_recent_memories TO authenticated;

-- ================================================
-- COMMENTS FOR DOCUMENTATION
-- ================================================

COMMENT ON TABLE public.memories IS 'Stores user memories with vector embeddings for semantic search';
COMMENT ON COLUMN public.memories.embedding IS 'Vector embedding (1536 dimensions for OpenAI ada-002)';
COMMENT ON COLUMN public.memories.metadata IS 'Additional context: tags, source, conversation_id, etc.';
COMMENT ON FUNCTION match_memories IS 'Performs cosine similarity search on memory embeddings';
