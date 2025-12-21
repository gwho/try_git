-- =================================================================
-- SEMANTIC SEARCH FUNCTIONS
-- =================================================================
-- These functions enable the AI agent to find relevant memories
-- by MEANING, not just keyword matching.
--
-- Example: Query "What food does the user like?"
--          Finds: "I love margherita pizza"
--          Even though "food" and "like" aren't in the memory!

-- =================================================================
-- FUNCTION: match_memories
-- =================================================================
-- The CORE of semantic search - finds similar memories using vector math
--
-- HOW IT WORKS:
-- 1. Takes a query embedding (the AI representation of a question)
-- 2. Compares it to ALL user memories using cosine distance
-- 3. Returns the most similar ones, ranked by relevance
--
-- COSINE DISTANCE: Measures angle between vectors
-- - 0 = identical meaning
-- - 1 = completely different meaning
-- - We use (1 - distance) to get similarity (higher = better)

CREATE OR REPLACE FUNCTION match_memories(
    -- The embedding of what we're searching for
    query_embedding vector(1536),

    -- Minimum similarity threshold (0.0 to 1.0)
    -- WHY: Filters out irrelevant results
    -- 0.7 is a good default (70% similar)
    match_threshold float DEFAULT 0.7,

    -- How many results to return
    -- WHY: Prevents returning thousands of low-quality matches
    match_count int DEFAULT 10,

    -- Optional: Filter to specific user
    -- WHY: Allows admin queries across all users (with proper auth)
    -- NULL means "use current user from RLS"
    filter_user_id uuid DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    user_id uuid,
    content text,
    metadata jsonb,
    similarity float,  -- 0.0 to 1.0, higher is better
    created_at timestamp with time zone
)
LANGUAGE plpgsql
SECURITY DEFINER  -- Runs with function creator's privileges
SET search_path = public  -- Security: Prevent schema injection
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.user_id,
        m.content,
        m.metadata,
        -- Convert distance (lower=better) to similarity (higher=better)
        -- <=> is the cosine distance operator from pgvector
        1 - (m.embedding <=> query_embedding) as similarity,
        m.created_at
    FROM public.memories m
    WHERE
        -- Optional user filter (for admin queries)
        (filter_user_id IS NULL OR m.user_id = filter_user_id)

        -- Similarity threshold: Only return relevant results
        AND (1 - (m.embedding <=> query_embedding)) > match_threshold

        -- RLS enforcement: Even with SECURITY DEFINER, respect RLS
        -- This ensures users only see their own memories
        AND m.user_id = auth.uid()

    -- Sort by most similar first
    ORDER BY m.embedding <=> query_embedding

    -- Limit results
    LIMIT match_count;
END;
$$;

-- =================================================================
-- FUNCTION: get_recent_memories
-- =================================================================
-- Simple time-based retrieval - get the latest memories
--
-- WHY THIS EXISTS:
-- - Useful for showing "conversation history"
-- - Faster than semantic search (just uses timestamp index)
-- - LangGraph agents often need recent context

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
    WHERE m.user_id = auth.uid()  -- RLS enforcement
    ORDER BY m.created_at DESC
    LIMIT limit_count;
END;
$$;

-- =================================================================
-- FUNCTION: get_user_memory_count
-- =================================================================
-- Simple counter - how many memories does this user have?
--
-- WHY THIS EXISTS:
-- - UI can show "You have 47 memories"
-- - Agents can gauge how much context they have
-- - Useful for analytics and debugging

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

-- =================================================================
-- FUNCTION: search_by_metadata
-- =================================================================
-- Search memories by their metadata tags/properties
--
-- WHY THIS EXISTS:
-- - Sometimes you want exact filters: "Show me all 'important' memories"
-- - Combine with semantic search for powerful queries
-- - Example: "Find memories tagged 'work' that mention 'deadlines'"

CREATE OR REPLACE FUNCTION search_by_metadata(
    metadata_query jsonb,
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
    WHERE
        m.user_id = auth.uid()  -- RLS enforcement
        AND m.metadata @> metadata_query  -- JSONB containment operator
    ORDER BY m.created_at DESC
    LIMIT limit_count;
END;
$$;

-- =================================================================
-- GRANT: Give authenticated users permission to call these functions
-- =================================================================

GRANT EXECUTE ON FUNCTION match_memories TO authenticated;
GRANT EXECUTE ON FUNCTION get_recent_memories TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_memory_count TO authenticated;
GRANT EXECUTE ON FUNCTION search_by_metadata TO authenticated;

-- =================================================================
-- COMMENTS: Documentation for database tools
-- =================================================================

COMMENT ON FUNCTION match_memories IS
'Semantic search: Find memories by meaning using vector similarity';

COMMENT ON FUNCTION get_recent_memories IS
'Temporal search: Get the most recent memories chronologically';

COMMENT ON FUNCTION get_user_memory_count IS
'Analytics: Count total memories for current user';

COMMENT ON FUNCTION search_by_metadata IS
'Metadata search: Filter memories by JSONB properties (tags, source, etc)';
