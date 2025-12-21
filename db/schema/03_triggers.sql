-- =================================================================
-- DATABASE TRIGGERS & AUTOMATION
-- =================================================================
-- Triggers are "automatic behaviors" that run when data changes
--
-- WHY TRIGGERS EXIST:
-- - Consistency: Ensures data is always valid
-- - DRY: Don't repeat logic in Python, TypeScript, AND SQL
-- - Reliability: Can't forget to update a timestamp
-- - Performance: Database-level operations are faster

-- =================================================================
-- TRIGGER FUNCTION: Auto-update timestamp
-- =================================================================
-- Automatically sets updated_at whenever a memory is modified
--
-- WHY THIS IS IMPORTANT:
-- - Without this, we'd need EVERY client (Python, TypeScript, mobile)
--   to remember to set updated_at
-- - Bugs happen: Someone forgets, and updated_at becomes unreliable
-- - Database guarantee: This ALWAYS runs, no exceptions

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    -- NEW is the row being inserted/updated
    -- Set its updated_at to right now
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =================================================================
-- TRIGGER: Apply timestamp update to memories table
-- =================================================================
-- BEFORE UPDATE: Runs before the row is written to disk
-- FOR EACH ROW: Runs once per row (vs once per statement)

CREATE TRIGGER trigger_memories_updated_at
    BEFORE UPDATE ON public.memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =================================================================
-- TRIGGER FUNCTION: Validate embedding dimensions
-- =================================================================
-- Ensures embeddings are always the correct size
--
-- WHY THIS IS CRITICAL:
-- - Mismatched dimensions cause vector math to fail
-- - Better to fail fast with clear error than corrupt data
-- - Protects against bugs in embedding generation

CREATE OR REPLACE FUNCTION validate_embedding_dimension()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    -- Only validate if embedding is provided
    IF NEW.embedding IS NOT NULL THEN
        -- Check if dimension count matches expected (1536 for OpenAI)
        IF array_length(NEW.embedding::float[], 1) != 1536 THEN
            RAISE EXCEPTION
                'Invalid embedding dimension: expected 1536, got %',
                array_length(NEW.embedding::float[], 1);
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

-- =================================================================
-- TRIGGER: Validate embeddings before insert or update
-- =================================================================

CREATE TRIGGER trigger_validate_embedding
    BEFORE INSERT OR UPDATE ON public.memories
    FOR EACH ROW
    EXECUTE FUNCTION validate_embedding_dimension();

-- =================================================================
-- TRIGGER FUNCTION: Sanitize content
-- =================================================================
-- Cleans up memory content before storage
--
-- WHY THIS EXISTS:
-- - Prevents storing excessive whitespace
-- - Ensures content is never empty
-- - Normalizes data for consistent search results

CREATE OR REPLACE FUNCTION sanitize_memory_content()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    -- Trim leading/trailing whitespace
    NEW.content = TRIM(NEW.content);

    -- Ensure content is not empty after trimming
    IF LENGTH(NEW.content) = 0 THEN
        RAISE EXCEPTION 'Memory content cannot be empty';
    END IF;

    -- Normalize multiple spaces to single space
    NEW.content = REGEXP_REPLACE(NEW.content, '\s+', ' ', 'g');

    RETURN NEW;
END;
$$;

-- =================================================================
-- TRIGGER: Sanitize content before storing
-- =================================================================

CREATE TRIGGER trigger_sanitize_content
    BEFORE INSERT OR UPDATE ON public.memories
    FOR EACH ROW
    EXECUTE FUNCTION sanitize_memory_content();

-- =================================================================
-- TRIGGER FUNCTION: Initialize metadata
-- =================================================================
-- Ensures metadata always has required fields
--
-- WHY THIS IS USEFUL:
-- - Guarantees certain fields always exist
-- - Prevents "undefined" errors in application code
-- - Tracks creation source automatically

CREATE OR REPLACE FUNCTION initialize_metadata()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    -- Ensure metadata is at least an empty object
    IF NEW.metadata IS NULL THEN
        NEW.metadata = '{}'::jsonb;
    END IF;

    -- Add creation timestamp to metadata if not present
    IF NOT (NEW.metadata ? 'created_by_trigger') THEN
        NEW.metadata = NEW.metadata || jsonb_build_object(
            'created_by_trigger', true,
            'schema_version', '1.0'
        );
    END IF;

    RETURN NEW;
END;
$$;

-- =================================================================
-- TRIGGER: Initialize metadata on insert
-- =================================================================

CREATE TRIGGER trigger_initialize_metadata
    BEFORE INSERT ON public.memories
    FOR EACH ROW
    EXECUTE FUNCTION initialize_metadata();

-- =================================================================
-- COMMENTS: Documentation
-- =================================================================

COMMENT ON FUNCTION update_updated_at_column IS
'Automatically updates the updated_at timestamp when a row is modified';

COMMENT ON FUNCTION validate_embedding_dimension IS
'Validates that vector embeddings have exactly 1536 dimensions';

COMMENT ON FUNCTION sanitize_memory_content IS
'Cleans and normalizes memory content before storage';

COMMENT ON FUNCTION initialize_metadata IS
'Ensures metadata JSONB field always has required properties';
