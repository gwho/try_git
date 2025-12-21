/**
 * =================================================================
 * REACT HOOKS: Real-Time Memory Operations
 * =================================================================
 * These hooks provide React components with live-updating memory data
 *
 * WHY HOOKS EXIST:
 * - Encapsulate complex subscription logic
 * - Handle loading/error states automatically
 * - Provide clean, reusable APIs
 * - Enable real-time UI updates without manual polling
 *
 * REAL-TIME MAGIC:
 * When LangGraph agent creates a memory → UI updates INSTANTLY
 * No refresh needed, no polling, pure WebSocket magic
 */

'use client'

import { useEffect, useState, useCallback } from 'react'
import { supabase } from '@/lib/supabase/browser-client'
import type {
  Memory,
  MemorySearchResult,
  MemoryMetadata
} from '@/types/database.types'
import type { RealtimePostgresChangesPayload } from '@supabase/supabase-js'

// =================================================================
// HOOK: useMemories() - Real-time memory list
// =================================================================
/**
 * Subscribe to all memories for the current user
 * Updates automatically when memories are created/updated/deleted
 *
 * WHY THIS EXISTS:
 * - Shows user's memory timeline
 * - Updates live when LangGraph agent saves memories
 * - Handles loading states automatically
 *
 * EXAMPLE USAGE:
 * ```tsx
 * function MemoryList() {
 *   const { memories, loading, error } = useMemories()
 *
 *   if (loading) return <Spinner />
 *   if (error) return <Error message={error.message} />
 *
 *   return memories.map(m => <MemoryCard key={m.id} memory={m} />)
 * }
 * ```
 *
 * HOW REAL-TIME WORKS:
 * 1. Initial fetch: Get all existing memories
 * 2. Subscribe to changes: Listen for INSERT/UPDATE/DELETE
 * 3. Update React state: Trigger re-render when data changes
 * 4. Cleanup: Unsubscribe when component unmounts
 */
export function useMemories() {
  const [memories, setMemories] = useState<Memory[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  // Fetch initial data
  const fetchMemories = useCallback(async () => {
    try {
      setLoading(true)

      const { data, error } = await supabase
        .from('memories')
        .select('*')
        .order('created_at', { ascending: false })

      if (error) throw error

      setMemories(data || [])
      setError(null)
    } catch (err) {
      setError(err as Error)
      console.error('[useMemories] Fetch error:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    // Initial fetch
    fetchMemories()

    // Set up real-time subscription
    // WHY channel name: Descriptive, helps with debugging
    const channel = supabase
      .channel('memories-realtime')
      .on(
        'postgres_changes',
        {
          event: '*',  // Listen to ALL events (INSERT, UPDATE, DELETE)
          schema: 'public',
          table: 'memories',
        },
        (payload: RealtimePostgresChangesPayload<Memory>) => {
          // Handle INSERT: Add new memory to top of list
          if (payload.eventType === 'INSERT') {
            setMemories((current) => [payload.new, ...current])
          }

          // Handle UPDATE: Replace old version with new
          else if (payload.eventType === 'UPDATE') {
            setMemories((current) =>
              current.map((m) =>
                m.id === payload.new.id ? payload.new : m
              )
            )
          }

          // Handle DELETE: Remove from list
          else if (payload.eventType === 'DELETE') {
            setMemories((current) =>
              current.filter((m) => m.id !== payload.old.id)
            )
          }
        }
      )
      .subscribe()

    // Cleanup: Unsubscribe when component unmounts
    // WHY: Prevents memory leaks
    return () => {
      supabase.removeChannel(channel)
    }
  }, [fetchMemories])

  return {
    memories,
    loading,
    error,
    refetch: fetchMemories,  // Manual refresh if needed
  }
}

// =================================================================
// HOOK: useMemoryActions() - CRUD operations
// =================================================================
/**
 * Provides functions to create, update, and delete memories
 *
 * WHY THIS EXISTS:
 * - Encapsulates database operations
 * - Handles errors consistently
 * - Type-safe parameters
 * - Automatically gets current user
 *
 * EXAMPLE USAGE:
 * ```tsx
 * function CreateMemoryForm() {
 *   const { createMemory } = useMemoryActions()
 *   const [content, setContent] = useState('')
 *
 *   const handleSubmit = async () => {
 *     await createMemory(content, null, { source: 'web_ui' })
 *   }
 * }
 * ```
 */
export function useMemoryActions() {
  const createMemory = useCallback(
    async (
      content: string,
      embedding?: number[],
      metadata?: Partial<MemoryMetadata>
    ) => {
      // Get current user from session
      const {
        data: { user },
        error: authError,
      } = await supabase.auth.getUser()

      if (authError || !user) {
        throw new Error('User not authenticated')
      }

      // Insert memory
      // WHY user_id required: RLS policy checks this
      const { data, error } = await supabase
        .from('memories')
        .insert({
          user_id: user.id,
          content,
          embedding,
          metadata,
        })
        .select()
        .single()

      if (error) throw error
      return data
    },
    []
  )

  const updateMemory = useCallback(
    async (
      id: string,
      updates: {
        content?: string
        embedding?: number[]
        metadata?: Partial<MemoryMetadata>
      }
    ) => {
      const { data, error } = await supabase
        .from('memories')
        .update(updates)
        .eq('id', id)
        .select()
        .single()

      if (error) throw error
      return data
    },
    []
  )

  const deleteMemory = useCallback(async (id: string) => {
    const { error } = await supabase
      .from('memories')
      .delete()
      .eq('id', id)

    if (error) throw error
  }, [])

  return {
    createMemory,
    updateMemory,
    deleteMemory,
  }
}

// =================================================================
// HOOK: useMemorySearch() - Semantic search
// =================================================================
/**
 * Search memories by meaning using vector similarity
 *
 * WHY THIS EXISTS:
 * - Enables "Ask your memories" feature
 * - Finds relevant context for AI responses
 * - Manages search state (loading, results, errors)
 *
 * EXAMPLE USAGE:
 * ```tsx
 * function MemorySearch() {
 *   const { results, searching, searchMemories } = useMemorySearch()
 *
 *   const handleSearch = async (query: string) => {
 *     const embedding = await generateEmbedding(query)
 *     await searchMemories(embedding, { matchThreshold: 0.75 })
 *   }
 *
 *   return results.map(r => (
 *     <div>{r.content} ({r.similarity * 100}% match)</div>
 *   ))
 * }
 * ```
 */
export function useMemorySearch() {
  const [results, setResults] = useState<MemorySearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const searchMemories = useCallback(
    async (
      queryEmbedding: number[],
      options?: {
        matchThreshold?: number
        matchCount?: number
      }
    ) => {
      try {
        setSearching(true)
        setError(null)

        // Call PostgreSQL function
        const { data, error } = await supabase.rpc('match_memories', {
          query_embedding: queryEmbedding,
          match_threshold: options?.matchThreshold || 0.7,
          match_count: options?.matchCount || 10,
        })

        if (error) throw error

        setResults(data || [])
        return data || []
      } catch (err) {
        setError(err as Error)
        console.error('[useMemorySearch] Search error:', err)
        return []
      } finally {
        setSearching(false)
      }
    },
    []
  )

  const clearResults = useCallback(() => {
    setResults([])
    setError(null)
  }, [])

  return {
    results,
    searching,
    error,
    searchMemories,
    clearResults,
  }
}
