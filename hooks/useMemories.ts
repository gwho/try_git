/**
 * React Hooks for Memory Management
 * Provides real-time subscriptions, CRUD operations, and vector search
 */

'use client';

import { useEffect, useState, useCallback } from 'react';
import { createClient } from '@/lib/supabase/client';
import type { Memory, MemorySearchResult } from '@/lib/types/memory';
import type { RealtimePostgresChangesPayload } from '@supabase/supabase-js';

/**
 * Hook for real-time memory subscriptions
 * Automatically updates when new memories are created, updated, or deleted
 */
export function useMemories() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const supabase = createClient();

  // Fetch initial memories
  const fetchMemories = useCallback(async () => {
    try {
      setLoading(true);
      const { data, error } = await supabase
        .from('memories')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;
      setMemories(data || []);
      setError(null);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching memories:', err);
    } finally {
      setLoading(false);
    }
  }, [supabase]);

  useEffect(() => {
    fetchMemories();

    // Set up real-time subscription
    const channel = supabase
      .channel('memories-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'memories',
        },
        (payload: RealtimePostgresChangesPayload<Memory>) => {
          if (payload.eventType === 'INSERT') {
            setMemories((current) => [payload.new, ...current]);
          } else if (payload.eventType === 'UPDATE') {
            setMemories((current) =>
              current.map((memory) =>
                memory.id === payload.new.id ? payload.new : memory
              )
            );
          } else if (payload.eventType === 'DELETE') {
            setMemories((current) =>
              current.filter((memory) => memory.id !== payload.old.id)
            );
          }
        }
      )
      .subscribe();

    // Cleanup subscription on unmount
    return () => {
      supabase.removeChannel(channel);
    };
  }, [supabase, fetchMemories]);

  return { memories, loading, error, refetch: fetchMemories };
}

/**
 * Hook for memory CRUD operations
 */
export function useMemoryActions() {
  const supabase = createClient();

  const createMemory = useCallback(
    async (content: string, embedding?: number[], metadata?: Record<string, any>) => {
      const {
        data: { user },
      } = await supabase.auth.getUser();

      if (!user) throw new Error('User not authenticated');

      const { data, error } = await supabase
        .from('memories')
        .insert({
          user_id: user.id,
          content,
          embedding,
          metadata: metadata || {},
        })
        .select()
        .single();

      if (error) throw error;
      return data;
    },
    [supabase]
  );

  const updateMemory = useCallback(
    async (
      id: string,
      updates: { content?: string; embedding?: number[]; metadata?: Record<string, any> }
    ) => {
      const { data, error } = await supabase
        .from('memories')
        .update(updates)
        .eq('id', id)
        .select()
        .single();

      if (error) throw error;
      return data;
    },
    [supabase]
  );

  const deleteMemory = useCallback(
    async (id: string) => {
      const { error } = await supabase.from('memories').delete().eq('id', id);

      if (error) throw error;
    },
    [supabase]
  );

  return { createMemory, updateMemory, deleteMemory };
}

/**
 * Hook for vector similarity search
 */
export function useMemorySearch() {
  const [results, setResults] = useState<MemorySearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const supabase = createClient();

  const searchMemories = useCallback(
    async (
      queryEmbedding: number[],
      options?: {
        matchThreshold?: number;
        matchCount?: number;
      }
    ) => {
      try {
        setSearching(true);
        setError(null);

        const { data, error } = await supabase.rpc('match_memories', {
          query_embedding: queryEmbedding,
          match_threshold: options?.matchThreshold || 0.7,
          match_count: options?.matchCount || 10,
        });

        if (error) throw error;
        setResults(data || []);
        return data || [];
      } catch (err) {
        setError(err as Error);
        console.error('Error searching memories:', err);
        return [];
      } finally {
        setSearching(false);
      }
    },
    [supabase]
  );

  const clearResults = useCallback(() => {
    setResults([]);
    setError(null);
  }, []);

  return { results, searching, error, searchMemories, clearResults };
}

/**
 * Hook for getting recent memories
 */
export function useRecentMemories(limit: number = 10) {
  const [memories, setMemories] = useState<
    Pick<Memory, 'id' | 'content' | 'metadata' | 'created_at'>[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const supabase = createClient();

  useEffect(() => {
    const fetchRecent = async () => {
      try {
        setLoading(true);
        const { data, error } = await supabase.rpc('get_recent_memories', {
          limit_count: limit,
        });

        if (error) throw error;
        setMemories(data || []);
        setError(null);
      } catch (err) {
        setError(err as Error);
        console.error('Error fetching recent memories:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchRecent();
  }, [supabase, limit]);

  return { memories, loading, error };
}

/**
 * Hook for getting memory count
 */
export function useMemoryCount() {
  const [count, setCount] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const supabase = createClient();

  useEffect(() => {
    const fetchCount = async () => {
      try {
        setLoading(true);
        const { data, error } = await supabase.rpc('get_user_memory_count');

        if (error) throw error;
        setCount(data || 0);
      } catch (err) {
        console.error('Error fetching memory count:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchCount();
  }, [supabase]);

  return { count, loading };
}
