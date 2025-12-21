/**
 * Type definitions for the Memory Agent system
 */

export interface Memory {
  id: string;
  user_id: string;
  content: string;
  embedding: number[] | null;
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface MemoryInsert {
  user_id: string;
  content: string;
  embedding?: number[];
  metadata?: Record<string, any>;
}

export interface MemoryUpdate {
  content?: string;
  embedding?: number[];
  metadata?: Record<string, any>;
}

export interface MemorySearchResult {
  id: string;
  user_id: string;
  content: string;
  metadata: Record<string, any>;
  similarity: number;
  created_at: string;
}

export interface MemorySearchParams {
  query_embedding: number[];
  match_threshold?: number;
  match_count?: number;
  filter_user_id?: string;
}

// Database schema type (for Supabase client type safety)
export interface Database {
  public: {
    Tables: {
      memories: {
        Row: Memory;
        Insert: MemoryInsert;
        Update: MemoryUpdate;
      };
    };
    Functions: {
      match_memories: {
        Args: MemorySearchParams;
        Returns: MemorySearchResult[];
      };
      get_user_memory_count: {
        Args: Record<string, never>;
        Returns: number;
      };
      get_recent_memories: {
        Args: { limit_count?: number };
        Returns: Pick<Memory, 'id' | 'content' | 'metadata' | 'created_at'>[];
      };
    };
  };
}
