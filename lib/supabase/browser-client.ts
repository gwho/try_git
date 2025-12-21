/**
 * =================================================================
 * SUPABASE BROWSER CLIENT
 * =================================================================
 * This client runs in the USER'S BROWSER
 *
 * WHY THIS EXISTS:
 * - Handles authentication cookies automatically
 * - Enforces Row Level Security (users see only their data)
 * - Enables real-time subscriptions
 * - Type-safe database queries
 *
 * WHERE TO USE:
 * - React components marked 'use client'
 * - Client-side API calls
 * - Real-time subscriptions
 *
 * WHERE NOT TO USE:
 * - Server Components (use server-client.ts instead)
 * - API routes that need elevated permissions
 */

import { createBrowserClient } from '@supabase/ssr'
import type { Database } from '@/types/database.types'

/**
 * Creates a typed Supabase client for browser use
 *
 * IMPORTANT SECURITY NOTES:
 * - Uses ANON key (safe to expose publicly)
 * - All queries subject to RLS policies
 * - Users can only access their own data
 * - Real-time subscriptions automatically filtered by user
 *
 * EXAMPLE USAGE:
 * ```tsx
 * 'use client'
 *
 * import { createBrowserClient } from '@/lib/supabase/browser-client'
 *
 * export function MemoryList() {
 *   const supabase = createBrowserClient()
 *
 *   // Type-safe query with autocomplete
 *   const { data: memories } = await supabase
 *     .from('memories')
 *     .select('*')
 *
 *   // TypeScript knows: memories is Memory[] | null
 * }
 * ```
 */
export function createClient() {
  // WHY environment variables are prefixed with NEXT_PUBLIC_:
  // - Next.js only exposes env vars with this prefix to the browser
  // - Prevents accidentally exposing secrets (like SERVICE_ROLE_KEY)
  // - These values are SAFE to expose (designed for public use)

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

  // Validate environment variables
  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error(
      'Missing Supabase environment variables. ' +
      'Please set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY'
    )
  }

  // Create typed client
  // The <Database> generic provides full type safety
  return createBrowserClient<Database>(supabaseUrl, supabaseAnonKey)
}

/**
 * Singleton pattern: Reuse client instance
 *
 * WHY SINGLETON:
 * - Prevents creating multiple WebSocket connections
 * - Reuses authentication state
 * - Better performance (less overhead)
 *
 * USAGE:
 * ```tsx
 * import { supabase } from '@/lib/supabase/browser-client'
 *
 * // Reuses the same client instance
 * const { data } = await supabase.from('memories').select('*')
 * ```
 */
export const supabase = createClient()
