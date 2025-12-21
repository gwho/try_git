/**
 * =================================================================
 * SUPABASE ADMIN CLIENT
 * =================================================================
 * This client has ELEVATED PRIVILEGES and BYPASSES RLS
 *
 * ⚠️ SECURITY WARNING ⚠️
 * This client can access ALL data from ALL users!
 * NEVER expose this to the browser or client-side code!
 *
 * WHY THIS EXISTS:
 * - Background jobs that process all users' data
 * - Admin operations (moderation, analytics)
 * - LangGraph agents running server-side
 * - Edge Functions that need cross-user operations
 *
 * WHERE TO USE:
 * - Server-side background jobs
 * - Admin API routes (with additional auth checks!)
 * - Supabase Edge Functions
 * - LangGraph Python nodes (server-side)
 *
 * WHERE NEVER TO USE:
 * - Client Components ❌ NEVER!
 * - Public API endpoints ❌
 * - Any code that runs in browser ❌
 */

import { createClient } from '@supabase/supabase-js'
import type { Database } from '@/types/database.types'

/**
 * Creates an admin client with RLS bypass
 *
 * SECURITY MODEL:
 * - Uses SERVICE_ROLE_KEY (secret, server-only)
 * - Bypasses ALL Row Level Security policies
 * - Can read/write any user's data
 * - Can execute administrative functions
 *
 * WHEN TO USE:
 * 1. LangGraph agent needs to save memories for users
 *    (agent doesn't have browser session)
 * 2. Background job: "Delete old memories for all users"
 * 3. Admin dashboard: "Show statistics across all users"
 *
 * EXAMPLE - CORRECT USAGE:
 * ```typescript
 * // In Edge Function or API route with admin auth
 * import { createAdminClient } from '@/lib/supabase/admin-client'
 *
 * export async function POST(request: Request) {
 *   // ✅ Verify admin permission first!
 *   const isAdmin = await verifyAdminToken(request)
 *   if (!isAdmin) return Response.json({error: 'Unauthorized'}, {status: 401})
 *
 *   const supabase = createAdminClient()
 *
 *   // Can access any user's data
 *   const { data } = await supabase
 *     .from('memories')
 *     .select('*')  // Gets ALL users' memories!
 * }
 * ```
 *
 * EXAMPLE - INCORRECT USAGE:
 * ```typescript
 * // ❌ NEVER DO THIS:
 * 'use client'
 * import { createAdminClient } from '@/lib/supabase/admin-client'
 *
 * export function Component() {
 *   const supabase = createAdminClient()  // ❌ Exposes secret to browser!
 * }
 * ```
 */
export function createAdminClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

  // Validate environment
  if (!supabaseUrl || !supabaseServiceKey) {
    throw new Error(
      'Missing Supabase admin credentials. ' +
      'Ensure SUPABASE_SERVICE_ROLE_KEY is set in environment.'
    )
  }

  // Additional safety check: Ensure we're on server
  if (typeof window !== 'undefined') {
    throw new Error(
      '❌ SECURITY VIOLATION: ' +
      'Admin client cannot be created in browser environment! ' +
      'This would expose the service role key.'
    )
  }

  // Create admin client with no cookie handling
  // WHY: Admin client doesn't use user sessions
  return createClient<Database>(
    supabaseUrl,
    supabaseServiceKey,
    {
      auth: {
        autoRefreshToken: false,  // Not needed for service role
        persistSession: false,     // No user session to persist
      },
    }
  )
}

/**
 * Singleton admin client
 *
 * WHY SINGLETON:
 * - Reuse connection pool
 * - Better performance
 * - Consistent configuration
 *
 * SECURITY NOTE:
 * This is safe as a module-level export because:
 * 1. Only server-side code can import it
 * 2. Next.js doesn't bundle server code to browser
 * 3. The safety check prevents instantiation in browser
 */
export const adminSupabase = createAdminClient()
