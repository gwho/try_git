/**
 * =================================================================
 * SUPABASE SERVER CLIENT
 * =================================================================
 * This client runs on the SERVER (Node.js environment)
 *
 * WHY THIS EXISTS:
 * - Server Components need a different client than browser
 * - Handles cookies differently (Next.js server vs browser)
 * - Still enforces RLS (uses user's session from cookies)
 * - Can run in API routes, Server Components, Server Actions
 *
 * WHERE TO USE:
 * - Server Components (app directory, no 'use client')
 * - API route handlers (app/api/*/route.ts)
 * - Server Actions
 *
 * WHERE NOT TO USE:
 * - Client Components (use browser-client.ts)
 * - When you need to bypass RLS (use admin-client.ts)
 */

import { createServerClient, type CookieOptions } from '@supabase/ssr'
import { cookies } from 'next/headers'
import type { Database } from '@/types/database.types'

/**
 * Creates a typed Supabase client for server-side use
 *
 * COOKIE HANDLING:
 * - Reads user session from HTTP cookies
 * - Updates cookies when session changes
 * - Respects Next.js cookie limitations in different contexts
 *
 * WHY ASYNC:
 * - Next.js 15+ requires cookies() to be awaited
 * - Ensures cookie state is synchronized
 *
 * EXAMPLE USAGE:
 * ```tsx
 * import { createServerClient } from '@/lib/supabase/server-client'
 *
 * export default async function MemoryPage() {
 *   const supabase = await createServerClient()
 *
 *   // Runs on server, uses user's session from cookies
 *   const { data: memories } = await supabase
 *     .from('memories')
 *     .select('*')
 *
 *   // Still respects RLS - only user's memories returned
 * }
 * ```
 */
export async function createServerClient() {
  const cookieStore = await cookies()

  return createServerClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        /**
         * Get cookie value
         * WHY: Supabase needs to read auth session from cookies
         */
        get(name: string) {
          return cookieStore.get(name)?.value
        },

        /**
         * Set cookie value
         * WHY: Auth state changes (login, refresh) need to update cookies
         *
         * TRY-CATCH: Some Next.js contexts don't allow cookie setting
         * (e.g., Server Components during rendering)
         * We silently fail in these cases - it's expected behavior
         */
        set(name: string, value: string, options: CookieOptions) {
          try {
            cookieStore.set({ name, value, ...options })
          } catch {
            // Expected in some contexts (Server Components)
            // Middleware will handle session refresh
          }
        },

        /**
         * Remove cookie
         * WHY: Logout needs to clear auth cookies
         */
        remove(name: string, options: CookieOptions) {
          try {
            cookieStore.set({ name, value: '', ...options })
          } catch {
            // Expected in some contexts
          }
        },
      },
    }
  )
}
