/**
 * Supabase Edge Function: LangGraph Webhook Receiver
 *
 * This function receives webhook calls from LangGraph when the agent
 * state is updated, allowing you to process agent thoughts, decisions,
 * and memory operations in real-time.
 *
 * Deploy with: supabase functions deploy langgraph-webhook
 */

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

// Type definitions for webhook payloads
interface LangGraphWebhookPayload {
  agent_id: string;
  state: {
    messages?: Array<{
      role: string;
      content: string;
    }>;
    memory?: {
      content: string;
      should_persist: boolean;
      metadata?: Record<string, any>;
    };
    [key: string]: any;
  };
  event_type: 'state_update' | 'task_complete' | 'error';
  timestamp: string;
}

interface MemoryEmbeddingRequest {
  content: string;
}

/**
 * Generate embedding using OpenAI API
 * Note: In production, consider caching or batching embeddings
 */
async function generateEmbedding(text: string): Promise<number[]> {
  const openaiApiKey = Deno.env.get('OPENAI_API_KEY');

  if (!openaiApiKey) {
    console.error('OPENAI_API_KEY not configured');
    return [];
  }

  try {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openaiApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        input: text,
        model: 'text-embedding-3-small', // or 'text-embedding-ada-002'
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.data[0].embedding;
  } catch (error) {
    console.error('Error generating embedding:', error);
    return [];
  }
}

/**
 * Verify webhook signature/secret
 */
function verifyWebhookSecret(request: Request): boolean {
  const secret = request.headers.get('x-webhook-secret');
  const expectedSecret = Deno.env.get('LANGGRAPH_WEBHOOK_SECRET');

  if (!expectedSecret) {
    console.warn('LANGGRAPH_WEBHOOK_SECRET not configured - skipping verification');
    return true;
  }

  return secret === expectedSecret;
}

serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', {
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-webhook-secret',
      },
    });
  }

  try {
    // Verify webhook secret
    if (!verifyWebhookSecret(req)) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized: Invalid webhook secret' }),
        {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    // Parse webhook payload
    const payload: LangGraphWebhookPayload = await req.json();
    console.log('Received webhook:', payload.event_type, payload.agent_id);

    // Initialize Supabase client with service role
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Process memory if present in the state
    if (payload.state.memory && payload.state.memory.should_persist) {
      const { content, metadata } = payload.state.memory;

      // Generate embedding for the memory content
      const embedding = await generateEmbedding(content);

      // Extract user_id from metadata or use a default
      // In a real system, you'd have user context from the LangGraph state
      const userId = metadata?.user_id || payload.agent_id;

      // Insert memory into database
      const { data, error } = await supabase
        .from('memories')
        .insert({
          user_id: userId,
          content: content,
          embedding: embedding,
          metadata: {
            ...metadata,
            agent_id: payload.agent_id,
            event_type: payload.event_type,
            timestamp: payload.timestamp,
          },
        })
        .select()
        .single();

      if (error) {
        console.error('Error inserting memory:', error);
        return new Response(
          JSON.stringify({ error: 'Failed to persist memory', details: error.message }),
          {
            status: 500,
            headers: { 'Content-Type': 'application/json' },
          }
        );
      }

      console.log('Memory persisted:', data.id);
    }

    // You can add more event handlers here
    switch (payload.event_type) {
      case 'state_update':
        // Handle state updates
        console.log('Agent state updated');
        break;
      case 'task_complete':
        // Handle task completion
        console.log('Agent task completed');
        break;
      case 'error':
        // Handle errors from agent
        console.error('Agent reported error:', payload.state);
        break;
    }

    // Return success response
    return new Response(
      JSON.stringify({
        success: true,
        message: 'Webhook processed successfully',
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
      }
    );

  } catch (error) {
    console.error('Error processing webhook:', error);
    return new Response(
      JSON.stringify({
        error: 'Internal server error',
        details: error.message,
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
});
