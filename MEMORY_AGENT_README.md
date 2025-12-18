# Long-Term Memory Agent with LangGraph & Supabase

A production-ready boilerplate for building AI agents with persistent, searchable long-term memory using **LangGraph** for orchestration and **Supabase** as the Backend-as-a-Service platform.

## Features

- **Vector Memory Storage**: Store memories with embeddings using pgvector
- **Real-time Updates**: Live UI updates when memories are created/updated
- **Semantic Search**: Find relevant memories using cosine similarity
- **Multi-tenant Security**: Row-level security ensures data isolation
- **LangGraph Integration**: Ready-to-use Python agent nodes
- **Webhook System**: Real-time synchronization between agent and frontend
- **TypeScript Type Safety**: Fully typed Supabase client and database schema

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Next.js UI    │←────────│   Supabase       │←────────│  LangGraph      │
│  (React Hooks)  │ Realtime│  PostgreSQL      │ Webhook │  Agent (Python) │
│                 │         │  + pgvector      │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                     │
                                     │ RPC Functions
                                     │ Vector Search
                                     ▼
                            ┌──────────────────┐
                            │  Edge Functions  │
                            │  (Webhook recv)  │
                            └──────────────────┘
```

## Quick Start

### Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.10+
- Supabase account ([supabase.com](https://supabase.com))
- OpenAI API key (for embeddings)

### 1. Supabase Setup

#### Create a new Supabase project

```bash
# Install Supabase CLI
npm install -g supabase

# Login to Supabase
supabase login

# Initialize project (or link existing project)
supabase init

# Link to your remote project
supabase link --project-ref your-project-ref
```

#### Run migrations

```bash
# Apply the database schema
supabase db push
```

Alternatively, you can manually run the SQL migration in the Supabase SQL Editor:
- Go to your Supabase dashboard → SQL Editor
- Copy the contents of `supabase/migrations/20250101000000_initial_memory_agent_schema.sql`
- Execute the migration

#### Enable Realtime

In your Supabase dashboard:
1. Go to **Database** → **Replication**
2. Enable replication for the `memories` table

### 2. Next.js Frontend Setup

#### Install dependencies

```bash
npm install
```

#### Configure environment variables

```bash
cp .env.example .env.local
```

Edit `.env.local` with your Supabase credentials:

```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
LANGGRAPH_WEBHOOK_SECRET=your-secure-secret
OPENAI_API_KEY=sk-your-openai-key
```

#### Run the development server

```bash
npm run dev
```

Visit [http://localhost:3000/memories](http://localhost:3000/memories) to see the memory dashboard.

### 3. Deploy Edge Function (Optional)

The Edge Function receives webhooks from your LangGraph agent:

```bash
# Deploy the webhook receiver
supabase functions deploy langgraph-webhook

# Set environment variables for the Edge Function
supabase secrets set OPENAI_API_KEY=sk-your-key
supabase secrets set LANGGRAPH_WEBHOOK_SECRET=your-secret
```

### 4. Python Agent Setup

#### Install Python dependencies

```bash
cd python
pip install -r requirements.txt
```

#### Configure environment

Create a `.env` file in the `python` directory:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
OPENAI_API_KEY=sk-your-openai-key
LANGGRAPH_WEBHOOK_SECRET=your-secret
```

#### Run the example agent

```bash
python langgraph_memory_integration.py
```

## Project Structure

```
.
├── app/
│   └── memories/
│       └── page.tsx              # Memory dashboard UI
├── hooks/
│   └── useMemories.ts            # React hooks for memory operations
├── lib/
│   ├── supabase/
│   │   ├── client.ts             # Browser Supabase client
│   │   └── server.ts             # Server-side Supabase client
│   └── types/
│       └── memory.ts             # TypeScript type definitions
├── python/
│   ├── langgraph_memory_integration.py  # LangGraph agent example
│   └── requirements.txt          # Python dependencies
├── supabase/
│   ├── functions/
│   │   └── langgraph-webhook/
│   │       └── index.ts          # Edge Function for webhooks
│   └── migrations/
│       └── 20250101000000_initial_memory_agent_schema.sql
├── middleware.ts                 # Next.js middleware for auth
├── .env.example                  # Environment variables template
└── package.json
```

## Database Schema

### `memories` Table

| Column      | Type               | Description                          |
|-------------|--------------------|--------------------------------------|
| id          | uuid               | Primary key                          |
| user_id     | uuid               | Foreign key to auth.users (RLS)      |
| content     | text               | The memory content                   |
| embedding   | vector(1536)       | Vector embedding for semantic search |
| metadata    | jsonb              | Additional context (tags, source)    |
| created_at  | timestamp          | Creation timestamp                   |
| updated_at  | timestamp          | Last update timestamp                |

### PostgreSQL Functions

#### `match_memories()`
Performs vector similarity search using cosine distance.

**Parameters:**
- `query_embedding`: The embedding vector to search for
- `match_threshold`: Minimum similarity score (default: 0.7)
- `match_count`: Maximum results to return (default: 10)
- `filter_user_id`: Optional user filter

**Returns:** Array of memories with similarity scores

#### `get_recent_memories()`
Retrieves the most recent memories for the current user.

**Parameters:**
- `limit_count`: Number of memories to return (default: 10)

#### `get_user_memory_count()`
Returns the total number of memories for the current user.

## Usage Examples

### React: Real-time Memory Subscription

```tsx
'use client';

import { useMemories, useMemoryActions } from '@/hooks/useMemories';

export default function MemoryComponent() {
  const { memories, loading } = useMemories();
  const { createMemory, deleteMemory } = useMemoryActions();

  const handleCreate = async () => {
    await createMemory('I love pizza', undefined, {
      tags: ['food', 'preference']
    });
  };

  return (
    <div>
      {memories.map(memory => (
        <div key={memory.id}>{memory.content}</div>
      ))}
    </div>
  );
}
```

### React: Vector Search

```tsx
import { useMemorySearch } from '@/hooks/useMemories';

export default function SearchComponent() {
  const { results, searchMemories } = useMemorySearch();

  const handleSearch = async (embedding: number[]) => {
    const results = await searchMemories(embedding, {
      matchThreshold: 0.8,
      matchCount: 5
    });
  };

  return (
    <div>
      {results.map(result => (
        <div key={result.id}>
          {result.content} - Similarity: {result.similarity}
        </div>
      ))}
    </div>
  );
}
```

### Python: LangGraph Agent Node

```python
from langgraph_memory_integration import memory_manager

# Save a memory with embedding
memory_manager.save_memory(
    user_id="user-uuid",
    content="The user prefers dark mode",
    metadata={"source": "preferences"}
)

# Search for relevant memories
results = memory_manager.search_memories(
    user_id="user-uuid",
    query="What are the user's preferences?",
    match_threshold=0.75,
    match_count=5
)

for memory in results:
    print(f"{memory['content']} (similarity: {memory['similarity']})")
```

## Security

### Row Level Security (RLS)

All memory operations are protected by RLS policies:

- ✅ Users can only read their own memories
- ✅ Users can only create memories for themselves
- ✅ Users can only update their own memories
- ✅ Users can only delete their own memories

### Authentication

The system uses Supabase Auth. Users must be authenticated to:
- View the memory dashboard
- Create, update, or delete memories
- Search their memories

### API Security

- Edge Functions validate webhook secrets
- Service role keys should never be exposed to the client
- All database queries respect RLS policies

## Customization

### Adjust Embedding Dimensions

If using a different embedding model, update the vector dimension:

```sql
-- In the migration file
embedding vector(768)  -- For models like BERT
```

### Add Custom Metadata

Extend the metadata JSONB column:

```typescript
await createMemory('content', embedding, {
  tags: ['important'],
  conversation_id: 'conv-123',
  source: 'chat',
  priority: 'high'
});
```

### Custom Search Functions

Add specialized search functions in SQL:

```sql
CREATE FUNCTION search_by_tags(tag_array text[])
RETURNS TABLE (...) AS $$
  SELECT * FROM memories
  WHERE metadata->'tags' ?| tag_array
  AND user_id = auth.uid();
$$ LANGUAGE sql;
```

## Deployment

### Vercel (Recommended for Next.js)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

Add environment variables in Vercel dashboard.

### Supabase Edge Functions

```bash
# Deploy all functions
supabase functions deploy

# Or deploy specific function
supabase functions deploy langgraph-webhook
```

### Python Agent (Cloud)

Deploy your LangGraph agent to:
- **Modal**: Serverless Python functions
- **Railway**: Container hosting
- **AWS Lambda**: Serverless with container support
- **Google Cloud Run**: Container hosting

## Monitoring & Debugging

### Check Realtime Connections

In Supabase Dashboard → Database → Replication, verify:
- `memories` table has replication enabled
- Active subscriptions are showing in logs

### Debug Edge Functions

```bash
# View logs
supabase functions logs langgraph-webhook

# Serve locally for testing
supabase functions serve langgraph-webhook
```

### Monitor Vector Search Performance

```sql
-- Check index usage
EXPLAIN ANALYZE
SELECT * FROM memories
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

## Performance Optimization

### Vector Index

The HNSW index provides fast approximate nearest neighbor search:

```sql
CREATE INDEX idx_memories_embedding ON public.memories
USING hnsw (embedding vector_cosine_ops);
```

### Caching Embeddings

Consider caching frequently used embeddings:

```typescript
// Use React Query or SWR
import { useQuery } from '@tanstack/react-query';

const { data: embedding } = useQuery({
  queryKey: ['embedding', text],
  queryFn: () => generateEmbedding(text),
  staleTime: 1000 * 60 * 60 // Cache for 1 hour
});
```

## Troubleshooting

### "Permission denied for table memories"

Ensure RLS is properly configured and you're authenticated:

```typescript
const { data: { user } } = await supabase.auth.getUser();
console.log('Current user:', user?.id);
```

### "Function match_memories does not exist"

Run the migration again:

```bash
supabase db push
```

### Real-time not working

1. Check replication is enabled for `memories` table
2. Verify the channel subscription in browser console
3. Ensure cookies are enabled (required for auth)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - feel free to use this boilerplate for your projects!

## Resources

- [Supabase Documentation](https://supabase.com/docs)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

## Support

- GitHub Issues: [Report bugs or request features]
- Supabase Discord: [Join the community](https://discord.supabase.com)
- LangChain Discord: [Get help with LangGraph](https://discord.gg/langchain)

---

Built with ❤️ using Next.js, Supabase, and LangGraph
