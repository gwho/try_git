# Long-Term Memory Agent Architecture

**Built node-by-node from scratch with LangGraph + Supabase**

## System Overview

This is a **production-ready memory system** that enables AI agents to remember user conversations across sessions using semantic search.

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                         │
│                    (React + Next.js)                        │
└───────────────┬─────────────────────────────────────────────┘
                │ Real-time WebSocket
                ↓
┌─────────────────────────────────────────────────────────────┐
│                     SUPABASE LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ PostgreSQL   │  │  Realtime    │  │   Auth       │      │
│  │ + pgvector   │  │  WebSockets  │  │   RLS        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────┬─────────────────────────────────────────────┘
                │ Service Role API
                ↓
┌─────────────────────────────────────────────────────────────┐
│                   LANGGRAPH AGENT                           │
│              (Python + Memory Manager)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Node-by-Node Architecture

### **Node 1: Database Schema Foundation**
**Location**: `db/schema/01_core_table.sql`

**Why it exists**: The **persistence layer** - where memories live forever.

**What it provides**:
- `memories` table with vector embeddings
- Row Level Security (RLS) for multi-tenancy
- HNSW index for fast vector search
- Automatic ID generation

**Key decisions**:
- **1536 dimensions**: Matches OpenAI's text-embedding-3-small
- **JSONB metadata**: Flexible schema without migrations
- **RLS policies**: Security enforced at database level
- **HNSW index**: 1000x faster than brute-force search

**Dependencies**: None (foundation)
**Depended on by**: Everything

---

### **Node 2: Search Functions**
**Location**: `db/schema/02_search_functions.sql`

**Why it exists**: The **intelligence layer** - find memories by meaning, not keywords.

**What it provides**:
- `match_memories()`: Semantic search via cosine similarity
- `get_recent_memories()`: Fast temporal retrieval
- `get_user_memory_count()`: Analytics
- `search_by_metadata()`: Tag-based filtering

**Key decisions**:
- **Cosine distance**: `1 - (embedding <=> query)` = similarity
- **Default threshold 0.7**: 70% similar (good balance)
- **SECURITY DEFINER**: Allows admin queries but enforces RLS
- **Separate functions**: Different use cases, different indexes

**Example**:
```sql
-- Find memories about food
SELECT * FROM match_memories(
  query_embedding := [0.1, 0.4, ...],  -- "What food does user like?"
  match_threshold := 0.75,
  match_count := 5
);
-- Returns: ["User loves pizza", "User is vegetarian", ...]
```

**Dependencies**: Node 1 (table schema)
**Depended on by**: React hooks, Python agents

---

### **Node 3: Database Triggers**
**Location**: `db/schema/03_triggers.sql`

**Why it exists**: The **automation layer** - keeps data clean without application code remembering.

**What it provides**:
- Auto-update `updated_at` timestamp
- Validate embedding dimensions (must be 1536)
- Sanitize content (trim whitespace, prevent empty)
- Initialize metadata with schema version

**Key decisions**:
- **Triggers vs app code**: Database guarantees > developer memory
- **Fail fast**: Reject bad data immediately
- **Normalization**: Consistent data = better search

**Real bug prevented**:
```python
# Without trigger:
memory = {"content": "  ", "embedding": wrong_size}
supabase.insert(memory)  # ❌ Silently corrupts database

# With trigger:
# ⚠️ ERROR: Memory content cannot be empty
# ⚠️ ERROR: Invalid embedding dimension: expected 1536, got 768
```

**Dependencies**: Node 1 (table schema)
**Depended on by**: All clients (Python, TypeScript)

---

### **Node 4: TypeScript Types**
**Location**: `types/database.types.ts`

**Why it exists**: The **type-safety bridge** - catch bugs at compile time, not runtime.

**What it provides**:
- `Memory`, `MemoryInsert`, `MemoryUpdate` types
- `Database` interface for Supabase client
- Type guards for runtime validation
- Structured `MemoryMetadata` type

**Key decisions**:
- **Three types**: Different operations need different shapes
- **Type guards**: Runtime validation of external data
- **JSONB typing**: Structure for flexible metadata

**Bug prevented**:
```typescript
// Without types:
const memory = await supabase.from('memories').select('conten')  // Typo!
// ❌ Fails at RUNTIME

// With types:
const memory = await supabase.from('memories').select('conten')
//                                                     ^^^^^^
// ⚠️ TypeScript ERROR at compile time
```

**Dependencies**: None (pure types)
**Depended on by**: All TypeScript code

---

### **Node 5: Supabase Clients**
**Location**: `lib/supabase/`

**Why it exists**: The **connection layer** - typed database clients for different environments.

**What it provides**:
- **Browser client**: For React components, enforces RLS
- **Server client**: For Next.js server, handles cookies
- **Admin client**: For agents, bypasses RLS

**Key decisions**:
- **Three clients**: Different environments, different needs
- **Cookie handling**: Server needs explicit management
- **Admin security check**: Prevents browser usage

**Security boundary**:
```
Browser (ANON key)     Server (ANON key)      Server (SERVICE key)
└─ User sees own       └─ User sees own       └─ Sees ALL data
   RLS enforced           RLS enforced           RLS bypassed
   ✅ Safe                ✅ Safe                ❌ NEVER expose
```

**Dependencies**: Node 4 (TypeScript types)
**Depended on by**: React hooks, API routes, Python agents

---

### **Node 6: React Hooks**
**Location**: `hooks/use-memories.ts`

**Why it exists**: The **UI integration layer** - makes database accessible to React with real-time updates.

**What it provides**:
- `useMemories()`: Real-time subscription via WebSockets
- `useMemoryActions()`: Type-safe CRUD operations
- `useMemorySearch()`: Semantic search with state management

**Key decisions**:
- **Separate hooks**: Single responsibility, smaller bundles
- **Automatic state**: Loading, error, cleanup handled
- **Real-time**: WebSocket subscription, zero latency

**Real-time flow**:
```
Python agent saves → PostgreSQL → Realtime broadcast → React hook → UI updates
                                  ↑ < 100ms total ↑
```

**Example**:
```typescript
function MemoryDashboard() {
  const { memories, loading } = useMemories()
  // ✅ Live updates when agent saves memories
  // ✅ Automatic loading states
  // ✅ Cleanup on unmount
}
```

**Dependencies**: Node 5 (Supabase clients), Node 4 (types)
**Depended on by**: React UI components

---

### **Node 7: LangGraph Integration**
**Location**: `agents/`

**Why it exists**: The **AI agent layer** - enables agents to save/retrieve memories with semantic understanding.

**What it provides**:
- `MemoryManager`: Encapsulated database operations
- **4-node pattern**: retrieve → generate → decide → save
- Automatic embedding generation
- Example LangGraph workflow

**Key decisions**:
- **MemoryManager class**: Reusable, testable, clean API
- **Decide node**: LLM filters out noise
- **Automatic embeddings**: Agent developer doesn't handle vectors

**Agent flow**:
```python
# Day 1
User: "I love pizza"
→ retrieve: (no memories)
→ generate: "I'll remember that!"
→ decide: "SAVE: User loves pizza"
→ save: ✓ Database + UI update

# Day 7
User: "What should I eat?"
→ retrieve: Finds "User loves pizza" (85% match)
→ generate: "How about pizza?"
→ decide: "SKIP" (no new info)
```

**Dependencies**: Node 5 (admin client), Node 2 (search functions)
**Depended on by**: LangGraph workflows

---

## Data Flow Examples

### **Creating a Memory (Agent → UI)**
```
1. Python Agent
   memory_manager.save_memory(user_id, "User loves pizza")
   ↓
2. Generate Embedding
   OpenAI API: text → [0.1, 0.4, ...] (1536 dims)
   ↓
3. PostgreSQL INSERT
   Trigger validates, sanitizes, timestamps
   ↓
4. Realtime Broadcast
   WebSocket: "INSERT event on memories table"
   ↓
5. React Hook (useMemories)
   setMemories(current => [newMemory, ...current])
   ↓
6. UI Re-renders
   User sees new memory instantly
```

### **Searching Memories (UI → Agent)**
```
1. User types: "What food do I like?"
   ↓
2. React Component
   const { searchMemories } = useMemorySearch()
   ↓
3. Generate Embedding
   API: /api/embeddings → OpenAI → [0.2, 0.5, ...]
   ↓
4. PostgreSQL Function
   match_memories(embedding, threshold: 0.7)
   Uses HNSW index for fast search
   ↓
5. Results Returned
   [
     {content: "User loves pizza", similarity: 0.92},
     {content: "User is vegetarian", similarity: 0.85}
   ]
   ↓
6. UI Displays
   Results shown with similarity scores
```

---

## Security Model

### **Multi-Tenant Isolation (RLS)**
```sql
-- Users can ONLY see their own memories
CREATE POLICY "select_own_memories"
  ON memories FOR SELECT
  USING (auth.uid() = user_id);
```

**How it works**:
1. User logs in → Supabase sets `auth.uid()` in session
2. Query: `SELECT * FROM memories`
3. PostgreSQL rewrites: `SELECT * FROM memories WHERE user_id = auth.uid()`
4. User CANNOT see other users' data, even with SQL injection

### **Three Security Levels**

| Client | Key | RLS | Use Case |
|--------|-----|-----|----------|
| Browser | ANON | ✅ Enforced | User UI |
| Server | ANON | ✅ Enforced | Server Components |
| Admin | SERVICE | ❌ Bypassed | Agent operations |

**Critical**: Admin client has runtime check preventing browser usage.

---

## Performance Optimizations

### **Indexes**
```sql
-- User lookup: O(log n)
CREATE INDEX idx_memories_user_id ON memories(user_id);

-- Recent memories: O(log n)
CREATE INDEX idx_memories_created_at ON memories(created_at DESC);

-- Vector search: O(log n) approximate
CREATE INDEX idx_memories_embedding USING hnsw (embedding vector_cosine_ops);
```

### **Why HNSW Index?**
- **Without**: 1M memories × 1536 dimensions = 2 billion comparisons
- **With HNSW**: ~100 comparisons (99%+ accuracy)
- **Speed**: 10,000x faster at scale

---

## Technology Stack

| Layer | Technology | Why This Choice |
|-------|-----------|-----------------|
| Database | PostgreSQL + pgvector | Production-ready, vector support |
| Backend | Supabase | Auth, realtime, RLS out-of-box |
| Frontend | Next.js 14 (App Router) | Server components, type safety |
| AI Agent | LangGraph + OpenAI | Flexible workflows, best embeddings |
| Real-time | Supabase Realtime | WebSocket, automatic filtering |
| Types | TypeScript | Compile-time safety |

---

## Environment Variables

```bash
# Next.js (Browser + Server)
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# Server-only (NEVER expose to browser!)
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
OPENAI_API_KEY=sk-your-openai-key

# Python Agent
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
OPENAI_API_KEY=sk-your-openai-key
```

---

## File Structure

```
.
├── db/schema/
│   ├── 01_core_table.sql          # Node 1: Tables, RLS, indexes
│   ├── 02_search_functions.sql    # Node 2: Semantic search
│   └── 03_triggers.sql            # Node 3: Automation
│
├── types/
│   └── database.types.ts          # Node 4: TypeScript types
│
├── lib/supabase/
│   ├── browser-client.ts          # Node 5a: Browser client
│   ├── server-client.ts           # Node 5b: Server client
│   └── admin-client.ts            # Node 5c: Admin client
│
├── hooks/
│   └── use-memories.ts            # Node 6: React hooks
│
└── agents/
    ├── memory_manager.py          # Node 7a: Python manager
    └── langgraph_example.py       # Node 7b: Agent workflow
```

---

## Quick Start

### 1. Database Setup
```bash
# Apply migrations
psql -f db/schema/01_core_table.sql
psql -f db/schema/02_search_functions.sql
psql -f db/schema/03_triggers.sql

# Or use Supabase CLI
supabase db push
```

### 2. Next.js Setup
```bash
npm install
cp .env.example .env.local
# Edit .env.local with your credentials
npm run dev
```

### 3. Python Agent Setup
```bash
cd agents
pip install supabase-py langchain-openai langgraph
python langgraph_example.py
```

---

## Testing the System

### Test 1: Real-time Updates
```typescript
// Terminal 1: Start Next.js
npm run dev

// Terminal 2: Run Python agent
python agents/langgraph_example.py

// Browser: Open http://localhost:3000
// Watch UI update in real-time when agent saves memory!
```

### Test 2: Semantic Search
```python
# Save memories
memory_manager.save_memory(user_id, "I love margherita pizza")
memory_manager.save_memory(user_id, "I'm allergic to peanuts")

# Search by meaning
results = memory_manager.search_memories(
    user_id,
    "What food restrictions does the user have?"
)
# Returns: "I'm allergic to peanuts" (high similarity)
# Even though query doesn't contain "allergic" or "peanuts"!
```

---

## Production Checklist

- [ ] Enable Supabase replication for `memories` table
- [ ] Set up row-level monitoring for RLS policy hits
- [ ] Configure connection pooling (Supavisor)
- [ ] Add rate limiting to embedding generation
- [ ] Set up error tracking (Sentry)
- [ ] Monitor vector search performance (pg_stat_statements)
- [ ] Implement memory retention policy (delete old memories)
- [ ] Add backup strategy for PostgreSQL
- [ ] Test RLS policies thoroughly
- [ ] Audit admin client usage

---

## Key Insights

1. **RLS is not optional**: Multi-tenant security MUST be at database level
2. **Triggers prevent bugs**: Automatic validation > developer discipline
3. **Real-time is powerful**: Zero-latency UI updates feel magical
4. **Types everywhere**: TypeScript + Python types catch bugs early
5. **Vector search scales**: HNSW index is essential for production
6. **Separate clients**: Browser/Server/Admin have different security needs
7. **LLM decides what to save**: Prevents memory database bloat

---

## Built With

This system was built **node-by-node** to demonstrate:
- Why each component exists
- How components depend on each other
- Design decisions and trade-offs
- Real-world usage patterns

Each node is documented with inline comments explaining the "why", not just the "what".
