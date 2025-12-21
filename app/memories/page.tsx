/**
 * Memories Dashboard Page
 * Demonstrates real-time memory subscriptions and vector search
 */

'use client';

import { useState } from 'react';
import { useMemories, useMemoryActions, useMemorySearch, useMemoryCount } from '@/hooks/useMemories';
import type { Memory } from '@/lib/types/memory';

export default function MemoriesPage() {
  const { memories, loading, error } = useMemories();
  const { createMemory, deleteMemory } = useMemoryActions();
  const { results, searching, searchMemories, clearResults } = useMemorySearch();
  const { count } = useMemoryCount();

  const [newMemory, setNewMemory] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  // Handle creating a new memory
  const handleCreateMemory = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMemory.trim()) return;

    setIsCreating(true);
    try {
      // In a real app, you'd generate embeddings here
      // For now, we'll create without embeddings
      await createMemory(newMemory, undefined, {
        source: 'web_ui',
        created_via: 'manual_entry',
      });
      setNewMemory('');
    } catch (err) {
      console.error('Error creating memory:', err);
      alert('Failed to create memory');
    } finally {
      setIsCreating(false);
    }
  };

  // Handle deleting a memory
  const handleDeleteMemory = async (id: string) => {
    if (!confirm('Are you sure you want to delete this memory?')) return;

    try {
      await deleteMemory(id);
    } catch (err) {
      console.error('Error deleting memory:', err);
      alert('Failed to delete memory');
    }
  };

  // Handle searching memories
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      clearResults();
      return;
    }

    // Note: In a real app, you'd generate embeddings for the search query
    // This is a placeholder - you'll need to implement embedding generation
    console.log('Search functionality requires embedding generation');
    alert('To use search, implement embedding generation in your API route');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading memories...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center text-red-600">
          <p className="text-xl font-semibold">Error loading memories</p>
          <p className="mt-2">{error.message}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Memory Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Real-time long-term memory powered by LangGraph and Supabase
          </p>
          <div className="mt-4 flex items-center gap-4">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
              {count} Total Memories
            </span>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
              ✓ Real-time Updates
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Create Memory Form */}
            <div className="bg-white shadow rounded-lg p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Create New Memory
              </h2>
              <form onSubmit={handleCreateMemory}>
                <textarea
                  value={newMemory}
                  onChange={(e) => setNewMemory(e.target.value)}
                  placeholder="Enter a memory to save..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isCreating}
                />
                <button
                  type="submit"
                  disabled={isCreating || !newMemory.trim()}
                  className="mt-3 w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {isCreating ? 'Saving...' : 'Save Memory'}
                </button>
              </form>
            </div>

            {/* Memories List */}
            <div className="bg-white shadow rounded-lg p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Your Memories ({memories.length})
              </h2>
              {memories.length === 0 ? (
                <p className="text-gray-500 text-center py-8">
                  No memories yet. Create your first memory above!
                </p>
              ) : (
                <div className="space-y-4">
                  {memories.map((memory) => (
                    <MemoryCard
                      key={memory.id}
                      memory={memory}
                      onDelete={handleDeleteMemory}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Search */}
            <div className="bg-white shadow rounded-lg p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Search Memories
              </h2>
              <form onSubmit={handleSearch}>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search by meaning..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  type="submit"
                  disabled={searching}
                  className="mt-3 w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-400 transition-colors"
                >
                  {searching ? 'Searching...' : 'Search'}
                </button>
              </form>

              {/* Search Results */}
              {results.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium text-gray-900 mb-2">
                    Results:
                  </h3>
                  <div className="space-y-2">
                    {results.map((result) => (
                      <div
                        key={result.id}
                        className="p-3 bg-gray-50 rounded-md text-sm"
                      >
                        <p className="text-gray-900">{result.content}</p>
                        <p className="mt-1 text-xs text-gray-500">
                          Similarity: {(result.similarity * 100).toFixed(1)}%
                        </p>
                      </div>
                    ))}
                  </div>
                  <button
                    onClick={clearResults}
                    className="mt-3 text-sm text-blue-600 hover:text-blue-700"
                  >
                    Clear results
                  </button>
                </div>
              )}
            </div>

            {/* Info Box */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
              <h3 className="text-sm font-semibold text-blue-900 mb-2">
                💡 How it works
              </h3>
              <ul className="text-sm text-blue-800 space-y-2">
                <li>• Memories are stored with vector embeddings</li>
                <li>• Real-time updates via Supabase subscriptions</li>
                <li>• Semantic search using cosine similarity</li>
                <li>• LangGraph agent integration ready</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Memory Card Component
function MemoryCard({
  memory,
  onDelete,
}: {
  memory: Memory;
  onDelete: (id: string) => void;
}) {
  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <p className="text-gray-900">{memory.content}</p>
          <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
            <span>
              {new Date(memory.created_at).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
            {memory.metadata?.source && (
              <span className="px-2 py-1 bg-gray-100 rounded">
                {memory.metadata.source}
              </span>
            )}
          </div>
        </div>
        <button
          onClick={() => onDelete(memory.id)}
          className="ml-4 text-red-600 hover:text-red-700 text-sm"
          title="Delete memory"
        >
          Delete
        </button>
      </div>
    </div>
  );
}
