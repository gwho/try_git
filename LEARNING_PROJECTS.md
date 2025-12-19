# Learning Projects: GRU Sequential Recommender System

## 📚 High-Level Architecture Explanation

### What Does This Recommender System Do?

This system predicts **what item a user will interact with next** based on their recent browsing/purchase history. Think of it like Amazon's "Customers who viewed this item also viewed..." feature, but powered by deep learning.

### The Three Main Components

#### 1. **Item Embedding Layer**
**What it does:** Converts item IDs (just numbers like 42, 137) into dense vector representations.

**Why it matters:**
- Raw item IDs have no inherent meaning - item 42 isn't "closer" to item 43 than to item 100
- Embeddings learn meaningful relationships: similar items get similar vectors
- Example: "Winter Coat" and "Snow Boots" might have similar embeddings because users often view them together

**Analogy:** Like converting words into vectors in natural language processing, where "king" and "queen" have similar representations because they're semantically related.

#### 2. **GRU (Gated Recurrent Unit) Layer**
**What it does:** Processes the sequence of item embeddings and learns temporal patterns in user behavior.

**Why it matters:**
- Captures patterns like: "Users who view A → B usually view C next"
- Maintains a "hidden state" that represents the user's evolving interests as they browse
- Handles sequences of different lengths (3 items vs 10 items)

**How it works:**
- The GRU reads items one by one, left to right
- At each step, it updates its internal "memory" based on the current item and previous memory
- The final memory state captures the essence of the entire browsing session
- **Gates** (reset and update gates) help it decide what to remember and what to forget

**Why GRU instead of a simple average?**
- Order matters! Viewing [phone case → phone → charger] suggests different intent than [phone → phone case → charger]
- Recent items matter more than older items
- GRU learns these patterns automatically from data

**GRU vs LSTM:**
- GRU is simpler (fewer parameters) → trains faster
- LSTM has more gates → can capture more complex patterns
- For most recommendation tasks, GRU works great and is more efficient

#### 3. **Dense Output Layer**
**What it does:** Takes the GRU's final hidden state and produces a probability distribution over all items.

**Why it matters:**
- Converts the abstract hidden state into concrete predictions
- Outputs one score for each item in the catalog
- Softmax converts scores into probabilities (sum to 1)

**How we use it:**
- Sort items by probability (highest first)
- Return top-K items as recommendations
- Can exclude items the user already viewed

### The Complete Flow

```
User's History: [Item_1, Item_2, Item_3, Item_4, Item_5]
                     ↓
        [Embedding_1, Embedding_2, ..., Embedding_5]
                     ↓
                   GRU Layer
    (processes sequence, maintains hidden state)
                     ↓
              Final Hidden State
            (captures user intent)
                     ↓
              Dense Output Layer
                     ↓
      [Prob(Item_1), Prob(Item_2), ..., Prob(Item_N)]
                     ↓
         Top-10 Highest Probability Items
                     ↓
            RECOMMENDATIONS!
```

### Training Process

**Input:** Sequence of items [A, B, C, D]
**Target:** Next item E

The model learns to predict E given [A, B, C, D]. Over thousands of examples, it learns:
- Which items are frequently viewed together
- What sequences lead to specific items
- How to use order and context to make better predictions

**Loss Function:** Sparse Categorical Cross-Entropy
- Measures how different the model's predictions are from the actual next item
- Lower loss = better predictions

---

## 🔨 15 Mini-Projects to Deepen Your Understanding

These projects are designed to be completed in order of increasing difficulty, but feel free to jump around based on your interests!

### Difficulty: ⭐ Beginner

---

### 1. **Sequence Length Experimentation**

**What to change:**
Modify the `sequence_length` parameter in `prepare_sequential_data()` to different values (3, 5, 10, 15, 20).

**What you'll learn:**
- How sequence length affects model accuracy (does more context always help?)
- Trade-off between context and data sparsity (longer sequences = fewer training examples)
- How training time changes with sequence length

**Expected observations:**
- Very short sequences (2-3): Fast training but may miss important context
- Moderate sequences (5-10): Usually optimal balance
- Very long sequences (20+): Slower training, may overfit on sparse data

**Bonus:** Plot accuracy vs sequence length to find the optimal value for this dataset.

---

### 2. **Replace GRU with LSTM**

**What to change:**
Replace `layers.GRU(...)` with `layers.LSTM(...)` in the model architecture.

**What you'll learn:**
- Practical differences between GRU and LSTM
- How model complexity affects training time
- Whether additional LSTM gates improve accuracy for this task

**Expected observations:**
- LSTM has more parameters (check with `model.summary()`)
- Training time per epoch increases
- Accuracy may improve slightly or stay similar

**Investigation questions:**
- Which converges faster?
- Which achieves better validation accuracy?
- Is the accuracy improvement (if any) worth the computational cost?

---

### 3. **Add a Simple Popularity Baseline**

**What to change:**
Before training the GRU model, implement a simple baseline that always recommends the most popular items (items that appear most frequently in the training data).

**What you'll learn:**
- How to properly evaluate recommender systems
- Why we need baselines (is our fancy model actually better?)
- How much sequential modeling helps vs just recommending popular items

**Implementation hint:**
```python
def get_popularity_baseline(df, k=10):
    """Returns top-k most popular items."""
    item_counts = df['item_id'].value_counts()
    return item_counts.head(k).index.tolist()
```

**Expected observations:**
- Popularity baseline gets surprisingly decent accuracy (20-30%)
- GRU should significantly outperform it
- Some users are well-served by popular items, others need personalization

---

### Difficulty: ⭐⭐ Intermediate

---

### 4. **Experiment with Embedding Dimensions**

**What to change:**
Try different `embedding_dim` values: 8, 16, 32, 64, 128.

**What you'll learn:**
- How embedding size affects model capacity
- Trade-off between expressiveness and overfitting
- Impact on training speed and memory usage

**Investigation:**
- Create a table:
  - Embedding Dim | Train Time | Train Accuracy | Val Accuracy | # Parameters
- Find the sweet spot for this dataset

**Expected observations:**
- Too small (8): Model may underfit, can't capture item relationships
- Too large (128): Slower training, may overfit on small dataset
- Just right (32-64): Best validation accuracy

---

### 5. **Implement Top-K Accuracy Logging**

**What to change:**
Already implemented in the model! But enhance it by logging top-1, top-3, top-5, and top-10 accuracy each epoch. Create plots showing how these metrics evolve during training.

**What you'll learn:**
- Difference between exact prediction and "good enough" recommendation
- How to evaluate ranking quality
- How different metrics tell different stories about model performance

**Implementation hint:**
```python
for k in [1, 3, 5, 10]:
    model.compile(
        metrics=[keras.metrics.SparseTopKCategoricalAccuracy(k=k, name=f'top{k}_accuracy')]
    )
```

**Expected observations:**
- Top-1 accuracy: 10-30% (exact prediction is hard!)
- Top-5 accuracy: 30-50% (much better, useful for showing 5 recommendations)
- Top-10 accuracy: 40-60% (very practical for real systems)

---

### 6. **Add Dropout and Layer Normalization**

**What to change:**
- Increase dropout rate in GRU layer (try 0.1, 0.3, 0.5)
- Add `layers.LayerNormalization()` after the GRU layer
- Add dropout to the dense layers

**What you'll learn:**
- Regularization techniques to prevent overfitting
- How to identify if your model is overfitting (train acc >> val acc)
- When dropout helps vs hurts

**Implementation:**
```python
self.gru = layers.GRU(units=gru_units, dropout=0.3, recurrent_dropout=0.2)
self.layer_norm = layers.LayerNormalization()
self.dropout = layers.Dropout(0.3)
```

**Expected observations:**
- Higher dropout: Slower learning, better generalization
- Layer normalization: Faster convergence, more stable training
- Watch the gap between train and validation accuracy

---

### 7. **Add User Embeddings**

**What to change:**
Extend the model to include user embeddings. Concatenate user embedding with the GRU output before the final dense layer.

**What you'll learn:**
- How to incorporate user-level personalization
- Multi-input models in Keras
- Whether user identity helps beyond behavioral sequences

**Implementation hint:**
```python
# In model __init__:
self.user_embedding = layers.Embedding(num_users, user_embedding_dim)

# In call():
user_embed = self.user_embedding(user_ids)
x = layers.Concatenate()([gru_output, user_embed])
```

**Expected observations:**
- Model now needs both sequence AND user_id as input
- May improve accuracy, especially for users with consistent preferences
- Helps with cold-start items (but not cold-start users!)

---

### Difficulty: ⭐⭐⭐ Advanced

---

### 8. **Implement Bidirectional GRU**

**What to change:**
Wrap the GRU layer with `layers.Bidirectional()` to process sequences in both forward and backward directions.

**What you'll learn:**
- When bidirectional processing helps (it's overkill for next-item prediction!)
- How to think about causality in sequence models
- Why this might hurt performance for next-item prediction

**Implementation:**
```python
self.gru = layers.Bidirectional(
    layers.GRU(units=gru_units, return_sequences=False)
)
```

**Expected observations:**
- Model has 2x parameters (forward + backward)
- Training is slower
- Accuracy may actually DECREASE (why? because future items leak information!)
- Great learning moment: not all "advanced" techniques help every task

---

### 9. **Implement Sampled Softmax Loss**

**What to change:**
Replace the full softmax over all items with sampled softmax (only compute loss over a sample of negative items + the true positive).

**What you'll learn:**
- How to scale to millions of items (full softmax becomes impractical)
- Negative sampling strategies
- Trade-off between exact computation and approximation

**Implementation hint:**
```python
# Use tf.nn.sampled_softmax_loss in a custom training loop
# Or use keras-nlp's sampled softmax layer
```

**Why it matters:**
- With 200 items, full softmax is fine
- With 1M items, full softmax is too slow
- Sampled softmax approximates the loss using only ~100 items

**Expected observations:**
- Training is much faster with large item catalogs
- Final accuracy is similar to full softmax
- Essential technique for production recommendation systems

---

### 10. **Add Stacked GRU Layers**

**What to change:**
Add 2-3 GRU layers on top of each other. Set `return_sequences=True` for all but the last layer.

**What you'll learn:**
- How depth affects model capacity
- When multiple layers help vs just adding more parameters to one layer
- Hierarchical feature learning (lower layers = simple patterns, upper layers = complex patterns)

**Implementation:**
```python
self.gru1 = layers.GRU(units=64, return_sequences=True, dropout=0.2)
self.gru2 = layers.GRU(units=64, return_sequences=True, dropout=0.2)
self.gru3 = layers.GRU(units=64, return_sequences=False, dropout=0.2)
```

**Expected observations:**
- Model is much deeper
- Training is slower
- May improve accuracy on complex patterns
- Diminishing returns after 2-3 layers

---

### 11. **Implement Attention Mechanism**

**What to change:**
Add an attention layer that lets the model focus on specific items in the sequence when making predictions.

**What you'll learn:**
- How attention improves sequence modeling
- Why attention is the foundation of Transformers
- How to visualize what the model is "paying attention to"

**Implementation hint:**
```python
# Use layers.MultiHeadAttention or layers.Attention
self.attention = layers.MultiHeadAttention(num_heads=2, key_dim=32)

# In call():
x_gru = self.gru(x, return_sequences=True)  # Keep all hidden states
x_attention = self.attention(x_gru, x_gru)  # Self-attention
x_final = x_attention[:, -1, :]  # Take last timestep
```

**Expected observations:**
- Model learns to focus on most relevant items in history
- Can visualize attention weights to understand predictions
- Modern recommender systems (SASRec, BERT4Rec) use attention extensively

---

### 12. **Cold-Start Handling**

**What to change:**
Implement a function to generate recommendations for users with only 1-2 interactions (cold-start problem).

**What you'll learn:**
- Real-world challenges in recommender systems
- Fallback strategies when you have insufficient data
- Hybrid approaches (content + collaborative filtering)

**Implementation ideas:**
```python
def recommend_cold_start(model, short_sequence, popular_items, k=10):
    """Blend model predictions with popular items for cold-start users."""
    if len(short_sequence) < 3:
        # Mostly use popular items with a small model component
        model_recs = get_top_k_recommendations(model, short_sequence, k=k//2)
        return model_recs + popular_items[:k//2]
    else:
        return get_top_k_recommendations(model, short_sequence, k=k)
```

**Expected observations:**
- Very short sequences lead to generic predictions
- Hybrid approach improves coverage
- This is a major challenge in production systems

---

### 13. **Add Temporal Features**

**What to change:**
Incorporate time information (e.g., time since last interaction, hour of day, day of week) into the model.

**What you'll learn:**
- How temporal patterns affect user behavior
- Feature engineering for deep learning
- Multi-modal input in recommender systems

**Implementation hint:**
```python
# Add time_delta as an additional feature
time_deltas = compute_time_since_last_interaction(df)
time_embedding = layers.Dense(16)(time_deltas)
combined = layers.Concatenate()([gru_output, time_embedding])
```

**Expected observations:**
- Users behave differently at different times (lunch browsing vs evening shopping)
- Recency matters (recent items more relevant than old ones)
- Can capture session boundaries

---

### 14. **Implement Negative Sampling During Training**

**What to change:**
Instead of computing loss over all items, explicitly sample negative items (items the user didn't interact with) and train the model to rank positives higher than negatives.

**What you'll learn:**
- Pairwise and listwise ranking losses
- How to create informative negative samples
- Differences between classification and ranking approaches

**Implementation:**
```python
# Sample hard negatives (popular items user didn't interact with)
# Use triplet loss or BPR (Bayesian Personalized Ranking) loss
def triplet_loss(anchor, positive, negative, margin=1.0):
    return max(0, margin + distance(anchor, negative) - distance(anchor, positive))
```

**Why it matters:**
- More efficient than softmax for large item catalogs
- Can focus on difficult negatives (hard negative mining)
- Used in two-tower models and retrieval systems

---

### 15. **Model Interpretation and Analysis**

**What to change:**
Add visualization and analysis tools to understand what the model learned.

**What you'll learn:**
- How to debug and interpret deep learning models
- What patterns the embeddings capture
- Model behavior on different user segments

**Implementation ideas:**
```python
# 1. Visualize item embeddings with t-SNE/UMAP
from sklearn.manifold import TSNE
embeddings = model.item_embedding.get_weights()[0]
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings)
# Plot and color by item category

# 2. Find similar items using embedding cosine similarity
def find_similar_items(item_id, embeddings, k=10):
    item_vec = embeddings[item_id]
    similarities = cosine_similarity([item_vec], embeddings)[0]
    return np.argsort(similarities)[::-1][1:k+1]

# 3. Analyze model performance by user segment
def analyze_by_segment(model, X_test, y_test):
    # Active users vs passive users
    # Different item categories
    # Different sequence lengths
    pass
```

**Expected insights:**
- Items in same category cluster together in embedding space
- Model performs better for active users
- Some item categories are harder to predict
- Embeddings capture meaningful semantic relationships

---

## 🚀 Additional Challenge Projects

If you've completed the above and want more:

### 16. **Session-Based Recommendations**
Split user interactions into sessions (e.g., 30-minute gaps). Train the model to predict next item within a session. This better captures short-term intent.

### 17. **Multi-Task Learning**
Predict both next item AND next category simultaneously. Use shared GRU encoder with two output heads.

### 18. **Incorporate Item Features**
Add item metadata (category, price, brand) using additional embeddings or concatenated features.

### 19. **Deploy as a REST API**
Use FastAPI or Flask to create an API endpoint that serves real-time recommendations.

### 20. **A/B Testing Simulation**
Simulate A/B testing by training two models with different architectures and comparing their offline metrics.

---

## 📊 How to Track Your Learning

For each project, keep a learning journal with:

1. **Hypothesis**: What do you expect to happen?
2. **Implementation**: What did you change?
3. **Results**: What actually happened? (Include metrics!)
4. **Analysis**: Why did you see these results?
5. **Takeaways**: What did you learn?

Example entry:
```
Project: #4 - Embedding Dimensions
Hypothesis: Larger embeddings will improve accuracy
Implementation: Tested dims [8, 16, 32, 64, 128]
Results:
  - 8: Val acc 0.23
  - 16: Val acc 0.28
  - 32: Val acc 0.31
  - 64: Val acc 0.32
  - 128: Val acc 0.30 (overfitting!)
Analysis: 64 is optimal for this dataset size. 128 starts overfitting because
we don't have enough data to learn meaningful 128-d representations.
Takeaways: Model capacity must match dataset size. More parameters ≠ better.
```

---

## 🎯 Learning Path Recommendation

**Week 1: Foundations**
- Projects 1, 2, 3 (understand baseline and architecture)

**Week 2: Regularization and Features**
- Projects 4, 5, 6, 7 (improve the model)

**Week 3: Advanced Techniques**
- Projects 8, 9, 10, 11 (cutting-edge methods)

**Week 4: Real-World Challenges**
- Projects 12, 13, 14, 15 (production considerations)

---

## 📚 Additional Resources

**Papers to Read:**
1. **GRU4Rec** (2015): Session-based recommendations with RNNs
2. **SASRec** (2018): Self-attention for sequential recommendation
3. **BERT4Rec** (2019): Bidirectional encoder for recommendations

**Key Concepts to Study:**
- Collaborative filtering vs content-based filtering
- Implicit vs explicit feedback
- Cold-start problem
- Diversity vs accuracy trade-off
- Online learning and real-time updates

**Tools to Explore:**
- TensorFlow Recommenders (TFRS)
- PyTorch with RecSys libraries
- Production serving: TensorFlow Serving, TorchServe
- Experiment tracking: Weights & Biases, MLflow

---

## ✅ Success Criteria

You'll know you've mastered this material when you can:

1. ✅ Explain why we use embeddings and sequential models for recommendations
2. ✅ Implement a recommender from scratch without looking at reference code
3. ✅ Debug common issues (e.g., why is accuracy low? why is it overfitting?)
4. ✅ Make informed architecture decisions based on dataset characteristics
5. ✅ Understand the trade-offs between different approaches
6. ✅ Read and implement ideas from recent research papers
7. ✅ Deploy a working recommender system to production

---

## 💡 Final Tips

- **Don't rush**: Understanding is more valuable than completing all projects quickly
- **Experiment freely**: Break things! That's how you learn
- **Compare everything**: Always measure before/after when making changes
- **Visualize results**: Plots and charts reveal patterns metrics alone miss
- **Read the errors**: Error messages are your friends
- **Ask "why"**: Don't just observe results, understand the mechanisms

**Most importantly**: Have fun! Recommender systems are fascinating, widely used, and deeply rewarding to master.

Happy learning! 🎓
