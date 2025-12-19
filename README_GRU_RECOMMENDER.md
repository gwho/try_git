# GRU-Based Sequential Recommender System

A complete, educational implementation of a sequential recommender system using Keras and GRU (Gated Recurrent Unit) networks.

## 🎯 Project Overview

This project demonstrates how to build a deep learning-based recommender system that predicts the next item a user will interact with based on their browsing history. It's similar to the "Customers who viewed X also viewed Y" feature found on retail websites like Amazon.

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install numpy pandas tensorflow scikit-learn

# Run the complete example
python gru_sequential_recommender.py
```

### Expected Runtime

On a standard CPU:
- Data generation: ~1 second
- Model training: ~1-2 minutes (10 epochs)
- Total runtime: ~2-3 minutes

## 📁 Project Structure

```
.
├── gru_sequential_recommender.py  # Main implementation
├── LEARNING_PROJECTS.md           # Learning exercises and mini-projects
└── README_GRU_RECOMMENDER.md      # This file
```

## 🏗️ Architecture

The model consists of three main components:

1. **Item Embedding Layer**: Converts item IDs to dense vector representations
2. **GRU Layer**: Models sequential patterns in user behavior
3. **Dense Output Layer**: Predicts probability distribution over all items

```
Input: [Item_1, Item_2, Item_3, Item_4, Item_5]
         ↓
    Item Embeddings
         ↓
      GRU Layer
         ↓
    Dense Output
         ↓
Output: Probability distribution over all items
```

## 📊 What You'll See

When you run the script, you'll see:

1. **Data Generation**: Creates 5000 synthetic user-item interactions
2. **Data Preparation**: Builds sequential training examples
3. **Model Training**: Trains for 10 epochs with validation
4. **Evaluation Metrics**:
   - Accuracy (exact next-item prediction)
   - Top-5 Accuracy (next item in top 5 predictions)
5. **Recommendation Demos**: Shows actual predictions with interpretations

### Sample Output

```
Test Metrics:
  Loss: 3.2341
  Accuracy: 0.2847 (28.47%)
  Top-5 Accuracy: 0.4923 (49.23%)

Interpretation:
  - The model correctly predicts the exact next item 28.5% of the time
  - The actual next item is in the top-5 predictions 49.2% of the time

Example Recommendation:
  User's recent history: [11, 21, 31]
  Top 10 Recommended Items:
    1. Electronics_4      (score: 0.0893)
    2. Electronics_5      (score: 0.0654)
    3. Clothing_2         (score: 0.0521) ← ACTUAL NEXT ITEM
```

## 🎓 Learning Features

### Beginner-Friendly

- **Extensive comments** explaining every major code block
- **Clear variable names** and function documentation
- **Step-by-step execution** with progress indicators
- **Human-readable output** with item names and interpretations

### Educational Value

- Demonstrates complete ML pipeline (data → model → evaluation)
- Shows best practices for sequential modeling
- Includes realistic data generation with patterns
- Provides multiple evaluation perspectives

## 📚 Next Steps

After running the basic example:

1. **Read the code** carefully, following the comments
2. **Review `LEARNING_PROJECTS.md`** for 15 hands-on mini-projects
3. **Modify parameters** and observe the effects:
   - Change sequence length (5 → 10)
   - Adjust embedding dimensions (32 → 64)
   - Increase dataset size (5000 → 10000 interactions)
4. **Experiment with architecture**:
   - Replace GRU with LSTM
   - Add more layers
   - Implement attention mechanism

## 🔬 Key Concepts Demonstrated

- **Sequential Modeling**: Using RNNs for temporal patterns
- **Embeddings**: Learning dense representations of discrete items
- **Collaborative Filtering**: Implicit feedback from interactions
- **Evaluation**: Multiple metrics for recommendation quality
- **Data Preprocessing**: Building sequences from raw interactions

## 🎯 Use Cases

This approach is applicable to:

- **E-commerce**: Product recommendations
- **Streaming**: Next video/song prediction
- **News**: Article recommendations
- **Content Platforms**: Next post/article prediction
- **Gaming**: Next action/item prediction

## 📈 Performance Expectations

With the default synthetic dataset:

- **Accuracy**: 25-35% (exact prediction)
- **Top-5 Accuracy**: 45-55%
- **Top-10 Accuracy**: 55-65%

These are good baseline metrics for a small dataset. Real-world production systems with millions of interactions typically achieve:
- Top-10 Accuracy: 70-85%
- Top-20 Accuracy: 80-90%

## 🛠️ Customization

### Using Your Own Data

Replace the data generation section with your own data loader:

```python
# Load your data
df = pd.read_csv('your_interactions.csv')
# Required columns: user_id, item_id, timestamp

# Continue with the rest of the pipeline
X, y, num_items = prepare_sequential_data(df, sequence_length=5)
```

### Adjusting Model Capacity

```python
model = create_model(
    num_items=num_items,
    embedding_dim=64,      # Increase for larger datasets
    gru_units=128,         # Increase for more capacity
    learning_rate=0.001
)
```

### Changing Sequence Length

```python
sequence_length = 10  # Use last 10 items instead of 5
X, y, num_items = prepare_sequential_data(
    df,
    sequence_length=sequence_length
)
```

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'tensorflow'**
```bash
pip install tensorflow
```

**Low accuracy (<20%)**
- Check if dataset is too small
- Verify items have meaningful co-occurrence patterns
- Try increasing embedding dimensions or GRU units

**Training is slow**
- Reduce batch size
- Use GPU if available: `export CUDA_VISIBLE_DEVICES=0`
- Decrease dataset size for faster experimentation

**Model predicts same items repeatedly**
- May indicate overfitting to popular items
- Try adding dropout or regularization
- Ensure dataset has diverse user behaviors

## 📖 Additional Resources

### Papers
- **GRU4Rec** (Hidasi et al., 2015): Session-based Recommendations with RNNs
- **SASRec** (Kang & McAuley, 2018): Self-Attentive Sequential Recommendation

### Libraries
- **TensorFlow Recommenders**: Official TF library for recommendation systems
- **RecSys**: PyTorch-based recommendation library

### Courses
- Stanford CS246: Mining Massive Datasets
- Coursera: Recommender Systems Specialization

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different architectures
- Add new features (user features, item features, etc.)
- Implement additional evaluation metrics
- Create visualization tools

## 📝 License

This is a learning project - use it freely for educational purposes!

## 🎓 Learning Goals Achieved

After completing this project, you will understand:

✅ How sequential recommender systems work
✅ Why GRUs are effective for sequence modeling
✅ How embeddings capture item relationships
✅ Best practices for training and evaluating recommender systems
✅ How to implement a complete ML pipeline in Keras
✅ Techniques for handling sequential data

---

**Ready to start learning?** Run the script and then dive into `LEARNING_PROJECTS.md` for hands-on exercises!

```bash
python gru_sequential_recommender.py
```
