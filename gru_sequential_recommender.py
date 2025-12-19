"""
GRU-Based Sequential Recommender System
========================================

This script demonstrates a complete, minimal implementation of a sequential
recommender system using Keras and GRU (Gated Recurrent Unit) networks.

The model predicts the next item a user will interact with based on their
recent interaction history - similar to "customers who viewed X also viewed Y"
recommendations on retail websites.

Architecture Overview:
- Item Embedding Layer: Converts item IDs to dense vectors
- GRU Layer: Models sequential patterns in user behavior
- Dense Output Layer: Predicts probability distribution over all items

Author: ML Learning Project
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def generate_synthetic_clickstream_data(
    num_users=500,
    num_items=200,
    num_interactions=5000,
    item_categories=10
):
    """
    Generate synthetic clickstream data that mimics real user behavior.

    The data simulates realistic patterns:
    - Users tend to interact with items in certain categories (preferences)
    - Sessions have temporal locality (items viewed close in time are related)
    - Some items are more popular than others (power law distribution)

    Args:
        num_users: Number of unique users
        num_items: Number of unique items in the catalog
        num_interactions: Total number of user-item interactions
        item_categories: Number of item categories (for realistic grouping)

    Returns:
        pandas DataFrame with columns: user_id, item_id, timestamp
    """
    print(f"Generating synthetic clickstream data...")
    print(f"  Users: {num_users}, Items: {num_items}, Interactions: {num_interactions}")

    # Assign each item to a category (for realistic user preferences)
    item_to_category = {
        item_id: item_id % item_categories
        for item_id in range(num_items)
    }

    # Define user preferences (each user prefers 2-3 categories)
    user_preferences = {}
    for user_id in range(num_users):
        num_preferred_categories = random.randint(2, 3)
        preferred_cats = random.sample(range(item_categories), num_preferred_categories)
        user_preferences[user_id] = preferred_cats

    # Create item popularity weights (power law distribution)
    # More popular items get higher weights
    popularity_weights = np.array([1.0 / (i + 1) ** 0.5 for i in range(num_items)])
    popularity_weights = popularity_weights / popularity_weights.sum()

    interactions = []
    current_timestamp = 1000000  # Starting timestamp

    for _ in range(num_interactions):
        # Select a random user
        user_id = random.randint(0, num_users - 1)

        # 70% of the time, user interacts with items from their preferred categories
        # 30% of the time, user explores random items
        if random.random() < 0.7 and user_id in user_preferences:
            preferred_cats = user_preferences[user_id]
            # Get items from preferred categories
            preferred_items = [
                item_id for item_id in range(num_items)
                if item_to_category[item_id] in preferred_cats
            ]
            item_id = random.choice(preferred_items)
        else:
            # Random exploration based on popularity
            item_id = np.random.choice(num_items, p=popularity_weights)

        interactions.append({
            'user_id': user_id,
            'item_id': int(item_id),
            'timestamp': current_timestamp
        })

        # Increment timestamp (simulate time passing)
        current_timestamp += random.randint(1, 100)

    df = pd.DataFrame(interactions)
    print(f"Generated {len(df)} interactions")
    print(f"Unique users: {df['user_id'].nunique()}, Unique items: {df['item_id'].nunique()}")

    return df


def prepare_sequential_data(df, sequence_length=5, min_sequence_length=3):
    """
    Transform clickstream data into sequences for training.

    For each user, we:
    1. Sort interactions by timestamp
    2. Create sliding windows of interactions
    3. Use sequence_length-1 items as input, next item as target

    Args:
        df: DataFrame with user_id, item_id, timestamp
        sequence_length: Length of input sequences (e.g., last 5 items)
        min_sequence_length: Minimum interactions per user to include

    Returns:
        X: Input sequences (padded), shape (num_sequences, sequence_length)
        y: Target items (next item in sequence), shape (num_sequences,)
        num_items: Total number of unique items (for model architecture)
    """
    print(f"\nPreparing sequential data...")
    print(f"  Sequence length: {sequence_length}")

    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    # Group by user and get their interaction sequences
    user_sequences = df.groupby('user_id')['item_id'].apply(list).to_dict()

    X_sequences = []
    y_targets = []

    # For each user, create training examples using sliding window
    for user_id, item_sequence in user_sequences.items():
        # Skip users with too few interactions
        if len(item_sequence) < min_sequence_length:
            continue

        # Create sliding windows
        # Example: [1,2,3,4,5,6] with sequence_length=3 creates:
        #   Input: [0,0,1] -> Target: 2
        #   Input: [0,1,2] -> Target: 3
        #   Input: [1,2,3] -> Target: 4
        #   Input: [2,3,4] -> Target: 5
        #   Input: [3,4,5] -> Target: 6
        for i in range(1, len(item_sequence)):
            # Get the sequence leading up to current item
            end_idx = i
            start_idx = max(0, end_idx - sequence_length)
            input_seq = item_sequence[start_idx:end_idx]

            # Pad sequence if it's shorter than sequence_length
            if len(input_seq) < sequence_length:
                padding = [0] * (sequence_length - len(input_seq))
                input_seq = padding + input_seq

            target_item = item_sequence[i]

            X_sequences.append(input_seq)
            y_targets.append(target_item)

    X = np.array(X_sequences)
    y = np.array(y_targets)

    # Get number of unique items (add 1 for padding token at index 0)
    num_items = df['item_id'].max() + 1

    print(f"Created {len(X)} training examples")
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    print(f"Number of unique items: {num_items}")

    return X, y, num_items


class GRUSequentialRecommender(keras.Model):
    """
    GRU-based Sequential Recommender Model

    Architecture:
    1. Embedding Layer: Maps item IDs to dense vector representations
       - Each item gets a learned vector that captures its characteristics
       - Similar items will have similar embeddings

    2. GRU Layer: Models sequential dependencies
       - GRU (Gated Recurrent Unit) processes the sequence of item embeddings
       - It learns patterns like: "users who view item A then B often view item C next"
       - The GRU maintains a hidden state that captures the user's current interests

    3. Dense Output Layer: Predicts next item
       - Maps GRU output to logits over all possible items
       - Softmax converts logits to probability distribution

    Why GRU for recommendations?
    - Captures sequential patterns in user behavior
    - Can handle variable-length sequences
    - More efficient than LSTM (fewer parameters)
    - Works well for medium-length sequences (5-20 items)
    """

    def __init__(self, num_items, embedding_dim=32, gru_units=64, dropout_rate=0.2):
        """
        Initialize the GRU Sequential Recommender.

        Args:
            num_items: Total number of items in catalog
            embedding_dim: Dimension of item embedding vectors
            gru_units: Number of units in GRU layer
            dropout_rate: Dropout rate for regularization
        """
        super(GRUSequentialRecommender, self).__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Item embedding layer
        # mask_zero=True means padding tokens (0) are ignored by GRU
        self.item_embedding = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_dim,
            mask_zero=True,
            name='item_embedding'
        )

        # GRU layer for sequence modeling
        # return_sequences=False means we only use the final hidden state
        self.gru = layers.GRU(
            units=gru_units,
            return_sequences=False,
            dropout=dropout_rate,
            name='gru_layer'
        )

        # Optional: Add a dense layer before output for more capacity
        self.dense_intermediate = layers.Dense(
            units=gru_units,
            activation='relu',
            name='dense_intermediate'
        )

        # Output layer: produces logits for each item
        self.output_layer = layers.Dense(
            units=num_items,
            activation=None,  # No activation (logits)
            name='output_logits'
        )

    def call(self, inputs, training=False):
        """
        Forward pass of the model.

        Args:
            inputs: Tensor of shape (batch_size, sequence_length) containing item IDs
            training: Boolean indicating training vs inference mode

        Returns:
            Logits of shape (batch_size, num_items)
        """
        # Convert item IDs to embeddings: (batch_size, sequence_length, embedding_dim)
        x = self.item_embedding(inputs)

        # Process sequence with GRU: (batch_size, gru_units)
        x = self.gru(x, training=training)

        # Intermediate dense layer: (batch_size, gru_units)
        x = self.dense_intermediate(x)

        # Output logits: (batch_size, num_items)
        logits = self.output_layer(x)

        return logits

    def get_config(self):
        """Return model configuration for serialization."""
        return {
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim,
        }


def create_model(num_items, embedding_dim=32, gru_units=64, learning_rate=0.001):
    """
    Create and compile the GRU recommender model.

    Args:
        num_items: Total number of items
        embedding_dim: Dimension of item embeddings
        gru_units: Number of GRU units
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = GRUSequentialRecommender(
        num_items=num_items,
        embedding_dim=embedding_dim,
        gru_units=gru_units,
        dropout_rate=0.2
    )

    # Compile model
    # Loss: Sparse categorical crossentropy (targets are integer class labels)
    # Metrics: Accuracy and top-5 accuracy
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Train the recommender model.

    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Training history
    """
    print(f"\nTraining model...")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return history


def get_top_k_recommendations(model, sequence, k=10, exclude_items=None):
    """
    Generate top-K item recommendations for a given sequence.

    Args:
        model: Trained model
        sequence: Input sequence of item IDs (list or array)
        k: Number of recommendations to return
        exclude_items: Set of item IDs to exclude (e.g., already interacted items)

    Returns:
        List of (item_id, score) tuples, sorted by score descending
    """
    # Convert sequence to numpy array and add batch dimension
    sequence = np.array(sequence).reshape(1, -1)

    # Get model predictions (logits)
    logits = model.predict(sequence, verbose=0)[0]

    # Convert logits to probabilities
    probabilities = tf.nn.softmax(logits).numpy()

    # Exclude items if specified
    if exclude_items:
        for item_id in exclude_items:
            if 0 <= item_id < len(probabilities):
                probabilities[item_id] = -1

    # Get top-k items
    top_k_indices = np.argsort(probabilities)[::-1][:k]
    top_k_scores = probabilities[top_k_indices]

    recommendations = [
        (int(item_id), float(score))
        for item_id, score in zip(top_k_indices, top_k_scores)
    ]

    return recommendations


def demonstrate_recommendations(model, X_test, y_test, item_names=None, num_examples=3):
    """
    Demonstrate the model's recommendations with human-readable output.

    Args:
        model: Trained model
        X_test: Test sequences
        y_test: True next items
        item_names: Optional dict mapping item_id to item name
        num_examples: Number of examples to show
    """
    print("\n" + "="*80)
    print("RECOMMENDATION DEMONSTRATIONS")
    print("="*80)

    for i in range(min(num_examples, len(X_test))):
        sequence = X_test[i]
        true_next_item = y_test[i]

        # Remove padding (zeros) from sequence for display
        sequence_no_padding = [item for item in sequence if item > 0]

        print(f"\nExample {i+1}:")
        print(f"  User's recent history: {sequence_no_padding}")

        if item_names:
            history_names = [item_names.get(item, f"Item_{item}") for item in sequence_no_padding]
            print(f"  Item names: {history_names}")

        # Get recommendations
        recommendations = get_top_k_recommendations(
            model,
            sequence,
            k=10,
            exclude_items=set(sequence_no_padding)
        )

        print(f"\n  Top 10 Recommended Items:")
        for rank, (item_id, score) in enumerate(recommendations, 1):
            item_name = item_names.get(item_id, f"Item_{item_id}") if item_names else f"Item_{item_id}"
            marker = " ← ACTUAL NEXT ITEM" if item_id == true_next_item else ""
            print(f"    {rank:2d}. {item_name:20s} (score: {score:.4f}){marker}")

        print(f"\n  Actual next item: Item_{true_next_item}")

        # Check if we got it right in top-k
        top_k_items = [item_id for item_id, _ in recommendations[:10]]
        if true_next_item in top_k_items:
            rank = top_k_items.index(true_next_item) + 1
            print(f"  ✓ Model correctly predicted this item in top-{rank}")
        else:
            print(f"  ✗ Actual item not in top-10")

        print("-" * 80)


def create_item_names(num_items, categories=['Electronics', 'Clothing', 'Books',
                                            'Home', 'Sports', 'Toys', 'Food',
                                            'Beauty', 'Garden', 'Automotive']):
    """
    Create fictional item names for better readability.

    Args:
        num_items: Number of items to name
        categories: List of category names

    Returns:
        Dictionary mapping item_id to item_name
    """
    item_names = {}
    for item_id in range(num_items):
        category = categories[item_id % len(categories)]
        item_num = item_id // len(categories) + 1
        item_names[item_id] = f"{category}_{item_num}"

    return item_names


def main():
    """
    Main execution function: generate data, train model, demonstrate recommendations.
    """
    print("="*80)
    print("GRU-BASED SEQUENTIAL RECOMMENDER SYSTEM")
    print("="*80)

    # -------------------------------------------------------------------------
    # 1. GENERATE SYNTHETIC DATA
    # -------------------------------------------------------------------------
    df = generate_synthetic_clickstream_data(
        num_users=500,
        num_items=200,
        num_interactions=5000,
        item_categories=10
    )

    # -------------------------------------------------------------------------
    # 2. PREPARE SEQUENTIAL DATA
    # -------------------------------------------------------------------------
    sequence_length = 5  # Use last 5 items to predict next item
    X, y, num_items = prepare_sequential_data(
        df,
        sequence_length=sequence_length,
        min_sequence_length=3
    )

    # -------------------------------------------------------------------------
    # 3. TRAIN/VALIDATION/TEST SPLIT
    # -------------------------------------------------------------------------
    print("\nSplitting data into train/validation/test sets...")

    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Second split: 80% train, 20% val (from the temp set)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )

    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # -------------------------------------------------------------------------
    # 4. CREATE AND TRAIN MODEL
    # -------------------------------------------------------------------------
    model = create_model(
        num_items=num_items,
        embedding_dim=32,
        gru_units=64,
        learning_rate=0.001
    )

    # Build model by calling it once (to display summary)
    dummy_input = X_train[:1]
    _ = model(dummy_input)

    print("\nModel Architecture:")
    model.summary()

    # Train model
    history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=64
    )

    # -------------------------------------------------------------------------
    # 5. EVALUATE MODEL
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)

    test_loss, test_accuracy, test_top5_accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    print(f"\nTest Metrics:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Top-5 Accuracy: {test_top5_accuracy:.4f} ({test_top5_accuracy*100:.2f}%)")
    print("\nInterpretation:")
    print(f"  - The model correctly predicts the exact next item {test_accuracy*100:.1f}% of the time")
    print(f"  - The actual next item is in the top-5 predictions {test_top5_accuracy*100:.1f}% of the time")

    # -------------------------------------------------------------------------
    # 6. DEMONSTRATE RECOMMENDATIONS
    # -------------------------------------------------------------------------
    # Create item names for better readability
    item_names = create_item_names(num_items)

    demonstrate_recommendations(
        model,
        X_test,
        y_test,
        item_names=item_names,
        num_examples=5
    )

    # -------------------------------------------------------------------------
    # 7. ADDITIONAL DEMO: GENERATE RECOMMENDATIONS FOR SPECIFIC SEQUENCE
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("CUSTOM RECOMMENDATION DEMO")
    print("="*80)

    # Create a custom sequence (user browsed Electronics items)
    custom_sequence = [0, 0, 1, 11, 21]  # Padding + 3 Electronics items
    print(f"\nCustom sequence: {custom_sequence}")
    print(f"Item names: {[item_names.get(item, 'Padding') for item in custom_sequence]}")

    recommendations = get_top_k_recommendations(
        model,
        custom_sequence,
        k=10,
        exclude_items={1, 11, 21}
    )

    print(f"\nTop 10 Recommendations:")
    for rank, (item_id, score) in enumerate(recommendations, 1):
        item_name = item_names.get(item_id, f"Item_{item_id}")
        print(f"  {rank:2d}. {item_name:20s} (score: {score:.4f})")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Review the code and comments to understand each component")
    print("  2. Check LEARNING_PROJECTS.md for hands-on learning exercises")
    print("  3. Modify the model architecture and observe the effects")
    print("  4. Try with your own dataset!")
    print("\n")


if __name__ == "__main__":
    main()
