"""
Two-Tower B2B Recommendation Model with Mixed Data Types
Supports numeric, categorical (low/high cardinality), and text features
Fixed: Uses print() instead of logging for console output
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from collections import defaultdict
from typing import Dict, Tuple, List, Any
import random
import string
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA GENERATION
# ============================================================================

def create_sample_b2b_dataset(
    n_users: int = 500,
    n_items: int = 300,
    n_interactions: int = 2000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Create a comprehensive B2B dataset with numeric, categorical, and text features.
    
    Returns:
        Dict containing:
        - users_df: User data with features
        - items_df: Item data with features
        - interactions: User-item interaction pairs
        - feature_config: Metadata about feature types and cardinalities
        - data_stats: Statistics for normalization
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # ========== USER FEATURES ==========
    user_ids = np.arange(n_users)
    
    # Numeric features for users
    user_budget = np.random.uniform(10000, 1000000, n_users)  # Budget in dollars
    user_company_age = np.random.uniform(1, 50, n_users)  # Years
    user_employee_count = np.random.exponential(100, n_users) + 10  # Heavily skewed
    
    # Low cardinality categorical (one-hot: <10 categories)
    user_industry = np.random.choice(
        ['Tech', 'Finance', 'Retail', 'Healthcare', 'Manufacturing'],
        n_users
    )
    user_region = np.random.choice(
        ['North America', 'Europe', 'Asia', 'Latin America'],
        n_users
    )
    
    # High cardinality categorical (embedding: 50+ categories)
    user_company_types = np.random.choice(
        ['Type_' + str(i) for i in range(80)],  # 80 unique company types
        n_users
    )
    user_account_types = np.random.choice(
        ['Account_' + str(i) for i in range(60)],  # 60 unique account types
        n_users
    )
    
    # Text features
    user_descriptions = [
        generate_company_description() for _ in range(n_users)
    ]
    
    users_df = pd.DataFrame({
        'user_id': user_ids,
        'budget_numeric': user_budget,
        'company_age_numeric': user_company_age,
        'employee_count_numeric': user_employee_count,
        'industry_categorical_low': user_industry,
        'region_categorical_low': user_region,
        'company_type_categorical_high': user_company_types,
        'account_type_categorical_high': user_account_types,
        'description_text': user_descriptions,
    })
    
    # ========== ITEM FEATURES ==========
    item_ids = np.arange(n_items)
    
    # Numeric features for items
    item_price = np.random.uniform(1000, 500000, n_items)
    item_rating = np.random.uniform(1.0, 5.0, n_items)
    item_monthly_cost = np.random.uniform(100, 50000, n_items)
    
    # Low cardinality categorical (one-hot)
    item_category = np.random.choice(
        ['SaaS', 'Consulting', 'Hardware', 'Services', 'Data'],
        n_items
    )
    item_deployment = np.random.choice(
        ['Cloud', 'On-Premise', 'Hybrid'],
        n_items
    )
    
    # High cardinality categorical (embedding)
    item_vendor = np.random.choice(
        ['Vendor_' + str(i) for i in range(100)],  # 100 unique vendors
        n_items
    )
    item_solution_id = np.random.choice(
        ['Solution_' + str(i) for i in range(150)],  # 150 unique solutions
        n_items
    )
    
    # Text features
    item_descriptions = [
        generate_product_description() for _ in range(n_items)
    ]
    
    items_df = pd.DataFrame({
        'item_id': item_ids,
        'price_numeric': item_price,
        'rating_numeric': item_rating,
        'monthly_cost_numeric': item_monthly_cost,
        'category_categorical_low': item_category,
        'deployment_categorical_low': item_deployment,
        'vendor_categorical_high': item_vendor,
        'solution_categorical_high': item_solution_id,
        'description_text': item_descriptions,
    })
    
    # ========== INTERACTIONS ==========
    interactions = []
    for _ in range(n_interactions):
        user_id = np.random.randint(0, n_users)
        item_id = np.random.randint(0, n_items)
        label = np.random.randint(0, 2)  # 0 or 1 (negative/positive)
        interactions.append({'user_id': user_id, 'item_id': item_id, 'label': label})
    
    interactions_df = pd.DataFrame(interactions)
    
    # ========== FEATURE CONFIGURATION ==========
    feature_config = {
        'user_features': {
            'numeric': ['budget_numeric', 'company_age_numeric', 'employee_count_numeric'],
            'categorical_low_cardinality': {
                'industry_categorical_low': list(users_df['industry_categorical_low'].unique()),
                'region_categorical_low': list(users_df['region_categorical_low'].unique()),
            },
            'categorical_high_cardinality': {
                'company_type_categorical_high': list(users_df['company_type_categorical_high'].unique()),
                'account_type_categorical_high': list(users_df['account_type_categorical_high'].unique()),
            },
            'text': ['description_text'],
        },
        'item_features': {
            'numeric': ['price_numeric', 'rating_numeric', 'monthly_cost_numeric'],
            'categorical_low_cardinality': {
                'category_categorical_low': list(items_df['category_categorical_low'].unique()),
                'deployment_categorical_low': list(items_df['deployment_categorical_low'].unique()),
            },
            'categorical_high_cardinality': {
                'vendor_categorical_high': list(items_df['vendor_categorical_high'].unique()),
                'solution_categorical_high': list(items_df['solution_categorical_high'].unique()),
            },
            'text': ['description_text'],
        },
    }
    
    # ========== DATA STATISTICS FOR NORMALIZATION ==========
    data_stats = {
        'user_numeric_stats': {},
        'item_numeric_stats': {},
        'vocabulary': None,  # Will be built during preprocessing
    }
    
    # Compute normalization stats for numeric features
    for col in feature_config['user_features']['numeric']:
        data_stats['user_numeric_stats'][col] = {
            'mean': users_df[col].mean(),
            'std': users_df[col].std() + 1e-8,  # Avoid division by zero
            'min': users_df[col].min(),
            'max': users_df[col].max(),
        }
    
    for col in feature_config['item_features']['numeric']:
        data_stats['item_numeric_stats'][col] = {
            'mean': items_df[col].mean(),
            'std': items_df[col].std() + 1e-8,
            'min': items_df[col].min(),
            'max': items_df[col].max(),
        }
    
    return {
        'users_df': users_df,
        'items_df': items_df,
        'interactions': interactions_df,
        'feature_config': feature_config,
        'data_stats': data_stats,
    }


def generate_company_description() -> str:
    """Generate a sample company description text."""
    industries = ['tech', 'finance', 'retail', 'healthcare', 'manufacturing']
    sizes = ['startup', 'scale-up', 'enterprise', 'mid-market', 'fortune500']
    focuses = ['innovation', 'efficiency', 'quality', 'customer-centric', 'data-driven']
    
    industry = random.choice(industries)
    size = random.choice(sizes)
    focus = random.choice(focuses)
    
    return f"A {size} {industry} company focused on {focus} solutions. We operate globally with strong market presence."


def generate_product_description() -> str:
    """Generate a sample product description text."""
    features = ['secure', 'scalable', 'enterprise-grade', 'cloud-native', 'ai-powered']
    benefits = ['reduces costs', 'improves efficiency', 'enhances security', 'accelerates growth']
    
    feature1, feature2 = random.sample(features, 2)
    benefit = random.choice(benefits)
    
    return f"{feature1.capitalize()} and {feature2} platform that {benefit} for your organization."


# ============================================================================
# DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================

class FeatureProcessor:
    """Handles preprocessing of numeric, categorical, and text features."""
    
    def __init__(self, feature_config: Dict, data_stats: Dict, vocab_size: int = 5000):
        self.feature_config = feature_config
        self.data_stats = data_stats
        self.vocab_size = vocab_size
        self.text_vectorizer = None
        self.categorical_mappings = {}
        self._build_categorical_mappings()
    
    def _build_categorical_mappings(self):
        """Create ID mappings for categorical features."""
        # Low cardinality: will use one-hot
        for feature_name, values in self.feature_config['user_features']['categorical_low_cardinality'].items():
            self.categorical_mappings[feature_name] = {val: idx for idx, val in enumerate(values)}
        
        for feature_name, values in self.feature_config['item_features']['categorical_low_cardinality'].items():
            self.categorical_mappings[feature_name] = {val: idx for idx, val in enumerate(values)}
        
        # High cardinality: create ID mappings (embedding)
        for feature_name, values in self.feature_config['user_features']['categorical_high_cardinality'].items():
            self.categorical_mappings[feature_name] = {val: idx for idx, val in enumerate(values)}
        
        for feature_name, values in self.feature_config['item_features']['categorical_high_cardinality'].items():
            self.categorical_mappings[feature_name] = {val: idx for idx, val in enumerate(values)}
    
    def build_text_vectorizer(self, all_texts: List[str]):
        """Build TextVectorization layer from text corpus."""
        self.text_vectorizer = layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode='int',
            output_sequence_length=50,  # Pad/truncate to 50 tokens
        )
        self.text_vectorizer.adapt(all_texts)
    
    def normalize_numeric(self, value: float, feature_name: str, tower_type: str) -> float:
        """Normalize numeric feature using z-score normalization."""
        stats_dict = (self.data_stats['user_numeric_stats'] if tower_type == 'user' 
                     else self.data_stats['item_numeric_stats'])
        if feature_name not in stats_dict:
            return value
        
        stats = stats_dict[feature_name]
        return (value - stats['mean']) / stats['std']
    
    def encode_categorical_low(self, value: str, feature_name: str) -> int:
        """Get one-hot index for low cardinality categorical."""
        return self.categorical_mappings.get(feature_name, {}).get(value, 0)
    
    def encode_categorical_high(self, value: str, feature_name: str) -> int:
        """Get embedding ID for high cardinality categorical."""
        return self.categorical_mappings.get(feature_name, {}).get(value, 0)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Tokenize and vectorize text."""
        if self.text_vectorizer is None:
            raise ValueError("Text vectorizer not built. Call build_text_vectorizer first.")
        result = self.text_vectorizer([text])
        return result.numpy()[0]
    
    def process_user_features(self, user_row: pd.Series) -> Dict[str, Any]:
        """Process all user features."""
        processed = {}
        
        # Numeric
        for col in self.feature_config['user_features']['numeric']:
            processed[col] = self.normalize_numeric(user_row[col], col, 'user')
        
        # Low cardinality categorical (one-hot indices)
        for col in self.feature_config['user_features']['categorical_low_cardinality'].keys():
            processed[col] = self.encode_categorical_low(user_row[col], col)
        
        # High cardinality categorical (embedding IDs)
        for col in self.feature_config['user_features']['categorical_high_cardinality'].keys():
            processed[col] = self.encode_categorical_high(user_row[col], col)
        
        # Text (tokenized)
        for col in self.feature_config['user_features']['text']:
            processed[col] = self.encode_text(user_row[col])
        
        return processed
    
    def process_item_features(self, item_row: pd.Series) -> Dict[str, Any]:
        """Process all item features."""
        processed = {}
        
        # Numeric
        for col in self.feature_config['item_features']['numeric']:
            processed[col] = self.normalize_numeric(item_row[col], col, 'item')
        
        # Low cardinality categorical
        for col in self.feature_config['item_features']['categorical_low_cardinality'].keys():
            processed[col] = self.encode_categorical_low(item_row[col], col)
        
        # High cardinality categorical
        for col in self.feature_config['item_features']['categorical_high_cardinality'].keys():
            processed[col] = self.encode_categorical_high(item_row[col], col)
        
        # Text
        for col in self.feature_config['item_features']['text']:
            processed[col] = self.encode_text(item_row[col])
        
        return processed


# ============================================================================
# DATA LOADING & SAMPLING
# ============================================================================

class DatasetBuilder:
    """Create TensorFlow datasets with proper batching and preprocessing."""
    
    def __init__(self, 
                 users_df: pd.DataFrame,
                 items_df: pd.DataFrame,
                 interactions_df: pd.DataFrame,
                 feature_processor: FeatureProcessor,
                 feature_config: Dict):
        self.users_df = users_df
        self.items_df = items_df
        self.interactions_df = interactions_df
        self.feature_processor = feature_processor
        self.feature_config = feature_config
        
        # Create ID to index mappings for fast lookup
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(users_df['user_id'])}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(items_df['item_id'])}
    
    def create_training_dataset(self, 
                               batch_size: int = 32,
                               negative_sample_ratio: int = 1) -> tf.data.Dataset:
        """
        Create training dataset with hard negative sampling.
        
        Args:
            batch_size: Training batch size
            negative_sample_ratio: Number of negative samples per positive
        """
        user_features_list = []
        item_features_list = []
        labels_list = []
        
        # Process interactions
        for _, row in self.interactions_df.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            label = int(row['label'])
            
            # Get data rows
            user_row = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
            item_row = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
            
            # Process features
            user_features = self.feature_processor.process_user_features(user_row)
            item_features = self.feature_processor.process_item_features(item_row)
            
            user_features_list.append(user_features)
            item_features_list.append(item_features)
            labels_list.append(label)
            
            # Add negative samples
            for _ in range(negative_sample_ratio):
                neg_item_id = np.random.randint(0, len(self.items_df))
                neg_item_row = self.items_df[self.items_df['item_id'] == neg_item_id].iloc[0]
                neg_item_features = self.feature_processor.process_item_features(neg_item_row)
                
                user_features_list.append(user_features)
                item_features_list.append(neg_item_features)
                labels_list.append(0)
        
        # Convert to tensors
        user_features_tensor = self._dict_list_to_tensor(user_features_list)
        item_features_tensor = self._dict_list_to_tensor(item_features_list)
        labels_tensor = tf.constant(labels_list, dtype=tf.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            user_features_tensor,
            item_features_tensor,
            labels_tensor
        )).shuffle(len(labels_list)).batch(batch_size)
        
        return dataset
    
    def _dict_list_to_tensor(self, dict_list: List[Dict]) -> Dict[str, tf.Tensor]:
        """Convert list of feature dicts to dict of tensors."""
        result = {}
        for key in dict_list[0].keys():
            values = [d[key] for d in dict_list]
            if isinstance(values[0], np.ndarray):
                result[key] = tf.constant(np.array(values), dtype=tf.int32)
            else:
                result[key] = tf.constant(values, dtype=tf.float32)
        return result
    
    def get_user_by_id(self, user_id: int) -> Dict[str, Any]:
        """Get processed features for a specific user."""
        user_row = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        return self.feature_processor.process_user_features(user_row)
    
    def get_item_by_id(self, item_id: int) -> Dict[str, Any]:
        """Get processed features for a specific item."""
        item_row = self.items_df[self.items_df['item_id'] == item_id].iloc[0]
        return self.feature_processor.process_item_features(item_row)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_two_tower_model(feature_config: Dict, 
                         feature_processor: FeatureProcessor,
                         embedding_dim: int = 64) -> Model:
    """
    Build Two-Tower model with mixed feature types.
    
    Architecture:
    - User Tower: numeric + one-hot categoricals + embedding for high-cardinality + text
    - Item Tower: numeric + one-hot categoricals + embedding for high-cardinality + text
    - Interaction: Dot product between towers
    """
    
    # ========== USER TOWER INPUTS ==========
    user_numeric_inputs = {}
    for col in feature_config['user_features']['numeric']:
        user_numeric_inputs[col] = keras.Input(shape=(1,), dtype=tf.float32, name=f'user_{col}')
    
    user_low_cat_inputs = {}
    for col in feature_config['user_features']['categorical_low_cardinality'].keys():
        user_low_cat_inputs[col] = keras.Input(shape=(1,), dtype=tf.int32, name=f'user_{col}')
    
    user_high_cat_inputs = {}
    user_high_cat_embeddings = {}
    for col, values in feature_config['user_features']['categorical_high_cardinality'].items():
        user_high_cat_inputs[col] = keras.Input(shape=(1,), dtype=tf.int32, name=f'user_{col}')
        user_high_cat_embeddings[col] = layers.Embedding(
            input_dim=len(values) + 1,
            output_dim=embedding_dim,
            name=f'user_{col}_embedding'
        )(user_high_cat_inputs[col])
        user_high_cat_embeddings[col] = layers.Flatten()(user_high_cat_embeddings[col])
    
    user_text_inputs = {}
    user_text_embeddings = {}
    for col in feature_config['user_features']['text']:
        user_text_inputs[col] = keras.Input(shape=(50,), dtype=tf.int32, name=f'user_{col}')
        user_text_embeddings[col] = layers.Embedding(
            input_dim=5000,  # vocab_size
            output_dim=embedding_dim,
            name=f'user_{col}_embedding'
        )(user_text_inputs[col])
        user_text_embeddings[col] = layers.GlobalAveragePooling1D()(user_text_embeddings[col])
    
    # ========== ITEM TOWER INPUTS ==========
    item_numeric_inputs = {}
    for col in feature_config['item_features']['numeric']:
        item_numeric_inputs[col] = keras.Input(shape=(1,), dtype=tf.float32, name=f'item_{col}')
    
    item_low_cat_inputs = {}
    item_low_cat_one_hot = {}
    for col, values in feature_config['item_features']['categorical_low_cardinality'].items():
        item_low_cat_inputs[col] = keras.Input(shape=(1,), dtype=tf.int32, name=f'item_{col}')
        item_low_cat_one_hot[col] = layers.CategoryEncoding(
            num_tokens=len(values) + 1,
            output_mode='one_hot'
        )(item_low_cat_inputs[col])
    
    item_high_cat_inputs = {}
    item_high_cat_embeddings = {}
    for col, values in feature_config['item_features']['categorical_high_cardinality'].items():
        item_high_cat_inputs[col] = keras.Input(shape=(1,), dtype=tf.int32, name=f'item_{col}')
        item_high_cat_embeddings[col] = layers.Embedding(
            input_dim=len(values) + 1,
            output_dim=embedding_dim,
            name=f'item_{col}_embedding'
        )(item_high_cat_inputs[col])
        item_high_cat_embeddings[col] = layers.Flatten()(item_high_cat_embeddings[col])
    
    item_text_inputs = {}
    item_text_embeddings = {}
    for col in feature_config['item_features']['text']:
        item_text_inputs[col] = keras.Input(shape=(50,), dtype=tf.int32, name=f'item_{col}')
        item_text_embeddings[col] = layers.Embedding(
            input_dim=5000,
            output_dim=embedding_dim,
            name=f'item_{col}_embedding'
        )(item_text_inputs[col])
        item_text_embeddings[col] = layers.GlobalAveragePooling1D()(item_text_embeddings[col])
    
    # ========== USER TOWER ASSEMBLY ==========
    user_features = []
    
    # Add numeric features
    user_numeric_concat = layers.Concatenate()(list(user_numeric_inputs.values())) if user_numeric_inputs else None
    if user_numeric_concat is not None:
        user_features.append(user_numeric_concat)
    
    # Add low cardinality one-hot
    for col, inp in user_low_cat_inputs.items():
        one_hot = layers.CategoryEncoding(
            num_tokens=len(feature_config['user_features']['categorical_low_cardinality'][col]) + 1,
            output_mode='one_hot'
        )(inp)
        user_features.append(one_hot)
    
    # Add high cardinality embeddings
    user_features.extend(user_high_cat_embeddings.values())
    
    # Add text embeddings
    user_features.extend(user_text_embeddings.values())
    
    user_tower = layers.Concatenate()(user_features)
    user_tower = layers.Dense(256, activation='relu')(user_tower)
    user_tower = layers.Dropout(0.3)(user_tower)
    user_tower = layers.Dense(128, activation='relu')(user_tower)
    user_tower = layers.Dropout(0.3)(user_tower)
    user_embedding = layers.Dense(embedding_dim, activation='relu', name='user_embedding')(user_tower)
    
    # ========== ITEM TOWER ASSEMBLY ==========
    item_features = []
    
    # Add numeric features
    item_numeric_concat = layers.Concatenate()(list(item_numeric_inputs.values())) if item_numeric_inputs else None
    if item_numeric_concat is not None:
        item_features.append(item_numeric_concat)
    
    # Add low cardinality one-hot
    item_features.extend(item_low_cat_one_hot.values())
    
    # Add high cardinality embeddings
    item_features.extend(item_high_cat_embeddings.values())
    
    # Add text embeddings
    item_features.extend(item_text_embeddings.values())
    
    item_tower = layers.Concatenate()(item_features)
    item_tower = layers.Dense(256, activation='relu')(item_tower)
    item_tower = layers.Dropout(0.3)(item_tower)
    item_tower = layers.Dense(128, activation='relu')(item_tower)
    item_tower = layers.Dropout(0.3)(item_tower)
    item_embedding = layers.Dense(embedding_dim, activation='relu', name='item_embedding')(item_tower)
    
    # ========== INTERACTION & OUTPUT ==========
    # Dot product similarity
    similarity = layers.Dot(axes=1)([user_embedding, item_embedding])
    output = layers.Dense(1, activation='sigmoid', name='output')(similarity)
    
    # Compile model
    all_inputs = (
        list(user_numeric_inputs.values()) +
        list(user_low_cat_inputs.values()) +
        list(user_high_cat_inputs.values()) +
        list(user_text_inputs.values()) +
        list(item_numeric_inputs.values()) +
        list(item_low_cat_inputs.values()) +
        list(item_high_cat_inputs.values()) +
        list(item_text_inputs.values())
    )
    
    model = Model(inputs=all_inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC', 'Precision', 'Recall']
    )
    
    return model


# ============================================================================
# TRAINING & INFERENCE
# ============================================================================

def train_model(model: Model,
                dataset_builder: DatasetBuilder,
                epochs: int = 5,
                validation_split: float = 0.2):
    """Train the two-tower model."""
    
    # Create training dataset
    train_dataset = dataset_builder.create_training_dataset(batch_size=32)
    
    # Train
    history = model.fit(
        train_dataset,
        epochs=epochs,
        verbose=1,
    )
    
    return history


def get_recommendations(model: Model,
                       user_id: int,
                       item_ids: List[int],
                       dataset_builder: DatasetBuilder,
                       top_k: int = 5) -> List[Tuple[int, float]]:
    """Get top-k recommendations for a user."""
    
    # Get user features
    user_features = dataset_builder.get_user_by_id(user_id)
    
    scores = []
    for item_id in item_ids:
        item_features = dataset_builder.get_item_by_id(item_id)
        
        # Prepare input tensors
        model_inputs = {}
        model_inputs.update({f'user_{k}': tf.expand_dims(tf.constant(v), 0) 
                           for k, v in user_features.items()})
        model_inputs.update({f'item_{k}': tf.expand_dims(tf.constant(v), 0) 
                           for k, v in item_features.items()})
        
        # Get prediction
        score = model.predict(model_inputs, verbose=0)[0][0]
        scores.append((item_id, float(score)))
    
    # Sort and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("TWO-TOWER B2B RECOMMENDATION MODEL")
    print("=" * 80 + "\n")
    
    # Step 1: Create dataset
    print("[1] Creating sample B2B dataset...")
    data = create_sample_b2b_dataset(n_users=100, n_items=50, n_interactions=300)
    users_df = data['users_df']
    items_df = data['items_df']
    interactions_df = data['interactions']
    feature_config = data['feature_config']
    data_stats = data['data_stats']
    
    print(f"    ✓ Users: {len(users_df)}")
    print(f"    ✓ Items: {len(items_df)}")
    print(f"    ✓ Interactions: {len(interactions_df)}")
    
    print("\n    User Features:")
    print(f"      • Numeric: {feature_config['user_features']['numeric']}")
    print(f"      • Low Cardinality (one-hot): {list(feature_config['user_features']['categorical_low_cardinality'].keys())}")
    print(f"      • High Cardinality (embedding): {list(feature_config['user_features']['categorical_high_cardinality'].keys())}")
    print(f"      • Text: {feature_config['user_features']['text']}")
    
    print("\n    Item Features:")
    print(f"      • Numeric: {feature_config['item_features']['numeric']}")
    print(f"      • Low Cardinality (one-hot): {list(feature_config['item_features']['categorical_low_cardinality'].keys())}")
    print(f"      • High Cardinality (embedding): {list(feature_config['item_features']['categorical_high_cardinality'].keys())}")
    print(f"      • Text: {feature_config['item_features']['text']}")
    
    # Step 2: Create feature processor
    print("\n[2] Building feature processor...")
    feature_processor = FeatureProcessor(feature_config, data_stats, vocab_size=5000)
    
    # Build text vectorizer
    all_texts = list(users_df['description_text']) + list(items_df['description_text'])
    feature_processor.build_text_vectorizer(all_texts)
    print("    ✓ Feature processor ready")
    print("    ✓ Text vectorizer built")
    
    # Step 3: Create dataset builder
    print("\n[3] Creating dataset builder...")
    dataset_builder = DatasetBuilder(
        users_df, items_df, interactions_df,
        feature_processor, feature_config
    )
    print("    ✓ Dataset builder ready")
    
    # Step 4: Build model
    print("\n[4] Building two-tower model...")
    model = build_two_tower_model(feature_config, feature_processor, embedding_dim=64)
    print("    ✓ Model built successfully")
    print("\n    Model Summary:")
    model.summary()
    
    # Step 5: Train model
    print("\n[5] Training model...")
    history = train_model(model, dataset_builder, epochs=3)
    print("    ✓ Training complete")
    
    # Step 6: Get recommendations
    print("\n[6] Getting recommendations...")
    sample_user_id = 0
    sample_item_ids = list(range(10))
    
    recommendations = get_recommendations(
        model, sample_user_id, sample_item_ids,
        dataset_builder, top_k=3
    )
    
    print(f"\n    Top 3 recommendations for User {sample_user_id}:")
    for rank, (item_id, score) in enumerate(recommendations, 1):
        print(f"      {rank}. Item {item_id}: Score = {score:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ TWO-TOWER B2B RECOMMENDATION SYSTEM COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
