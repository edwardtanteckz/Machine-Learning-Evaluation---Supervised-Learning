import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, data_path, target_column='target'):
        self.data_path = data_path
        self.raw_data = None
        self.scalar_data = None
        self.label_encoded_data = None
        self.target = target_column
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()

    def load_data(self):
        self.raw_data = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.raw_data)} samples")
        print(f"✓ Features: {len(self.raw_data.columns) - 1}")
        print(f"✓ Shape: {self.raw_data.shape}")

    def preprocess_data(self, categorical_features=None, numeric_features=None):
        self.load_data()
        X = self.raw_data.drop(columns=[self.target])

        self.raw_data[self.target] = self.target_encoder.fit_transform(self.raw_data[self.target])

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            drop='first', sparse_output=False, handle_unknown='ignore'
        )
        if categorical_features:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
            self.scalar_data = preprocessor.fit_transform(X)

            X_label_encoded = X.copy()
            for col in categorical_features:
                le = LabelEncoder()
                X_label_encoded[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
            self.label_encoded_data = X_label_encoded[numeric_features + categorical_features].values
        else:
            self.scalar_data = numeric_transformer.fit_transform(X)
            self.label_encoded_data = self.scalar_data.copy()

        print(f"✓ Preprocessed scaled data shape: {self.scalar_data.shape}")
        print(f"✓ Preprocessed label-encoded data shape: {self.label_encoded_data.shape}")
        return self
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        y = self.raw_data[self.target]

        indices = np.arange(len(self.raw_data))
        train_idx, test_idx = train_test_split(
            indices, test_size = test_size
            , random_state = random_state, stratify = y
        )
        X_train_scaled = self.scalar_data[train_idx]
        X_test_scaled = self.scalar_data[test_idx]

        X_train_encoded = self.label_encoded_data[train_idx]
        X_test_encoded = self.label_encoded_data[test_idx]

        self.verify_no_leakage(X_train_encoded, X_test_encoded)

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train_encoded, X_test_encoded

    def verify_no_leakage(self, X_train, X_test):
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        train_hashes = pd.util.hash_pandas_object(X_train, index=False)
        test_hashes = pd.util.hash_pandas_object(X_test, index=False)
        overlap = set(train_hashes) & set(test_hashes)
        
        if len(overlap) > 0:
            print(f"⚠ WARNING: {len(overlap)} overlapping samples detected!")
        else:
            print(f"✓ No overlapping samples between train and test")

    def get_class_weights(self, y_train):
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        
        print(f"\nClass weights:")
        for cls, weight in class_weight_dict.items():
            print(f"  Class {cls}: {weight:.3f}")
        
        return class_weight_dict