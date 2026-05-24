import os
import warnings
warnings.filterwarnings('ignore')

from config.config import *
import data.data_processor as dp
from trainer import execute_experiments

def main():
    print("\n" + "="*60)
    print("Data Preprocessing...")
    print("="*60)

    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/adult.csv')
    print(f"\nLoading and processing data from: {filepath}\n")

    processor = dp.DataProcessor(filepath, target_column = 'class')
    processor = processor.preprocess_data(
        categorical_features = ADULT_CONFIG['categorical_features']
        , numeric_features = ADULT_CONFIG['numeric_features']
    )

    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = processor.get_train_test_split(test_size=TEST_SIZE, random_state=RANDOM_SEED)
    class_weights = processor.get_class_weights(y_train)

    print(f"\nFinal shapes (Adult Income):")
    print(f"  Scaled X data (Train : Test): {X_train_scaled.shape} : {X_test_scaled.shape}")
    print(f"  X data (Train : Test): {X_train.shape} : {X_test.shape}")
    print(f"  Y data (Train : Test): {y_train.shape} : {y_test.shape}")

    execute_experiments('adult', X_train_scaled, X_test_scaled, y_train, y_test, 
                       X_train, X_test, class_weights)

    print("\n" + "="*60)
    print("Data Preprocessing...")
    print("="*60)

    filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/wine.csv')
    print(f"\nLoading and processing data from: {filepath}")

    processor = dp.DataProcessor(filepath, target_column = 'class')
    processor = processor.preprocess_data(
        categorical_features = None
        , numeric_features = WINE_CONFIG['numeric_features'])

    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = processor.get_train_test_split(test_size=TEST_SIZE, random_state=RANDOM_SEED)
    class_weights = processor.get_class_weights(y_train)

    print(f"\nFinal shapes (Wine Quality):")
    print(f"  Scaled X data (Train : Test): {X_train_scaled.shape} : {X_test_scaled.shape}")
    print(f"  X data (Train : Test): {X_train.shape} : {X_test.shape}")
    print(f"  Y data (Train : Test): {y_train.shape} : {y_test.shape}")

    execute_experiments('wine', X_train_scaled, X_test_scaled, y_train, y_test, 
                       X_train, X_test, class_weights)

if __name__ == "__main__":
    main()