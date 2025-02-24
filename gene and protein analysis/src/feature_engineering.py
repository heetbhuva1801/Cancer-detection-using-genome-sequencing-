import pandas as pd
import os  # Ensure os module is imported
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Paths
PROCESSED_DATA_PATH = "../data/processed/processed_data.csv"
FEATURES_OUTPUT_PATH = "../data/processed/features.csv"
LABELS_OUTPUT_PATH = "../data/processed/labels.csv"

def load_data(filepath):
    """Load the processed dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed dataset not found at {filepath}")
    print(f"Loading processed dataset from {filepath}...")
    return pd.read_csv(filepath)

def prepare_features_and_labels(df):
    """Select important features, encode categorical variables, and scale numerical features."""
    # Define important features and target
    important_features = [
        "age_at_diagnosis",
        "tumor_size",
        "lymph_nodes_examined_positive",
        "mutation_count",
        "hormone_therapy",
        "radio_therapy",
        "chemotherapy",
    ]
    target_column = "tumor_stage"

    # Ensure all selected features exist in the dataset
    important_features = [col for col in important_features if col in df.columns]
    print(f"Selected Features: {important_features}")

    # Split into features and target
    X = df[important_features]
    y = df[target_column]

    # Handle categorical variables
    categorical_features = ["hormone_therapy", "radio_therapy", "chemotherapy"]
    numerical_features = [col for col in important_features if col not in categorical_features]

    # Define transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )

    # Apply transformations
    print("Applying transformations...")
    X_transformed = preprocessor.fit_transform(X)

    # Return processed features and target
    return X_transformed, y

def save_data(X, y):
    """Save features and labels to CSV files."""
    print(f"Saving features to {FEATURES_OUTPUT_PATH}...")
    pd.DataFrame(X).to_csv(FEATURES_OUTPUT_PATH, index=False)
    print(f"Saving labels to {LABELS_OUTPUT_PATH}...")
    y.to_csv(LABELS_OUTPUT_PATH, index=False)

def main():
    # Load processed dataset
    df = load_data(PROCESSED_DATA_PATH)

    # Prepare features and labels
    X, y = prepare_features_and_labels(df)

    # Save processed data
    save_data(X, y)
    print("Feature engineering completed successfully.")

if __name__ == "__main__":
    main()
