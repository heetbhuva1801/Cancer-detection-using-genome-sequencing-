import pandas as pd
import os

# Paths
RAW_DATA_PATH = "../data/raw/METABRIC_RNA_Mutation.csv"  # Replace with your dataset file name
PROCESSED_DATA_PATH = "../data/processed/processed_data.csv"

def load_dataset(filepath):
    """Load the dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    print(f"Loading dataset from {filepath}...")
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            # Fill missing values in numeric columns with the mean
            if df[col].isnull().sum() > 0:
                print(f"Filling missing values in numeric column '{col}' with mean.")
                df[col] = df[col].fillna(df[col].mean())
        elif df[col].dtype == "object":
            # Fill missing values in categorical columns with the mode
            if df[col].isnull().sum() > 0:
                print(f"Filling missing values in categorical column '{col}' with mode.")
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

def preprocess_columns(df):
    """Convert columns to numeric where possible and retain all others as is."""
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # Attempt to convert object columns to numeric
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except:
                print(f"Could not convert column '{col}' to numeric.")
    return df

def save_processed_data(df, filepath):
    """Save the processed dataset to a CSV file."""
    print(f"Saving processed dataset to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print("Processed dataset saved successfully.")

def main():
    # Step 1: Load the dataset
    df = load_dataset(RAW_DATA_PATH)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Preprocess columns
    df = preprocess_columns(df)

    # Step 4: Save the processed dataset
    save_processed_data(df, PROCESSED_DATA_PATH)

if __name__ == "__main__":
    main()
