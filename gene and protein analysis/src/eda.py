import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
PROCESSED_DATA_PATH = "../data/processed/processed_data.csv"
EDA_OUTPUT_PATH = "../reports/figures/"  # Directory to save EDA visualizations

# List of important columns for your project
IMPORTANT_COLUMNS = [
    "tumor_stage",  # Target column
    "age_at_diagnosis",
    "tumor_size",
    "lymph_nodes_examined_positive",
    "mutation_count",
    "hormone_therapy",
    "radio_therapy",
    "chemotherapy"
]

def load_data(filepath):
    """Load the processed dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed dataset not found at {filepath}")
    print(f"Loading processed dataset from {filepath}...")
    return pd.read_csv(filepath)

def summarize_data(df):
    """Print basic summary statistics of the dataset."""
    print("\nDataset Summary:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

def visualize_distributions(df, columns):
    """Plot histograms for important columns."""
    for col in columns:
        if col in df.columns:  # Ensure the column exists
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid()
            plt.savefig(os.path.join(EDA_OUTPUT_PATH, f"{col}_distribution.png"))
            plt.close()
            print(f"Saved distribution plot for {col}.")
        else:
            print(f"Column '{col}' not found in the dataset. Skipping.")

def visualize_correlations(df, columns):
    """Generate a heatmap for correlations between important numeric features."""
    numeric_df = df[columns].select_dtypes(include=['float64', 'int64'])

    if numeric_df.empty:
        print("No numeric columns available for correlation analysis. Skipping heatmap generation.")
        return

    # Generate the correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(EDA_OUTPUT_PATH, "correlation_matrix.png"))
    plt.close()
    print("Saved correlation matrix heatmap.")

def visualize_boxplots(df, numeric_columns, target_column):
    """Generate boxplots for numeric features against the target variable."""
    for col in numeric_columns:
        if col != target_column and col in df.columns:  # Skip the target column itself
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df[target_column], y=df[col])
            plt.title(f"{col} by {target_column}")
            plt.xlabel(target_column)
            plt.ylabel(col)
            plt.grid()
            plt.savefig(os.path.join(EDA_OUTPUT_PATH, f"{col}_boxplot.png"))
            plt.close()
            print(f"Saved boxplot for {col} by {target_column}.")

def preprocess_columns(df):
    """Convert string columns to numeric where possible."""
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                print(f"Converted column '{col}' to numeric.")
            except:
                print(f"Column '{col}' could not be converted to numeric.")
    return df

def main():
    # Step 1: Load the processed dataset
    df = load_data(PROCESSED_DATA_PATH)

    # Debug column types
    print("Column Names:", df.columns)
    print("Column Data Types:\n", df.dtypes)

    # Step 2: Preprocess columns
    df = preprocess_columns(df)

    # Step 3: Summarize the data
    summarize_data(df)

    # Step 4: Create output directory for visualizations
    os.makedirs(EDA_OUTPUT_PATH, exist_ok=True)

    # Step 5: Visualize distributions of important columns
    visualize_distributions(df, IMPORTANT_COLUMNS)

    # Step 6: Visualize correlations between important numeric columns
    visualize_correlations(df, IMPORTANT_COLUMNS)

    # Step 7: Visualize boxplots for numeric features against the target variable (tumor_stage)
    target_column = "tumor_stage"
    if target_column in df.columns:
        visualize_boxplots(df, IMPORTANT_COLUMNS, target_column)
    else:
        print(f"Target column '{target_column}' not found. Skipping boxplot generation.")

if __name__ == "__main__":
    main()
