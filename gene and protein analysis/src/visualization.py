import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Function to create a distribution plot for gene expression
def plot_gene_distribution(data, output_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(data.mean(axis=0), kde=True, color='blue')
    plt.title("Gene Expression Distribution")
    plt.xlabel("Expression Level")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()
    print(f"Gene expression distribution plot saved to {output_path}")

# Function to create a distribution plot for protein expression
def plot_protein_distribution(data, output_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(data.mean(axis=0), kde=True, color='green')
    plt.title("Protein Expression Distribution")
    plt.xlabel("Expression Level")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()
    print(f"Protein expression distribution plot saved to {output_path}")

# Function to create a heatmap for gene correlations
def plot_gene_correlation_heatmap(data, output_path):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
    plt.title("Gene Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()
    print(f"Gene correlation heatmap saved to {output_path}")

# Function to create a heatmap for protein correlations
def plot_protein_correlation_heatmap(data, output_path):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap="viridis", annot=False)
    plt.title("Protein Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()
    print(f"Protein correlation heatmap saved to {output_path}")

# Main function
def main():
    # Load preprocessed gene and protein data
    gene_data_path = "../data/processed/gene_data.csv"
    protein_data_path = "../data/processed/protein_data.csv"
    gene_data = pd.read_csv(gene_data_path, index_col=0)
    protein_data = pd.read_csv(protein_data_path, index_col=0)

    # Create directories for visualization outputs
    figures_dir = "../reports/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Generate gene visualizations
    plot_gene_distribution(
        gene_data, os.path.join(figures_dir, "visualization_gene_expression_distribution.png")
    )
    plot_gene_correlation_heatmap(
        gene_data, os.path.join(figures_dir, "visualization_gene_correlation_heatmap.png")
    )

    # Generate protein visualizations
    plot_protein_distribution(
        protein_data, os.path.join(figures_dir, "visualization_protein_expression_distribution.png")
    )
    plot_protein_correlation_heatmap(
        protein_data, os.path.join(figures_dir, "visualization_protein_correlation_heatmap.png")
    )

if __name__ == "__main__":
    main()
