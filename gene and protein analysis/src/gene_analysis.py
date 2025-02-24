import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# File paths
GENE_DATA_PATH = "../data/processed/gene_data.csv"
OUTPUT_PATH = "../reports/figures/"

def load_gene_data():
    """Load the preprocessed gene data."""
    if not os.path.exists(GENE_DATA_PATH):
        raise FileNotFoundError(f"Gene data file not found: {GENE_DATA_PATH}")
    return pd.read_csv(GENE_DATA_PATH)

def perform_pca(data, n_components=2):
    """Perform PCA for dimensionality reduction."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_
    return principal_components, explained_variance

def perform_clustering(data, n_clusters=3):
    """Perform clustering using K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans.cluster_centers_

def plot_pca_results(principal_components, clusters, output_path):
    """Plot PCA results with clustering information."""
    plt.figure(figsize=(8, 6))
    plt.scatter(
        principal_components[:, 0],
        principal_components[:, 1],
        c=clusters,
        cmap="viridis",
        s=50,
        alpha=0.7
    )
    plt.colorbar(label="Cluster")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Gene Data with Clustering")
    plt.savefig(os.path.join(output_path, "gene_pca_clustering.png"))
    plt.close()

def main():
    print("Loading gene data...")
    data = load_gene_data()

    # Drop non-numeric columns if any
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        raise ValueError("No numeric columns found in the gene data.")

    print("Performing PCA...")
    principal_components, explained_variance = perform_pca(numeric_data)
    print(f"Explained variance by components: {explained_variance}")

    print("Performing clustering...")
    clusters, _ = perform_clustering(principal_components)

    print("Generating PCA plot with clustering...")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    plot_pca_results(principal_components, clusters, OUTPUT_PATH)

    print(f"Analysis complete. Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
