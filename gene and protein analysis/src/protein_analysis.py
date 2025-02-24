import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# Function to calculate statistical summaries for proteins
def calculate_protein_statistics(data):
    stats = data.describe().transpose()
    stats['variance'] = data.var()
    return stats

# Function to perform clustering on protein data
def cluster_proteins(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    data['Cluster'] = clusters
    return data, kmeans

# Function to visualize protein correlation
def visualize_protein_correlation(data, output_path):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
    plt.title("Protein Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")

# Function to visualize PCA
def visualize_protein_pca(data, output_path):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)
    plt.title("PCA of Protein Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(output_path)
    plt.close()
    print(f"PCA plot saved to {output_path}")

# Function to generate a report for protein analysis
def generate_protein_report(data, stats, clusters, output_path):
    with open(output_path, "w") as report:
        report.write("Protein Analysis Report\n")
        report.write("=" * 80 + "\n\n")
        report.write("1. Protein Data Summary\n")
        report.write(f"Number of Samples: {data.shape[0]}\n")
        report.write(f"Number of Proteins: {data.shape[1] - 1}\n\n")  # Excluding 'Cluster'

        report.write("2. Protein Statistical Summary\n")
        report.write(stats.to_string() + "\n\n")

        report.write("3. Clustering Analysis\n")
        report.write(f"Number of Clusters: {clusters.n_clusters}\n")
        report.write("Cluster Distribution:\n")
        report.write(data['Cluster'].value_counts().to_string() + "\n\n")

        report.write("4. Key Observations\n")
        report.write("- Top proteins with highest variance:\n")
        top_proteins = stats.nlargest(5, 'variance')
        for protein, row in top_proteins.iterrows():
            report.write(f"  {protein}: Variance = {row['variance']:.2f}\n")

    print(f"Protein analysis report saved to {output_path}")

# Main function
def main():
    # Load the preprocessed protein data
    protein_data_path = "../data/processed/protein_data.csv"
    protein_data = pd.read_csv(protein_data_path, index_col=0)

    # Calculate statistics
    protein_stats = calculate_protein_statistics(protein_data)

    # Perform clustering
    clustered_data, kmeans_model = cluster_proteins(protein_data)

    # Visualize protein correlation
    correlation_heatmap_path = "../reports/figures/protein_correlation_heatmap.png"
    visualize_protein_correlation(protein_data, correlation_heatmap_path)

    # Visualize PCA
    pca_plot_path = "../reports/figures/protein_pca_plot.png"
    visualize_protein_pca(protein_data, pca_plot_path)

    # Generate text-based report
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    report_output_path = os.path.join(downloads_folder, "protein_analysis_report.txt")
    generate_protein_report(clustered_data, protein_stats, kmeans_model, report_output_path)

if __name__ == "__main__":
    main()
