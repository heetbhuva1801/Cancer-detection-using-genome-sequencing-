import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, gene_columns, protein_columns):
    """Preprocess the dataset for gene and protein analysis."""
    print("Inspecting dataset...")
    print(f"Total columns in the dataset: {data.columns.tolist()}")

    # Filter out gene and protein columns
    missing_genes = [col for col in gene_columns if col not in data.columns]
    missing_proteins = [col for col in protein_columns if col not in data.columns]

    # Adjust dynamically based on available columns
    if missing_genes:
        print(f"WARNING: The following gene columns are missing: {missing_genes}")
        gene_columns = [col for col in gene_columns if col in data.columns]
    if missing_proteins:
        print(f"WARNING: The following protein columns are missing: {missing_proteins}")
        protein_columns = [col for col in protein_columns if col in data.columns]

    # Extract gene and protein data
    gene_data = data[gene_columns]
    protein_data = data[protein_columns]

    # Handle missing values
    print("Handling missing values for gene and protein data...")
    gene_data = gene_data.fillna(gene_data.mean())  # Replace missing values with column means
    protein_data = protein_data.fillna(protein_data.mean())

    # Normalize the data
    print("Normalizing gene and protein features...")
    scaler = MinMaxScaler()
    gene_data_scaled = scaler.fit_transform(gene_data)
    protein_data_scaled = scaler.fit_transform(protein_data)

    # Convert back to DataFrame for saving
    gene_data = pd.DataFrame(gene_data_scaled, columns=gene_columns)
    protein_data = pd.DataFrame(protein_data_scaled, columns=protein_columns)

    print("Preprocessing complete!")
    return gene_data, protein_data

def main():
    # Load the dataset
    data_path = "../data/raw/brca_data_w_subtypes.csv"  # Use the updated dataset
    print(f"Loading dataset from {data_path}...")
    data = pd.read_csv(data_path)

    # Define gene and protein columns based on prefixes
    gene_columns = [col for col in data.columns if col.startswith(('rs_', 'cn_', 'mu_'))]
    protein_columns = [col for col in data.columns if col.startswith('pp_')]

    # Debug: Print detected columns
    print(f"Detected {len(gene_columns)} gene-related columns: {gene_columns[:5]}... (truncated)")
    print(f"Detected {len(protein_columns)} protein-related columns: {protein_columns[:5]}... (truncated)")

    if not gene_columns and not protein_columns:
        raise ValueError("No gene or protein features found in the dataset. Check the dataset structure.")

    # Preprocess data
    gene_data, protein_data = preprocess_data(data, gene_columns, protein_columns)

    # Save the preprocessed data
    gene_output_path = "../data/processed/gene_data.csv"
    protein_output_path = "../data/processed/protein_data.csv"

    print(f"Saving preprocessed gene data to {gene_output_path}...")
    gene_data.to_csv(gene_output_path, index=False)

    print(f"Saving preprocessed protein data to {protein_output_path}...")
    protein_data.to_csv(protein_output_path, index=False)

    print("Gene and protein data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
