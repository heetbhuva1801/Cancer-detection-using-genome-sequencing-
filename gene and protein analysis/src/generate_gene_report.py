import os
import pandas as pd

# Function to calculate summary statistics
def generate_gene_statistics(data):
    statistics = data.describe().transpose()
    statistics['variance'] = data.var()
    return statistics

# Function to generate gene expression levels
def categorize_gene_expression(data):
    # Calculate thresholds
    low_threshold = data.quantile(0.25, axis=0)
    high_threshold = data.quantile(0.75, axis=0)

    # Categorize each value as low, moderate, or high
    expression_levels = data.apply(
        lambda col: col.apply(
            lambda x: "Low" if x < low_threshold[col.name] 
            else "High" if x > high_threshold[col.name] 
            else "Moderate"
        )
    )
    return expression_levels.apply(pd.Series.value_counts).fillna(0).sum(axis=1)

# Function to generate the report content
def generate_text_report(data, output_path):
    # Prepare statistical summary
    gene_statistics = generate_gene_statistics(data)
    expression_levels = categorize_gene_expression(data)

    # Write report
    with open(output_path, 'w') as report:
        report.write("Gene Analysis Report\n")
        report.write("=" * 80 + "\n\n")
        report.write("1. Data Summary\n")
        report.write(f"Total Samples: {data.shape[0]}\n")
        report.write(f"Total Genes: {data.shape[1]}\n\n")

        report.write("2. Gene Statistical Measures\n")
        report.write("Gene-wise Mean, Median, Standard Deviation, and Variance:\n")
        report.write(gene_statistics.to_string() + "\n\n")

        report.write("3. Gene Expression Levels\n")
        report.write("Low, Moderate, and High Expression Counts:\n")
        for level, count in expression_levels.items():
            report.write(f"{level} Expression: {int(count)}\n")
        report.write("\n")

        report.write("4. Key Observations\n")
        report.write("Based on statistical analysis, the following observations were made:\n")
        report.write("- Significant variation was observed in genes with high variance.\n")
        report.write("- Top 5 Genes with highest variance:\n")
        top_genes = gene_statistics.nlargest(5, 'variance')
        for gene, row in top_genes.iterrows():
            report.write(f"  {gene}: Variance = {row['variance']:.2f}\n")

    print(f"Text-based gene report generated: {output_path}")

# Main function
def main():
    # Load the processed gene data
    gene_data_path = "../data/processed/gene_data.csv"
    gene_data = pd.read_csv(gene_data_path, index_col=0)

    # Output report path
    report_output_path = os.path.expanduser("~/Downloads/gene_analysis_report.txt")

    # Generate the report
    generate_text_report(gene_data, report_output_path)

if __name__ == "__main__":
    main()
