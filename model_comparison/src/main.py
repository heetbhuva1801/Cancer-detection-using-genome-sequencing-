from preprocess import load_data, preprocess_data
from clustering import detect_stages, save_clustered_data
from evaluate import evaluate_clustering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load and preprocess data
data = load_data('data/data.csv')
X_scaled = preprocess_data(data)

# Set the number of clusters to 3 to predict exactly 3 stages
n_clusters = 3
print(f"Predicting with {n_clusters} clusters...")

# Perform clustering to detect stages
predicted_labels = detect_stages(X_scaled, n_clusters=n_clusters)

# Evaluate clustering quality
score = evaluate_clustering(X_scaled, predicted_labels)
print(f"Silhouette Score with {n_clusters} clusters: {score}")

# Count the number of samples in each predicted stage
unique, counts = np.unique(predicted_labels, return_counts=True)
stage_labels = [f"Stage {i+1}" for i in unique]
percentages = (counts / counts.sum()) * 100

# Plotting the bar chart
plt.figure(figsize=(12, 7))
plt.bar(unique, counts, tick_label=stage_labels)
plt.xlabel("Predicted Stage")
plt.ylabel("Number of Patients")
plt.title(f"Distribution of Patients Across Predicted Stages (3 Clusters)")
plt.show()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=stage_labels, autopct='%1.1f%%', startangle=140)
plt.title("Proportion of Patients Across Predicted Stages (3 Clusters)")
plt.show()

# Creating and displaying a table
table_data = {
    "Predicted Stage": stage_labels,
    "Number of Patients": counts,
    "Percentage (%)": percentages
}
table_df = pd.DataFrame(table_data)
print(table_df)  # Print to console

# Display the table as a plot
fig, ax = plt.subplots(figsize=(6, 2))  # Adjust figure size as needed
ax.axis('tight')
ax.axis('off')
ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')
plt.title("Summary Table for Predicted Stages")
plt.show()

# Optionally save the final clustered data
save_clustered_data(X_scaled, predicted_labels, f'data/clustered_data_{n_clusters}_clusters.csv')
