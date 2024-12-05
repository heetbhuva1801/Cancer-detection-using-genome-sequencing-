from sklearn.cluster import KMeans
import pandas as pd

def detect_stages(X, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    stage_labels = kmeans.fit_predict(X)
    return stage_labels

def save_clustered_data(X, stage_labels, output_path='data/clustered_data.csv'):
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    df['Predicted_Stage'] = stage_labels
    df.to_csv(output_path, index=False)
