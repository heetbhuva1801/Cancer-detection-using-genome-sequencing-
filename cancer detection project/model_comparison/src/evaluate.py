from sklearn.metrics import silhouette_score

def evaluate_clustering(X, labels):
    score = silhouette_score(X, labels)
    print(f"Silhouette Score: {score}")
    return score
