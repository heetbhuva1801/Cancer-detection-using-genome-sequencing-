import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os

# Paths
FEATURES_PATH = "../data/processed/features.csv"
LABELS_PATH = "../data/processed/labels.csv"
MODEL_RESULTS_PATH = "../reports/figures/"

def load_data(features_path, labels_path):
    """Load features and labels."""
    print(f"Loading features from {features_path}...")
    X = pd.read_csv(features_path)
    print(f"Loading labels from {labels_path}...")
    y = pd.read_csv(labels_path).squeeze()  # Ensure y is a 1D array

    # Ensure the target variable contains discrete values
    print("Converting target variable to discrete categories...")
    if y.dtype in ['float64', 'float32']:
        y = y.round(0).astype(int)  # Round continuous values and convert to integer
    elif y.dtype not in ['int64', 'int32']:
        raise ValueError("Target variable must be numeric. Please check the dataset.")

    print(f"Unique labels after conversion: {y.unique()}")
    return X, y

def calculate_metrics(model, X_test, y_test, model_name):
    """Calculate and return performance metrics for a given model."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1-Score": f1_score(y_test, y_pred, average="macro"),
    }

    # ROC-AUC Score
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        if len(model.classes_) == 2:
            metrics["AUC-ROC"] = roc_auc_score(y_test, y_prob[:, 1])
        else:
            metrics["AUC-ROC"] = roc_auc_score(y_test, y_prob, multi_class="ovr")
    else:
        metrics["AUC-ROC"] = "N/A"  # Not available if predict_proba is not supported

    return metrics

def main():
    # Step 1: Load models, features, and labels
    print("Loading data...")
    X, y = load_data(FEATURES_PATH, LABELS_PATH)

    # Splitting data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    }

    # Train models and calculate metrics
    metrics_list = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = calculate_metrics(model, X_test, y_test, name)
        metrics_list.append(metrics)

    # Step 2: Create a DataFrame for metrics
    metrics_df = pd.DataFrame(metrics_list)

    # Step 3: Save or display the results
    os.makedirs(MODEL_RESULTS_PATH, exist_ok=True)
    metrics_path = f"{MODEL_RESULTS_PATH}model_performance_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Model performance comparison saved at {metrics_path}")

    # Display the table
    print(metrics_df)

if __name__ == "__main__":
    main()
