import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Convert labels to discrete categories
    print("Converting target variable to discrete categories...")
    y = y.round(0).astype(int)  # Example: rounding continuous values to nearest integer
    print(f"Unique labels after conversion: {y.unique()}")
    return X, y

def split_data(X, y):
    """Split the data into training and testing sets."""
    print("Splitting data into training and testing sets...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train a model and evaluate its performance."""
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Classification Report
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs(MODEL_RESULTS_PATH, exist_ok=True)
    cm_path = f"{MODEL_RESULTS_PATH}{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix for {model_name} at {cm_path}.")

    # ROC-AUC Score
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

        if len(model.classes_) == 2:
            # Binary classification
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])  # Use probabilities for the positive class
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
        else:
            # Multi-class classification
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
            for i, label in enumerate(model.classes_):
                fpr, tpr, _ = roc_curve(y_test == label, y_prob[:, i])
                plt.plot(fpr, tpr, label=f"Class {label} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        roc_path = f"{MODEL_RESULTS_PATH}{model_name}_roc_curve.png"
        plt.savefig(roc_path)
        plt.close()
        print(f"Saved ROC curve for {model_name} at {roc_path}.")
    else:
        print(f"{model_name} does not support predict_proba for ROC-AUC.")

def main():
    # Step 1: Load features and labels
    X, y = load_data(FEATURES_PATH, LABELS_PATH)

    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Train and evaluate models
    os.makedirs(MODEL_RESULTS_PATH, exist_ok=True)

    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    train_and_evaluate_model(logistic_model, X_train, X_test, y_train, y_test, "Logistic_Regression")

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random_Forest")

    # XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    train_and_evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")

    print("Model training and evaluation completed.")

if __name__ == "__main__":
    main()
