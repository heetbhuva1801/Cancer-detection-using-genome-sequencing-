import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
FEATURES_PATH = "../data/processed/features.csv"
LABELS_PATH = "../data/processed/labels.csv"
MODEL_PATH = "../models/xgboost_best_model.pkl"
RESULTS_PATH = "../reports/figures/"

def load_data(features_path, labels_path):
    """Load features and labels."""
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()

    # Ensure the target variable is discrete
    y = y.round(0).astype(int)  # Convert to discrete classes if necessary
    print(f"Unique labels in target: {y.unique()}")
    return X, y

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and generate performance metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Classification Report
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix - Tuned Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs(RESULTS_PATH, exist_ok=True)
    cm_path = os.path.join(RESULTS_PATH, "tuned_model_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ROC Curve (for multi-class)
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")  # One-vs-Rest approach
        print(f"Multi-Class AUC-ROC Score (One-vs-Rest): {roc_auc:.2f}")

        # Plot ROC curve for each class
        for i, class_label in enumerate(model.classes_):
            fpr, tpr, _ = roc_curve(y_test == class_label, y_prob[:, i])
            plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC Curve - Tuned Model (Multi-Class)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        roc_path = os.path.join(RESULTS_PATH, "tuned_model_roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        print(f"ROC curve saved to {roc_path}")
    else:
        print("ROC curve not available for this model.")

def plot_feature_importance(model, feature_names):
    """Plot feature importance for the model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure(figsize=(10, 8))
        plt.barh(feature_names, importances)
        plt.title("Feature Importance - Tuned Model")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        importance_path = os.path.join(RESULTS_PATH, "tuned_model_feature_importance.png")
        plt.savefig(importance_path)
        plt.close()
        print(f"Feature importance plot saved to {importance_path}")
    else:
        print("Feature importance is not available for this model.")

def main():
    # Load data
    X, y = load_data(FEATURES_PATH, LABELS_PATH)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load the best model
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Plot feature importance
    plot_feature_importance(model, X.columns)

if __name__ == "__main__":
    main()
