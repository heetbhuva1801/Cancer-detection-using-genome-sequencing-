import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
import joblib
import os

# Paths
FEATURES_PATH = "../data/processed/features.csv"
LABELS_PATH = "../data/processed/labels.csv"
MODEL_RESULTS_PATH = "../models/"

def load_data(features_path, labels_path):
    """Load features and labels."""
    print(f"Loading features from {features_path}...")
    X = pd.read_csv(features_path)
    print(f"Loading labels from {labels_path}...")
    y = pd.read_csv(labels_path).squeeze()

    # Ensure target variable is discrete
    print("Converting target variable to discrete categories...")
    y = y.round(0).astype(int)
    print(f"Unique labels in target: {y.unique()}")
    return X, y

def perform_grid_search(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    print("Starting hyperparameter tuning...")

    # Define the model
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    # Define the parameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }

    # GridSearchCV setup
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="f1_macro",  # You can change the metric based on your needs
        cv=3,  # 3-fold cross-validation
        verbose=2,
        n_jobs=-1,  # Use all available processors
    )

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best F1-Score:", grid_search.best_score_)
    return grid_search.best_estimator_

def main():
    # Step 1: Load features and labels
    X, y = load_data(FEATURES_PATH, LABELS_PATH)

    # Step 2: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 3: Perform hyperparameter tuning
    best_model = perform_grid_search(X_train, y_train)

    # Step 4: Save the best model
    os.makedirs(MODEL_RESULTS_PATH, exist_ok=True)
    model_path = os.path.join(MODEL_RESULTS_PATH, "xgboost_best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    main()
