"""Supervised classification experiment.

Trains RandomForest, GradientBoosting, and XGBoost classifiers on audio features
to predict consolidated genre labels. Handles class imbalance via class weighting
and stratified splits. Includes DummyClassifier baseline and RandomizedSearchCV
hyperparameter tuning.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import model_training.train as train


# Hyperparameter search spaces
PARAM_SPACES = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced"],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
}


def train_models(X, df, audio_features):
    """Train supervised classifiers and return results dict.

    Args:
        X: Scaled feature matrix (numpy array).
        df: Cleaned DataFrame with metadata (needs genres_consolidated column).
        audio_features: List of audio feature names.

    Returns:
        dict with model_results, best_model_name, best_model, best_params,
        and feature_importances.
    """
    # Extract target labels
    y = train.get_primary_genre(df)

    # Filter out genres with fewer than 30 samples
    genre_counts = y.value_counts()
    valid_genres = genre_counts[genre_counts >= 30].index
    mask = y.isin(valid_genres)
    X_filtered = X[mask]
    y_filtered = y[mask].reset_index(drop=True)

    dropped_genres = genre_counts[genre_counts < 30].index.tolist()
    print(f"\n{'=' * 60}")
    print("Genre Filtering")
    print("=" * 60)
    print(f"Kept {len(valid_genres)} genres ({len(X_filtered)} tracks)")
    if dropped_genres:
        print(f"Dropped (< 30 samples): {dropped_genres}")

    # Show class distribution
    print(f"\nClass distribution (train set will be stratified):")
    dist = y_filtered.value_counts()
    print(f"  Largest class:  {dist.iloc[0]:>5} ({dist.index[0]})")
    print(f"  Smallest class: {dist.iloc[-1]:>5} ({dist.index[-1]})")
    print(f"  Imbalance ratio: {dist.iloc[0] / dist.iloc[-1]:.1f}x")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, stratify=y_filtered, random_state=42
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)} (stratified)")

    # Encode labels for XGBoost (requires integer targets)
    le = LabelEncoder()
    le.fit(y_train)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Compute sample weights for models that don't support class_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    results = {}
    best_params = {}
    tuned_models = {}

    # --- Baseline: DummyClassifier ---
    print(f"\n{'=' * 60}")
    print("Baseline (DummyClassifier, strategy='stratified')")
    print("=" * 60)
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    acc_dummy = accuracy_score(y_test, y_pred_dummy)
    bal_acc_dummy = balanced_accuracy_score(y_test, y_pred_dummy)
    results["Baseline"] = {"accuracy": acc_dummy, "balanced_accuracy": bal_acc_dummy}
    print(classification_report(y_test, y_pred_dummy, zero_division=0))

    # --- Hyperparameter tuning with RandomizedSearchCV ---
    model_configs = {
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "X_train": X_train,
            "y_train": y_train,
            "fit_params": {},
            "X_test": X_test,
            "y_test": y_test,
            "decode": False,
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "X_train": X_train,
            "y_train": y_train,
            "fit_params": {"sample_weight": sample_weights},
            "X_test": X_test,
            "y_test": y_test,
            "decode": False,
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                random_state=42, eval_metric="mlogloss", verbosity=0
            ),
            "X_train": X_train,
            "y_train": y_train_encoded,
            "fit_params": {"sample_weight": sample_weights},
            "X_test": X_test,
            "y_test": y_test,
            "decode": True,
        },
    }

    for name, cfg in model_configs.items():
        print(f"\n{'=' * 60}")
        print(f"{name} — RandomizedSearchCV (cv=3, n_iter=20)")
        print("=" * 60)

        search = RandomizedSearchCV(
            cfg["estimator"],
            PARAM_SPACES[name],
            n_iter=20,
            scoring="balanced_accuracy",
            cv=3,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(cfg["X_train"], cfg["y_train"], **cfg["fit_params"])

        best_params[name] = search.best_params_
        print(f"Best params: {search.best_params_}")
        print(f"Best CV balanced accuracy: {search.best_score_:.4f}")

        # Evaluate on held-out test set
        model = search.best_estimator_
        tuned_models[name] = model

        if cfg["decode"]:
            y_pred_raw = model.predict(cfg["X_test"])
            y_pred = le.inverse_transform(y_pred_raw)
        else:
            y_pred = model.predict(cfg["X_test"])

        acc = accuracy_score(cfg["y_test"], y_pred)
        bal_acc = balanced_accuracy_score(cfg["y_test"], y_pred)
        results[name] = {"accuracy": acc, "balanced_accuracy": bal_acc}

        print(f"\nTest Set — Classification Report")
        print(classification_report(cfg["y_test"], y_pred, zero_division=0))

    # Accuracy comparison
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':<12} {'Balanced Acc':<12}")
    print(f"  {'-'*49}")
    for name, metrics in results.items():
        print(f"  {name:<25} {metrics['accuracy']:<12.4f} {metrics['balanced_accuracy']:<12.4f}")

    # Feature importances (from best tuned model by balanced accuracy)
    real_models = {k: v for k, v in results.items() if k != "Baseline"}
    best_model_name = max(real_models, key=lambda n: real_models[n]["balanced_accuracy"])
    best_model = tuned_models[best_model_name]
    importances = pd.Series(best_model.feature_importances_, index=audio_features)
    importances = importances.sort_values(ascending=False)

    print(f"\n{'=' * 60}")
    print(f"Feature Importances ({best_model_name})")
    print("=" * 60)
    for feat, imp in importances.items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:<18} {imp:.4f}  {bar}")

    return {
        "model_results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_params": best_params,
        "feature_importances": importances.to_dict(),
    }


def main():
    """Standalone entry point — loads preprocessed artifacts from disk."""
    artifacts_dir = Path("artifacts/preprocessed")

    X = np.load(artifacts_dir / "features.npy")
    df = pd.read_csv(artifacts_dir / "metadata.csv")
    with open(artifacts_dir / "config.json") as f:
        config = json.load(f)

    train_models(X, df, config["audio_features"])


if __name__ == "__main__":
    main()
