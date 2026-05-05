"""Standalone supervised classification experiment.

Trains RandomForest, GradientBoosting, and XGBoost classifiers on audio features
to predict consolidated genre labels. Handles class imbalance via class weighting
and stratified splits. Uses pipeline.prepare_training_data() for data prep.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import pipeline
import train


def main():
    # Data preparation
    data = pipeline.prepare_training_data()
    X = data["X"]
    df = data["df"]
    audio_features = data["audio_features"]

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

    # Train models — all handle class imbalance
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="mlogloss",
            verbosity=0
        ),
    }

    # GradientBoosting and XGBoost use sample_weight for class balancing
    fit_params = {
        "RandomForest": {"X": X_train, "y": y_train},
        "GradientBoosting": {"X": X_train, "y": y_train, "sample_weight": sample_weights},
        "XGBoost": {"X": X_train, "y": y_train_encoded, "sample_weight": sample_weights},
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"{name} (class-balanced) — Classification Report")
        print("=" * 60)
        params = fit_params[name]
        model.fit(params["X"], params["y"], **{k: v for k, v in params.items() if k not in ("X", "y")})

        if name == "XGBoost":
            y_pred_raw = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_raw)
        else:
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        results[name] = {"accuracy": acc, "balanced_accuracy": bal_acc}
        print(classification_report(y_test, y_pred, zero_division=0))

    # Accuracy comparison
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':<12} {'Balanced Acc':<12}")
    print(f"  {'-'*49}")
    for name, metrics in results.items():
        print(f"  {name:<25} {metrics['accuracy']:<12.4f} {metrics['balanced_accuracy']:<12.4f}")

    # Feature importances (from best model by balanced accuracy)
    best_model_name = max(results, key=lambda n: results[n]["balanced_accuracy"])
    best_model = models[best_model_name]
    importances = pd.Series(best_model.feature_importances_, index=audio_features)
    importances = importances.sort_values(ascending=False)

    print(f"\n{'=' * 60}")
    print(f"Feature Importances ({best_model_name})")
    print("=" * 60)
    for feat, imp in importances.items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:<18} {imp:.4f}  {bar}")


if __name__ == "__main__":
    main()
