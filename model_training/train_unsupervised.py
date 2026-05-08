"""Unsupervised K-Means clustering experiment.

Replicates the notebook's unsupervised workflow as a terminal-friendly script.
Handles class imbalance by downsampling to a balanced dataset before clustering,
so that dominant genres don't dominate cluster assignments. Includes random
assignment baseline for comparison.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

import model_training.train as train


def balance_by_genre(X, df, genre_col="genres_consolidated", random_state=42):
    """Downsample to the median genre count so no single genre dominates clusters."""
    primary_genres = train.get_primary_genre(df, genre_col)
    genre_counts = primary_genres.value_counts()

    cap = int(genre_counts.median())
    print(f"Balancing: capping each genre to {cap} tracks (median count)")
    print(f"  Before: {len(X)} tracks, genre sizes {genre_counts.min()}-{genre_counts.max()}")

    rng = np.random.default_rng(random_state)
    keep_indices = []
    for genre in genre_counts.index:
        genre_idx = np.where(primary_genres.values == genre)[0]
        if len(genre_idx) > cap:
            genre_idx = rng.choice(genre_idx, size=cap, replace=False)
        keep_indices.extend(genre_idx)

    keep_indices = sorted(keep_indices)
    X_balanced = X[keep_indices]
    df_balanced = df.iloc[keep_indices].reset_index(drop=True)

    new_counts = train.get_primary_genre(df_balanced, genre_col).value_counts()
    print(f"  After:  {len(X_balanced)} tracks, genre sizes {new_counts.min()}-{new_counts.max()}")
    return X_balanced, df_balanced


def compute_random_baseline(X, k_values, primary_genres, random_state=42):
    """Compute baseline metrics using random cluster assignments for each k."""
    rng = np.random.default_rng(random_state)
    baseline_results = []

    for k in k_values:
        random_labels = rng.integers(0, k, size=len(X))
        sil = silhouette_score(X, random_labels)
        ari = adjusted_rand_score(primary_genres, random_labels)
        nmi = normalized_mutual_info_score(primary_genres, random_labels)
        baseline_results.append({
            "k": k,
            "silhouette": sil,
            "ari": ari,
            "nmi": nmi,
        })

    return baseline_results


def train_clusters(X, df, scaler, audio_features):
    """Run K-Means clustering experiment and return results dict.

    Args:
        X: Scaled feature matrix (numpy array).
        df: Cleaned DataFrame with metadata.
        scaler: Fitted StandardScaler for inverse-transforming centroids.
        audio_features: List of audio feature names.

    Returns:
        dict with summary metrics, best_k, baseline metrics, and detailed results per k.
    """
    # Balance dataset to mitigate genre imbalance in cluster evaluation
    print("\n" + "=" * 60)
    print("Class Balancing (downsample to median)")
    print("=" * 60)
    X, df = balance_by_genre(X, df)

    # Find optimal k via silhouette score
    print("\n" + "=" * 60)
    print("Finding Optimal K (silhouette method)")
    print("=" * 60)
    optimal = train.find_optimal_k(X)
    best_k = optimal["best_k"]
    print(f"Best k = {best_k} (silhouette = {optimal['best_score']:.4f})")

    # Compare multiple k values
    k_values = sorted(set([2, best_k, 10, 15]))
    print(f"\nComparing k values: {k_values}")
    print("-" * 60)

    # Random baseline for each k
    primary_genres = train.get_primary_genre(df)
    baseline_results = compute_random_baseline(X, k_values, primary_genres)

    # K-Means results
    results = train.compare_k_values(X, k_values, df, audio_features, scaler)

    # Print both tables
    print("\n" + "=" * 60)
    print("Baseline (Random Assignment)")
    print("=" * 60)
    print(f"{'k':<5} {'Silhouette':<12} {'ARI':<10} {'NMI':<10}")
    print("-" * 37)
    for b in baseline_results:
        print(f"{b['k']:<5} {b['silhouette']:<12.4f} {b['ari']:<10.4f} {b['nmi']:<10.4f}")

    print("\n" + "=" * 60)
    print("K-Means Results")
    print("=" * 60)
    print(f"{'k':<5} {'Silhouette':<12} {'ARI':<10} {'NMI':<10}")
    print("-" * 37)
    for r in results:
        print(f"{r['k']:<5} {r['silhouette']:<12.4f} {r['ari']:<10.4f} {r['nmi']:<10.4f}")

    # Find best NMI result for detailed output
    best_nmi_result = max(results, key=lambda r: r["nmi"])
    best_nmi_k = best_nmi_result["k"]

    # Cluster centroids for best-NMI k
    print(f"\n{'=' * 60}")
    print(f"Cluster Centroids (k={best_nmi_k}, original units)")
    print("=" * 60)
    print(best_nmi_result["profiles"].round(3).to_string())

    # Genre-cluster crosstab for best-NMI k
    print(f"\n{'=' * 60}")
    print(f"Genre-Cluster Crosstab (k={best_nmi_k})")
    print("=" * 60)
    print(best_nmi_result["crosstab"].to_string())

    return {
        "best_k": best_k,
        "best_silhouette": optimal["best_score"],
        "baseline": baseline_results,
        "k_results": [
            {
                "k": r["k"],
                "silhouette": r["silhouette"],
                "ari": r["ari"],
                "nmi": r["nmi"],
            }
            for r in results
        ],
    }


def main():
    """Standalone entry point — loads preprocessed artifacts from disk."""
    artifacts_dir = Path("artifacts/preprocessed")

    X = np.load(artifacts_dir / "features.npy")
    df = pd.read_csv(artifacts_dir / "metadata.csv")
    with open(artifacts_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(artifacts_dir / "config.json") as f:
        config = json.load(f)

    train_clusters(X, df, scaler, config["audio_features"])


if __name__ == "__main__":
    main()
