"""Standalone K-Means clustering experiment.

Replicates the notebook's unsupervised workflow as a terminal-friendly script.
Uses pipeline.prepare_training_data() for data prep and train.py functions for clustering.

Handles class imbalance by downsampling to a balanced dataset before clustering,
so that dominant genres don't dominate cluster assignments.
"""

import numpy as np
import pandas as pd

import pipeline
import train


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


def main():
    # Data preparation
    data = pipeline.prepare_training_data()
    X = data["X"]
    df = data["df"]
    scaler = data["scaler"]
    audio_features = data["audio_features"]

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

    results = train.compare_k_values(X, k_values, df, audio_features, scaler)

    # Summary metrics table
    print("\n" + "=" * 60)
    print("Summary Metrics")
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


if __name__ == "__main__":
    main()
