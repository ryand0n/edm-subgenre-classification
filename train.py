import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA


def find_optimal_k(X, k_range=range(2, 16), n_init=10, random_state=42):
    """Run K-Means for each k in range, return inertias, silhouettes, and best k."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    best_idx = np.argmax(silhouettes)
    best_k = list(k_range)[best_idx]

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
        "best_score": silhouettes[best_idx],
    }


def train_kmeans(X, k, n_init=10, random_state=42):
    """Fit K-Means for a specific k. Returns model, labels, and metrics."""
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)

    return {
        "model": km,
        "labels": labels,
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(X, labels),
    }


def get_primary_genre(df, genre_col="genres_consolidated"):
    """Extract first genre from comma-separated genre column for single-label evaluation."""
    return df[genre_col].str.split(",").str[0].str.strip()


def evaluate_clusters(labels, primary_genres):
    """Compute ARI and NMI between cluster assignments and genre labels."""
    return {
        "ari": adjusted_rand_score(primary_genres, labels),
        "nmi": normalized_mutual_info_score(primary_genres, labels),
    }


def build_crosstab(labels, primary_genres):
    """Cluster-vs-genre cross-tabulation matrix."""
    return pd.crosstab(
        pd.Series(labels, name="cluster"),
        pd.Series(primary_genres, name="genre"),
    )


def profile_clusters(X, labels, feature_names, scaler):
    """Compute centroid values per cluster, inverse-transformed to original units."""
    df = pd.DataFrame(X, columns=feature_names)
    df["cluster"] = labels
    centroids_scaled = df.groupby("cluster")[feature_names].mean()
    centroids_original = pd.DataFrame(
        scaler.inverse_transform(centroids_scaled.values),
        index=centroids_scaled.index,
        columns=feature_names,
    )
    return centroids_original


def compare_k_values(X, k_values, df, feature_names, scaler,
                     genre_col="genres_consolidated", n_init=10, random_state=42):
    """Train K-Means for multiple k values, compute metrics + crosstab + profiles."""
    primary_genres = get_primary_genre(df, genre_col)
    results = []

    for k in k_values:
        km_result = train_kmeans(X, k, n_init=n_init, random_state=random_state)
        eval_result = evaluate_clusters(km_result["labels"], primary_genres)
        crosstab = build_crosstab(km_result["labels"], primary_genres)
        profiles = profile_clusters(X, km_result["labels"], feature_names, scaler)

        results.append({
            "k": k,
            "model": km_result["model"],
            "labels": km_result["labels"],
            "inertia": km_result["inertia"],
            "silhouette": km_result["silhouette"],
            "ari": eval_result["ari"],
            "nmi": eval_result["nmi"],
            "crosstab": crosstab,
            "profiles": profiles,
        })

    return results


def reduce_pca(X, n_components=2):
    """PCA for visualization. Returns projected data, model, and explained variance."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    return {
        "X_pca": X_pca,
        "pca": pca,
        "explained_variance": pca.explained_variance_ratio_,
    }
