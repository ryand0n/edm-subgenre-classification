"""Stage 3: Model Training — load preprocessed artifacts, train models, save results.

Reads preprocessed data from artifacts/preprocessed/, runs unsupervised and
supervised experiments, and saves results to artifacts/models/.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from model_training.train_unsupervised import train_clusters
from model_training.train_supervised import train_models


PREPROCESSED_DIR = Path("artifacts/preprocessed")
MODELS_DIR = Path("artifacts/models")


def run():
    """Load preprocessed artifacts, train all models, save results."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load preprocessed artifacts
    print("Loading preprocessed artifacts...")
    X = np.load(PREPROCESSED_DIR / "features.npy")
    df = pd.read_csv(PREPROCESSED_DIR / "metadata.csv")
    with open(PREPROCESSED_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(PREPROCESSED_DIR / "config.json") as f:
        config = json.load(f)

    audio_features = config["audio_features"]
    print(f"Loaded {X.shape[0]} tracks x {X.shape[1]} features")

    # Unsupervised clustering
    print("\n" + "#" * 60)
    print("# UNSUPERVISED CLUSTERING")
    print("#" * 60)
    unsupervised_results = train_clusters(X, df, scaler, audio_features)

    unsupervised_path = MODELS_DIR / "unsupervised_results.json"
    with open(unsupervised_path, "w") as f:
        json.dump(unsupervised_results, f, indent=2)

    # Supervised classification
    print("\n" + "#" * 60)
    print("# SUPERVISED CLASSIFICATION")
    print("#" * 60)
    supervised_results = train_models(X, df, audio_features)

    # Save supervised results (metrics only, not the model object)
    supervised_path = MODELS_DIR / "supervised_results.json"
    with open(supervised_path, "w") as f:
        json.dump({
            "model_results": supervised_results["model_results"],
            "best_model_name": supervised_results["best_model_name"],
            "feature_importances": supervised_results["feature_importances"],
        }, f, indent=2)

    # Save best model
    best_model_path = MODELS_DIR / "best_model.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(supervised_results["best_model"], f)

    print(f"\n{'=' * 60}")
    print("Training Complete — Artifacts Saved")
    print("=" * 60)
    print(f"  {unsupervised_path}")
    print(f"  {supervised_path}")
    print(f"  {best_model_path}")


if __name__ == "__main__":
    run()
