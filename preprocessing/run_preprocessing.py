"""Stage 2: Preprocessing — raw CSVs to training-ready artifacts.

Reads per-artist CSVs from data/raw/, runs the preprocessing pipeline
(feature selection, genre consolidation, outlier removal, scaling),
and saves outputs to artifacts/preprocessed/.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.pipeline import prepare_training_data


ARTIFACTS_DIR = Path("artifacts/preprocessed")


def run(data_dir="data/raw"):
    """Run preprocessing and save artifacts to disk."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_training_data(data_dir)

    X = data["X"]
    df = data["df"]
    scaler = data["scaler"]
    audio_features = data["audio_features"]

    # Save scaled feature matrix
    features_path = ARTIFACTS_DIR / "features.npy"
    np.save(features_path, X)

    # Save cleaned metadata DataFrame
    metadata_path = ARTIFACTS_DIR / "metadata.csv"
    df.to_csv(metadata_path, index=False)

    # Save fitted scaler
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save config (feature names, params, timestamp)
    config_path = ARTIFACTS_DIR / "config.json"
    config = {
        "audio_features": audio_features,
        "feature_names": data["feature_names"],
        "n_tracks": X.shape[0],
        "n_features": X.shape[1],
        "data_dir": str(data_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print(f"  features.npy   ({X.shape[0]} x {X.shape[1]})")
    print(f"  metadata.csv   ({len(df)} rows)")
    print(f"  scaler.pkl")
    print(f"  config.json")

    return data


if __name__ == "__main__":
    run()
