import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# Features to drop: redundant (correlated with energy), non-discriminative, or categorical
DROP_FEATURES = ["loudness", "acousticness", "liveness", "key", "mode"]

# Genre consolidation mapping: granular subgenres -> broader families
GENRE_CONSOLIDATION = {
    # Hardcore family
    "hardcore techno": "hardcore",
    "frenchcore": "hardcore",
    "gabber": "hardcore",
    "speedcore": "hardcore",
    "happy hardcore": "hardcore",
    "hardcore": "hardcore",
    "j-dance": "hardcore",
    # Hardstyle stays distinct
    "hardstyle": "hardstyle",
    # Techno family
    "techno": "techno",
    "hard techno": "techno",
    "acid techno": "techno",
    "hypertechno": "techno",
    "tekno": "techno",
    # Bass music family
    "dubstep": "bass music",
    "riddim": "bass music",
    "deathstep": "bass music",
    "bass music": "bass music",
    "melodic bass": "bass music",
    "chillstep": "bass music",
    # House family
    "bass house": "house",
    "bassline": "house",
    "stutter house": "house",
    "melbourne bounce": "house",
    "big room": "house",
    # Trap
    "edm trap": "trap",
    # Future bass
    "future bass": "future bass",
    # Electronic / pop-electronic
    "edm": "edm",
    "hyperpop": "hyperpop",
    "electroclash": "electroclash",
    "witch house": "witch house",
}


def load_raw_data(data_dir="data"):
    """Load all artist CSVs and merge genre labels from genres.csv.

    Drops rows with no genre (NaN or 'N/A') since we need labels for
    post-training cluster validation.
    """
    data_dir = Path(data_dir)
    genres_csv = data_dir / "genres.csv"

    dfs = []
    for csv_file in sorted(data_dir.glob("*.csv")):
        if csv_file.name == "genres.csv":
            continue
        df = pd.read_csv(csv_file)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    # Drop any existing genre column from artist CSVs to avoid merge conflicts
    if "genre" in combined.columns:
        combined = combined.drop(columns="genre")

    genres_df = pd.read_csv(genres_csv)
    genres_df["join_key"] = genres_df["artist_name"].str.lower()
    combined["join_key"] = combined["artist_name"].str.lower()
    combined = combined.merge(genres_df[["join_key", "genre"]], on="join_key", how="left")
    combined = combined.drop(columns="join_key").rename(columns={"genre": "genres"})

    # Drop rows without genre labels
    before = len(combined)
    combined = combined[combined["genres"].notna() & (combined["genres"] != "N/A")]
    combined = combined.reset_index(drop=True)
    dropped = before - len(combined)
    print(f"Loaded {len(combined)} tracks ({dropped} dropped — no genre label)")

    return combined


def select_features(df):
    """Drop non-discriminative and redundant features.

    Removes: loudness (correlated with energy), acousticness (correlated with
    energy), liveness (low between-genre variance), key (categorical),
    mode (binary categorical).
    """
    existing_drops = [f for f in DROP_FEATURES if f in df.columns]
    df = df.drop(columns=existing_drops)

    kept = df.select_dtypes(include=np.number).columns.tolist()
    print(f"Dropped {existing_drops} -> kept {kept}")
    return df


def consolidate_genres(df):
    """Add genres_consolidated column by mapping granular subgenres to families."""

    def _consolidate(genre_str):
        if pd.isna(genre_str) or genre_str == "N/A":
            return genre_str
        genres = [g.strip() for g in genre_str.split(",")]
        consolidated = list(dict.fromkeys(
            GENRE_CONSOLIDATION.get(g, g) for g in genres
        ))
        return ", ".join(consolidated)

    df = df.copy()
    df["genres_consolidated"] = df["genres"].apply(_consolidate)

    n_original = df["genres"].str.split(", ").explode().nunique()
    n_consolidated = df["genres_consolidated"].str.split(", ").explode().nunique()
    print(f"Genre consolidation: {n_original} -> {n_consolidated} unique genres")
    return df


def remove_outliers(df, features, threshold=1.5):
    """Remove outlier tracks using IQR method across audio features.

    A track is removed if any of its audio features fall outside
    Q1 - threshold*IQR or Q3 + threshold*IQR.
    """
    outlier_mask = pd.Series(False, index=df.index)
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outlier_mask |= (df[feature] < lower) | (df[feature] > upper)

    n_outliers = outlier_mask.sum()
    print(f"Removed {n_outliers} outlier tracks ({n_outliers/len(df)*100:.1f}%) "
          f"[IQR threshold={threshold}]")
    return df[~outlier_mask].reset_index(drop=True)


def scale_features(df, features):
    """StandardScaler on audio features. Returns scaled array + fitted scaler."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].values)
    print(f"Scaled {len(features)} features: mean={X.mean(axis=0).round(6).tolist()}, "
          f"std={X.std(axis=0).round(4).tolist()}")
    return X, scaler


def prepare_training_data(data_dir="data"):
    """Main entry point: raw CSVs -> training-ready dataset.

    Returns dict with:
        X              - scaled feature matrix (numpy array)
        df             - cleaned DataFrame with metadata (before scaling)
        scaler         - fitted StandardScaler
        feature_names  - all column names in cleaned df
        audio_features - list of audio feature names used for training
    """
    print("=" * 60)
    print("Data Preparation Pipeline")
    print("=" * 60)

    df = load_raw_data(data_dir)
    df = select_features(df)
    df = consolidate_genres(df)

    audio_features = df.select_dtypes(include=np.number).columns.tolist()
    df = remove_outliers(df, audio_features)

    X, scaler = scale_features(df, audio_features)

    print("=" * 60)
    print(f"Final: {X.shape[0]} tracks x {X.shape[1]} features")
    print(f"Audio features: {audio_features}")
    print("=" * 60)

    return {
        "X": X,
        "df": df,
        "scaler": scaler,
        "feature_names": df.columns.tolist(),
        "audio_features": audio_features,
    }
