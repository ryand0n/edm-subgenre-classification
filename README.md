# EDM Subgenre Classification

A data pipeline for collecting audio features from Spotify/ReccoBeats and training ML models to classify electronic dance music subgenres.

The project explores whether measurable audio features (tempo, energy, danceability, etc.) can distinguish EDM subgenres like hardstyle, dubstep, techno, and house. See [EXPERIMENT.md](EXPERIMENT.md) for the full experiment writeup, results, and conclusions.

## Setup

```bash
python -m venv venv_linux
source venv_linux/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Spotify API credentials:

```
CLIENT_ID=your_spotify_client_id
CLIENT_SECRET=your_spotify_client_secret
```

## Usage

### Full Pipeline

```bash
python run_pipeline.py                 # All stages
python run_pipeline.py --stage 1       # Collection only
python run_pipeline.py --stage 2       # Preprocessing only
python run_pipeline.py --stage 3       # Training only
python run_pipeline.py --stage 2 3     # Preprocessing + Training
```

### Individual Scripts

```bash
# Collect audio features for all EDC 2026 artists
python -m data_collection.collect_edc_2026

# Collect audio features for a single artist
python -m data_collection.collect

# Run supervised classification standalone
python -m model_training.train_supervised

# Run unsupervised clustering standalone
python -m model_training.train_unsupervised

# Run analysis notebook
jupyter notebook notebooks/eda.ipynb
```

## Pipeline Stages

The pipeline has three stages with disk-based contracts between them:

### Stage 1: Data Collection

Fetches audio features for 284 EDC 2026 artists via the Spotify and ReccoBeats APIs. Skips automatically if `data/raw/` already has CSV files.

- **Input:** `.env` (Spotify credentials)
- **Output:** `data/raw/*.csv` (one per artist), `data/raw/genres.csv`

### Stage 2: Preprocessing

Loads raw CSVs, selects features, consolidates genre labels, removes outliers, and scales features.

- **Input:** `data/raw/*.csv`, `data/raw/genres.csv`
- **Output:**
  - `artifacts/preprocessed/features.npy` — scaled feature matrix
  - `artifacts/preprocessed/metadata.csv` — cleaned DataFrame with metadata
  - `artifacts/preprocessed/scaler.pkl` — fitted StandardScaler
  - `artifacts/preprocessed/config.json` — feature names, parameters, timestamp

### Stage 3: Model Training

Loads preprocessed artifacts and trains both unsupervised (K-Means) and supervised (RandomForest, GradientBoosting, XGBoost) models.

- **Input:** `artifacts/preprocessed/*`
- **Output:**
  - `artifacts/models/unsupervised_results.json`
  - `artifacts/models/supervised_results.json`
  - `artifacts/models/best_model.pkl`

## Project Structure

```
├── run_pipeline.py              # Top-level orchestrator
├── util.py                      # Spotify & ReccoBeats API integration
├── data_collection/
│   ├── __init__.py              # Shared auth/collection helpers
│   ├── collect.py               # Single-artist collection
│   ├── collect_edc_2026.py      # Batch collection (284 artists)
│   └── run_collection.py        # Stage 1 orchestrator
├── preprocessing/
│   ├── __init__.py
│   ├── pipeline.py              # Feature selection, genre consolidation, outlier removal, scaling
│   └── run_preprocessing.py     # Stage 2 orchestrator
├── model_training/
│   ├── __init__.py
│   ├── train.py                 # K-Means utilities (find_optimal_k, compare_k_values, PCA)
│   ├── train_supervised.py      # RandomForest, GradientBoosting, XGBoost
│   ├── train_unsupervised.py    # K-Means clustering experiment
│   └── run_training.py          # Stage 3 orchestrator
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis
├── data/
│   └── raw/                     # Per-artist CSV files + genres.csv
└── artifacts/                   # Pipeline outputs (gitignored)
    ├── preprocessed/            # Stage 2 outputs
    └── models/                  # Stage 3 outputs
```

## API Workflow

1. **Spotify API** — get artist/album/track metadata and Spotify track IDs
2. **ReccoBeats `/v1/track?ids=`** — look up Spotify IDs to get ReccoBeats IDs
3. **ReccoBeats `/v1/track/{id}/audio-features`** — get audio features using ReccoBeats IDs

ReccoBeats has ~70-80% coverage for most artists. Tracks not in their database are logged and skipped.

## Audio Features

| Feature | Description |
|---------|-------------|
| danceability | How suitable a track is for dancing (0-1) |
| energy | Perceptual intensity and activity (0-1) |
| speechiness | Presence of spoken words (0-1) |
| instrumentalness | Likelihood of no vocal content (0-1) |
| valence | Musical positiveness — happy vs. sad (0-1) |
| tempo | Estimated BPM |

Additional features collected but dropped during preprocessing (redundant or non-discriminative): key, mode, loudness, acousticness, liveness.
