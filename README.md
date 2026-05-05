# EDM Subgenre Classification

Can unsupervised clustering recover EDM subgenre boundaries from audio features alone?

## Hypothesis

Electronic dance music subgenres (hardstyle, dubstep, techno, house, etc.) have distinct audio fingerprints — characteristic ranges of tempo, energy, danceability, and other measurable features. If these fingerprints are strong enough, an unsupervised clustering algorithm like K-Means should naturally group tracks into clusters that align with known genre labels, without ever seeing those labels during training.

## How the Experiment Was Conducted

### 1. Data Collection

Audio features were collected for ~18,500 tracks across 284 artists spanning 25+ EDM subgenres.

- **Spotify API** provided artist/album/track metadata and Spotify track IDs
- **ReccoBeats API** provided audio features (danceability, energy, tempo, valence, etc.) using those track IDs
- Genre labels were manually assigned per artist in `data/genres.csv` (many artists have multiple genre tags, e.g. "dubstep, riddim")
- Each artist's tracks were written to individual CSVs in `data/`

**Scripts:** `util.py` (API integration), `data_collection.py` (orchestration with token refresh)

### 2. Exploratory Data Analysis

The notebook (`notebooks/eda.ipynb`) explored the raw data before any modeling:

- **Feature distributions** — bar charts of top/bottom artists by danceability, energy, loudness, tempo
- **Correlation heatmap** — identified redundant feature pairs (energy-loudness r=0.65, energy-acousticness r=-0.52)
- **Box plots by genre** — showed how each audio feature distributes within each subgenre
- **Radar charts by genre** — visualized each genre's average audio profile as a fingerprint
- **Scatter plots** — plotted tracks in 2D feature pairs colored by genre to preview separability
- **PCA projection** — reduced all features to 2D to see if genres form visible groups
- **Feature variance by genre** — identified which genres are tightly vs. loosely defined per feature

### 3. Data Preparation Pipeline (`pipeline.py`)

The pipeline transforms raw CSVs into a training-ready dataset:

1. **Load & merge** — combines all artist CSVs, joins genre labels from `genres.csv`, drops tracks with no genre label (403 dropped)
2. **Feature selection** — drops 5 features that are redundant, non-discriminative, or categorical:
   - `loudness` — highly correlated with energy (r=0.65), redundant
   - `acousticness` — highly correlated with energy (r=-0.52), redundant
   - `liveness` — low between-genre variance, not useful for discrimination
   - `key` — categorical (pitch class), not meaningful for distance-based clustering
   - `mode` — binary (major/minor), limited signal
   - **Remaining 6 features:** danceability, energy, speechiness, instrumentalness, valence, tempo
3. **Genre consolidation** — maps 25+ granular subgenres to broader families (e.g. dubstep/riddim/deathstep -> "bass music", frenchcore/gabber/speedcore -> "hardcore") to reduce label noise
4. **Outlier removal** — IQR method (1.5x threshold) removes tracks with extreme feature values (~19.7%), which would distort cluster centroids
5. **Feature scaling** — StandardScaler normalizes all features to zero mean and unit variance so no single feature (like tempo at ~130) dominates distance calculations

**Output:** 14,531 tracks x 6 features (scaled), plus the cleaned DataFrame with metadata and consolidated genre labels.

### 4. K-Means Training & Evaluation (`train.py`)

1. **Optimal k search** — ran K-Means for k=2 through k=15, tracking inertia (elbow method) and silhouette score for each
2. **Multi-k comparison** — trained K-Means at k=2, 6, 10, and 15 to compare metrics across scales
3. **Cluster-genre evaluation** — computed ARI and NMI to measure how well clusters align with genre labels
4. **Cross-tabulation heatmap** — shows what percentage of each genre lands in each cluster
5. **Cluster profiling** — inverse-transformed centroids back to original units to interpret what each cluster represents sonically
6. **PCA visualization** — side-by-side 2D scatter plots colored by cluster vs. colored by genre for direct visual comparison

## Results

### Unsupervised Clustering (K-Means)

#### Evaluation Metrics (class-balanced)

The unsupervised experiment downsamples to the median genre count before clustering to prevent dominant genres (house: 3,255 tracks) from dominating cluster assignments.

| k | Silhouette | ARI | NMI |
|---|-----------|-----|-----|
| 2 | 0.206 | 0.021 | 0.083 |
| 10 | 0.191 | 0.056 | 0.166 |
| 15 | 0.181 | 0.056 | 0.182 |

- **Silhouette scores (0.18–0.21)** are low-positive across all k values, indicating the audio feature space doesn't contain tight, well-separated clusters. Tracks from different clusters overlap significantly.
- **ARI scores (0.02–0.06)** are barely above zero (where zero = random assignment). Clusters do not meaningfully reproduce genre boundaries.
- **NMI scores (0.08–0.18)** show that knowing a track's cluster only weakly predicts its genre. NMI increases with k because more clusters can capture finer distinctions, but returns are diminishing.

#### Genre-Cluster Heatmap

The cross-tabulation heatmap confirms what the low ARI suggests: most genres are spread across multiple clusters rather than concentrated in one. The model is partitioning the space, but not along genre lines.

#### Cluster Profiles

Despite poor genre alignment, the clusters are **musically interpretable**. The centroids reveal distinct audio archetypes:

- **High-tempo, high-energy, instrumental clusters** (~159 BPM) capture the hardcore/hardstyle zone
- **Low-tempo, high-energy clusters** (~103 BPM) capture half-time bass music (dubstep/riddim)
- **Mid-tempo, high-danceability, high-valence clusters** (~126 BPM) capture house/pop-EDM
- **Mid-tempo, high-instrumentalness, low-energy clusters** capture deep/minimal techno
- **High-speechiness clusters** separate vocal-heavy tracks from instrumentals

K-Means found real structure in the data — it correctly separates fast from slow, dark from bright, vocal from instrumental. These are valid audio dimensions, but they don't map to genre labels.

#### PCA Visualization

The side-by-side PCA scatter plots make the fundamental issue visible: the cluster-colored plot shows clean spatial partitions, but the genre-colored plot shows genres spread across those same regions with heavy overlap.

### Supervised Classification

Three supervised models were trained on the same 6 audio features with class-imbalance handling:

- **RandomForest** — `class_weight="balanced"` (inversely weights each class by frequency)
- **GradientBoosting** — balanced via `sample_weight` (computed from sklearn's `compute_sample_weight("balanced")`)
- **XGBoost** — same `sample_weight` approach

All use stratified 80/20 train/test splits. The dataset has severe class imbalance (75.7x ratio between largest and smallest genre), so **balanced accuracy** (macro-averaged recall across all classes) is the primary metric.

#### Model Comparison

| Model | Accuracy | Balanced Accuracy |
|-------|----------|-------------------|
| RandomForest | 0.432 | 0.228 |
| GradientBoosting | **0.328** | **0.332** |
| XGBoost | 0.388 | 0.261 |

- **GradientBoosting** achieves the best balanced accuracy (0.332), meaning it's most equitable across all genres including rare ones. The tradeoff is lower raw accuracy since it sacrifices performance on dominant genres to improve recall on smaller ones.
- **RandomForest** has the highest raw accuracy (0.432) but the lowest balanced accuracy — it's biased toward majority classes.
- All models are well below 50% balanced accuracy, confirming that 6 audio features are insufficient for reliable genre classification regardless of the algorithm.

#### Feature Importances

| Feature | Importance |
|---------|-----------|
| tempo | 0.381 |
| danceability | 0.149 |
| instrumentalness | 0.136 |
| valence | 0.133 |
| energy | 0.109 |
| speechiness | 0.091 |

Tempo is by far the most discriminative feature (~38% importance), which makes sense — it's the one audio feature that hard-separates certain genres (hardstyle at ~150 BPM vs. dubstep at ~140 half-time vs. house at ~128). The remaining features contribute roughly equally.

## Conclusion

The hypothesis that unsupervised clustering would recover EDM subgenre boundaries from audio features was **not supported**. K-Means successfully identified musically meaningful audio archetypes, but these archetypes don't correspond to genre labels. Supervised classification (RandomForest, GradientBoosting, XGBoost) with class-imbalance handling confirms the ceiling: the best model achieves only 33% balanced accuracy across 22 genre classes — better than the ~4.5% random baseline, but far from usable.

The core issue is that genre identity in EDM is defined by attributes these audio features can't capture: sound design (the difference between a dubstep wobble bass and a hardstyle kick), production techniques, drop structure, cultural context, and scene affiliation. A dubstep track and a hardstyle track can have similar energy and tempo values but sound completely different because of *how* the bass and drums are constructed — and that information isn't encoded in high-level features like danceability or valence.

These 6 audio features describe *what* a track feels like (fast/slow, happy/dark, vocal/instrumental) but not *how* it sounds — and genre boundaries in EDM are drawn along the "how," not the "what."

## Future Work

### Quick Wins (same data, different methods)

- **Track-level genre labels via playlist mapping** — genres are currently assigned per artist, so every track by an artist gets the same label even if their catalog spans multiple styles. Collecting Spotify playlist IDs curated by subgenre and tagging each track with its playlist's genre would provide per-track labels and reduce label noise significantly.
- **More aggressive genre consolidation** — the current 22 classes after consolidation still include genres that are nearly impossible to separate by audio features alone (e.g. melodic house vs. progressive house). Reducing to 8-10 broader families would improve both supervised and unsupervised performance.
- **Ensemble or stacking** — combine the three supervised models (RF, GBM, XGBoost) via soft-voting or stacking to extract marginal gains from model diversity.

### Medium Effort (new features, same APIs)

- **Spotify audio analysis** — Spotify provides beat-level timing, section boundaries (intro/verse/drop/outro), loudness curves over time, and timbral vectors per segment. From this, we could engineer structural features like "energy jump at drop" (loudness difference between buildup and drop), "drop frequency" (number of high-energy sections per track), and "rhythmic regularity" (beat timing variance). This gets closer to drop structure without needing raw audio files.
- **Re-add dropped features conditionally** — loudness and acousticness were dropped for being correlated with energy, but supervised models can handle correlated features. Re-introducing them (possibly with the dropped key/mode as categorical inputs) could provide extra signal for tree-based classifiers.

### Raw Audio Analysis (highest effort, highest potential)

- **Spectral features** — MFCCs, spectral centroid, spectral rolloff, and chroma features capture timbre — the quality that makes a reese bass sound different from a supersaw, even at the same pitch and volume. Extractable from audio files via `librosa`. Would add ~20-30 features describing *how* a track sounds.
- **Mel spectrogram + deep learning** — convert audio to mel spectrograms and use a CNN to learn visual patterns (the "wall of bass" in dubstep drops vs. sharp transient kicks in hardstyle). Pre-trained audio models like OpenL3, VGGish, or CLAP can produce embeddings without training from scratch.
- **Drop and structure quantification** — detect drop boundaries via sudden energy/spectral shifts, then characterize each drop by its spectral content, bass frequency distribution, rhythmic pattern, and transient sharpness. Most genre-specific and most powerful, but requires significant domain-specific feature engineering.

All raw audio approaches require actual audio files (Spotify 30-second previews, purchased tracks, or YouTube), which means a different data pipeline entirely.

## Project Structure

```
data_collection.py      # Data collection entry point
util.py                 # Spotify & ReccoBeats API integration
pipeline.py             # Data preparation pipeline
train.py                # K-Means training & evaluation functions
train_unsupervised.py   # Standalone K-Means experiment (class-balanced)
train_supervised.py     # Standalone supervised classification (RF, GBM, XGBoost)
notebooks/eda.ipynb     # EDA + clustering analysis notebook
data/                   # Artist CSVs + genres.csv
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib scikit-learn xgboost requests python-dotenv jupyter
```

Requires a `.env` file with Spotify API credentials:
```
CLIENT_ID=your_spotify_client_id
CLIENT_SECRET=your_spotify_client_secret
```

## Usage

```bash
# Collect audio features for an artist
python data_collection.py

# Run unsupervised K-Means experiment (terminal output)
python train_unsupervised.py

# Run supervised classification experiment (terminal output)
python train_supervised.py

# Run the full pipeline + clustering from the notebook
jupyter notebook notebooks/eda.ipynb
```
