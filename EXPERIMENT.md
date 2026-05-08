# ML Experiment: EDM Subgenre Classification

Can unsupervised clustering recover EDM subgenre boundaries from audio features alone?

## Hypothesis

Electronic dance music subgenres (hardstyle, dubstep, techno, house, etc.) have distinct audio fingerprints — characteristic ranges of tempo, energy, danceability, and other measurable features. If these fingerprints are strong enough, an unsupervised clustering algorithm like K-Means should naturally group tracks into clusters that align with known genre labels, without ever seeing those labels during training.

## How the Experiment Was Conducted

### 1. Data Collection

Audio features were collected for ~18,500 tracks across 284 artists spanning 25+ EDM subgenres.

- **Spotify API** provided artist/album/track metadata and Spotify track IDs
- **ReccoBeats API** provided audio features (danceability, energy, tempo, valence, etc.) using those track IDs
- Genre labels were manually assigned per artist in `data/raw/genres.csv` (many artists have multiple genre tags, e.g. "dubstep, riddim")
- Each artist's tracks were written to individual CSVs in `data/raw/`

### 2. Exploratory Data Analysis

The notebook (`notebooks/eda.ipynb`) explored the raw data before any modeling:

- **Feature distributions** — bar charts of top/bottom artists by danceability, energy, loudness, tempo
- **Correlation heatmap** — identified redundant feature pairs (energy-loudness r=0.65, energy-acousticness r=-0.52)
- **Box plots by genre** — showed how each audio feature distributes within each subgenre
- **Radar charts by genre** — visualized each genre's average audio profile as a fingerprint
- **Scatter plots** — plotted tracks in 2D feature pairs colored by genre to preview separability
- **PCA projection** — reduced all features to 2D to see if genres form visible groups
- **Feature variance by genre** — identified which genres are tightly vs. loosely defined per feature

### 3. Data Preparation Pipeline

The pipeline (`preprocessing/pipeline.py`) transforms raw CSVs into a training-ready dataset:

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

### 4. K-Means Training & Evaluation

1. **Optimal k search** — ran K-Means for k=2 through k=15, tracking inertia (elbow method) and silhouette score for each
2. **Multi-k comparison** — trained K-Means at k=2, 6, 10, and 15 to compare metrics across scales
3. **Cluster-genre evaluation** — computed ARI and NMI to measure how well clusters align with genre labels
4. **Cross-tabulation heatmap** — shows what percentage of each genre lands in each cluster
5. **Cluster profiling** — inverse-transformed centroids back to original units to interpret what each cluster represents sonically
6. **PCA visualization** — side-by-side 2D scatter plots colored by cluster vs. colored by genre for direct visual comparison

## Results

### Unsupervised Clustering (K-Means)

#### Evaluation Metrics (class-balanced)

The unsupervised experiment downsamples to the median genre count before clustering to prevent dominant genres (house: 3,255 tracks) from dominating cluster assignments. A random assignment baseline (assigning each track to a random cluster) is included for comparison.

| k | Method | Silhouette | ARI | NMI |
|---|--------|-----------|-----|-----|
| 2 | Random Baseline | -0.000 | 0.000 | 0.004 |
| 2 | K-Means | 0.203 | 0.024 | 0.089 |
| 10 | Random Baseline | -0.027 | 0.000 | 0.020 |
| 10 | K-Means | 0.185 | 0.059 | 0.174 |
| 15 | Random Baseline | -0.048 | -0.001 | 0.026 |
| 15 | K-Means | 0.177 | 0.060 | 0.196 |

- **Silhouette scores (0.18–0.20)** are low-positive across all k values, indicating the audio feature space doesn't contain tight, well-separated clusters. Random assignment produces negative silhouette scores, confirming K-Means finds real (if weak) structure.
- **ARI scores (0.02–0.06)** are barely above zero (where zero = random assignment). The random baseline confirms this — ARI ~0.000 for random labels. Clusters do not meaningfully reproduce genre boundaries, but K-Means is consistently better than chance.
- **NMI scores (0.08–0.20)** show that knowing a track's cluster only weakly predicts its genre. NMI increases with k because more clusters can capture finer distinctions, but returns are diminishing. The random baseline NMI (0.004–0.026) confirms K-Means captures real mutual information.

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

All use stratified 80/20 train/test splits. The dataset has severe class imbalance (76.7x ratio between largest and smallest genre), so **balanced accuracy** (macro-averaged recall across all classes) is the primary metric.

A `DummyClassifier(strategy="stratified")` baseline is included — it predicts classes proportionally to training distribution (pure random guessing given class frequencies). Each real model was tuned using `RandomizedSearchCV` with `scoring="balanced_accuracy"`, `cv=3` (stratified k-fold), and `n_iter=20` random parameter combinations.

#### Model Comparison

| Model | Accuracy | Balanced Accuracy |
|-------|----------|-------------------|
| Baseline (Dummy) | 0.117 | 0.046 |
| RandomForest | 0.425 | 0.277 |
| GradientBoosting | 0.297 | **0.323** |
| XGBoost | 0.262 | 0.320 |

- **Baseline** achieves 4.6% balanced accuracy — roughly 1/22 (random chance across 22 classes). All real models substantially outperform this.
- **GradientBoosting** achieves the best balanced accuracy (0.323), meaning it's most equitable across all genres including rare ones. The tradeoff is lower raw accuracy since it sacrifices performance on dominant genres to improve recall on smaller ones.
- **RandomForest** has the highest raw accuracy (0.425) but the lowest balanced accuracy — it's biased toward majority classes despite `class_weight="balanced"`.
- All models are well below 50% balanced accuracy, confirming that 6 audio features are insufficient for reliable genre classification regardless of the algorithm.

#### Feature Importances

| Feature | Importance |
|---------|-----------|
| tempo | 0.368 |
| danceability | 0.147 |
| instrumentalness | 0.139 |
| valence | 0.132 |
| energy | 0.121 |
| speechiness | 0.093 |

Tempo is by far the most discriminative feature (~37% importance), which makes sense — it's the one audio feature that hard-separates certain genres (hardstyle at ~150 BPM vs. dubstep at ~140 half-time vs. house at ~128). The remaining features contribute roughly equally.

## Conclusion

The hypothesis that unsupervised clustering would recover EDM subgenre boundaries from audio features was **not supported**. K-Means successfully identified musically meaningful audio archetypes, but these archetypes don't correspond to genre labels. Random assignment baselines confirm that both K-Means and supervised models are learning real structure — but not enough. Supervised classification with hyperparameter tuning (HalvingRandomSearchCV for RF/GB, RandomizedSearchCV for XGBoost with GPU acceleration) confirms the ceiling: the best tuned model (GradientBoosting) achieves only 32.3% balanced accuracy across 22 genre classes — better than the 4.6% random baseline, but far from usable.

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
