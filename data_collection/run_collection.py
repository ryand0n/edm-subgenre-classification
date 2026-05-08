"""Stage 1: Data Collection — fetch audio features from Spotify/ReccoBeats.

Checks if data/raw/ already has CSV files. If yes, skips collection.
If not, runs batch collection for all EDC 2026 artists.
"""

from pathlib import Path

from data_collection.collect_edc_2026 import collect_all


DATA_DIR = Path("data/raw")


def run():
    """Run data collection if needed."""
    existing_csvs = list(DATA_DIR.glob("*.csv"))
    # Exclude genres.csv from the count
    artist_csvs = [f for f in existing_csvs if f.name != "genres.csv"]

    if artist_csvs:
        print(f"Stage 1: Skipping — {len(artist_csvs)} artist CSVs already exist in {DATA_DIR}/")
        return

    print("Stage 1: No artist CSVs found, starting collection...")
    collect_all(str(DATA_DIR))


if __name__ == "__main__":
    run()
