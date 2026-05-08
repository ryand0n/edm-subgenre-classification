"""Single-artist data collection entry point.

Collects audio features for a single artist via Spotify + ReccoBeats APIs
and writes a CSV to data/raw/.
"""

import os

from data_collection import initialize_token, collect_artist_safe


if __name__ == "__main__":
    initialize_token()
    os.makedirs("data/raw", exist_ok=True)

    # Collect all unique tracks from Ninajirachi
    collect_artist_safe("Ninajirachi", "data/raw/ninajirachi.csv")
