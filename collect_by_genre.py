"""
Collect audio features for EDM tracks labeled by genre via Spotify playlists.

For each unique genre in data/genres.csv, searches Spotify for playlists
matching that genre, pulls tracks, fetches audio features from ReccoBeats,
and writes a single CSV with per-track genre labels.
"""

import os
import csv
import time
from util import (
    get_spotify_token,
    search_playlists,
    get_playlist_tracks,
    get_reccobeats_track_ids,
    get_reccobeats_audio_features,
)


def load_env():
    """Load .env file into environment."""
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ[key] = val


def get_unique_genres(genres_csv="data/genres.csv"):
    """Extract unique genre labels from genres.csv."""
    genres = set()
    with open(genres_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["genre"].strip()
            if raw and raw != "N/A":
                for g in raw.split(", "):
                    genres.add(g.strip())
    return sorted(genres)


def fetch_audio_features_batch(tracks, seen_ids):
    """
    Fetch ReccoBeats audio features for a list of tracks.
    Skips tracks already in seen_ids. Returns list of row dicts.
    """
    # Filter to unseen tracks
    new_tracks = [t for t in tracks if t["id"] not in seen_ids]
    if not new_tracks:
        return []

    # Look up in ReccoBeats (batch by 40)
    spotify_to_recco = {}
    for i in range(0, len(new_tracks), 40):
        batch = new_tracks[i:i + 40]
        batch_ids = [t["id"] for t in batch]
        mapping = get_reccobeats_track_ids(batch_ids)
        spotify_to_recco.update(mapping)

    # Fetch audio features
    rows = []
    for track in new_tracks:
        recco_id = spotify_to_recco.get(track["id"])
        if not recco_id:
            continue

        features = get_reccobeats_audio_features(recco_id)
        if features is None:
            continue

        rows.append({
            "track_name": track["name"],
            "track_id": track["id"],
            "artist_name": track["artist_name"],
            "album_name": track["album_name"],
            "danceability": features.get("danceability"),
            "energy": features.get("energy"),
            "key": features.get("key"),
            "loudness": features.get("loudness"),
            "mode": features.get("mode"),
            "speechiness": features.get("speechiness"),
            "acousticness": features.get("acousticness"),
            "instrumentalness": features.get("instrumentalness"),
            "liveness": features.get("liveness"),
            "valence": features.get("valence"),
            "tempo": features.get("tempo"),
        })

    return rows


def collect_all_genres(
    genres_csv="data/genres.csv",
    output_file="data/genre_tracks.csv",
    playlists_per_genre=2,
):
    """
    Main collection loop: for each genre, find playlists, pull tracks,
    fetch audio features, write to CSV.
    """
    load_env()

    # Authenticate
    token_data = get_spotify_token(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"])
    access_token = token_data["access_token"]
    token_time = time.time()

    genres = get_unique_genres(genres_csv)
    print(f"Found {len(genres)} unique genres to process\n")

    seen_track_ids = set()  # For deduplication (keep first seen)
    all_rows = []
    genre_stats = {}

    for i, genre in enumerate(genres):
        # Refresh token if older than 50 minutes
        if time.time() - token_time > 3000:
            print("Refreshing Spotify token...")
            token_data = get_spotify_token(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"])
            access_token = token_data["access_token"]
            token_time = time.time()

        print(f"[{i + 1}/{len(genres)}] {genre}")

        # Search for playlists
        playlists = search_playlists(genre, access_token, limit=playlists_per_genre)
        if not playlists:
            print(f"  No playlists found, skipping")
            genre_stats[genre] = {"playlists": 0, "tracks": 0, "with_features": 0}
            continue

        playlist_names = [f"'{p['name']}' ({p['track_count']} tracks)" for p in playlists]
        print(f"  Playlists: {', '.join(playlist_names)}")

        # Collect tracks from all playlists for this genre
        genre_tracks = []
        for pl in playlists:
            tracks = get_playlist_tracks(pl["id"], access_token)
            # Filter to tracks we haven't seen (keep first genre assignment)
            new_tracks = [t for t in tracks if t["id"] not in seen_track_ids]
            genre_tracks.extend(new_tracks)

        if not genre_tracks:
            print(f"  No new tracks (all duplicates)")
            genre_stats[genre] = {"playlists": len(playlists), "tracks": 0, "with_features": 0}
            continue

        print(f"  {len(genre_tracks)} new tracks to process")

        # Fetch audio features
        rows = fetch_audio_features_batch(genre_tracks, seen_track_ids)

        # Add genre label and mark as seen
        for row in rows:
            row["genre"] = genre
            seen_track_ids.add(row["track_id"])
        # Also mark tracks without features as seen to avoid re-processing
        for t in genre_tracks:
            seen_track_ids.add(t["id"])

        all_rows.extend(rows)
        genre_stats[genre] = {
            "playlists": len(playlists),
            "tracks": len(genre_tracks),
            "with_features": len(rows),
        }
        print(f"  Got audio features for {len(rows)}/{len(genre_tracks)} tracks")

    # Write CSV
    if all_rows:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fieldnames = [
            "track_name", "track_id", "artist_name", "album_name", "genre",
            "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo",
        ]
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} tracks to {output_file}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(all_rows)} total tracks across {len(genres)} genres")
    print(f"{'=' * 60}")
    for genre in genres:
        stats = genre_stats.get(genre, {})
        pl = stats.get("playlists", 0)
        tr = stats.get("tracks", 0)
        feat = stats.get("with_features", 0)
        print(f"  {genre:30s}  {pl} playlists  {tr:5d} tracks  {feat:5d} w/ features")


if __name__ == "__main__":
    collect_all_genres()
