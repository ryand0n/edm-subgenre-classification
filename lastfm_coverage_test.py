"""
Last.fm track-level genre tag coverage test.

Samples ~100 tracks across all genres and measures Last.fm tag coverage
to decide whether per-track genre tagging is viable.
"""

import os
import csv
import random
from collections import Counter
from util import get_lastfm_tags


def load_env_file(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ[key] = val


def load_genres():
    """Load artist -> genre mapping from genres.csv."""
    genres = {}
    with open("data/genres.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genres[row["artist_name"].lower()] = row["genre"]
    return genres


def load_all_tracks():
    """Load all tracks from artist CSVs, keyed by artist name."""
    tracks_by_artist = {}
    data_dir = "data"
    for filename in os.listdir(data_dir):
        if filename == "genres.csv" or not filename.endswith(".csv"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                artist = rows[0].get("artist_name", "")
                if artist:
                    tracks_by_artist[artist] = rows
    return tracks_by_artist


# Genre-relevant keywords: if a tag (lowercased) contains any of these, it's genre-relevant
GENRE_KEYWORDS = [
    "house", "techno", "trance", "bass", "step", "core",
    "drum", "dnb", "d&b", "d'n'b", "jungle",
    "electro", "edm", "dance", "rave",
    "dub", "garage", "grime",
    "ambient", "chill", "downtempo", "lounge",
    "industrial", "acid", "minimal", "progressive",
    "breakbeat", "breaks", "big beat",
    "psytrance", "psy", "goa",
    "disco", "funk", "synthwave", "synthpop", "synth",
    "bounce", "hardstyle", "gabber", "speed",
    "trap", "riddim", "wobble", "brostep",
    "future", "melodic", "deep",
    "electronic", "club",
    "hyperpop", "witch",
]

# Noise tags to filter out (exact match, lowercased)
NOISE_TAGS = {
    "seen live", "favorites", "favourite", "favourite songs",
    "favorite", "favorite songs", "love", "loved",
    "awesome", "amazing", "beautiful", "cool",
    "my music", "check out", "under 2000 listeners",
    "spotify", "good", "great", "epic", "best",
    "male vocalists", "female vocalists",
    "sexy", "catchy", "fun", "chill", "relax",
    "summer", "party", "workout", "driving",
    "00s", "10s", "90s", "80s", "70s",
    "2024", "2023", "2022", "2021", "2020",
}


def is_genre_relevant(tag_name):
    """Check if a tag name is genre-relevant based on keyword matching."""
    lower = tag_name.lower()
    if lower in NOISE_TAGS:
        return False
    return any(kw in lower for kw in GENRE_KEYWORDS)


def sample_tracks(tracks_by_artist, genres, target=100, per_artist=2):
    """Sample tracks spread across artists and genres."""
    # Group artists by their primary genre family
    genre_artists = {}
    for artist, tracks in tracks_by_artist.items():
        genre = genres.get(artist.lower(), "N/A")
        if genre == "N/A":
            continue
        primary_genre = genre.split(",")[0].strip()
        genre_artists.setdefault(primary_genre, []).append((artist, tracks))

    sampled = []
    # Shuffle genre order for variety
    genre_list = list(genre_artists.items())
    random.shuffle(genre_list)

    for genre, artists in genre_list:
        for artist, tracks in artists:
            if len(sampled) >= target:
                break
            n = min(per_artist, len(tracks))
            picked = random.sample(tracks, n)
            for t in picked:
                sampled.append({
                    "track_name": t["track_name"],
                    "artist_name": t["artist_name"],
                    "genre": genres.get(artist.lower(), "N/A"),
                })
        if len(sampled) >= target:
            break

    return sampled[:target]


def run_test():
    load_env_file()
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        print("Error: LASTFM_API_KEY not set in .env file")
        print("Get one at https://www.last.fm/api/account/create")
        return

    random.seed(42)

    print("Loading data...")
    genres = load_genres()
    tracks_by_artist = load_all_tracks()
    print(f"  {len(tracks_by_artist)} artists, {sum(len(t) for t in tracks_by_artist.values())} total tracks")

    print("\nSampling tracks...")
    sampled = sample_tracks(tracks_by_artist, genres)
    print(f"  Selected {len(sampled)} tracks across genres")

    # Fetch tags
    print("\nFetching Last.fm tags...")
    results = []
    for i, track in enumerate(sampled):
        tags = get_lastfm_tags(track["track_name"], track["artist_name"], api_key)
        genre_tags = [t for t in tags if is_genre_relevant(t["name"])]
        noise_tags = [t for t in tags if not is_genre_relevant(t["name"])]
        results.append({
            **track,
            "all_tags": tags,
            "genre_tags": genre_tags,
            "noise_tags": noise_tags,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(sampled)} tracks processed...")

    print(f"  Done! {len(results)} tracks processed.\n")

    # --- Report ---
    print("=" * 70)
    print("LAST.FM TAG COVERAGE REPORT")
    print("=" * 70)

    total = len(results)
    has_any_tags = sum(1 for r in results if r["all_tags"])
    has_genre_tags = sum(1 for r in results if r["genre_tags"])

    print(f"\nTotal tracks sampled: {total}")
    print(f"Tracks with any tags: {has_any_tags} ({100 * has_any_tags / total:.1f}%)")
    print(f"Tracks with genre-relevant tags: {has_genre_tags} ({100 * has_genre_tags / total:.1f}%)")

    # Most common genre tags
    genre_counter = Counter()
    for r in results:
        for t in r["genre_tags"]:
            genre_counter[t["name"].lower()] += 1

    print(f"\nTop 20 genre-relevant tags:")
    for tag, count in genre_counter.most_common(20):
        print(f"  {tag:40s} {count}")

    # Most common noise tags
    noise_counter = Counter()
    for r in results:
        for t in r["noise_tags"]:
            noise_counter[t["name"].lower()] += 1

    print(f"\nTop 20 non-genre tags (potential noise):")
    for tag, count in noise_counter.most_common(20):
        print(f"  {tag:40s} {count}")

    # Per-genre breakdown
    print(f"\nPer-genre coverage:")
    genre_groups = {}
    for r in results:
        primary = r["genre"].split(",")[0].strip()
        genre_groups.setdefault(primary, []).append(r)

    for genre in sorted(genre_groups.keys()):
        tracks = genre_groups[genre]
        n = len(tracks)
        covered = sum(1 for t in tracks if t["genre_tags"])
        pct = 100 * covered / n if n else 0
        print(f"  {genre:30s} {covered}/{n} ({pct:.0f}%)")

    # Example tracks for manual inspection
    print(f"\nExample tracks with full tag lists:")
    examples = random.sample(results, min(10, len(results)))
    for ex in examples:
        print(f"\n  '{ex['track_name']}' by {ex['artist_name']} [{ex['genre']}]")
        if ex["all_tags"]:
            tag_strs = [f"{t['name']}({t['count']})" for t in ex["all_tags"][:15]]
            print(f"    Tags: {', '.join(tag_strs)}")
        else:
            print(f"    Tags: (none)")
        if ex["genre_tags"]:
            genre_strs = [t["name"] for t in ex["genre_tags"]]
            print(f"    Genre-relevant: {', '.join(genre_strs)}")

    # Decision guidance
    print(f"\n{'=' * 70}")
    if has_genre_tags / total >= 0.6:
        print(f"RESULT: {100 * has_genre_tags / total:.0f}% genre-relevant coverage — PROCEED with Last.fm collection")
    else:
        print(f"RESULT: {100 * has_genre_tags / total:.0f}% genre-relevant coverage — consider pivoting to playlist mapping")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
