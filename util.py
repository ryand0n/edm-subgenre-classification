import requests
import os
import csv
import time
from typing import Optional

RECCOBEATS_BASE_URL = "https://api.reccobeats.com/v1"
LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"


def _lastfm_get(params, max_retries=3):
    """Make a GET request to Last.fm API with rate limit retry handling."""
    for attempt in range(max_retries):
        response = requests.get(LASTFM_BASE_URL, params=params)
        if response.status_code == 429:
            retry_after = min(int(response.headers.get("Retry-After", 5)), 60)
            print(f"  Rate limited by Last.fm, waiting {retry_after}s...")
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        time.sleep(0.2)  # Last.fm allows ~5 req/sec
        return response
    # Final attempt without catching
    response = requests.get(LASTFM_BASE_URL, params=params)
    response.raise_for_status()
    return response


def get_lastfm_tags(track_name: str, artist_name: str, api_key: str) -> list[dict]:
    """
    Get top tags for a track from Last.fm.

    Args:
        track_name: Name of the track
        artist_name: Name of the artist
        api_key: Last.fm API key

    Returns:
        list: List of {name, count} tag dicts, or empty list on failure
    """
    params = {
        "method": "track.gettoptags",
        "track": track_name,
        "artist": artist_name,
        "api_key": api_key,
        "autocorrect": 1,
        "format": "json",
    }
    try:
        response = _lastfm_get(params)
        data = response.json()
        tags = data.get("toptags", {}).get("tag", [])
        if isinstance(tags, list):
            return [{"name": t["name"], "count": int(t["count"])} for t in tags]
        return []
    except Exception as e:
        print(f"  Last.fm error for '{track_name}' by '{artist_name}': {e}")
        return []


def _spotify_get(url, headers, params=None, max_retries=3):
    """Make a GET request to Spotify API with rate limit retry handling."""
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            retry_after = min(int(response.headers.get("Retry-After", 5)), 60)
            print(f"  Rate limited by Spotify, waiting {retry_after}s...")
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        time.sleep(0.1)  # Small delay to stay under rate limits
        return response
    # Final attempt without catching
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response

def get_spotify_token(client_id: str, client_secret: str) -> dict:
    """
    Get Spotify access token using client credentials flow.
    Also saves the access token as an environment variable named ACCESS_TOKEN.
    
    Args:
        client_id: Spotify client ID
        client_secret: Spotify client secret
        
    Returns:
        dict: Response containing access token and other metadata
    """
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()  # Raise an exception for bad status codes
    token_data = response.json()
    
    # Save the access token as an environment variable
    os.environ['ACCESS_TOKEN'] = token_data['access_token']
    
    return token_data


def get_reccobeats_track_ids(spotify_track_ids: list[str]) -> dict[str, str]:
    """
    Look up Spotify track IDs in ReccoBeats and return mapping to ReccoBeats IDs.

    Args:
        spotify_track_ids: List of Spotify track IDs

    Returns:
        dict: Mapping of Spotify track ID -> ReccoBeats track ID for found tracks
    """
    if not spotify_track_ids:
        return {}

    url = f"{RECCOBEATS_BASE_URL}/track"
    params = {"ids": ",".join(spotify_track_ids)}

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"ReccoBeats track lookup failed: {response.status_code}")
        return {}

    data = response.json()
    mapping = {}

    for track in data.get("content", []):
        recco_id = track.get("id")
        # Extract Spotify ID from the href field
        href = track.get("href", "")
        if href and recco_id:
            spotify_id = href.split("/")[-1]
            mapping[spotify_id] = recco_id

    return mapping


def get_reccobeats_audio_features(reccobeats_track_id: str) -> Optional[dict]:
    """
    Get audio features for a track from ReccoBeats API using ReccoBeats track ID.

    Args:
        reccobeats_track_id: ReccoBeats track ID (not Spotify ID)

    Returns:
        dict: Audio features if found, None otherwise
    """
    url = f"{RECCOBEATS_BASE_URL}/track/{reccobeats_track_id}/audio-features"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return None
    else:
        print(f"ReccoBeats API error for track {reccobeats_track_id}: {response.status_code}")
        return None


def get_artist_id(artist_name: str, access_token: str) -> Optional[str]:
    """
    Search for an artist by name and return their Spotify ID.

    Args:
        artist_name: Name of the artist to search for
        access_token: Spotify API access token

    Returns:
        str: Artist ID if found, None otherwise
    """
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"q": artist_name, "type": "artist", "limit": 1}

    response = _spotify_get(url, headers, params)
    data = response.json()

    if data.get("artists", {}).get("items"):
        return data["artists"]["items"][0]["id"]
    return None


def get_all_artist_tracks(artist_id: str, access_token: str, max_tracks: int = 0) -> list[dict]:
    """
    Get all unique tracks from an artist across all their releases.

    Args:
        artist_id: Spotify artist ID
        access_token: Spotify API access token
        max_tracks: Stop early once this many unique tracks are found (0 for no limit)

    Returns:
        list: List of unique track dicts with name, id, and album info
    """
    headers = {"Authorization": f"Bearer {access_token}"}

    # Get all albums, singles, and compilations
    albums = []
    url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
    params = {"include_groups": "album,single,compilation", "limit": 50}

    while url:
        response = _spotify_get(url, headers, params)
        data = response.json()
        albums.extend(data["items"])
        url = data.get("next")
        params = None

    print(f"Found {len(albums)} releases")

    # Get tracks from each album
    all_tracks = []
    seen_names = set()  # Track names we've seen (normalized for deduplication)

    for album in albums:
        if max_tracks and len(all_tracks) >= max_tracks:
            break

        album_id = album["id"]
        album_name = album["name"]

        url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
        params = {"limit": 50}

        while url:
            response = _spotify_get(url, headers, params)
            data = response.json()

            for track in data["items"]:
                # Normalize track name for deduplication (lowercase, strip whitespace)
                normalized_name = track["name"].lower().strip()

                # Skip if we've seen this track name (likely same song on different releases)
                if normalized_name in seen_names:
                    continue

                seen_names.add(normalized_name)
                all_tracks.append({
                    "name": track["name"],
                    "id": track["id"],
                    "album_name": album_name,
                    "album_id": album_id,
                })

            url = data.get("next")
            params = None

    return all_tracks


def search_artist_tracks(artist_name: str, access_token: str, max_tracks: int = 100) -> list[dict]:
    """
    Find tracks by an artist using Spotify's search endpoint.
    Fallback when albums/tracks endpoints are rate limited.

    Args:
        artist_name: Name of the artist
        access_token: Spotify API access token
        max_tracks: Maximum number of tracks to return

    Returns:
        list: List of unique track dicts with name, id, and album info
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    all_tracks = []
    seen_names = set()
    offset = 0
    limit = 50

    while len(all_tracks) < max_tracks:
        url = "https://api.spotify.com/v1/search"
        params = {
            "q": f"artist:\"{artist_name}\"",
            "type": "track",
            "limit": limit,
            "offset": offset,
        }

        response = _spotify_get(url, headers, params)
        data = response.json()
        items = data.get("tracks", {}).get("items", [])

        if not items:
            break

        for track in items:
            # Only include tracks where the artist is a primary artist
            artist_names = [a["name"].lower() for a in track.get("artists", [])]
            if artist_name.lower() not in artist_names:
                continue

            normalized_name = track["name"].lower().strip()
            if normalized_name in seen_names:
                continue

            seen_names.add(normalized_name)
            all_tracks.append({
                "name": track["name"],
                "id": track["id"],
                "album_name": track.get("album", {}).get("name", ""),
                "album_id": track.get("album", {}).get("id", ""),
            })

            if len(all_tracks) >= max_tracks:
                break

        offset += limit
        if offset >= 1000:  # Spotify search offset limit
            break

    return all_tracks


def get_artist_audio_features(artist_name: str, output_file: str, access_token: str, max_tracks: int = 100) -> None:
    """
    Get audio features for all unique tracks from an artist available in ReccoBeats.

    Args:
        artist_name: Name of the artist
        output_file: Path to output CSV file
        access_token: Spotify API access token
        max_tracks: Maximum number of tracks to collect per artist (0 for no limit)
    """
    print(f"Searching for artist: {artist_name}")

    # Use search-based track discovery (avoids albums endpoint rate limits)
    print("Searching for tracks...")
    tracks = search_artist_tracks(artist_name, access_token, max_tracks=max_tracks)
    print(f"Found {len(tracks)} unique tracks")

    if not tracks:
        print("No tracks found")
        return

    # Look up tracks in ReccoBeats (batch by 40 - ReccoBeats API limit)
    print("\nLooking up tracks in ReccoBeats...")
    spotify_to_recco = {}
    batch_size = 40

    for i in range(0, len(tracks), batch_size):
        batch = tracks[i:i + batch_size]
        batch_ids = [t["id"] for t in batch]
        batch_mapping = get_reccobeats_track_ids(batch_ids)
        spotify_to_recco.update(batch_mapping)

    print(f"Found {len(spotify_to_recco)}/{len(tracks)} tracks in ReccoBeats")

    # Get audio features for tracks in ReccoBeats
    print("\nFetching audio features...")
    rows = []
    skipped = []

    for i, track in enumerate(tracks):
        spotify_id = track["id"]
        recco_id = spotify_to_recco.get(spotify_id)

        if not recco_id:
            skipped.append(track["name"])
            continue

        audio_features = get_reccobeats_audio_features(recco_id)
        if audio_features is None:
            skipped.append(track["name"])
            continue

        row = {
            "track_name": track["name"],
            "track_id": spotify_id,
            "artist_name": artist_name,
            "album_name": track["album_name"],
            "danceability": audio_features.get("danceability"),
            "energy": audio_features.get("energy"),
            "key": audio_features.get("key"),
            "loudness": audio_features.get("loudness"),
            "mode": audio_features.get("mode"),
            "speechiness": audio_features.get("speechiness"),
            "acousticness": audio_features.get("acousticness"),
            "instrumentalness": audio_features.get("instrumentalness"),
            "liveness": audio_features.get("liveness"),
            "valence": audio_features.get("valence"),
            "tempo": audio_features.get("tempo"),
        }
        rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(tracks)} tracks...")

    # Write to CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSuccessfully wrote {len(rows)} tracks to {output_file}")
    else:
        print("\nNo tracks with audio features found.")

    if skipped:
        print(f"Skipped {len(skipped)} tracks (not in ReccoBeats)")


def search_playlists(query: str, access_token: str, limit: int = 2) -> list[dict]:
    """
    Search Spotify for playlists matching a query.

    Args:
        query: Search query (e.g. genre name)
        access_token: Spotify API access token
        limit: Maximum number of playlists to return

    Returns:
        list: Playlist dicts with id, name, owner, and track count
    """
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"q": query, "type": "playlist", "limit": limit}

    response = _spotify_get(url, headers, params)
    data = response.json()

    playlists = []
    for item in data.get("playlists", {}).get("items", []) or []:
        if item is None:
            continue
        playlists.append({
            "id": item["id"],
            "name": item["name"],
            "owner": item.get("owner", {}).get("display_name", ""),
            "track_count": item.get("tracks", {}).get("total", 0),
        })
    return playlists


def get_playlist_tracks(playlist_id: str, access_token: str) -> list[dict]:
    """
    Get all tracks from a Spotify playlist.

    Args:
        playlist_id: Spotify playlist ID
        access_token: Spotify API access token

    Returns:
        list: Track dicts with name, id, artist_name, album_name
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    tracks = []
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    params = {"limit": 100}

    while url:
        response = _spotify_get(url, headers, params)
        data = response.json()

        for item in data.get("items", []):
            track = item.get("track")
            if not track or not track.get("id"):
                continue
            tracks.append({
                "name": track["name"],
                "id": track["id"],
                "artist_name": ", ".join(a["name"] for a in track.get("artists", [])),
                "album_name": track.get("album", {}).get("name", ""),
            })

        url = data.get("next")
        params = None

    return tracks


def get_album_id_from_name(album_name: str, access_token: str, artist_name: Optional[str] = None) -> Optional[str]:
    """
    Search for an album by name and return its ID.

    Args:
        album_name: Name of the album to search for
        access_token: Spotify API access token
        artist_name: Optional artist name to make search more specific

    Returns:
        str: Album ID if found, None otherwise
    """
    url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # Build search query - include artist if provided
    if artist_name:
        query = f"album:{album_name} artist:{artist_name}"
    else:
        query = album_name

    params = {
        "q": query,
        "type": "album",
        "limit": 1
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    if data.get("albums", {}).get("items"):
        return data["albums"]["items"][0]["id"]
    return None


def get_album_tracks_audio_features(album_identifier: str, output_file: str = "album_audio_features.csv",
                                     access_token: Optional[str] = None, is_album_id: bool = True,
                                     append: bool = False, artist_name: Optional[str] = None) -> None:
    """
    Get all tracks from an album and their audio features, then write to CSV.

    Args:
        album_identifier: Album ID or album name
        output_file: Path to output CSV file
        access_token: Spotify API access token (if None, will try to get from environment)
        is_album_id: If True, treat album_identifier as album ID; if False, treat as album name
        append: If True, append to existing CSV file instead of overwriting
        artist_name: Optional artist name to make album search more specific (only used when is_album_id=False)

    Raises:
        ValueError: If access token is not available
        requests.HTTPError: If API request fails
    """
    # Get access token
    if access_token is None:
        access_token = os.getenv("ACCESS_TOKEN")
        if access_token is None:
            raise ValueError("Access token not found. Please set ACCESS_TOKEN environment variable or pass it as argument.")
    
    # If album name provided, search for album ID
    album_id = album_identifier
    if not is_album_id:
        album_id = get_album_id_from_name(album_identifier, access_token, artist_name)
        if album_id is None:
            raise ValueError(f"Album '{album_identifier}' not found.")
    
    # Get album information
    url = f"https://api.spotify.com/v1/albums/{album_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    album_data = response.json()
    
    album_name = album_data["name"]
    artist_name = ", ".join([artist["name"] for artist in album_data["artists"]])
    
    # Get all tracks from the album (handle pagination)
    tracks = []
    url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
    params = {"limit": 50}  # Max limit per request
    
    while url:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        tracks.extend(data["items"])
        url = data.get("next")  # Get next page URL if exists
        params = None  # Next URL already has params
    
    if not tracks:
        raise ValueError(f"No tracks found for album '{album_name}'.")
    
    # Get track IDs (filter out any None or missing IDs)
    spotify_track_ids = [track["id"] for track in tracks if track.get("id")]

    # Step 1: Look up Spotify IDs in ReccoBeats to get ReccoBeats IDs
    print(f"Looking up {len(spotify_track_ids)} tracks in ReccoBeats...")
    spotify_to_recco = get_reccobeats_track_ids(spotify_track_ids)
    print(f"  Found {len(spotify_to_recco)}/{len(spotify_track_ids)} tracks in ReccoBeats")

    # Step 2: Get audio features for tracks that exist in ReccoBeats
    print("Fetching audio features...")
    rows = []
    skipped_count = 0

    for i, track in enumerate(tracks):
        spotify_id = track.get("id")
        if not spotify_id:
            continue

        recco_id = spotify_to_recco.get(spotify_id)
        if not recco_id:
            print(f"  [SKIP] '{track['name']}' - not found in ReccoBeats")
            skipped_count += 1
            continue

        audio_features = get_reccobeats_audio_features(recco_id)
        if audio_features is None:
            print(f"  [SKIP] '{track['name']}' - no audio features available")
            skipped_count += 1
            continue

        row = {
            "track_name": track["name"],
            "track_id": spotify_id,
            "artist_name": artist_name,
            "album_name": album_name,
            "danceability": audio_features.get("danceability"),
            "energy": audio_features.get("energy"),
            "key": audio_features.get("key"),
            "loudness": audio_features.get("loudness"),
            "mode": audio_features.get("mode"),
            "speechiness": audio_features.get("speechiness"),
            "acousticness": audio_features.get("acousticness"),
            "instrumentalness": audio_features.get("instrumentalness"),
            "liveness": audio_features.get("liveness"),
            "valence": audio_features.get("valence"),
            "tempo": audio_features.get("tempo"),
        }
        rows.append(row)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(tracks)} tracks...")

    if skipped_count > 0:
        print(f"  Skipped {skipped_count} tracks (not in ReccoBeats)")
    
    # Write to CSV
    if rows:
        fieldnames = list(rows[0].keys())
        file_exists = os.path.exists(output_file)
        mode = "a" if append and file_exists else "w"
        write_header = not (append and file_exists)

        with open(output_file, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        action = "appended" if append and file_exists else "wrote"
        print(f"Successfully {action} {len(rows)} tracks to {output_file}")
    else:
        print("No tracks with audio features found.")


