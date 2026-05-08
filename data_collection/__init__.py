"""Shared authentication and collection helpers for data collection scripts."""

import os
import time
import requests

from util import get_spotify_token, get_artist_audio_features


# Global token variable
_token = None


def load_env_file(filepath=".env"):
    """Load environment variables from a .env file."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ[key] = val


def initialize_token():
    """Initialize or refresh the global access token."""
    global _token
    load_env_file()

    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("CLIENT_ID and CLIENT_SECRET must be set in .env file")

    print("Getting new Spotify access token...")
    token_data = get_spotify_token(client_id, client_secret)
    _token = token_data["access_token"]
    print("Token obtained successfully!")


def get_token():
    """Get the current token, initializing if needed."""
    global _token
    if _token is None:
        initialize_token()
    return _token


def collect_artist_safe(artist_name, output_file):
    """Collect audio features for an artist with token refresh and rate limit handling."""
    global _token
    if _token is None:
        initialize_token()

    try:
        get_artist_audio_features(artist_name, output_file, _token)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Token expired, refreshing...")
            initialize_token()
            get_artist_audio_features(artist_name, output_file, _token)
        elif e.response.status_code == 429:
            retry_after = int(e.response.headers.get("Retry-After", 30))
            retry_after = min(retry_after, 120)
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            get_artist_audio_features(artist_name, output_file, _token)
        else:
            raise
