import os
import requests
from util import get_spotify_token, get_artist_audio_features

def load_env_file(filepath=".env"):
    """Load environment variables from a .env file."""
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ[key] = val

# Global token variable
token = None

def initialize_token():
    """Initialize or refresh the global access token."""
    global token
    load_env_file()

    # Always get a fresh token (tokens expire after 1 hour)
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise ValueError("CLIENT_ID and CLIENT_SECRET must be set in .env file")
    
    print("Getting new Spotify access token...")
    token_data = get_spotify_token(client_id, client_secret)
    token = token_data['access_token']
    print("Token obtained successfully!")

def collect_artist_tracks(artist_name: str, output_file: str):
    """Collect all unique tracks from an artist available in ReccoBeats."""
    global token
    if token is None:
        initialize_token()

    try:
        get_artist_audio_features(artist_name, output_file, token)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Token expired, refreshing...")
            initialize_token()
            get_artist_audio_features(artist_name, output_file, token)
        else:
            raise

if __name__ == "__main__":
    initialize_token()
    os.makedirs("data", exist_ok=True)

    # Collect all unique tracks from Ninajirachi
    collect_artist_tracks("Ninajirachi", "data/ninajirachi.csv")
