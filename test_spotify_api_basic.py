import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load environment variables
load_dotenv()

# Spotify API credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

print("Testing Spotify API credentials...")
print(f"Client ID: {CLIENT_ID[:5]}..." if CLIENT_ID else "Client ID: Not found")
print(f"Client Secret: {CLIENT_SECRET[:5]}..." if CLIENT_SECRET else "Client Secret: Not found")

# Set up Spotify client
try:
    client_credentials_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID, 
        client_secret=CLIENT_SECRET
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Try a simple search
    print("\nSearching for 'Bohemian Rhapsody'...")
    results = sp.search(q="Bohemian Rhapsody", type='track', limit=3)
    
    if results and 'tracks' in results and 'items' in results['tracks']:
        print("\nResults:")
        for i, track in enumerate(results['tracks']['items']):
            print(f"{i+1}. {track['name']} by {', '.join([artist['name'] for artist in track['artists']])} (Popularity: {track['popularity']})")
    else:
        print("No results found or unexpected response format.")
        
    print("\nSpotify API is working!")
except Exception as e:
    print(f"\nError: {e}")
    print("\nSpotify API connection failed.")
