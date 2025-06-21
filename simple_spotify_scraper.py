import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
try:
    load_dotenv()
except ImportError:
    print("dotenv not installed. Continuing without loading .env file.")

class SimpleSpotifyScraper:
    def __init__(self, client_id=None, client_secret=None):
        """Initialize the Spotify scraper with credentials"""
        # Get credentials from environment variables or parameters
        self.client_id = client_id or os.environ.get('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('SPOTIFY_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            print("Warning: Spotify API credentials not found.")
            print("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables or pass them as parameters.")
            self.access_token = None
        else:
            # Get access token
            self.access_token = self._get_access_token()
            if self.access_token:
                print("Successfully authenticated with Spotify API!")
    
    def _get_access_token(self):
        """Get access token from Spotify API using client credentials flow"""
        try:
            auth_url = 'https://accounts.spotify.com/api/token'
            auth_response = requests.post(auth_url, {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
            })
            
            if auth_response.status_code != 200:
                print(f"Authentication failed with status code {auth_response.status_code}")
                print(f"Response: {auth_response.text}")
                return None
                
            auth_data = auth_response.json()
            return auth_data['access_token']
        except Exception as e:
            print(f"Error getting access token: {e}")
            return None
    
    def search_tracks(self, query, limit=10):
        """Search for tracks on Spotify"""
        if not self.access_token:
            print("Cannot search: No access token available")
            return None
            
        try:
            search_url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit={limit}"
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            }
            
            response = requests.get(search_url, headers=headers)
            
            if response.status_code != 200:
                print(f"Search failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
            results = response.json()
            return results['tracks']['items']
        except Exception as e:
            print(f"Error searching tracks: {e}")
            return None
    
    def get_audio_features(self, track_ids):
        """Get audio features for a list of track IDs"""
        if not self.access_token:
            print("Cannot get audio features: No access token available")
            return None
            
        try:
            # Spotify API allows up to 100 track IDs per request
            # Join multiple IDs with comma
            ids_str = ','.join(track_ids[:100])  # Limit to 100 IDs
            
            features_url = f"https://api.spotify.com/v1/audio-features?ids={ids_str}"
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            }
            
            response = requests.get(features_url, headers=headers)
            
            if response.status_code != 200:
                print(f"Get audio features failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
            results = response.json()
            return results['audio_features']
        except Exception as e:
            print(f"Error getting audio features: {e}")
            return None
    
    def get_playlist_tracks(self, playlist_id, limit=100):
        """Get tracks from a Spotify playlist"""
        if not self.access_token:
            print("Cannot get playlist tracks: No access token available")
            return None
            
        try:
            tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit={limit}"
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            }
            
            response = requests.get(tracks_url, headers=headers)
            
            if response.status_code != 200:
                print(f"Get playlist tracks failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
            results = response.json()
            return results['items']
        except Exception as e:
            print(f"Error getting playlist tracks: {e}")
            return None
    
    def get_playlist_info(self, playlist_id):
        """Get information about a Spotify playlist"""
        if not self.access_token:
            print("Cannot get playlist info: No access token available")
            return None
            
        try:
            playlist_url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            }
            
            response = requests.get(playlist_url, headers=headers)
            
            if response.status_code != 200:
                print(f"Get playlist info failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
            return response.json()
        except Exception as e:
            print(f"Error getting playlist info: {e}")
            return None
    
    def get_track_info(self, track_id):
        """Get detailed information about a track"""
        if not self.access_token:
            print("Cannot get track info: No access token available")
            return None
            
        try:
            track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            }
            
            response = requests.get(track_url, headers=headers)
            
            if response.status_code != 200:
                print(f"Get track info failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
            return response.json()
        except Exception as e:
            print(f"Error getting track info: {e}")
            return None
    
    def get_recommendations(self, seed_tracks=None, seed_artists=None, seed_genres=None, limit=20):
        """Get track recommendations based on seeds"""
        if not self.access_token:
            print("Cannot get recommendations: No access token available")
            return None
            
        try:
            # Prepare parameters
            params = {'limit': limit}
            
            if seed_tracks:
                params['seed_tracks'] = ','.join(seed_tracks[:5])  # Max 5 seeds
            if seed_artists:
                params['seed_artists'] = ','.join(seed_artists[:5])  # Max 5 seeds
            if seed_genres:
                params['seed_genres'] = ','.join(seed_genres[:5])  # Max 5 seeds
                
            # Build URL with parameters
            recommendations_url = "https://api.spotify.com/v1/recommendations"
            headers = {
                'Authorization': f'Bearer {self.access_token}'
            }
            
            response = requests.get(recommendations_url, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"Get recommendations failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
            results = response.json()
            return results['tracks']
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None
    
    def fetch_playlist_data(self, playlist_id, save_to_csv=True, filename=None):
        """Fetch all data for a playlist and optionally save to CSV"""
        if not self.access_token:
            print("Cannot fetch playlist data: No access token available")
            return None
            
        # Get playlist info
        playlist_info = self.get_playlist_info(playlist_id)
        if not playlist_info:
            return None
            
        playlist_name = playlist_info['name']
        print(f"Fetching data for playlist: {playlist_name}")
        
        # Get playlist tracks
        playlist_tracks = self.get_playlist_tracks(playlist_id)
        if not playlist_tracks:
            return None
            
        print(f"Found {len(playlist_tracks)} tracks in playlist")
        
        # Extract track IDs
        track_ids = []
        track_data = []
        
        for item in playlist_tracks:
            track = item['track']
            if not track:  # Skip None tracks
                continue
                
            track_ids.append(track['id'])
            
            # Basic track info
            track_info = {
                'track_id': track['id'],
                'track_name': track['name'],
                'track_popularity': track['popularity'],
                'track_href': track['href'],
                'uri': track['uri'],
                'track_artist': ', '.join([artist['name'] for artist in track['artists']]),
                'track_album_name': track['album']['name'],
                'track_album_id': track['album']['id'],
                'track_album_release_date': track['album']['release_date'],
                'duration_ms': track['duration_ms'],
                'playlist_id': playlist_id,
                'playlist_name': playlist_name
            }
            
            track_data.append(track_info)
        
        # Get audio features for the tracks
        if track_ids:
            print(f"Fetching audio features for {len(track_ids)} tracks")
            
            # Process in batches of 100 (API limit)
            all_audio_features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                features = self.get_audio_features(batch)
                if features:
                    all_audio_features.extend(features)
                # Rate limiting
                if i + 100 < len(track_ids):
                    time.sleep(1)
            
            # Combine track data with audio features
            for i, track in enumerate(track_data):
                if i < len(all_audio_features) and all_audio_features[i]:
                    # Add audio features to track data
                    features = all_audio_features[i]
                    for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                  'speechiness', 'acousticness', 'instrumentalness',
                                  'liveness', 'valence', 'tempo', 'time_signature',
                                  'analysis_url', 'id', 'type']:
                        if feature in features:
                            track[feature] = features[feature]
        
        # Create DataFrame
        df = pd.DataFrame(track_data)
        print(f"Created DataFrame with {len(df)} tracks")
        
        # Save to CSV if requested
        if save_to_csv:
            if not filename:
                filename = f"{playlist_name.replace(' ', '_')}_spotify_data.csv"
            
            try:
                df.to_csv(filename, index=False)
                print(f"Data saved to {filename}")
            except Exception as e:
                print(f"Error saving data to CSV: {e}")
        
        return df

def main():
    # Create the scraper
    scraper = SimpleSpotifyScraper()
    
    if not scraper.access_token:
        print("\nNo valid Spotify API credentials found.")
        print("You can still search and download playlist data if you provide credentials.")
    
    # Menu
    while True:
        print("\n===== Simple Spotify Data Scraper =====")
        print("1. Search for tracks")
        print("2. Get audio features for a track")
        print("3. Fetch playlist data")
        print("4. Get track recommendations")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            # Search for tracks
            query = input("Enter search query: ")
            limit = 10
            try:
                limit_input = input("Number of results (default: 10): ")
                if limit_input:
                    limit = int(limit_input)
            except ValueError:
                print("Invalid input, using default (10)")
            
            tracks = scraper.search_tracks(query, limit=limit)
            
            if tracks:
                print(f"\nFound {len(tracks)} tracks:")
                for i, track in enumerate(tracks, 1):
                    artists = ", ".join([artist['name'] for artist in track['artists']])
                    print(f"{i}. {track['name']} by {artists} (Popularity: {track['popularity']})")
                    
                # Option to get audio features
                track_idx = input("\nEnter track number to get audio features (or press Enter to skip): ")
                if track_idx:
                    try:
                        idx = int(track_idx) - 1
                        if 0 <= idx < len(tracks):
                            track_id = tracks[idx]['id']
                            features = scraper.get_audio_features([track_id])
                            
                            if features and features[0]:
                                print("\nAudio Features:")
                                for key, value in features[0].items():
                                    print(f"{key}: {value}")
                    except ValueError:
                        print("Invalid input")
        
        elif choice == '2':
            # Get audio features for a track
            track_id = input("Enter Spotify track ID: ")
            if track_id:
                features = scraper.get_audio_features([track_id])
                
                if features and features[0]:
                    print("\nAudio Features:")
                    for key, value in features[0].items():
                        print(f"{key}: {value}")
        
        elif choice == '3':
            # Fetch playlist data
            playlist_id = input("Enter Spotify playlist ID: ")
            if playlist_id:
                filename = input("Enter output CSV filename (or press Enter for default): ")
                
                df = scraper.fetch_playlist_data(playlist_id, 
                                              save_to_csv=True, 
                                              filename=filename if filename else None)
                
                if df is not None and not df.empty:
                    # Show sample of the data
                    print("\nSample of the data:")
                    print(df.head())
                    
                    # Plot some basic stats if there's enough data
                    if len(df) > 5:
                        # Check which columns exist
                        audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                                        'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness']
                        available_features = [f for f in audio_features if f in df.columns]
                        
                        if available_features:
                            try:
                                # Create a simple bar chart of average feature values
                                plt.figure(figsize=(10, 6))
                                means = df[available_features].mean()
                                means.plot(kind='bar', color='skyblue')
                                plt.title(f'Average Audio Features in "{df["playlist_name"].iloc[0]}"')
                                plt.ylabel('Value')
                                plt.xlabel('Audio Feature')
                                plt.grid(axis='y', alpha=0.3)
                                plt.tight_layout()
                                plt.show()
                            except Exception as e:
                                print(f"Error creating visualization: {e}")
        
        elif choice == '4':
            # Get track recommendations
            seed_type = input("Seed type (track, artist, or genre): ").lower()
            
            if seed_type == 'track':
                track_id = input("Enter Spotify track ID: ")
                recommendations = scraper.get_recommendations(seed_tracks=[track_id])
            elif seed_type == 'artist':
                artist_id = input("Enter Spotify artist ID: ")
                recommendations = scraper.get_recommendations(seed_artists=[artist_id])
            elif seed_type == 'genre':
                genre = input("Enter genre (e.g., pop, rock, hip-hop): ")
                recommendations = scraper.get_recommendations(seed_genres=[genre])
            else:
                print("Invalid seed type")
                continue
            
            if recommendations:
                print(f"\nRecommended tracks:")
                for i, track in enumerate(recommendations, 1):
                    artists = ", ".join([artist['name'] for artist in track['artists']])
                    print(f"{i}. {track['name']} by {artists}")
        
        elif choice == '5':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
