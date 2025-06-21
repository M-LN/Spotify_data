import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import time

# Load environment variables
load_dotenv()

# Spotify API credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

class SpotifyLiveScraper:
    def __init__(self):
        """Initialize the Spotify scraper with credentials"""
        self.client_id = CLIENT_ID
        self.client_secret = CLIENT_SECRET
        
        if not self.client_id or not self.client_secret:
            print("Error: Spotify API credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file.")
            self.sp = None
        else:
            # Set up Spotify client
            try:
                client_credentials_manager = SpotifyClientCredentials(
                    client_id=self.client_id, 
                    client_secret=self.client_secret
                )
                self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
                print("Successfully connected to Spotify API!")
            except Exception as e:
                print(f"Error connecting to Spotify API: {e}")
                self.sp = None
        
        # Storage for scraped data
        self.tracks_df = pd.DataFrame()
    
    def search_tracks(self, query, limit=10):
        """Search for tracks on Spotify"""
        if not self.sp:
            print("Error: Spotify API not connected.")
            return pd.DataFrame()
            
        try:
            print(f"Searching for: {query}")
            results = self.sp.search(q=query, type='track', limit=limit)
            
            tracks = results['tracks']['items']
            if not tracks:
                print("No tracks found.")
                return pd.DataFrame()
                
            # Extract basic info from tracks
            track_data = []
            track_ids = []
            
            for track in tracks:
                track_ids.append(track['id'])
                
                # Basic track info
                track_info = {
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'track_popularity': track['popularity'],
                    'track_href': track['href'],
                    'uri': track['uri'],
                    'track_artist': ', '.join([artist['name'] for artist in track['artists']]),
                    'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                    'track_album_name': track['album']['name'],
                    'track_album_id': track['album']['id'],
                    'track_album_release_date': track['album']['release_date'],
                    'duration_ms': track['duration_ms'],
                    'album_image_url': track['album']['images'][0]['url'] if track['album']['images'] else None
                }
                
                track_data.append(track_info)
            
            # Create initial DataFrame
            tracks_df = pd.DataFrame(track_data)
            
            # Try to get audio features for these tracks
            try:
                print(f"Getting audio features for {len(track_ids)} tracks...")
                audio_features = self.sp.audio_features(track_ids)
                
                # Add audio features to DataFrame
                for i, features in enumerate(audio_features):
                    if features and i < len(tracks_df):
                        for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                        'speechiness', 'acousticness', 'instrumentalness',
                                        'liveness', 'valence', 'tempo', 'time_signature']:
                            tracks_df.loc[i, feature] = features[feature]
            except Exception as e:
                print(f"Warning: Could not get audio features: {e}")
            
            # Try to get artist genres
            try:
                # Get unique artist IDs
                artist_ids = tracks_df['artist_id'].dropna().unique().tolist()
                
                if artist_ids:
                    # Spotify API allows up to 50 artist IDs per request
                    artist_data = {}
                    for i in range(0, len(artist_ids), 50):
                        batch = artist_ids[i:i+50]
                        artists_info = self.sp.artists(batch)
                        
                        for artist in artists_info['artists']:
                            artist_data[artist['id']] = {
                                'artist_name': artist['name'],
                                'artist_genres': ', '.join(artist['genres']) if artist['genres'] else 'Unknown',
                                'artist_popularity': artist['popularity']
                            }
                    
                    # Add artist info to tracks
                    for i, row in tracks_df.iterrows():
                        if row['artist_id'] in artist_data:
                            for key, value in artist_data[row['artist_id']].items():
                                tracks_df.loc[i, key] = value
            except Exception as e:
                print(f"Warning: Could not get artist information: {e}")
            
            print(f"Found {len(tracks_df)} tracks with data.")
            return tracks_df
        
        except Exception as e:
            print(f"Error searching tracks: {e}")
            return pd.DataFrame()
    
    def get_related_artists_top_tracks(self, artist_id, country='US', limit=5):
        """Get top tracks from related artists"""
        if not self.sp:
            print("Error: Spotify API not connected.")
            return pd.DataFrame()
            
        try:
            # Get related artists
            related = self.sp.artist_related_artists(artist_id)
            
            if not related or 'artists' not in related or not related['artists']:
                print("No related artists found.")
                return pd.DataFrame()
            
            all_tracks = []
            
            # Get top tracks for each related artist (up to limit)
            for artist in related['artists'][:limit]:
                try:
                    print(f"Getting top tracks for {artist['name']}...")
                    top_tracks = self.sp.artist_top_tracks(artist['id'], country=country)
                    
                    for track in top_tracks['tracks']:
                        track_info = {
                            'track_id': track['id'],
                            'track_name': track['name'],
                            'track_popularity': track['popularity'],
                            'track_href': track['href'],
                            'uri': track['uri'],
                            'track_artist': artist['name'],
                            'artist_id': artist['id'],
                            'artist_genres': ', '.join(artist['genres']) if artist['genres'] else 'Unknown',
                            'track_album_name': track['album']['name'],
                            'track_album_id': track['album']['id'],
                            'track_album_release_date': track['album']['release_date'],
                            'duration_ms': track['duration_ms'],
                            'album_image_url': track['album']['images'][0]['url'] if track['album']['images'] else None
                        }
                        all_tracks.append(track_info)
                except Exception as e:
                    print(f"Warning: Could not get top tracks for {artist['name']}: {e}")
                    continue
            
            if not all_tracks:
                print("No tracks found from related artists.")
                return pd.DataFrame()
                
            # Create DataFrame
            tracks_df = pd.DataFrame(all_tracks)
            
            # Try to get audio features
            try:
                track_ids = tracks_df['track_id'].tolist()
                audio_features = []
                
                # Process in batches of 100 (Spotify API limit)
                for i in range(0, len(track_ids), 100):
                    batch_ids = track_ids[i:i+100]
                    batch_features = self.sp.audio_features(batch_ids)
                    audio_features.extend(batch_features)
                    time.sleep(1)  # Avoid rate limiting
                
                # Add audio features to DataFrame
                for i, features in enumerate(audio_features):
                    if features and i < len(tracks_df):
                        for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                        'speechiness', 'acousticness', 'instrumentalness',
                                        'liveness', 'valence', 'tempo', 'time_signature']:
                            tracks_df.loc[i, feature] = features[feature]
            except Exception as e:
                print(f"Warning: Could not get audio features: {e}")
            
            print(f"Found {len(tracks_df)} tracks from related artists.")
            return tracks_df
        
        except Exception as e:
            print(f"Error getting related artists: {e}")
            return pd.DataFrame()
    
    def get_playlist_tracks(self, playlist_id, limit=100):
        """Get tracks from a Spotify playlist"""
        if not self.sp:
            print("Error: Spotify API not connected.")
            return pd.DataFrame()
            
        try:
            # Get playlist details first
            playlist = self.sp.playlist(playlist_id)
            playlist_name = playlist['name']
            playlist_owner = playlist['owner']['display_name']
            playlist_desc = playlist['description']
            
            print(f"Getting tracks from playlist: {playlist_name} by {playlist_owner}")
            print(f"Description: {playlist_desc}")
            
            # Get tracks
            results = self.sp.playlist_tracks(playlist_id, limit=limit)
            tracks = results['items']
            
            # Handle pagination for large playlists (up to the limit)
            while results['next'] and len(tracks) < limit:
                results = self.sp.next(results)
                tracks.extend(results['items'])
                if len(tracks) >= limit:
                    tracks = tracks[:limit]
                    break
            
            if not tracks:
                print("No tracks found in playlist.")
                return pd.DataFrame()
                
            # Extract track data
            track_data = []
            track_ids = []
            
            for item in tracks:
                track = item['track']
                
                if not track:  # Skip None tracks (can happen with removed tracks)
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
                    'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                    'track_album_name': track['album']['name'],
                    'track_album_id': track['album']['id'],
                    'track_album_release_date': track['album']['release_date'],
                    'duration_ms': track['duration_ms'],
                    'album_image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'added_at': item['added_at'],
                    'playlist_name': playlist_name,
                    'playlist_owner': playlist_owner
                }
                
                track_data.append(track_info)
            
            # Create DataFrame
            tracks_df = pd.DataFrame(track_data)
            
            # Try to get audio features
            try:
                audio_features = []
                
                # Process in batches of 100 (Spotify API limit)
                for i in range(0, len(track_ids), 100):
                    batch_ids = track_ids[i:i+100]
                    batch_features = self.sp.audio_features(batch_ids)
                    audio_features.extend(batch_features)
                    time.sleep(1)  # Avoid rate limiting
                
                # Add audio features to DataFrame
                for i, features in enumerate(audio_features):
                    if features and i < len(tracks_df):
                        for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                        'speechiness', 'acousticness', 'instrumentalness',
                                        'liveness', 'valence', 'tempo', 'time_signature']:
                            tracks_df.loc[i, feature] = features[feature]
            except Exception as e:
                print(f"Warning: Could not get audio features: {e}")
            
            print(f"Found {len(tracks_df)} tracks in playlist.")
            return tracks_df
        
        except Exception as e:
            print(f"Error getting playlist: {e}")
            return pd.DataFrame()
    
    def get_recommendations(self, seed_tracks=None, seed_artists=None, seed_genres=None, limit=20, **kwargs):
        """Get track recommendations based on seeds"""
        if not self.sp:
            print("Error: Spotify API not connected.")
            return pd.DataFrame()
            
        # Validate seed inputs
        seed_count = sum(1 for seed in [seed_tracks, seed_artists, seed_genres] if seed)
        if seed_count == 0:
            print("Error: At least one seed (tracks, artists, or genres) is required")
            return pd.DataFrame()
            
        # Format seed inputs
        if seed_tracks and isinstance(seed_tracks, list):
            seed_tracks = seed_tracks[:5]  # API allows up to 5 seed tracks
        if seed_artists and isinstance(seed_artists, list):
            seed_artists = seed_artists[:5]  # API allows up to 5 seed artists
        if seed_genres and isinstance(seed_genres, list):
            seed_genres = seed_genres[:5]  # API allows up to 5 seed genres
            
        try:
            # Get recommendations
            print("Getting recommendations...")
            results = self.sp.recommendations(
                seed_tracks=seed_tracks,
                seed_artists=seed_artists,
                seed_genres=seed_genres,
                limit=limit,
                **kwargs
            )
            
            if not results or 'tracks' not in results or not results['tracks']:
                print("No recommendations found")
                return pd.DataFrame()
                
            tracks = results['tracks']
            
            # Extract track data
            track_data = []
            track_ids = []
            
            for track in tracks:
                track_ids.append(track['id'])
                
                # Basic track info
                track_info = {
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'track_popularity': track['popularity'],
                    'track_href': track['href'],
                    'uri': track['uri'],
                    'track_artist': ', '.join([artist['name'] for artist in track['artists']]),
                    'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                    'track_album_name': track['album']['name'],
                    'track_album_id': track['album']['id'],
                    'track_album_release_date': track['album']['release_date'],
                    'duration_ms': track['duration_ms'],
                    'album_image_url': track['album']['images'][0]['url'] if track['album']['images'] else None
                }
                
                track_data.append(track_info)
            
            # Create DataFrame
            tracks_df = pd.DataFrame(track_data)
            
            # Try to get audio features
            try:
                audio_features = self.sp.audio_features(track_ids)
                
                # Add audio features to DataFrame
                for i, features in enumerate(audio_features):
                    if features and i < len(tracks_df):
                        for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                        'speechiness', 'acousticness', 'instrumentalness',
                                        'liveness', 'valence', 'tempo', 'time_signature']:
                            tracks_df.loc[i, feature] = features[feature]
            except Exception as e:
                print(f"Warning: Could not get audio features: {e}")
            
            print(f"Found {len(tracks_df)} recommended tracks.")
            return tracks_df
        
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return pd.DataFrame()
    
    def get_new_releases(self, country='US', limit=20):
        """Get new releases from Spotify"""
        if not self.sp:
            print("Error: Spotify API not connected.")
            return pd.DataFrame()
            
        try:
            print(f"Getting new releases for country: {country}")
            results = self.sp.new_releases(country=country, limit=limit)
            
            if not results or 'albums' not in results or 'items' not in results['albums'] or not results['albums']['items']:
                print("No new releases found")
                return pd.DataFrame()
                
            albums = results['albums']['items']
            
            # Get tracks for each album
            all_tracks = []
            
            for album in albums:
                try:
                    print(f"Getting tracks for album: {album['name']} by {album['artists'][0]['name']}")
                    album_tracks = self.sp.album_tracks(album['id'])
                    
                    for track in album_tracks['items']:
                        track_info = {
                            'track_id': track['id'],
                            'track_name': track['name'],
                            'track_href': track['href'],
                            'uri': track['uri'],
                            'track_artist': ', '.join([artist['name'] for artist in track['artists']]),
                            'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                            'track_album_name': album['name'],
                            'track_album_id': album['id'],
                            'track_album_release_date': album['release_date'],
                            'duration_ms': track['duration_ms'],
                            'album_image_url': album['images'][0]['url'] if album['images'] else None,
                            'album_type': album['album_type'],
                            'album_genres': ', '.join(album.get('genres', [])) if 'genres' in album and album['genres'] else 'Unknown'
                        }
                        all_tracks.append(track_info)
                except Exception as e:
                    print(f"Warning: Could not get tracks for album {album['name']}: {e}")
                    continue
            
            if not all_tracks:
                print("No tracks found in new releases.")
                return pd.DataFrame()
                
            # Create DataFrame
            tracks_df = pd.DataFrame(all_tracks)
            
            # Try to get audio features and popularity
            try:
                # Get track IDs
                track_ids = tracks_df['track_id'].tolist()
                
                # Get track popularity (by getting full track objects)
                track_info = {}
                for i in range(0, len(track_ids), 50):
                    batch_ids = track_ids[i:i+50]
                    batch_tracks = self.sp.tracks(batch_ids)
                    
                    for track in batch_tracks['tracks']:
                        track_info[track['id']] = {
                            'track_popularity': track['popularity']
                        }
                
                # Add popularity to DataFrame
                for i, row in tracks_df.iterrows():
                    if row['track_id'] in track_info:
                        for key, value in track_info[row['track_id']].items():
                            tracks_df.loc[i, key] = value
                
                # Get audio features
                audio_features = []
                
                # Process in batches of 100 (Spotify API limit)
                for i in range(0, len(track_ids), 100):
                    batch_ids = track_ids[i:i+100]
                    batch_features = self.sp.audio_features(batch_ids)
                    audio_features.extend(batch_features)
                    time.sleep(1)  # Avoid rate limiting
                
                # Add audio features to DataFrame
                for i, features in enumerate(audio_features):
                    if features and i < len(tracks_df):
                        for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                        'speechiness', 'acousticness', 'instrumentalness',
                                        'liveness', 'valence', 'tempo', 'time_signature']:
                            tracks_df.loc[i, feature] = features[feature]
            except Exception as e:
                print(f"Warning: Could not get audio features or popularity: {e}")
            
            print(f"Found {len(tracks_df)} tracks in new releases.")
            return tracks_df
        
        except Exception as e:
            print(f"Error getting new releases: {e}")
            return pd.DataFrame()
    
    def visualize_features(self, df=None, features=None):
        """Visualize audio features distribution"""
        if df is None:
            if self.tracks_df.empty:
                print("No data available for visualization")
                return
            df = self.tracks_df
        
        if features is None:
            features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence']
        
        # Filter to include only columns that exist in the DataFrame
        features = [f for f in features if f in df.columns]
        
        if not features:
            print("No audio features found in DataFrame for visualization")
            return
            
        # Set up the figure
        fig, axes = plt.subplots(len(features), 1, figsize=(12, 3*len(features)))
        
        # If only one feature, make axes iterable
        if len(features) == 1:
            axes = [axes]
            
        # Plot histograms for each feature
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.histplot(df[feature], kde=True, ax=ax)
            ax.set_title(f'Distribution of {feature.capitalize()}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            
        plt.tight_layout()
        plt.show()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation between Audio Features')
        plt.tight_layout()
        plt.show()
    
    def visualize_track_comparison(self, tracks_df, track_indices=None):
        """Visualize a comparison of specific tracks"""
        if tracks_df.empty:
            print("No tracks available for visualization")
            return
            
        # If no specific tracks are selected, use all tracks (up to 5)
        if track_indices is None:
            if len(tracks_df) > 5:
                print("Too many tracks to compare. Using the first 5 tracks.")
                tracks_df = tracks_df.head(5)
        else:
            # Filter to selected tracks
            valid_indices = [idx for idx in track_indices if 0 <= idx < len(tracks_df)]
            if not valid_indices:
                print("No valid track indices provided")
                return
            tracks_df = tracks_df.iloc[valid_indices]
        
        # Features for the radar chart
        features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                  'instrumentalness', 'liveness', 'valence']
        
        # Filter to include only columns that exist in the DataFrame
        features = [f for f in features if f in tracks_df.columns]
        
        if not features:
            print("No audio features found in tracks for visualization")
            return
            
        # Number of variables
        N = len(features)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], features, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each track
        for i, (_, track) in enumerate(tracks_df.iterrows()):
            values = [track[feature] for feature in features]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"{track['track_name']} - {track['track_artist']}")
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Track Audio Features Comparison", size=20, y=1.1)
        plt.tight_layout()
        plt.show()
    
    def analyze_genre_features(self, df=None):
        """Analyze audio features by genre"""
        if df is None:
            if self.tracks_df.empty:
                print("No data available for analysis")
                return
            df = self.tracks_df
        
        # Check if we have genre information
        if 'artist_genres' not in df.columns:
            print("No genre information available for analysis")
            return
            
        # Explode genres (a track can have multiple genres)
        # First, convert the comma-separated genre string to a list
        df['genres_list'] = df['artist_genres'].str.split(', ')
        exploded_df = df.explode('genres_list')
        
        # Remove 'Unknown' genre
        exploded_df = exploded_df[exploded_df['genres_list'] != 'Unknown']
        
        # Get the top genres by count
        top_genres = exploded_df['genres_list'].value_counts().head(10).index.tolist()
        
        # Filter to top genres only
        filtered_df = exploded_df[exploded_df['genres_list'].isin(top_genres)]
        
        if filtered_df.empty:
            print("No genre data available after filtering")
            return
            
        # Features to analyze
        features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                   'instrumentalness', 'liveness', 'valence']
        
        # Filter to include only columns that exist in the DataFrame
        features = [f for f in features if f in filtered_df.columns]
        
        if not features:
            print("No audio features found for genre analysis")
            return
            
        # Group by genre and calculate mean for each feature
        grouped_data = filtered_df.groupby('genres_list')[features].mean().reset_index()
        
        # Create a grouped bar chart
        plt.figure(figsize=(14, 8))
        bar_width = 0.1
        x = np.arange(len(top_genres))
        
        for i, feature in enumerate(features):
            plt.bar(x + i*bar_width, grouped_data[feature], width=bar_width, label=feature)
            
        plt.xlabel('Genre')
        plt.ylabel('Average Value')
        plt.title('Audio Features by Genre')
        plt.xticks(x + bar_width * (len(features) - 1) / 2, top_genres, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Create radar chart for each genre
        # Number of variables
        N = len(features)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], features, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each genre
        for i, (genre, row) in enumerate(grouped_data.iterrows()):
            values = [row[feature] for feature in features]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=genre)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Genre Audio Features Comparison", size=20, y=1.1)
        plt.tight_layout()
        plt.show()
    
    def get_top_tracks_by_feature(self, df=None, feature='danceability', top_n=10, ascending=False):
        """Get top tracks by a specific audio feature"""
        if df is None:
            if self.tracks_df.empty:
                print("No data available for analysis")
                return pd.DataFrame()
            df = self.tracks_df
        
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in data")
            return pd.DataFrame()
            
        # Sort by feature
        sorted_df = df.sort_values(by=feature, ascending=ascending)
        
        # Select top N tracks
        top_tracks = sorted_df.head(top_n)[['track_name', 'track_artist', feature, 'track_popularity']]
        
        return top_tracks
    
    def save_to_csv(self, df=None, filename='spotify_scraped_data.csv'):
        """Save the DataFrame to a CSV file"""
        if df is None:
            if self.tracks_df.empty:
                print("No data available to save")
                return False
            df = self.tracks_df
            
        try:
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

def main():
    # Create scraper instance
    scraper = SpotifyLiveScraper()
    
    # Check if the API is connected
    if scraper.sp is None:
        print("Cannot continue without Spotify API connection. Please check your credentials in .env file.")
        return
    
    # Main menu
    while True:
        print("\n===== Spotify Live Data Scraper =====")
        print("1. Search for tracks")
        print("2. Get tracks from a playlist")
        print("3. Get new releases")
        print("4. Get related artist tracks")
        print("5. Get recommendations")
        print("6. Visualize audio features")
        print("7. Compare specific tracks")
        print("8. Analyze genre features")
        print("9. Get top tracks by feature")
        print("10. Save current data to CSV")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-10): ")
        
        if choice == '0':
            print("Exiting...")
            break
            
        elif choice == '1':
            # Search for tracks
            query = input("Enter search query: ")
            limit = int(input("Number of results (default: 10): ") or 10)
            
            results = scraper.search_tracks(query, limit=limit)
            scraper.tracks_df = results
            
            if not results.empty:
                print("\nSearch results:")
                for i, (_, track) in enumerate(results.iterrows()):
                    print(f"{i+1}. {track['track_name']} by {track['track_artist']} (Popularity: {track['track_popularity']})")
            
        elif choice == '2':
            # Get tracks from a playlist
            playlist_id = input("Enter Spotify playlist ID: ")
            limit = int(input("Maximum number of tracks to fetch (default: 100): ") or 100)
            
            results = scraper.get_playlist_tracks(playlist_id, limit=limit)
            scraper.tracks_df = results
            
        elif choice == '3':
            # Get new releases
            country = input("Enter country code (default: US): ") or 'US'
            limit = int(input("Number of albums to fetch (default: 20): ") or 20)
            
            results = scraper.get_new_releases(country=country, limit=limit)
            scraper.tracks_df = results
            
        elif choice == '4':
            # Get related artist tracks
            # First search for an artist
            artist_query = input("Enter artist name: ")
            
            try:
                artist_results = scraper.sp.search(q=artist_query, type='artist', limit=5)
                
                if artist_results and 'artists' in artist_results and 'items' in artist_results['artists']:
                    artists = artist_results['artists']['items']
                    
                    if artists:
                        print("\nArtist search results:")
                        for i, artist in enumerate(artists):
                            print(f"{i+1}. {artist['name']} (Popularity: {artist['popularity']})")
                            
                        artist_idx = int(input("\nSelect artist (number): ")) - 1
                        
                        if 0 <= artist_idx < len(artists):
                            artist = artists[artist_idx]
                            print(f"Getting related artists and their top tracks for {artist['name']}...")
                            
                            results = scraper.get_related_artists_top_tracks(artist['id'])
                            scraper.tracks_df = results
                        else:
                            print("Invalid selection")
                    else:
                        print("No artists found")
                else:
                    print("No artists found")
            except Exception as e:
                print(f"Error searching for artist: {e}")
            
        elif choice == '5':
            # Get recommendations
            if scraper.tracks_df.empty:
                print("Please search for tracks first to use as seeds")
                continue
                
            # Show current tracks
            print("\nCurrent tracks (potential seeds):")
            for i, (_, track) in enumerate(scraper.tracks_df.iterrows()):
                print(f"{i+1}. {track['track_name']} by {track['track_artist']}")
                
            # Select seed tracks
            seed_indices = input("\nSelect seed tracks (comma-separated numbers, max 5): ")
            try:
                indices = [int(idx.strip()) - 1 for idx in seed_indices.split(",")]
                seed_tracks = [scraper.tracks_df.iloc[idx]['track_id'] for idx in indices if 0 <= idx < len(scraper.tracks_df)]
                
                if not seed_tracks or len(seed_tracks) > 5:
                    print("Invalid selection. Please select between 1 and 5 tracks.")
                    continue
                    
                # Get recommendations
                limit = int(input("Number of recommendations (default: 20): ") or 20)
                
                results = scraper.get_recommendations(seed_tracks=seed_tracks, limit=limit)
                
                if not results.empty:
                    print("\nRecommendations:")
                    for i, (_, track) in enumerate(results.iterrows()):
                        print(f"{i+1}. {track['track_name']} by {track['track_artist']} (Popularity: {track['track_popularity']})")
                        
                    # Ask if user wants to replace current data with recommendations
                    replace = input("\nReplace current data with these recommendations? (y/n): ").lower() == 'y'
                    if replace:
                        scraper.tracks_df = results
            except Exception as e:
                print(f"Error: {e}")
            
        elif choice == '6':
            # Visualize audio features
            if scraper.tracks_df.empty:
                print("No data available. Please fetch some tracks first.")
                continue
                
            scraper.visualize_features()
            
        elif choice == '7':
            # Compare specific tracks
            if scraper.tracks_df.empty:
                print("No data available. Please fetch some tracks first.")
                continue
                
            # Show current tracks
            print("\nCurrent tracks:")
            for i, (_, track) in enumerate(scraper.tracks_df.head(20).iterrows()):
                print(f"{i+1}. {track['track_name']} by {track['track_artist']}")
                
            if len(scraper.tracks_df) > 20:
                print(f"...and {len(scraper.tracks_df) - 20} more tracks")
                
            # Select tracks to compare
            track_indices = input("\nSelect tracks to compare (comma-separated numbers, max 5): ")
            try:
                indices = [int(idx.strip()) - 1 for idx in track_indices.split(",")]
                if indices:
                    scraper.visualize_track_comparison(scraper.tracks_df, indices)
                else:
                    print("No tracks selected")
            except Exception as e:
                print(f"Error: {e}")
            
        elif choice == '8':
            # Analyze genre features
            if scraper.tracks_df.empty:
                print("No data available. Please fetch some tracks first.")
                continue
                
            scraper.analyze_genre_features()
            
        elif choice == '9':
            # Get top tracks by feature
            if scraper.tracks_df.empty:
                print("No data available. Please fetch some tracks first.")
                continue
                
            # Available features
            features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                       'instrumentalness', 'liveness', 'valence', 'tempo', 'track_popularity']
            
            # Filter to available features
            available_features = [f for f in features if f in scraper.tracks_df.columns]
            
            print("\nAvailable features:")
            for i, feature in enumerate(available_features):
                print(f"{i+1}. {feature}")
                
            feature_idx = int(input("\nSelect feature (number): ")) - 1
            
            if 0 <= feature_idx < len(available_features):
                feature = available_features[feature_idx]
                
                # Ascending or descending
                ascending = input("Sort ascending (low to high)? (y/n): ").lower() == 'y'
                
                # Number of tracks
                top_n = int(input("Number of tracks to show (default: 10): ") or 10)
                
                top_tracks = scraper.get_top_tracks_by_feature(feature=feature, top_n=top_n, ascending=ascending)
                
                if not top_tracks.empty:
                    print(f"\n{'Bottom' if ascending else 'Top'} {top_n} tracks by {feature}:")
                    for i, (_, track) in enumerate(top_tracks.iterrows()):
                        print(f"{i+1}. {track['track_name']} by {track['track_artist']} ({feature}: {track[feature]:.3f})")
            else:
                print("Invalid feature selection")
            
        elif choice == '10':
            # Save current data to CSV
            if scraper.tracks_df.empty:
                print("No data available to save. Please fetch some tracks first.")
                continue
                
            filename = input("Enter filename (default: spotify_scraped_data.csv): ") or 'spotify_scraped_data.csv'
            scraper.save_to_csv(filename=filename)
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
