import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import requests
from PIL import Image
from io import BytesIO
import time

# Load environment variables (create a .env file with your Spotify API credentials)
load_dotenv()

# Spotify API credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

class SpotifyDataScraper:
    def __init__(self, client_id=None, client_secret=None):
        """Initialize the Spotify scraper with credentials"""
        self.client_id = client_id or CLIENT_ID
        self.client_secret = client_secret or CLIENT_SECRET
        
        if not self.client_id or not self.client_secret:
            print("Warning: Spotify API credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
            print("Using local CSV data instead of live Spotify API.")
            self.sp = None
        else:
            # Set up Spotify client
            client_credentials_manager = SpotifyClientCredentials(
                client_id=self.client_id, 
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        
        # Load existing data if available
        self.high_popularity_df = None
        self.low_popularity_df = None
        self.combined_df = None
        self.try_load_csv_data()
    
    def try_load_csv_data(self):
        """Try to load existing CSV data files"""
        try:
            self.high_popularity_df = pd.read_csv("high_popularity_spotify_data.csv")
            print(f"Loaded high popularity data: {len(self.high_popularity_df)} tracks")
        except FileNotFoundError:
            print("High popularity data file not found.")
        
        try:
            self.low_popularity_df = pd.read_csv("low_popularity_spotify_data.csv")
            print(f"Loaded low popularity data: {len(self.low_popularity_df)} tracks")
        except FileNotFoundError:
            print("Low popularity data file not found.")
            
        # Combine datasets if both are available
        if self.high_popularity_df is not None and self.low_popularity_df is not None:
            self.combined_df = pd.concat([self.high_popularity_df, self.low_popularity_df], ignore_index=True)
            print(f"Combined dataset has {len(self.combined_df)} tracks")
    
    def get_playlist_tracks(self, playlist_id):
        """Get all tracks from a playlist"""
        if not self.sp:
            print("Cannot get playlist tracks: Spotify API not initialized")
            return []
            
        results = self.sp.playlist_tracks(playlist_id)
        tracks = results['items']
        
        # Handle pagination for large playlists
        while results['next']:
            results = self.sp.next(results)
            tracks.extend(results['items'])
            
        return tracks
    
    def get_audio_features(self, track_ids):
        """Get audio features for a list of track IDs"""
        if not self.sp:
            print("Cannot get audio features: Spotify API not initialized")
            return []
            
        # Spotify API allows up to 100 track IDs per request
        audio_features = []
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            audio_features.extend(self.sp.audio_features(batch))
            # Rate limiting - avoid hitting API limits
            if i + 100 < len(track_ids):
                time.sleep(1)
                
        return audio_features
    
    def get_playlist_data(self, playlist_id, playlist_name=None, playlist_genre=None, playlist_subgenre=None):
        """Get all data for tracks in a playlist"""
        if not self.sp:
            print("Cannot get playlist data: Spotify API not initialized")
            return pd.DataFrame()
            
        # Get playlist details if not provided
        if not playlist_name:
            playlist = self.sp.playlist(playlist_id)
            playlist_name = playlist['name']
            
        # Get all tracks
        print(f"Fetching tracks for playlist: {playlist_name}")
        tracks = self.get_playlist_tracks(playlist_id)
        
        if not tracks:
            print("No tracks found in playlist")
            return pd.DataFrame()
            
        # Extract track IDs and details
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
                'track_album_name': track['album']['name'],
                'track_album_id': track['album']['id'],
                'track_album_release_date': track['album']['release_date'],
                'duration_ms': track['duration_ms'],
                'playlist_id': playlist_id,
                'playlist_name': playlist_name
            }
            
            # Add genre/subgenre if provided
            if playlist_genre:
                track_info['playlist_genre'] = playlist_genre
            if playlist_subgenre:
                track_info['playlist_subgenre'] = playlist_subgenre
                
            track_data.append(track_info)
        
        # Get audio features for all tracks
        print(f"Fetching audio features for {len(track_ids)} tracks")
        audio_features = self.get_audio_features(track_ids)
        
        # Combine track data with audio features
        for i, features in enumerate(audio_features):
            if features and i < len(track_data):
                # Add audio features to track data
                for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                               'speechiness', 'acousticness', 'instrumentalness',
                               'liveness', 'valence', 'tempo', 'time_signature',
                               'analysis_url', 'id', 'type']:
                    track_data[i][feature] = features[feature]
        
        # Create DataFrame
        df = pd.DataFrame(track_data)
        print(f"Created DataFrame with {len(df)} tracks")
        return df
    
    def search_and_get_tracks(self, query, limit=10):
        """Search for tracks and get their audio features"""
        if not self.sp:
            print("Cannot search tracks: Spotify API not initialized")
            return pd.DataFrame()
            
        # Search for tracks
        results = self.sp.search(q=query, type='track', limit=limit)
        tracks = results['tracks']['items']
        
        if not tracks:
            print(f"No tracks found for query: {query}")
            return pd.DataFrame()
        
        # Extract track IDs
        track_ids = [track['id'] for track in tracks]
        
        # Get audio features
        audio_features = self.get_audio_features(track_ids)
        
        # Combine track data with audio features
        track_data = []
        for i, track in enumerate(tracks):
            if i < len(audio_features) and audio_features[i]:
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
                    'album_image_url': track['album']['images'][0]['url'] if track['album']['images'] else None
                }
                
                # Add audio features
                features = audio_features[i]
                for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                              'speechiness', 'acousticness', 'instrumentalness',
                              'liveness', 'valence', 'tempo', 'time_signature',
                              'analysis_url', 'id', 'type']:
                    track_info[feature] = features[feature]
                    
                track_data.append(track_info)
        
        # Create DataFrame
        df = pd.DataFrame(track_data)
        return df
    
    def get_recommendations(self, seed_tracks=None, seed_artists=None, seed_genres=None, limit=20, **kwargs):
        """Get track recommendations based on seeds and optional parameters"""
        if not self.sp:
            print("Cannot get recommendations: Spotify API not initialized")
            return pd.DataFrame()
            
        # Validate seed inputs
        seed_count = sum(1 for seed in [seed_tracks, seed_artists, seed_genres] if seed)
        if seed_count == 0:
            print("At least one seed (tracks, artists, or genres) is required")
            return pd.DataFrame()
            
        # Format seed inputs
        if seed_tracks and isinstance(seed_tracks, list):
            seed_tracks = seed_tracks[:5]  # API allows up to 5 seed tracks
        if seed_artists and isinstance(seed_artists, list):
            seed_artists = seed_artists[:5]  # API allows up to 5 seed artists
        if seed_genres and isinstance(seed_genres, list):
            seed_genres = seed_genres[:5]  # API allows up to 5 seed genres
            
        # Get recommendations
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
        
        # Extract track IDs
        track_ids = [track['id'] for track in tracks]
        
        # Get audio features
        audio_features = self.get_audio_features(track_ids)
        
        # Combine track data with audio features
        track_data = []
        for i, track in enumerate(tracks):
            if i < len(audio_features) and audio_features[i]:
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
                    'album_image_url': track['album']['images'][0]['url'] if track['album']['images'] else None
                }
                
                # Add audio features
                features = audio_features[i]
                for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                              'speechiness', 'acousticness', 'instrumentalness',
                              'liveness', 'valence', 'tempo', 'time_signature',
                              'analysis_url', 'id', 'type']:
                    track_info[feature] = features[feature]
                    
                track_data.append(track_info)
        
        # Create DataFrame
        df = pd.DataFrame(track_data)
        return df
    
    def analyze_track_features(self, df=None):
        """Analyze audio features and return statistics"""
        if df is None:
            if self.combined_df is not None:
                df = self.combined_df
            elif self.high_popularity_df is not None:
                df = self.high_popularity_df
            elif self.low_popularity_df is not None:
                df = self.low_popularity_df
            else:
                print("No data available for analysis")
                return None
        
        # Select numeric audio features
        features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                   'instrumentalness', 'liveness', 'valence', 'tempo', 
                   'loudness', 'track_popularity']
        
        # Filter to include only columns that exist in the DataFrame
        features = [f for f in features if f in df.columns]
        
        if not features:
            print("No audio features found in DataFrame")
            return None
            
        # Calculate statistics
        stats = df[features].describe()
        
        # Calculate correlations
        correlations = df[features].corr()
        
        return {
            'statistics': stats,
            'correlations': correlations
        }
    
    def cluster_tracks(self, df=None, n_clusters=5):
        """Cluster tracks based on audio features"""
        if df is None:
            if self.combined_df is not None:
                df = self.combined_df
            elif self.high_popularity_df is not None:
                df = self.high_popularity_df
            elif self.low_popularity_df is not None:
                df = self.low_popularity_df
            else:
                print("No data available for clustering")
                return None
        
        # Select features for clustering
        features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                   'instrumentalness', 'liveness', 'valence']
        
        # Filter to include only columns that exist in the DataFrame
        features = [f for f in features if f in df.columns]
        
        if not features:
            print("No audio features found in DataFrame for clustering")
            return None
            
        # Handle missing values
        df_clean = df.dropna(subset=features)
        
        if len(df_clean) < n_clusters:
            print(f"Not enough data points ({len(df_clean)}) for {n_clusters} clusters")
            return None
            
        # Scale features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df_clean[features])
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_clean['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Calculate cluster centers in original scale
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Create cluster profile
        cluster_profile = pd.DataFrame(cluster_centers, columns=features)
        cluster_profile.index.name = 'cluster'
        
        # Get representative tracks for each cluster
        representative_tracks = []
        
        for cluster_id in range(n_clusters):
            cluster_tracks = df_clean[df_clean['cluster'] == cluster_id]
            
            # Find track closest to cluster center
            if len(cluster_tracks) > 0:
                # Get scaled features for this cluster
                cluster_scaled = scaled_features[df_clean['cluster'] == cluster_id]
                # Calculate distance to cluster center
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.sqrt(((cluster_scaled - center) ** 2).sum(axis=1))
                # Get index of track with minimum distance
                representative_idx = distances.argmin()
                representative = cluster_tracks.iloc[representative_idx]
                
                representative_tracks.append({
                    'cluster': cluster_id,
                    'track_id': representative['track_id'],
                    'track_name': representative['track_name'],
                    'track_artist': representative['track_artist'],
                    'track_popularity': representative.get('track_popularity', 'N/A'),
                    'distance': distances[representative_idx]
                })
        
        return {
            'clustered_data': df_clean,
            'cluster_profile': cluster_profile,
            'representative_tracks': representative_tracks
        }
    
    def visualize_features(self, df=None, features=None):
        """Visualize audio features distribution"""
        if df is None:
            if self.combined_df is not None:
                df = self.combined_df
            elif self.high_popularity_df is not None:
                df = self.high_popularity_df
            elif self.low_popularity_df is not None:
                df = self.low_popularity_df
            else:
                print("No data available for visualization")
                return
        
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
    
    def visualize_clusters(self, cluster_results):
        """Visualize clustering results"""
        if not cluster_results or 'clustered_data' not in cluster_results:
            print("No clustering results to visualize")
            return
            
        df = cluster_results['clustered_data']
        cluster_profile = cluster_results['cluster_profile']
        
        # 1. Create radar chart for cluster profiles
        categories = cluster_profile.columns
        n_clusters = len(cluster_profile)
        
        # Set up the figure
        fig = plt.figure(figsize=(12, 8))
        
        # Create a color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        # Number of variables
        N = len(categories)
        
        # What will be the angle of each axis in the plot (divide the plot into equal parts)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create the subplot
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each cluster
        for i, row in cluster_profile.iterrows():
            values = row.values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i])
            ax.fill(angles, values, color=colors[i], alpha=0.1)
            
        # Add legend
        plt.legend([f"Cluster {i}" for i in range(n_clusters)], loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Cluster Profiles", size=20, y=1.1)
        plt.tight_layout()
        plt.show()
        
        # 2. Create 2D scatter plot with PCA for dimension reduction
        from sklearn.decomposition import PCA
        
        # Select features for PCA
        features = cluster_profile.columns.tolist()
        
        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df[features])
        
        # Create a DataFrame with the principal components
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['cluster'] = df['cluster']
        
        # Plot
        plt.figure(figsize=(10, 8))
        for i in range(n_clusters):
            cluster_data = pca_df[pca_df['cluster'] == i]
            plt.scatter(cluster_data['PC1'], cluster_data['PC2'], c=[colors[i]], label=f'Cluster {i}')
            
        plt.title('Track Clusters (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # 3. Bar chart of cluster sizes
        cluster_sizes = df['cluster'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(cluster_sizes.index, cluster_sizes.values, color=colors[:len(cluster_sizes)])
        
        plt.title('Number of Tracks in Each Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Tracks')
        plt.xticks(cluster_sizes.index)
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
            
        plt.tight_layout()
        plt.show()
    
    def compare_playlists(self, playlist_names):
        """Compare audio features across different playlists"""
        if self.combined_df is None and self.high_popularity_df is None and self.low_popularity_df is None:
            print("No data available for comparison")
            return
            
        df = self.combined_df if self.combined_df is not None else (
            self.high_popularity_df if self.high_popularity_df is not None else self.low_popularity_df
        )
        
        # Filter to include only specified playlists
        if playlist_names:
            df_filtered = df[df['playlist_name'].isin(playlist_names)]
            if len(df_filtered) == 0:
                print(f"No data found for playlists: {', '.join(playlist_names)}")
                return
            df = df_filtered
            
        # Features to compare
        features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                   'instrumentalness', 'liveness', 'valence']
        
        # Filter to include only columns that exist in the DataFrame
        features = [f for f in features if f in df.columns]
        
        if not features:
            print("No audio features found in DataFrame for comparison")
            return
            
        # Group by playlist and calculate mean for each feature
        grouped_data = df.groupby('playlist_name')[features].mean().reset_index()
        
        # Melt the data for easier plotting
        melted_data = pd.melt(grouped_data, id_vars=['playlist_name'], 
                             value_vars=features, 
                             var_name='Feature', value_name='Value')
        
        # Create grouped bar chart
        plt.figure(figsize=(14, 8))
        sns.barplot(x='playlist_name', y='Value', hue='Feature', data=melted_data)
        plt.title('Comparison of Audio Features Across Playlists')
        plt.xlabel('Playlist')
        plt.ylabel('Average Value')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        # Create radar chart for each playlist
        # Number of variables
        N = len(features)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Get unique playlists
        playlists = grouped_data['playlist_name'].unique()
        
        # Create a color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, len(playlists)))
        
        # Create the subplot
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], features, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each playlist
        for i, playlist in enumerate(playlists):
            values = grouped_data[grouped_data['playlist_name'] == playlist][features].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=playlist)
            ax.fill(angles, values, color=colors[i], alpha=0.1)
            
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Playlist Profiles", size=20, y=1.1)
        plt.tight_layout()
        plt.show()
    
    def get_top_tracks_by_feature(self, feature, top_n=10, ascending=False):
        """Get top tracks by a specific audio feature"""
        if self.combined_df is None and self.high_popularity_df is None and self.low_popularity_df is None:
            print("No data available")
            return pd.DataFrame()
            
        df = self.combined_df if self.combined_df is not None else (
            self.high_popularity_df if self.high_popularity_df is not None else self.low_popularity_df
        )
        
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in data")
            return pd.DataFrame()
            
        # Sort by feature
        sorted_df = df.sort_values(by=feature, ascending=ascending)
        
        # Select top N tracks
        top_tracks = sorted_df.head(top_n)[['track_name', 'track_artist', feature, 'track_popularity']]
        
        return top_tracks
    
    def get_track_album_art(self, track_id):
        """Get album art for a track"""
        if not self.sp:
            print("Cannot get album art: Spotify API not initialized")
            return None
            
        try:
            track = self.sp.track(track_id)
            if track and 'album' in track and 'images' in track['album'] and track['album']['images']:
                image_url = track['album']['images'][0]['url']
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                return img
        except Exception as e:
            print(f"Error getting album art: {e}")
            
        return None
    
    def get_track_analysis(self, track_id):
        """Get detailed audio analysis for a track"""
        if not self.sp:
            print("Cannot get track analysis: Spotify API not initialized")
            return None
            
        try:
            return self.sp.audio_analysis(track_id)
        except Exception as e:
            print(f"Error getting track analysis: {e}")
            return None
    
    def save_to_csv(self, df, filename):
        """Save DataFrame to CSV file"""
        try:
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def fetch_and_save_playlist(self, playlist_id, filename, playlist_name=None, playlist_genre=None, playlist_subgenre=None):
        """Fetch data for a playlist and save it to a CSV file"""
        if not self.sp:
            print("Cannot fetch playlist: Spotify API not initialized")
            return False
            
        df = self.get_playlist_data(playlist_id, playlist_name, playlist_genre, playlist_subgenre)
        
        if df.empty:
            print("No data to save")
            return False
            
        return self.save_to_csv(df, filename)
    
    def genre_analysis(self):
        """Analyze and compare different genres in the data"""
        if self.combined_df is None and self.high_popularity_df is None and self.low_popularity_df is None:
            print("No data available for genre analysis")
            return
            
        df = self.combined_df if self.combined_df is not None else (
            self.high_popularity_df if self.high_popularity_df is not None else self.low_popularity_df
        )
        
        if 'playlist_genre' not in df.columns:
            print("No genre information found in data")
            return
            
        # Features to analyze
        features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                   'instrumentalness', 'liveness', 'valence']
        
        # Filter to include only columns that exist in the DataFrame
        features = [f for f in features if f in df.columns]
        
        if not features:
            print("No audio features found in DataFrame for genre analysis")
            return
            
        # Group by genre and calculate mean for each feature
        grouped_data = df.groupby('playlist_genre')[features].mean().reset_index()
        
        # Melt the data for easier plotting
        melted_data = pd.melt(grouped_data, id_vars=['playlist_genre'], 
                             value_vars=features, 
                             var_name='Feature', value_name='Value')
        
        # Create grouped bar chart
        plt.figure(figsize=(14, 8))
        sns.barplot(x='playlist_genre', y='Value', hue='Feature', data=melted_data)
        plt.title('Comparison of Audio Features Across Genres')
        plt.xlabel('Genre')
        plt.ylabel('Average Value')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        # Create radar chart for each genre
        # Number of variables
        N = len(features)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Get unique genres
        genres = grouped_data['playlist_genre'].unique()
        
        # Create a color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, len(genres)))
        
        # Create the subplot
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], features, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each genre
        for i, genre in enumerate(genres):
            values = grouped_data[grouped_data['playlist_genre'] == genre][features].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=genre)
            ax.fill(angles, values, color=colors[i], alpha=0.1)
            
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Genre Profiles", size=20, y=1.1)
        plt.tight_layout()
        plt.show()
        
        # Count tracks in each genre
        genre_counts = df['playlist_genre'].value_counts()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=genre_counts.index, y=genre_counts.values)
        plt.title('Number of Tracks in Each Genre')
        plt.xlabel('Genre')
        plt.ylabel('Number of Tracks')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on top of bars
        for i, count in enumerate(genre_counts.values):
            plt.text(i, count + 10, str(count), ha='center')
            
        plt.tight_layout()
        plt.show()
        
        # Popularity by genre
        if 'track_popularity' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='playlist_genre', y='track_popularity', data=df)
            plt.title('Track Popularity by Genre')
            plt.xlabel('Genre')
            plt.ylabel('Popularity')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()


# Example usage
if __name__ == "__main__":
    # Create scraper instance
    scraper = SpotifyDataScraper()
    
    # If you want to use the Spotify API, set your credentials in a .env file
    # or pass them directly:
    # scraper = SpotifyDataScraper(client_id='your_client_id', client_secret='your_client_secret')
    
    # Analyze and visualize the data we already have
    print("\nAnalyzing data...")
    if scraper.high_popularity_df is not None:
        # Basic analysis
        analysis = scraper.analyze_track_features(scraper.high_popularity_df)
        if analysis:
            print("\nStatistics:")
            print(analysis['statistics'])
            
            print("\nCorrelations:")
            print(analysis['correlations'])
        
        # Visualize features
        print("\nVisualizing audio features...")
        scraper.visualize_features(scraper.high_popularity_df)
        
        # Cluster analysis
        print("\nPerforming cluster analysis...")
        clusters = scraper.cluster_tracks(scraper.high_popularity_df, n_clusters=4)
        if clusters:
            scraper.visualize_clusters(clusters)
            
            print("\nRepresentative tracks for each cluster:")
            for track in clusters['representative_tracks']:
                print(f"Cluster {track['cluster']}: {track['track_name']} by {track['track_artist']} (Popularity: {track['track_popularity']})")
        
        # Genre analysis (if genre data exists)
        if 'playlist_genre' in scraper.high_popularity_df.columns:
            print("\nPerforming genre analysis...")
            scraper.genre_analysis()
            
        # Get top tracks by feature
        print("\nTop 5 most danceable tracks:")
        top_danceable = scraper.get_top_tracks_by_feature('danceability', top_n=5)
        if not top_danceable.empty:
            print(top_danceable)
            
        print("\nTop 5 most energetic tracks:")
        top_energy = scraper.get_top_tracks_by_feature('energy', top_n=5)
        if not top_energy.empty:
            print(top_energy)
    
    # If you have Spotify API credentials, you can use these functions:
    # Get recommendations based on a seed track
    # recommendations = scraper.get_recommendations(seed_tracks=['2plbrEY59IikOBgBGLjaoe'], limit=10)
    
    # Search for tracks
    # search_results = scraper.search_and_get_tracks("billie eilish happier than ever", limit=5)
    
    # Fetch and save a playlist
    # scraper.fetch_and_save_playlist('37i9dQZF1DXcBWIGoYBM5M', 'new_playlist_data.csv', 'Today\'s Top Hits', 'pop', 'mainstream')
