import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Spotify API credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

class SpotifyAnalyzer:
    def __init__(self):
        """Initialize the Spotify analyzer with API credentials and data loading options"""
        self.client_id = CLIENT_ID
        self.client_secret = CLIENT_SECRET
        
        # Initialize dataframes
        self.csv_df = None
        self.api_df = None
        self.current_df = None
        self.data_source = None
        
        # Initialize Spotify client
        if not self.client_id or not self.client_secret:
            print("Warning: Spotify API credentials not found. API features will be unavailable.")
            print("Create a .env file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to enable API features.")
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
        
        # Important audio features for analysis
        self.audio_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'time_signature'
        ]
    
    def load_csv_data(self, file_path=None):
        """Load data from CSV file"""
        if file_path is None:
            # Check for default CSV files in the current directory
            possible_files = [
                "high_popularity_spotify_data.csv",
                "low_popularity_spotify_data.csv",
                "spotify_data.csv"
            ]
            
            for file in possible_files:
                if os.path.exists(file):
                    file_path = file
                    print(f"Found CSV file: {file_path}")
                    break
            
            if file_path is None:
                print("No default CSV files found. Please provide a file path.")
                return False
        
        try:
            print(f"Loading data from {file_path}...")
            self.csv_df = pd.read_csv(file_path)
            print(f"Successfully loaded {len(self.csv_df)} tracks from CSV.")
            
            # Set as current dataframe
            self.current_df = self.csv_df
            self.data_source = "CSV"
            
            return True
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return False
    
    def search_tracks(self, query, limit=10):
        """Search for tracks on Spotify API"""
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
                        for feature in self.audio_features:
                            if feature in features:
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
    
    def get_new_releases(self, country='US', limit=20):
        """Get new releases from Spotify"""
        if not self.sp:
            print("Error: Spotify API not connected.")
            return pd.DataFrame()
        
        try:
            print(f"Getting {limit} new releases...")
            new_releases = self.sp.new_releases(country=country, limit=limit)
            
            if 'albums' not in new_releases or not new_releases['albums']['items']:
                print("No new releases found.")
                return pd.DataFrame()
            
            # Get tracks from each album
            all_tracks = []
            album_ids = [album['id'] for album in new_releases['albums']['items']]
            
            for album_id in album_ids:
                try:
                    album_tracks = self.sp.album_tracks(album_id)
                    album_info = self.sp.album(album_id)
                    
                    for track in album_tracks['items']:
                        track_info = {
                            'track_id': track['id'],
                            'track_name': track['name'],
                            'track_href': track['href'],
                            'uri': track['uri'],
                            'track_artist': ', '.join([artist['name'] for artist in track['artists']]),
                            'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                            'track_album_name': album_info['name'],
                            'track_album_id': album_id,
                            'track_album_release_date': album_info['release_date'],
                            'duration_ms': track['duration_ms'],
                            'album_image_url': album_info['images'][0]['url'] if album_info['images'] else None,
                            'album_popularity': album_info.get('popularity', None)
                        }
                        all_tracks.append(track_info)
                        
                except Exception as e:
                    print(f"Warning: Could not get tracks for album {album_id}: {e}")
                    continue
            
            if not all_tracks:
                print("No tracks found in new releases.")
                return pd.DataFrame()
            
            # Create DataFrame
            tracks_df = pd.DataFrame(all_tracks)
            
            # Try to get audio features
            try:
                track_ids = tracks_df['track_id'].tolist()
                
                # Process in batches of 100 (Spotify API limit)
                all_features = []
                for i in range(0, len(track_ids), 100):
                    batch_ids = track_ids[i:i+100]
                    features = self.sp.audio_features(batch_ids)
                    all_features.extend(features)
                    time.sleep(0.5)  # Avoid rate limiting
                
                # Add audio features to DataFrame
                for i, features in enumerate(all_features):
                    if features and i < len(tracks_df):
                        for feature in self.audio_features:
                            if feature in features:
                                tracks_df.loc[i, feature] = features[feature]
            except Exception as e:
                print(f"Warning: Could not get audio features: {e}")
            
            # Try to get artist genres
            try:
                artist_ids = tracks_df['artist_id'].dropna().unique().tolist()
                
                if artist_ids:
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
            
            print(f"Found {len(tracks_df)} tracks in new releases.")
            return tracks_df
        
        except Exception as e:
            print(f"Error getting new releases: {e}")
            return pd.DataFrame()
    
    def get_recommendations(self, seed_tracks=None, seed_artists=None, limit=20):
        """Get recommendations based on seed tracks or artists"""
        if not self.sp:
            print("Error: Spotify API not connected.")
            return pd.DataFrame()
        
        if not seed_tracks and not seed_artists:
            print("Error: No seed tracks or artists provided.")
            return pd.DataFrame()
        
        try:
            print("Getting recommendations...")
            
            # Get recommendations
            recommendations = self.sp.recommendations(
                seed_tracks=seed_tracks if seed_tracks else None,
                seed_artists=seed_artists if seed_artists else None,
                limit=limit
            )
            
            if 'tracks' not in recommendations or not recommendations['tracks']:
                print("No recommendations found.")
                return pd.DataFrame()
            
            # Extract track info
            track_data = []
            track_ids = []
            
            for track in recommendations['tracks']:
                track_ids.append(track['id'])
                
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
            
            # Get audio features
            try:
                features = self.sp.audio_features(track_ids)
                
                for i, feat in enumerate(features):
                    if feat and i < len(tracks_df):
                        for feature in self.audio_features:
                            if feature in feat:
                                tracks_df.loc[i, feature] = feat[feature]
            except Exception as e:
                print(f"Warning: Could not get audio features: {e}")
            
            print(f"Found {len(tracks_df)} recommended tracks.")
            return tracks_df
        
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return pd.DataFrame()
    
    def get_playlist_tracks(self, playlist_id):
        """Get tracks from a Spotify playlist"""
        if not self.sp:
            print("Error: Spotify API not connected.")
            return pd.DataFrame()
        
        try:
            print(f"Getting tracks from playlist {playlist_id}...")
            
            # Get playlist info
            try:
                playlist_info = self.sp.playlist(playlist_id)
                playlist_name = playlist_info['name']
                print(f"Playlist: {playlist_name}")
            except:
                playlist_name = "Unknown Playlist"
            
            # Get all tracks (paginate through results)
            results = self.sp.playlist_tracks(playlist_id)
            tracks = results['items']
            
            while results['next']:
                results = self.sp.next(results)
                tracks.extend(results['items'])
            
            if not tracks:
                print("No tracks found in playlist.")
                return pd.DataFrame()
            
            # Extract track info
            track_data = []
            track_ids = []
            
            for item in tracks:
                track = item['track']
                if not track:
                    continue
                    
                track_ids.append(track['id'])
                
                track_info = {
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'track_popularity': track.get('popularity', None),
                    'track_href': track['href'],
                    'uri': track['uri'],
                    'track_artist': ', '.join([artist['name'] for artist in track['artists']]),
                    'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                    'track_album_name': track['album']['name'],
                    'track_album_id': track['album']['id'],
                    'track_album_release_date': track['album']['release_date'],
                    'duration_ms': track['duration_ms'],
                    'album_image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'playlist_name': playlist_name,
                    'playlist_id': playlist_id,
                    'added_at': item['added_at']
                }
                
                track_data.append(track_info)
            
            # Create DataFrame
            tracks_df = pd.DataFrame(track_data)
            
            # Get audio features in batches
            try:
                all_features = []
                for i in range(0, len(track_ids), 100):
                    batch_ids = track_ids[i:i+100]
                    features = self.sp.audio_features(batch_ids)
                    all_features.extend(features)
                    time.sleep(0.5)  # Avoid rate limiting
                
                for i, feat in enumerate(all_features):
                    if feat and i < len(tracks_df):
                        for feature in self.audio_features:
                            if feature in feat:
                                tracks_df.loc[i, feature] = feat[feature]
            except Exception as e:
                print(f"Warning: Could not get audio features: {e}")
            
            print(f"Found {len(tracks_df)} tracks in playlist.")
            return tracks_df
        
        except Exception as e:
            print(f"Error getting playlist tracks: {e}")
            return pd.DataFrame()
    
    def set_current_data(self, df, source):
        """Set the current dataframe for analysis and visualization"""
        self.current_df = df
        self.data_source = source
        print(f"Current data set to {source} with {len(df)} tracks.")
    
    def visualize_feature_distributions(self):
        """Visualize distributions of audio features"""
        if self.current_df is None or len(self.current_df) == 0:
            print("No data available for visualization.")
            return
        
        # Get numeric features
        numeric_features = [col for col in self.audio_features 
                           if col in self.current_df.columns 
                           and pd.api.types.is_numeric_dtype(self.current_df[col])]
        
        if not numeric_features:
            print("No numeric features available for visualization.")
            return
        
        print(f"Visualizing distributions for {len(numeric_features)} features...")
        
        # Create the subplots
        fig = make_subplots(
            rows=len(numeric_features), 
            cols=1, 
            subplot_titles=numeric_features,
            vertical_spacing=0.05
        )
        
        # Add histograms for each feature
        for i, feature in enumerate(numeric_features):
            fig.add_trace(
                go.Histogram(
                    x=self.current_df[feature],
                    name=feature,
                    marker_color='#1DB954'  # Spotify green
                ),
                row=i+1, 
                col=1
            )
            
            # Add mean and median lines
            mean_val = self.current_df[feature].mean()
            median_val = self.current_df[feature].median()
            
            fig.add_vline(
                x=mean_val,
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top right",
                row=i+1,
                col=1
            )
            
            fig.add_vline(
                x=median_val,
                line_width=2,
                line_dash="dot",
                line_color="blue",
                annotation_text=f"Median: {median_val:.2f}",
                annotation_position="top left",
                row=i+1,
                col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"Distribution of Audio Features ({len(self.current_df)} tracks)",
            height=300 * len(numeric_features),
            showlegend=False
        )
        
        # Show plot
        fig.show()
    
    def visualize_feature_correlations(self):
        """Visualize correlations between audio features"""
        if self.current_df is None or len(self.current_df) == 0:
            print("No data available for visualization.")
            return
        
        # Get numeric features
        numeric_features = [col for col in self.audio_features 
                           if col in self.current_df.columns 
                           and pd.api.types.is_numeric_dtype(self.current_df[col])]
        
        if 'track_popularity' in self.current_df.columns:
            numeric_features.append('track_popularity')
        
        if len(numeric_features) < 2:
            print("Not enough numeric features for correlation analysis.")
            return
        
        print("Calculating correlations...")
        
        # Calculate correlation matrix
        corr_matrix = self.current_df[numeric_features].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis',
            colorbar=dict(title='Correlation'),
            hoverongaps=False,
            text=np.around(corr_matrix.values, decimals=2),
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='Feature Correlation Heatmap',
            height=600,
            width=700
        )
        
        # Show plot
        fig.show()
    
    def visualize_scatter_plot(self, x_feature=None, y_feature=None, color_by=None):
        """Create interactive scatter plot of features"""
        if self.current_df is None or len(self.current_df) == 0:
            print("No data available for visualization.")
            return
        
        # Get available numeric features
        numeric_features = [col for col in self.current_df.columns 
                           if pd.api.types.is_numeric_dtype(self.current_df[col])]
        
        if len(numeric_features) < 2:
            print("Not enough numeric features for scatter plot.")
            return
        
        # If features not specified, use the first two
        if x_feature is None or x_feature not in self.current_df.columns:
            x_feature = numeric_features[0]
        
        if y_feature is None or y_feature not in self.current_df.columns:
            y_feature = numeric_features[1] if numeric_features[1] != x_feature else numeric_features[0]
        
        # Determine color variable
        if color_by is None:
            if 'track_popularity' in self.current_df.columns:
                color_by = 'track_popularity'
            elif 'artist_popularity' in self.current_df.columns:
                color_by = 'artist_popularity'
            else:
                # Use a third numeric feature if available
                remaining_features = [f for f in numeric_features if f not in [x_feature, y_feature]]
                color_by = remaining_features[0] if remaining_features else None
        
        # Create scatter plot
        if color_by and color_by in self.current_df.columns:
            # Create scatter plot with color
            fig = px.scatter(
                self.current_df,
                x=x_feature,
                y=y_feature,
                color=color_by,
                hover_name='track_name' if 'track_name' in self.current_df.columns else None,
                hover_data=['track_artist'] if 'track_artist' in self.current_df.columns else None,
                title=f"{y_feature} vs {x_feature} (colored by {color_by})"
            )
        else:
            # Create scatter plot without color
            fig = px.scatter(
                self.current_df,
                x=x_feature,
                y=y_feature,
                hover_name='track_name' if 'track_name' in self.current_df.columns else None,
                hover_data=['track_artist'] if 'track_artist' in self.current_df.columns else None,
                title=f"{y_feature} vs {x_feature}"
            )
        
        # Show plot
        fig.show()
    
    def visualize_track_comparison(self, track_indices=None, num_tracks=5):
        """Compare audio features of selected tracks using radar chart"""
        if self.current_df is None or len(self.current_df) == 0:
            print("No data available for visualization.")
            return
        
        # Get numeric features for comparison
        features_to_compare = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence'
        ]
        
        # Filter to features that exist in the dataframe
        features_to_compare = [f for f in features_to_compare if f in self.current_df.columns]
        
        if len(features_to_compare) < 3:
            print("Not enough audio features for comparison.")
            return
        
        # If no indices provided, use the first n tracks
        if track_indices is None:
            if len(self.current_df) <= num_tracks:
                track_indices = list(range(len(self.current_df)))
            else:
                # Select n most popular tracks if popularity is available
                if 'track_popularity' in self.current_df.columns:
                    track_indices = self.current_df.sort_values('track_popularity', ascending=False).head(num_tracks).index.tolist()
                else:
                    track_indices = list(range(num_tracks))
        
        # Ensure track_indices is a list and valid
        if isinstance(track_indices, int):
            track_indices = [track_indices]
        
        track_indices = [i for i in track_indices if i < len(self.current_df)]
        
        if not track_indices:
            print("No valid track indices provided.")
            return
        
        # Create radar chart
        fig = go.Figure()
        
        for idx in track_indices:
            track = self.current_df.iloc[idx]
            
            # Get track name and artist
            track_name = track.get('track_name', f"Track {idx}")
            track_artist = track.get('track_artist', '')
            
            # Get feature values
            values = [track[feature] for feature in features_to_compare]
            
            # Add to chart
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=features_to_compare,
                fill='toself',
                name=f"{track_name} - {track_artist}"
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Track Audio Feature Comparison",
            showlegend=True
        )
        
        # Show plot
        fig.show()
    
    def cluster_tracks(self, n_clusters=5):
        """Cluster tracks based on audio features using K-means"""
        if self.current_df is None or len(self.current_df) == 0:
            print("No data available for clustering.")
            return
        
        # Get numeric features for clustering
        features_for_clustering = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        # Filter to features that exist in the dataframe
        features_for_clustering = [f for f in features_for_clustering if f in self.current_df.columns]
        
        if len(features_for_clustering) < 3:
            print("Not enough audio features for clustering.")
            return
        
        print(f"Clustering tracks using {len(features_for_clustering)} features...")
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.current_df[features_for_clustering])
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to dataframe
        self.current_df['cluster'] = cluster_labels
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)
        
        # Create a dataframe for plotting
        pca_df = pd.DataFrame(
            data=principal_components, 
            columns=['PC1', 'PC2']
        )
        pca_df['cluster'] = cluster_labels
        
        # Add track info if available
        if 'track_name' in self.current_df.columns:
            pca_df['track_name'] = self.current_df['track_name'].values
        
        if 'track_artist' in self.current_df.columns:
            pca_df['track_artist'] = self.current_df['track_artist'].values
        
        # Create scatter plot
        fig = px.scatter(
            pca_df, 
            x='PC1', 
            y='PC2', 
            color='cluster',
            hover_name='track_name' if 'track_name' in pca_df.columns else None,
            hover_data=['track_artist'] if 'track_artist' in pca_df.columns else None,
            title=f"Track Clustering (K-means, {n_clusters} clusters)",
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        # Show plot
        fig.show()
        
        # Print cluster summary
        print("\nCluster Summary:")
        for cluster in range(n_clusters):
            cluster_tracks = self.current_df[self.current_df['cluster'] == cluster]
            print(f"\nCluster {cluster} ({len(cluster_tracks)} tracks):")
            
            # Calculate mean values for features
            means = cluster_tracks[features_for_clustering].mean()
            
            print("Average audio features:")
            for feature, mean in means.items():
                print(f"  {feature}: {mean:.3f}")
            
            # Show some example tracks
            if 'track_name' in self.current_df.columns and 'track_artist' in self.current_df.columns:
                print("\nExample tracks:")
                sample = cluster_tracks.sample(min(3, len(cluster_tracks)))
                for _, track in sample.iterrows():
                    print(f"  {track['track_name']} - {track['track_artist']}")
        
        return self.current_df
    
    def save_to_csv(self, filename=None):
        """Save current dataframe to CSV"""
        if self.current_df is None or len(self.current_df) == 0:
            print("No data available to save.")
            return
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"spotify_analysis_{timestamp}.csv"
        
        try:
            self.current_df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def display_track_details(self, track_index=0):
        """Display detailed information about a specific track"""
        if self.current_df is None or len(self.current_df) == 0:
            print("No data available.")
            return
        
        if track_index < 0 or track_index >= len(self.current_df):
            print(f"Invalid track index. Must be between 0 and {len(self.current_df)-1}")
            return
        
        track = self.current_df.iloc[track_index]
        
        print("\n==== Track Details ====")
        
        # Basic track info
        if 'track_name' in track:
            print(f"Name: {track['track_name']}")
        
        if 'track_artist' in track:
            print(f"Artist: {track['track_artist']}")
        
        if 'track_album_name' in track:
            print(f"Album: {track['track_album_name']}")
        
        if 'track_album_release_date' in track:
            print(f"Release Date: {track['track_album_release_date']}")
        
        if 'track_popularity' in track:
            print(f"Popularity: {track['track_popularity']}")
        
        if 'artist_genres' in track:
            print(f"Genres: {track['artist_genres']}")
        
        # Audio features
        print("\nAudio Features:")
        for feature in self.audio_features:
            if feature in track:
                print(f"  {feature}: {track[feature]}")
        
        # Links
        if 'uri' in track:
            print(f"\nSpotify URI: {track['uri']}")
        
        if 'track_href' in track:
            print(f"API Link: {track['track_href']}")
        
        # Duration
        if 'duration_ms' in track:
            duration_sec = track['duration_ms'] / 1000
            minutes = int(duration_sec // 60)
            seconds = int(duration_sec % 60)
            print(f"\nDuration: {minutes}:{seconds:02d}")


def main():
    # Initialize the analyzer
    analyzer = SpotifyAnalyzer()
    
    # Data source flag
    data_loaded = False
    
    # Main menu loop
    while True:
        print("\n" + "="*50)
        print("SPOTIFY DATA ANALYZER".center(50))
        print("="*50)
        
        # Show data source if data is loaded
        if analyzer.current_df is not None:
            print(f"Current data: {analyzer.data_source} ({len(analyzer.current_df)} tracks)")
        
        print("\nMAIN MENU:")
        print("1. Load Data")
        print("2. Search & Explore")
        print("3. Visualize")
        print("4. Analyze")
        print("5. Save Data")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-5): ")
        
        if choice == '0':
            print("Exiting...")
            break
        
        elif choice == '1':
            # Data loading submenu
            while True:
                print("\n--- DATA LOADING OPTIONS ---")
                print("1. Load from CSV file")
                print("2. Load from Spotify API (search)")
                print("3. Load new releases from Spotify")
                print("4. Load tracks from a playlist")
                print("0. Back to main menu")
                
                data_choice = input("\nEnter your choice (0-4): ")
                
                if data_choice == '0':
                    break
                
                elif data_choice == '1':
                    # Load from CSV
                    file_path = input("Enter CSV file path (or press Enter for default): ")
                    file_path = file_path.strip() if file_path.strip() else None
                    
                    if analyzer.load_csv_data(file_path):
                        data_loaded = True
                    
                    break
                
                elif data_choice == '2':
                    # Load from search
                    if not analyzer.sp:
                        print("Spotify API not connected. Please check your credentials.")
                        continue
                    
                    query = input("Enter search query: ")
                    limit = input("Enter number of results (default 10): ")
                    limit = int(limit) if limit.isdigit() else 10
                    
                    results_df = analyzer.search_tracks(query, limit)
                    
                    if len(results_df) > 0:
                        analyzer.set_current_data(results_df, "API Search")
                        data_loaded = True
                        break
                
                elif data_choice == '3':
                    # Load new releases
                    if not analyzer.sp:
                        print("Spotify API not connected. Please check your credentials.")
                        continue
                    
                    country = input("Enter country code (default US): ")
                    country = country.strip() if country.strip() else "US"
                    
                    limit = input("Enter number of releases (default 20): ")
                    limit = int(limit) if limit.isdigit() else 20
                    
                    results_df = analyzer.get_new_releases(country, limit)
                    
                    if len(results_df) > 0:
                        analyzer.set_current_data(results_df, "New Releases")
                        data_loaded = True
                        break
                
                elif data_choice == '4':
                    # Load from playlist
                    if not analyzer.sp:
                        print("Spotify API not connected. Please check your credentials.")
                        continue
                    
                    playlist_id = input("Enter Spotify playlist ID: ")
                    
                    if playlist_id.strip():
                        results_df = analyzer.get_playlist_tracks(playlist_id)
                        
                        if len(results_df) > 0:
                            analyzer.set_current_data(results_df, "Playlist")
                            data_loaded = True
                            break
                
                else:
                    print("Invalid choice. Please try again.")
        
        elif choice == '2':
            # Check if data is loaded
            if not data_loaded:
                print("No data loaded. Please load data first.")
                continue
            
            # Search & explore submenu
            while True:
                print("\n--- SEARCH & EXPLORE OPTIONS ---")
                print("1. Display track details")
                print("2. Find similar tracks (recommendations)")
                print("3. Display summary statistics")
                print("4. Show top tracks by feature")
                print("0. Back to main menu")
                
                explore_choice = input("\nEnter your choice (0-4): ")
                
                if explore_choice == '0':
                    break
                
                elif explore_choice == '1':
                    # Display track details
                    if analyzer.current_df is None:
                        print("No data available.")
                        continue
                    
                    # Show a few tracks to choose from
                    if 'track_name' in analyzer.current_df.columns and 'track_artist' in analyzer.current_df.columns:
                        print("\nAvailable tracks:")
                        for i, (_, track) in enumerate(analyzer.current_df.head(10).iterrows()):
                            print(f"{i}: {track['track_name']} - {track['track_artist']}")
                    
                    track_idx = input("Enter track index to view details: ")
                    track_idx = int(track_idx) if track_idx.isdigit() else 0
                    
                    analyzer.display_track_details(track_idx)
                
                elif explore_choice == '2':
                    # Get recommendations
                    if not analyzer.sp:
                        print("Spotify API not connected. Please check your credentials.")
                        continue
                    
                    if analyzer.current_df is None or 'track_id' not in analyzer.current_df.columns:
                        print("No valid track data available for recommendations.")
                        continue
                    
                    # Show a few tracks to choose from
                    if 'track_name' in analyzer.current_df.columns and 'track_artist' in analyzer.current_df.columns:
                        print("\nSelect seed tracks:")
                        for i, (_, track) in enumerate(analyzer.current_df.head(10).iterrows()):
                            print(f"{i}: {track['track_name']} - {track['track_artist']}")
                    
                    track_indices = input("Enter track indices separated by commas (max 5): ")
                    indices = [int(i) for i in track_indices.split(',') if i.strip().isdigit()]
                    
                    if not indices:
                        print("No valid indices provided.")
                        continue
                    
                    # Get track IDs for selected indices
                    seed_tracks = []
                    for idx in indices[:5]:  # Max 5 seed tracks
                        if idx < len(analyzer.current_df):
                            seed_tracks.append(analyzer.current_df.iloc[idx]['track_id'])
                    
                    if not seed_tracks:
                        print("No valid seed tracks selected.")
                        continue
                    
                    # Get recommendations
                    results_df = analyzer.get_recommendations(seed_tracks=seed_tracks)
                    
                    if len(results_df) > 0:
                        analyzer.set_current_data(results_df, "Recommendations")
                
                elif explore_choice == '3':
                    # Display summary statistics
                    if analyzer.current_df is None:
                        print("No data available.")
                        continue
                    
                    # Get numeric columns for statistics
                    numeric_cols = [col for col in analyzer.current_df.columns 
                                   if pd.api.types.is_numeric_dtype(analyzer.current_df[col])]
                    
                    if not numeric_cols:
                        print("No numeric columns available for statistics.")
                        continue
                    
                    print("\n=== Summary Statistics ===")
                    
                    # Calculate and display statistics
                    stats = analyzer.current_df[numeric_cols].describe()
                    print(stats)
                    
                    # Display additional info
                    if 'track_artist' in analyzer.current_df.columns:
                        top_artists = analyzer.current_df['track_artist'].value_counts().head(5)
                        print("\nTop Artists:")
                        for artist, count in top_artists.items():
                            print(f"  {artist}: {count} tracks")
                    
                    if 'artist_genres' in analyzer.current_df.columns:
                        # Split genres and count
                        all_genres = []
                        for genres in analyzer.current_df['artist_genres'].dropna():
                            all_genres.extend([g.strip() for g in genres.split(',')])
                        
                        if all_genres:
                            genre_counts = pd.Series(all_genres).value_counts().head(5)
                            print("\nTop Genres:")
                            for genre, count in genre_counts.items():
                                print(f"  {genre}: {count}")
                
                elif explore_choice == '4':
                    # Show top tracks by feature
                    if analyzer.current_df is None:
                        print("No data available.")
                        continue
                    
                    # Get numeric features
                    numeric_features = [col for col in analyzer.audio_features 
                                      if col in analyzer.current_df.columns 
                                      and pd.api.types.is_numeric_dtype(analyzer.current_df[col])]
                    
                    if not numeric_features:
                        print("No audio features available.")
                        continue
                    
                    # List features
                    print("\nAvailable features:")
                    for i, feature in enumerate(numeric_features):
                        print(f"{i}: {feature}")
                    
                    feature_idx = input("Enter feature index: ")
                    feature_idx = int(feature_idx) if feature_idx.isdigit() and int(feature_idx) < len(numeric_features) else 0
                    
                    selected_feature = numeric_features[feature_idx]
                    
                    # Ask whether to show highest or lowest
                    high_low = input("Show highest (h) or lowest (l) values? [h/l]: ").lower()
                    ascending = True if high_low == 'l' else False
                    
                    # Sort and display
                    n_tracks = 10
                    sorted_df = analyzer.current_df.sort_values(selected_feature, ascending=ascending).head(n_tracks)
                    
                    print(f"\nTop {n_tracks} tracks by {selected_feature} ({'lowest' if ascending else 'highest'}):")
                    
                    for i, (_, track) in enumerate(sorted_df.iterrows()):
                        track_info = []
                        
                        if 'track_name' in track:
                            track_info.append(track['track_name'])
                        
                        if 'track_artist' in track:
                            track_info.append(track['track_artist'])
                        
                        track_str = " - ".join(track_info) if track_info else f"Track {i}"
                        print(f"{i+1}. {track_str}: {track[selected_feature]:.3f}")
                
                else:
                    print("Invalid choice. Please try again.")
        
        elif choice == '3':
            # Check if data is loaded
            if not data_loaded:
                print("No data loaded. Please load data first.")
                continue
            
            # Visualization submenu
            while True:
                print("\n--- VISUALIZATION OPTIONS ---")
                print("1. Feature distributions")
                print("2. Feature correlations")
                print("3. Interactive scatter plot")
                print("4. Track feature comparison")
                print("0. Back to main menu")
                
                viz_choice = input("\nEnter your choice (0-4): ")
                
                if viz_choice == '0':
                    break
                
                elif viz_choice == '1':
                    # Feature distributions
                    analyzer.visualize_feature_distributions()
                
                elif viz_choice == '2':
                    # Feature correlations
                    analyzer.visualize_feature_correlations()
                
                elif viz_choice == '3':
                    # Interactive scatter plot
                    if analyzer.current_df is None:
                        print("No data available.")
                        continue
                    
                    # Get numeric features
                    numeric_features = [col for col in analyzer.current_df.columns 
                                      if pd.api.types.is_numeric_dtype(analyzer.current_df[col])]
                    
                    if len(numeric_features) < 2:
                        print("Not enough numeric features for scatter plot.")
                        continue
                    
                    # List features
                    print("\nAvailable features:")
                    for i, feature in enumerate(numeric_features):
                        print(f"{i}: {feature}")
                    
                    # Get x-axis feature
                    x_idx = input("Enter index for x-axis feature: ")
                    x_idx = int(x_idx) if x_idx.isdigit() and int(x_idx) < len(numeric_features) else 0
                    x_feature = numeric_features[x_idx]
                    
                    # Get y-axis feature
                    y_idx = input("Enter index for y-axis feature: ")
                    y_idx = int(y_idx) if y_idx.isdigit() and int(y_idx) < len(numeric_features) else 1
                    y_feature = numeric_features[y_idx]
                    
                    # Get color feature
                    print("\nAvailable color features (or enter to skip):")
                    for i, feature in enumerate(numeric_features):
                        if feature not in [x_feature, y_feature]:
                            print(f"{i}: {feature}")
                    
                    color_idx = input("Enter index for color feature (or press Enter for none): ")
                    
                    if color_idx.isdigit() and int(color_idx) < len(numeric_features):
                        color_feature = numeric_features[int(color_idx)]
                    else:
                        color_feature = None
                    
                    # Create scatter plot
                    analyzer.visualize_scatter_plot(x_feature, y_feature, color_feature)
                
                elif viz_choice == '4':
                    # Track feature comparison
                    if analyzer.current_df is None:
                        print("No data available.")
                        continue
                    
                    # Show a few tracks to choose from
                    if 'track_name' in analyzer.current_df.columns and 'track_artist' in analyzer.current_df.columns:
                        print("\nSelect tracks to compare:")
                        for i, (_, track) in enumerate(analyzer.current_df.head(10).iterrows()):
                            print(f"{i}: {track['track_name']} - {track['track_artist']}")
                    
                    track_indices = input("Enter track indices separated by commas (max 5): ")
                    indices = [int(i) for i in track_indices.split(',') if i.strip().isdigit()]
                    
                    if not indices:
                        print("No valid indices provided.")
                        continue
                    
                    # Compare tracks
                    analyzer.visualize_track_comparison(indices)
                
                else:
                    print("Invalid choice. Please try again.")
        
        elif choice == '4':
            # Check if data is loaded
            if not data_loaded:
                print("No data loaded. Please load data first.")
                continue
            
            # Analysis submenu
            while True:
                print("\n--- ANALYSIS OPTIONS ---")
                print("1. Cluster tracks")
                print("2. Find outliers")
                print("3. Create playlists by mood")
                print("0. Back to main menu")
                
                analysis_choice = input("\nEnter your choice (0-3): ")
                
                if analysis_choice == '0':
                    break
                
                elif analysis_choice == '1':
                    # Cluster tracks
                    n_clusters = input("Enter number of clusters (default 5): ")
                    n_clusters = int(n_clusters) if n_clusters.isdigit() else 5
                    
                    analyzer.cluster_tracks(n_clusters)
                
                elif analysis_choice == '2':
                    # Find outliers
                    if analyzer.current_df is None:
                        print("No data available.")
                        continue
                    
                    # Get numeric features
                    numeric_features = [col for col in analyzer.audio_features 
                                      if col in analyzer.current_df.columns 
                                      and pd.api.types.is_numeric_dtype(analyzer.current_df[col])]
                    
                    if not numeric_features:
                        print("No numeric features available for outlier detection.")
                        continue
                    
                    print("Finding outliers...")
                    
                    # Create a copy of the dataframe
                    df = analyzer.current_df.copy()
                    
                    # Calculate z-scores for each feature
                    z_scores = pd.DataFrame()
                    for feature in numeric_features:
                        z_scores[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
                    
                    # Find outliers (absolute z-score > 3)
                    outliers = pd.DataFrame()
                    for feature in numeric_features:
                        feature_outliers = df[abs(z_scores[feature]) > 3].copy()
                        feature_outliers['outlier_feature'] = feature
                        feature_outliers['z_score'] = z_scores.loc[feature_outliers.index, feature]
                        outliers = pd.concat([outliers, feature_outliers])
                    
                    # Display outliers
                    if len(outliers) == 0:
                        print("No outliers found.")
                        continue
                    
                    print(f"\nFound {len(outliers)} outliers across {numeric_features} features.")
                    
                    # Group by feature
                    feature_counts = outliers['outlier_feature'].value_counts()
                    print("\nOutliers by feature:")
                    for feature, count in feature_counts.items():
                        print(f"  {feature}: {count} outliers")
                    
                    # Display some examples
                    print("\nExample outliers:")
                    
                    if 'track_name' in outliers.columns and 'track_artist' in outliers.columns:
                        for i, (_, track) in enumerate(outliers.head(10).iterrows()):
                            feature = track['outlier_feature']
                            z_score = track['z_score']
                            value = track[feature]
                            print(f"{i+1}. {track['track_name']} - {track['track_artist']}")
                            print(f"   Feature: {feature}, Value: {value:.3f}, Z-score: {z_score:.3f}")
                    
                    # Visualize outliers
                    if 'track_name' in outliers.columns:
                        # Create box plots with outliers highlighted
                        fig = make_subplots(
                            rows=len(numeric_features), 
                            cols=1, 
                            subplot_titles=numeric_features,
                            vertical_spacing=0.05
                        )
                        
                        for i, feature in enumerate(numeric_features):
                            # Add box plot
                            fig.add_trace(
                                go.Box(
                                    y=df[feature],
                                    name=feature,
                                    boxpoints='outliers',
                                    marker_color='#1DB954'  # Spotify green
                                ),
                                row=i+1, 
                                col=1
                            )
                            
                            # Add outlier points
                            feature_outliers = outliers[outliers['outlier_feature'] == feature]
                            if len(feature_outliers) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        y=feature_outliers[feature],
                                        mode='markers+text',
                                        marker=dict(color='red', size=10),
                                        text=feature_outliers['track_name'],
                                        textposition='top center',
                                        name=f"{feature} outliers"
                                    ),
                                    row=i+1, 
                                    col=1
                                )
                        
                        # Update layout
                        fig.update_layout(
                            title_text="Feature Outliers",
                            height=300 * len(numeric_features),
                            showlegend=False
                        )
                        
                        # Show plot
                        fig.show()
                
                elif analysis_choice == '3':
                    # Create playlists by mood
                    if analyzer.current_df is None:
                        print("No data available.")
                        continue
                    
                    # Check for required features
                    required_features = ['valence', 'energy']
                    missing_features = [f for f in required_features if f not in analyzer.current_df.columns]
                    
                    if missing_features:
                        print(f"Missing required features: {', '.join(missing_features)}")
                        continue
                    
                    print("Creating mood-based playlists...")
                    
                    # Create a copy of the dataframe
                    df = analyzer.current_df.copy()
                    
                    # Define moods based on valence and energy
                    df['mood'] = 'Unknown'
                    
                    # Happy: high valence, high energy
                    df.loc[(df['valence'] > 0.5) & (df['energy'] > 0.5), 'mood'] = 'Happy'
                    
                    # Sad: low valence, low energy
                    df.loc[(df['valence'] < 0.5) & (df['energy'] < 0.5), 'mood'] = 'Sad'
                    
                    # Energetic: low-medium valence, high energy
                    df.loc[(df['valence'] < 0.6) & (df['energy'] > 0.7), 'mood'] = 'Energetic'
                    
                    # Relaxed: medium-high valence, low energy
                    df.loc[(df['valence'] > 0.4) & (df['energy'] < 0.4), 'mood'] = 'Relaxed'
                    
                    # Angry: low valence, high energy, high speechiness
                    if 'speechiness' in df.columns:
                        df.loc[(df['valence'] < 0.4) & (df['energy'] > 0.6) & (df['speechiness'] > 0.2), 'mood'] = 'Angry'
                    
                    # Count by mood
                    mood_counts = df['mood'].value_counts()
                    print("\nTracks by mood:")
                    for mood, count in mood_counts.items():
                        print(f"  {mood}: {count} tracks")
                    
                    # Visualize mood distribution
                    fig = px.scatter(
                        df,
                        x='valence',
                        y='energy',
                        color='mood',
                        hover_name='track_name' if 'track_name' in df.columns else None,
                        hover_data=['track_artist'] if 'track_artist' in df.columns else None,
                        title="Mood Classification by Valence and Energy",
                        labels={'valence': 'Valence (Positivity)', 'energy': 'Energy'}
                    )
                    
                    # Add mood regions
                    fig.add_shape(
                        type="rect",
                        x0=0.5, y0=0.5, x1=1, y1=1,
                        line=dict(color="rgba(0,0,0,0)"),
                        fillcolor="rgba(0,255,0,0.1)",
                        name="Happy"
                    )
                    
                    fig.add_shape(
                        type="rect",
                        x0=0, y0=0, x1=0.5, y1=0.5,
                        line=dict(color="rgba(0,0,0,0)"),
                        fillcolor="rgba(0,0,255,0.1)",
                        name="Sad"
                    )
                    
                    # Show plot
                    fig.show()
                    
                    # Show example tracks for each mood
                    if 'track_name' in df.columns and 'track_artist' in df.columns:
                        print("\nExample tracks by mood:")
                        for mood in mood_counts.index:
                            mood_tracks = df[df['mood'] == mood]
                            print(f"\n{mood} tracks:")
                            
                            for i, (_, track) in enumerate(mood_tracks.head(3).iterrows()):
                                print(f"  {i+1}. {track['track_name']} - {track['track_artist']}")
                                print(f"     Valence: {track['valence']:.2f}, Energy: {track['energy']:.2f}")
                    
                    # Set this as the current dataframe
                    analyzer.current_df = df
                    print("\nMood classifications added to current dataset.")
                
                else:
                    print("Invalid choice. Please try again.")
        
        elif choice == '5':
            # Save data
            if analyzer.current_df is None or len(analyzer.current_df) == 0:
                print("No data available to save.")
                continue
            
            filename = input("Enter filename to save (or press Enter for default): ")
            filename = filename.strip() if filename.strip() else None
            
            analyzer.save_to_csv(filename)
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
