import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import numpy as np
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
import base64
import io
import json
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Spotify API credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Initialize Dash app
app = dash.Dash(__name__, 
               external_stylesheets=[dbc.themes.DARKLY],
               suppress_callback_exceptions=True)

app.title = "Spotify Data Explorer Dashboard"
server = app.server

# Define the SpotifyClient class to handle API interactions
class SpotifyClient:
    def __init__(self):
        self.client_id = CLIENT_ID
        self.client_secret = CLIENT_SECRET
        
        if not self.client_id or not self.client_secret:
            self.sp = None
            self.connected = False
        else:
            try:
                client_credentials_manager = SpotifyClientCredentials(
                    client_id=self.client_id, 
                    client_secret=self.client_secret
                )
                self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
                self.connected = True
            except Exception as e:
                self.sp = None
                self.connected = False
    
    def search_tracks(self, query, limit=10):
        if not self.sp:
            return pd.DataFrame()
            
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            
            tracks = results['tracks']['items']
            if not tracks:
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
                audio_features = self.sp.audio_features(track_ids)
                
                # Add audio features to DataFrame
                for i, features in enumerate(audio_features):
                    if features and i < len(tracks_df):
                        for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                       'speechiness', 'acousticness', 'instrumentalness',
                                       'liveness', 'valence', 'tempo', 'time_signature']:
                            tracks_df.loc[i, feature] = features[feature]
            except Exception as e:
                # Continue without audio features
                pass
            
            return tracks_df
        
        except Exception as e:
            return pd.DataFrame()
    
    def get_new_releases(self, country='US', limit=20):
        if not self.sp:
            return pd.DataFrame()
        
        try:
            new_releases = self.sp.new_releases(country=country, limit=limit)
            
            if 'albums' not in new_releases or not new_releases['albums']['items']:
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
                        
                except Exception:
                    continue
            
            if not all_tracks:
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
                        for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                       'speechiness', 'acousticness', 'instrumentalness',
                                       'liveness', 'valence', 'tempo', 'time_signature']:
                            if feature in features:
                                tracks_df.loc[i, feature] = features[feature]
            except Exception:
                # Continue without audio features
                pass
            
            return tracks_df
        
        except Exception:
            return pd.DataFrame()
    
    def get_playlist_tracks(self, playlist_id):
        if not self.sp:
            return pd.DataFrame()
        
        try:
            # Get playlist info
            try:
                playlist_info = self.sp.playlist(playlist_id)
                playlist_name = playlist_info['name']
            except:
                playlist_name = "Unknown Playlist"
            
            # Get all tracks (paginate through results)
            results = self.sp.playlist_tracks(playlist_id)
            tracks = results['items']
            
            while results['next']:
                results = self.sp.next(results)
                tracks.extend(results['items'])
            
            if not tracks:
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
                        for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                       'speechiness', 'acousticness', 'instrumentalness',
                                       'liveness', 'valence', 'tempo', 'time_signature']:
                            if feature in feat:
                                tracks_df.loc[i, feature] = feat[feature]
            except Exception:
                # Continue without audio features
                pass
            
            return tracks_df
        
        except Exception:
            return pd.DataFrame()
    
    def get_recommendations(self, seed_tracks, limit=20):
        if not self.sp:
            return pd.DataFrame()
        
        try:
            # Get recommendations
            recommendations = self.sp.recommendations(
                seed_tracks=seed_tracks,
                limit=limit
            )
            
            if 'tracks' not in recommendations or not recommendations['tracks']:
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
                        for feature in ['danceability', 'energy', 'key', 'loudness', 'mode', 
                                       'speechiness', 'acousticness', 'instrumentalness',
                                       'liveness', 'valence', 'tempo', 'time_signature']:
                            if feature in feat:
                                tracks_df.loc[i, feature] = feat[feature]
            except Exception:
                # Continue without audio features
                pass
            
            return tracks_df
        
        except Exception:
            return pd.DataFrame()

# Initialize the Spotify client
spotify_client = SpotifyClient()

# Define app layout
app.layout = dbc.Container(
    [
        dcc.Store(id='data-store', storage_type='memory'),
        dcc.Store(id='filtered-data-store', storage_type='memory'),
        
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Spotify Data Explorer", className="text-center mb-4", style={'color': '#1DB954'}),
                html.Hr(style={'borderTop': '3px solid #1DB954'})
            ])
        ]),
        
        # Data loading section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Data Source", style={'backgroundColor': '#212121', 'color': '#1DB954'}),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(label="Load CSV", tab_id="csv-tab", children=[
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select CSV File')
                                    ]),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px'
                                    },
                                    multiple=False
                                ),
                                html.Div(id='upload-status')
                            ]),
                            dbc.Tab(label="Spotify API", tab_id="api-tab", children=[
                                dbc.Tabs([
                                    dbc.Tab(label="Search", tab_id="search-tab", children=[
                                        html.P("Search for tracks:", className="mt-2"),
                                        dbc.InputGroup([
                                            dbc.Input(id="search-query", placeholder="Enter artist, track, or album", type="text"),
                                            dbc.InputGroupText(
                                                dbc.Button("Search", id="search-button", color="success", className="ms-2")
                                            )
                                        ], className="mb-3"),
                                        html.P("Number of results:"),
                                        dcc.Slider(id="search-limit", min=5, max=50, step=5, value=10, 
                                                  marks={i: str(i) for i in range(5, 51, 5)})
                                    ]),
                                    dbc.Tab(label="New Releases", tab_id="new-releases-tab", children=[
                                        html.P("Get new releases:", className="mt-2"),
                                        dbc.InputGroup([
                                            dbc.Input(id="country-code", placeholder="Country code (e.g., US)", type="text", value="US"),
                                            dbc.InputGroupText(
                                                dbc.Button("Get Releases", id="releases-button", color="success", className="ms-2")
                                            )
                                        ], className="mb-3"),
                                        html.P("Number of releases:"),
                                        dcc.Slider(id="releases-limit", min=10, max=50, step=5, value=20, 
                                                  marks={i: str(i) for i in range(10, 51, 10)})
                                    ]),
                                    dbc.Tab(label="Playlist", tab_id="playlist-tab", children=[
                                        html.P("Enter Spotify playlist ID:", className="mt-2"),
                                        dbc.InputGroup([
                                            dbc.Input(id="playlist-id", placeholder="Playlist ID", type="text"),
                                            dbc.InputGroupText(
                                                dbc.Button("Load Playlist", id="playlist-button", color="success", className="ms-2")
                                            )
                                        ], className="mb-3"),
                                        html.P("Find playlist ID from playlist URL: spotify:playlist:PLAYLIST_ID"),
                                    ]),
                                    dbc.Tab(label="Recommendations", tab_id="recommendations-tab", children=[
                                        html.P("Get recommendations based on tracks:", className="mt-2"),
                                        html.P("Select seed tracks:"),
                                        dcc.Dropdown(id="seed-tracks", multi=True, options=[], placeholder="First load some tracks to select seeds"),
                                        html.P("Number of recommendations:"),
                                        dcc.Slider(id="recommendations-limit", min=5, max=50, step=5, value=10, 
                                                  marks={i: str(i) for i in range(5, 51, 5)}),
                                        dbc.Button("Get Recommendations", id="recommendations-button", color="success", className="mt-3")
                                    ]),
                                ], className="mt-2")
                            ]),
                        ], id="data-source-tabs"),
                          # Connection status
                        html.Div([
                            html.P([
                                "Spotify API Status: ",
                                html.Span("Connected", style={'color': '#1DB954'}) if spotify_client.connected 
                                else html.Span("Not Connected", style={'color': 'red'})
                            ], className="mt-3"),
                            html.P("Data loaded: ", id="data-status", className="mt-2")
                        ]),
                    ])
                ], className="mb-4", style={'backgroundColor': '#333333'})
            ], width=12)
        ]),
        
        # Main content area
        dbc.Row([
            # Left column for filters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filters & Analysis", style={'backgroundColor': '#212121', 'color': '#1DB954'}),
                    dbc.CardBody([
                        html.P("Filter by Feature Ranges:"),
                        html.Div(id="feature-sliders"),
                        
                        html.P("Filter by Text:", className="mt-3"),
                        dbc.InputGroup([
                            dbc.Input(id="text-filter", placeholder="Filter by track or artist name", type="text"),
                        ], className="mb-3"),
                        
                        dbc.Button("Apply Filters", id="apply-filters", color="primary", className="w-100 mb-3"),
                        
                        html.Hr(),
                        
                        html.P("Analysis Options:", className="mt-3"),
                        dbc.Button("Cluster Tracks", id="cluster-button", color="info", className="w-100 mb-2"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Clusters:"),
                            dbc.Input(id="n-clusters", type="number", value=4, min=2, max=10)
                        ], className="mb-3"),
                        
                        dbc.Button("Find Outliers", id="outliers-button", color="warning", className="w-100 mb-2"),
                        dbc.Button("Create Mood Playlists", id="mood-button", color="success", className="w-100 mb-2"),
                    ])
                ], style={'backgroundColor': '#333333', 'height': '100%'})
            ], width=3),
            
            # Right column for visualizations
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Tabs([
                            dbc.Tab(label="Scatter Plot", tab_id="scatter-tab"),
                            dbc.Tab(label="Feature Distribution", tab_id="distribution-tab"),
                            dbc.Tab(label="Feature Comparison", tab_id="comparison-tab"),
                            dbc.Tab(label="Track Comparison", tab_id="radar-tab"),
                            dbc.Tab(label="Correlation Matrix", tab_id="correlation-tab"),
                            dbc.Tab(label="Cluster View", tab_id="cluster-tab"),
                            dbc.Tab(label="Data Table", tab_id="table-tab"),
                        ], id="viz-tabs", active_tab="scatter-tab"),
                        style={'backgroundColor': '#212121', 'color': '#1DB954'}
                    ),
                    dbc.CardBody([
                        # Scatter Plot Tab
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.P("X-Axis Feature:"),
                                    dcc.Dropdown(id="scatter-x-feature", className="mb-2")
                                ], width=4),
                                dbc.Col([
                                    html.P("Y-Axis Feature:"),
                                    dcc.Dropdown(id="scatter-y-feature", className="mb-2")
                                ], width=4),
                                dbc.Col([
                                    html.P("Color By:"),
                                    dcc.Dropdown(id="scatter-color", className="mb-2")
                                ], width=4)
                            ]),
                            dcc.Graph(id="scatter-plot", style={"height": "60vh"})
                        ], id="scatter-content"),
                        
                        # Feature Distribution Tab
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.P("Select Feature:"),
                                    dcc.Dropdown(id="distribution-feature", className="mb-2")
                                ], width=6),
                                dbc.Col([
                                    html.P("Group By:"),
                                    dcc.Dropdown(id="distribution-group", className="mb-2")
                                ], width=6)
                            ]),
                            dcc.Graph(id="distribution-plot", style={"height": "60vh"})
                        ], id="distribution-content", style={"display": "none"}),
                        
                        # Feature Comparison Tab
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.P("Feature to Compare:"),
                                    dcc.Dropdown(id="comparison-feature", className="mb-2")
                                ], width=6),
                                dbc.Col([
                                    html.P("Group By:"),
                                    dcc.Dropdown(id="comparison-group", className="mb-2")
                                ], width=6)
                            ]),
                            dcc.Graph(id="comparison-plot", style={"height": "60vh"})
                        ], id="comparison-content", style={"display": "none"}),
                        
                        # Track Comparison Tab
                        html.Div([
                            html.P("Select Tracks to Compare:"),
                            dcc.Dropdown(id="radar-tracks", multi=True, className="mb-2"),
                            dcc.Graph(id="radar-plot", style={"height": "60vh"})
                        ], id="radar-content", style={"display": "none"}),
                        
                        # Correlation Matrix Tab
                        html.Div([
                            dcc.Graph(id="correlation-plot", style={"height": "65vh"})
                        ], id="correlation-content", style={"display": "none"}),
                        
                        # Cluster View Tab
                        html.Div([
                            dcc.Graph(id="cluster-plot", style={"height": "65vh"})
                        ], id="cluster-content", style={"display": "none"}),
                        
                        # Data Table Tab
                        html.Div([
                            dbc.Table(id="data-table", className="table-hover table-striped", style={"fontSize": "small"})
                        ], id="table-content", style={"display": "none", "maxHeight": "65vh", "overflow": "auto"}),
                    ])
                ], style={'backgroundColor': '#333333', 'height': '100%'})
            ], width=9)
        ]),
        
        # Track detail modal
        dbc.Modal([
            dbc.ModalHeader("Track Details", style={'backgroundColor': '#212121', 'color': '#1DB954'}),
            dbc.ModalBody(id="track-detail-content", style={'backgroundColor': '#333333', 'color': 'white'}),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-track-modal", className="ms-auto", color="secondary"),
                style={'backgroundColor': '#212121'}
            ),
        ], id="track-modal", size="lg", backdrop="static"),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(style={'borderTop': '1px solid #1DB954'}),
                html.P("Spotify Data Explorer Dashboard • Interactive Analysis and Visualization", 
                       className="text-center text-muted mb-0")
            ])
        ], className="mt-4")
    ],
    fluid=True,
    className="p-4",
    style={"backgroundColor": "#121212", "minHeight": "100vh", "color": "white"}
)

# Helper function to parse uploaded CSV files
def parse_csv(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
        else:
            return None
    except Exception as e:
        print(e)
        return None

# Helper function to get numeric features from dataframe
def get_numeric_features(df):
    if df is None or df.empty:
        return []
    
    # Standard audio features
    standard_features = [
        'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'loudness'
    ]
    
    # Get all numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Prioritize standard features
    features = [f for f in standard_features if f in numeric_cols]
    
    # Add other numeric columns that aren't standard features
    for col in numeric_cols:
        if col not in features and 'id' not in col.lower() and 'index' not in col.lower():
            features.append(col)
    
    return features

# Helper function to filter dataframe based on slider values
def filter_dataframe(df, filter_values, text_filter):
    if df is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply feature range filters
    if filter_values:
        for feature, (min_val, max_val) in filter_values.items():
            if feature in filtered_df.columns:
                filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]
    
    # Apply text filter
    if text_filter:
        text = text_filter.lower()
        text_cols = ['track_name', 'track_artist', 'track_album_name']
        mask = False
        for col in text_cols:
            if col in filtered_df.columns:
                mask = mask | filtered_df[col].str.lower().str.contains(text, na=False)
        filtered_df = filtered_df[mask]
    
    return filtered_df

# Callback to handle file upload
@app.callback(
    [Output('data-store', 'data'),
     Output('upload-status', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        raise PreventUpdate
    
    df = parse_csv(contents, filename)
    
    if df is not None:
        return df.to_json(date_format='iso', orient='split'), html.Div([
            html.P(f"Uploaded: {filename}", style={'color': '#1DB954'}),
            html.P(f"Loaded {len(df)} tracks with {len(df.columns)} features")
        ])
    else:
        return None, html.Div([
            html.P(f"Error processing file: {filename}", style={'color': 'red'})
        ])

# Callback to handle Spotify search
@app.callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('data-status', 'children')],
    [Input('search-button', 'n_clicks')],
    [State('search-query', 'value'),
     State('search-limit', 'value')],
    prevent_initial_call=True
)
def search_spotify(n_clicks, query, limit):
    if not n_clicks or not query:
        raise PreventUpdate
    
    if not spotify_client.connected:
        return None, html.Span("Spotify API not connected. Check your credentials.", style={'color': 'red'})
    
    df = spotify_client.search_tracks(query, limit)
    
    if df is not None and not df.empty:
        return df.to_json(date_format='iso', orient='split'), html.Span(f"Loaded {len(df)} tracks from search", style={'color': '#1DB954'})
    else:
        return None, html.Span("No tracks found or error occurred", style={'color': 'red'})

# Callback to handle new releases
@app.callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('data-status', 'children', allow_duplicate=True)],
    [Input('releases-button', 'n_clicks')],
    [State('country-code', 'value'),
     State('releases-limit', 'value')],
    prevent_initial_call=True
)
def get_new_releases(n_clicks, country, limit):
    if not n_clicks:
        raise PreventUpdate
    
    if not spotify_client.connected:
        return None, html.Span("Spotify API not connected. Check your credentials.", style={'color': 'red'})
    
    country = country if country else 'US'
    
    df = spotify_client.get_new_releases(country, limit)
    
    if df is not None and not df.empty:
        return df.to_json(date_format='iso', orient='split'), html.Span(f"Loaded {len(df)} new releases", style={'color': '#1DB954'})
    else:
        return None, html.Span("No new releases found or error occurred", style={'color': 'red'})

# Callback to handle playlist loading
@app.callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('data-status', 'children', allow_duplicate=True)],
    [Input('playlist-button', 'n_clicks')],
    [State('playlist-id', 'value')],
    prevent_initial_call=True
)
def load_playlist(n_clicks, playlist_id):
    if not n_clicks or not playlist_id:
        raise PreventUpdate
    
    if not spotify_client.connected:
        return None, html.Span("Spotify API not connected. Check your credentials.", style={'color': 'red'})
    
    df = spotify_client.get_playlist_tracks(playlist_id)
    
    if df is not None and not df.empty:
        return df.to_json(date_format='iso', orient='split'), html.Span(f"Loaded {len(df)} tracks from playlist", style={'color': '#1DB954'})
    else:
        return None, html.Span("No playlist tracks found or error occurred", style={'color': 'red'})

# Callback to update seed track options
@app.callback(
    Output('seed-tracks', 'options'),
    [Input('data-store', 'data')]
)
def update_seed_options(data):
    if not data:
        return []
    
    df = pd.read_json(data, orient='split')
    
    if 'track_name' in df.columns and 'track_artist' in df.columns and 'track_id' in df.columns:
        options = [
            {'label': f"{row['track_name']} - {row['track_artist']}", 'value': row['track_id']}
            for _, row in df.iterrows()
        ]
        return options
    
    return []

# Callback to handle recommendations
@app.callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('data-status', 'children', allow_duplicate=True)],
    [Input('recommendations-button', 'n_clicks')],
    [State('seed-tracks', 'value'),
     State('recommendations-limit', 'value')],
    prevent_initial_call=True
)
def get_recommendations(n_clicks, seed_tracks, limit):
    if not n_clicks or not seed_tracks:
        raise PreventUpdate
    
    if not spotify_client.connected:
        return None, html.Span("Spotify API not connected. Check your credentials.", style={'color': 'red'})
    
    # Limit to 5 seed tracks (Spotify API limit)
    seed_tracks = seed_tracks[:5] if len(seed_tracks) > 5 else seed_tracks
    
    df = spotify_client.get_recommendations(seed_tracks, limit)
    
    if df is not None and not df.empty:
        return df.to_json(date_format='iso', orient='split'), html.Span(f"Loaded {len(df)} recommended tracks", style={'color': '#1DB954'})
    else:
        return None, html.Span("No recommendations found or error occurred", style={'color': 'red'})

# Callback to generate feature sliders
@app.callback(
    Output('feature-sliders', 'children'),
    [Input('data-store', 'data')]
)
def generate_feature_sliders(data):
    if not data:
        return [html.P("No data loaded. Load data to enable filtering.", style={'color': 'gray'})]
    
    df = pd.read_json(data, orient='split')
    numeric_features = get_numeric_features(df)
    
    if not numeric_features:
        return [html.P("No numeric features found for filtering.", style={'color': 'gray'})]
    
    sliders = []
    for feature in numeric_features[:5]:  # Limit to top 5 features to avoid clutter
        if feature in df.columns:
            # Convert NumPy types to native Python types
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            
            # Ensure we have a reasonable step size
            step = float((max_val - min_val) / 100.0) if max_val > min_val else 0.1
            
            # Create marks at min and max, ensuring they're strings
            marks = {
                float(min_val): f"{min_val:.2f}",
                float(max_val): f"{max_val:.2f}"
            }
            
            sliders.append(html.Div([
                html.P(f"{feature.replace('_', ' ').title()}:"),
                dcc.RangeSlider(
                    id=f"slider-{feature}",
                    min=min_val,
                    max=max_val,
                    step=step,
                    marks=marks,
                    value=[min_val, max_val]
                )
            ], className="mb-3"))
    
    return sliders

# Callback to apply filters
@app.callback(
    Output('filtered-data-store', 'data'),
    [Input('apply-filters', 'n_clicks')],
    [State('data-store', 'data'),
     State('text-filter', 'value')] + 
    [State(f"slider-{feature}", "value") for feature in ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']]
)
def apply_filters(n_clicks, data, text_filter, *args):
    if not data:
        raise PreventUpdate
    
    df = pd.read_json(data, orient='split')
    numeric_features = get_numeric_features(df)[:5]  # Match the features in generate_feature_sliders
    
    # Create filter_values dictionary
    filter_values = {}
    for i, feature in enumerate(numeric_features):
        if i < len(args) and args[i] is not None:
            filter_values[feature] = args[i]
    
    filtered_df = filter_dataframe(df, filter_values, text_filter)
    
    return filtered_df.to_json(date_format='iso', orient='split')

# Callback to update dropdown options
@app.callback(
    [Output('scatter-x-feature', 'options'),
     Output('scatter-y-feature', 'options'),
     Output('scatter-color', 'options'),
     Output('distribution-feature', 'options'),
     Output('distribution-group', 'options'),
     Output('comparison-feature', 'options'),
     Output('comparison-group', 'options'),
     Output('radar-tracks', 'options')],
    [Input('filtered-data-store', 'data'),
     Input('data-store', 'data')]
)
def update_dropdown_options(filtered_data, original_data):
    data = filtered_data if filtered_data else original_data
    
    if not data:
        empty_options = []
        return empty_options, empty_options, empty_options, empty_options, empty_options, empty_options, empty_options, empty_options
    
    df = pd.read_json(data, orient='split')
    
    # Get numeric features
    numeric_features = get_numeric_features(df)
    numeric_options = [{'label': feature.capitalize(), 'value': feature} for feature in numeric_features]
    
    # Get categorical features
    categorical_features = []
    for col in df.columns:
        if col in ['playlist_name', 'playlist_genre', 'artist_name', 'track_album_name'] or 'genre' in col.lower():
            if df[col].nunique() < 50:  # Limit to reasonable number of categories
                categorical_features.append(col)
    
    categorical_options = [{'label': feature.replace('_', ' ').capitalize(), 'value': feature} 
                          for feature in categorical_features]
    
    # Color options include None, categorical features, and track_popularity
    color_options = [{'label': 'None', 'value': 'none'}]
    if 'track_popularity' in df.columns:
        color_options.append({'label': 'Popularity', 'value': 'track_popularity'})
    color_options.extend(categorical_options)
    
    # Track options for radar chart
    track_options = []
    if 'track_name' in df.columns and 'track_artist' in df.columns:
        track_options = [
            {'label': f"{row['track_name']} - {row['track_artist']}", 'value': str(i)}
            for i, row in df.iterrows()
        ]
    
    return (numeric_options, numeric_options, color_options, numeric_options, 
            categorical_options, numeric_options, categorical_options, track_options)

# Callback to set initial dropdown values
@app.callback(
    [Output('scatter-x-feature', 'value'),
     Output('scatter-y-feature', 'value'),
     Output('scatter-color', 'value'),
     Output('distribution-feature', 'value'),
     Output('comparison-feature', 'value')],
    [Input('scatter-x-feature', 'options'),
     Input('scatter-y-feature', 'options')]
)
def set_initial_values(x_options, y_options):
    if not x_options or not y_options:
        raise PreventUpdate
    
    x_val = x_options[0]['value'] if x_options else None
    y_val = x_options[1]['value'] if len(x_options) > 1 else x_options[0]['value'] if x_options else None
    color_val = 'none'
    dist_val = x_options[0]['value'] if x_options else None
    comp_val = x_options[0]['value'] if x_options else None
    
    return x_val, y_val, color_val, dist_val, comp_val

# Callback to update scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-x-feature', 'value'),
     Input('scatter-y-feature', 'value'),
     Input('scatter-color', 'value'),
     Input('filtered-data-store', 'data'),
     Input('data-store', 'data')]
)
def update_scatter_plot(x_feature, y_feature, color_feature, filtered_data, original_data):
    data = filtered_data if filtered_data else original_data
    
    if not data or not x_feature or not y_feature:
        return px.scatter(title="No data available")
    
    df = pd.read_json(data, orient='split')
    
    if color_feature and color_feature != 'none':
        fig = px.scatter(
            df, x=x_feature, y=y_feature, color=color_feature,
            hover_name=df['track_name'] if 'track_name' in df.columns else None,
            hover_data=['track_artist'] if 'track_artist' in df.columns else None,
            title=f"{y_feature.capitalize()} vs {x_feature.capitalize()}"
        )
    else:
        fig = px.scatter(
            df, x=x_feature, y=y_feature,
            hover_name=df['track_name'] if 'track_name' in df.columns else None,
            hover_data=['track_artist'] if 'track_artist' in df.columns else None,
            title=f"{y_feature.capitalize()} vs {x_feature.capitalize()}"
        )
    
    fig.update_layout(
        plot_bgcolor='#333333',
        paper_bgcolor='#333333',
        font_color='white'
    )
    
    return fig

# Callback to update distribution plot
@app.callback(
    Output('distribution-plot', 'figure'),
    [Input('distribution-feature', 'value'),
     Input('distribution-group', 'value'),
     Input('filtered-data-store', 'data'),
     Input('data-store', 'data')]
)
def update_distribution_plot(feature, group_by, filtered_data, original_data):
    data = filtered_data if filtered_data else original_data
    
    if not data or not feature:
        return px.histogram(title="No data available")
    
    df = pd.read_json(data, orient='split')
    
    if group_by:
        fig = px.histogram(
            df, x=feature, color=group_by,
            title=f"Distribution of {feature.capitalize()} by {group_by.replace('_', ' ').capitalize()}",
            marginal="box"
        )
    else:
        fig = px.histogram(
            df, x=feature,
            title=f"Distribution of {feature.capitalize()}",
            marginal="box"
        )
    
    fig.update_layout(
        plot_bgcolor='#333333',
        paper_bgcolor='#333333',
        font_color='white'
    )
    
    return fig

# Callback to update comparison plot
@app.callback(
    Output('comparison-plot', 'figure'),
    [Input('comparison-feature', 'value'),
     Input('comparison-group', 'value'),
     Input('filtered-data-store', 'data'),
     Input('data-store', 'data')]
)
def update_comparison_plot(feature, group_by, filtered_data, original_data):
    data = filtered_data if filtered_data else original_data
    
    if not data or not feature:
        return px.box(title="No data available")
    
    df = pd.read_json(data, orient='split')
    
    if not group_by:
        # If no group_by, use track_artist if available, else return histogram
        if 'track_artist' in df.columns and df['track_artist'].nunique() < 20:
            group_by = 'track_artist'
        else:
            return px.histogram(
                df, x=feature,
                title=f"Distribution of {feature.capitalize()} (No grouping available)",
                marginal="box"
            )
    
    # Get top 10 groups by count to avoid overcrowding
    top_groups = df[group_by].value_counts().head(10).index
    plot_df = df[df[group_by].isin(top_groups)]
    
    fig = px.box(
        plot_df, x=group_by, y=feature,
        title=f"Comparison of {feature.capitalize()} by {group_by.replace('_', ' ').capitalize()}",
        points="all", color=group_by
    )
    
    fig.update_layout(
        plot_bgcolor='#333333',
        paper_bgcolor='#333333',
        font_color='white',
        xaxis_tickangle=-45
    )
    
    return fig

# Callback to update radar plot
@app.callback(
    Output('radar-plot', 'figure'),
    [Input('radar-tracks', 'value'),
     Input('filtered-data-store', 'data'),
     Input('data-store', 'data')]
)
def update_radar_plot(selected_tracks, filtered_data, original_data):
    data = filtered_data if filtered_data else original_data
    
    if not data or not selected_tracks:
        return go.Figure(layout=dict(title="Select tracks to compare"))
    
    df = pd.read_json(data, orient='split')
    
    # Features to include in radar chart
    features = ['danceability', 'energy', 'speechiness', 'acousticness', 
               'instrumentalness', 'liveness', 'valence']
    
    # Filter to features that exist in the dataframe
    features = [f for f in features if f in df.columns]
    
    if not features:
        return go.Figure(layout=dict(title="No audio features available for comparison"))
    
    # Convert selected_tracks to integers
    try:
        indices = [int(idx) for idx in selected_tracks]
    except (ValueError, TypeError):
        return go.Figure(layout=dict(title="Invalid track selection"))
    
    # Filter to selected tracks
    selected_df = df.iloc[indices]
    
    if selected_df.empty:
        return go.Figure(layout=dict(title="No tracks selected"))
    
    # Create radar chart
    fig = go.Figure()
    
    for i, (idx, track) in enumerate(selected_df.iterrows()):
        track_name = track.get('track_name', f"Track {idx}")
        track_artist = track.get('track_artist', '')
        
        values = [track[feature] for feature in features]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=features,
            fill='toself',
            name=f"{track_name} - {track_artist}"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Track Audio Feature Comparison",
        plot_bgcolor='#333333',
        paper_bgcolor='#333333',
        font_color='white'
    )
    
    return fig

# Callback to update correlation plot
@app.callback(
    Output('correlation-plot', 'figure'),
    [Input('filtered-data-store', 'data'),
     Input('data-store', 'data')]
)
def update_correlation_plot(filtered_data, original_data):
    data = filtered_data if filtered_data else original_data
    
    if not data:
        return go.Figure(layout=dict(title="No data available"))
    
    df = pd.read_json(data, orient='split')
    
    # Get numeric features
    numeric_features = get_numeric_features(df)
    
    if len(numeric_features) < 2:
        return go.Figure(layout=dict(title="Not enough numeric features for correlation analysis"))
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_features].corr()
    
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
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        plot_bgcolor='#333333',
        paper_bgcolor='#333333',
        font_color='white'
    )
    
    return fig

# Callback to cluster tracks
@app.callback(
    [Output('cluster-plot', 'figure'),
     Output('data-store', 'data', allow_duplicate=True)],
    [Input('cluster-button', 'n_clicks')],
    [State('n-clusters', 'value'),
     State('filtered-data-store', 'data'),
     State('data-store', 'data')],
    prevent_initial_call=True
)
def cluster_tracks(n_clicks, n_clusters, filtered_data, original_data):
    if not n_clicks:
        raise PreventUpdate
    
    data = filtered_data if filtered_data else original_data
    
    if not data:
        return go.Figure(layout=dict(title="No data available")), original_data
    
    df = pd.read_json(data, orient='split')
    
    # Features for clustering
    features_for_clustering = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence'
    ]
    
    # Filter to features that exist in the dataframe
    features_for_clustering = [f for f in features_for_clustering if f in df.columns]
    
    if len(features_for_clustering) < 3:
        return go.Figure(layout=dict(title="Not enough audio features for clustering")), original_data
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features_for_clustering])
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels
    
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
    if 'track_name' in df.columns:
        pca_df['track_name'] = df['track_name'].values
    
    if 'track_artist' in df.columns:
        pca_df['track_artist'] = df['track_artist'].values
    
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
    
    fig.update_layout(
        plot_bgcolor='#333333',
        paper_bgcolor='#333333',
        font_color='white'
    )
    
    # Return the plot and the updated dataframe with cluster labels
    return fig, df.to_json(date_format='iso', orient='split')

# Callback to update data table
@app.callback(
    Output('data-table', 'children'),
    [Input('filtered-data-store', 'data'),
     Input('data-store', 'data')]
)
def update_data_table(filtered_data, original_data):
    data = filtered_data if filtered_data else original_data
    
    if not data:
        return [html.Tr([html.Td("No data available")])]
    
    df = pd.read_json(data, orient='split')
    
    # Select columns to display
    display_cols = ['track_name', 'track_artist', 'track_popularity', 'track_album_name', 
                   'danceability', 'energy', 'valence']
    display_cols = [col for col in display_cols if col in df.columns]
    
    if not display_cols:
        display_cols = df.columns[:5]  # Show first 5 columns if none of the preferred columns exist
    
    # Create table header
    header = html.Thead(html.Tr([html.Th(col.replace('_', ' ').title()) for col in display_cols]))
    
    # Create table body (show first 100 rows)
    rows = []
    for i, row in df.head(100).iterrows():
        rows.append(html.Tr([
            html.Td(row[col] if pd.notna(row[col]) else "") for col in display_cols
        ]))
    
    body = html.Tbody(rows)
    
    if len(df) > 100:
        footer = html.Tfoot(html.Tr([html.Td(f"Showing 100 of {len(df)} rows", colSpan=len(display_cols))]))
        return [header, body, footer]
    
    return [header, body]

# Callback to handle tab switches
@app.callback(
    [Output('scatter-content', 'style'),
     Output('distribution-content', 'style'),
     Output('comparison-content', 'style'),
     Output('radar-content', 'style'),
     Output('correlation-content', 'style'),
     Output('cluster-content', 'style'),
     Output('table-content', 'style')],
    [Input('viz-tabs', 'active_tab')]
)
def toggle_tab_content(active_tab):
    tab_contents = {
        'scatter-tab': 'scatter-content',
        'distribution-tab': 'distribution-content',
        'comparison-tab': 'comparison-content',
        'radar-tab': 'radar-content',
        'correlation-tab': 'correlation-content',
        'cluster-tab': 'cluster-content',
        'table-tab': 'table-content'
    }
    
    styles = []
    for tab in tab_contents.values():
        if tab == tab_contents.get(active_tab):
            styles.append({'display': 'block'})
        else:
            styles.append({'display': 'none'})
    
    return styles

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
