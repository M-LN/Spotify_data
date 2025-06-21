import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from spotify_scraper import SpotifyDataScraper
import numpy as np

# Create scraper instance to load data
scraper = SpotifyDataScraper()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Spotify Data Explorer"

# Check if data is loaded
if scraper.combined_df is not None:
    df = scraper.combined_df
elif scraper.high_popularity_df is not None:
    df = scraper.high_popularity_df
elif scraper.low_popularity_df is not None:
    df = scraper.low_popularity_df
else:
    # Handle case when no data is available
    df = pd.DataFrame()

# Get audio features for visualization
audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                 'instrumentalness', 'liveness', 'valence']

# Filter to features that exist in the DataFrame
audio_features = [f for f in audio_features if f in df.columns]

# Add popularity if available
if 'track_popularity' in df.columns:
    audio_features.append('track_popularity')

# Get playlist names if available
if 'playlist_name' in df.columns:
    playlist_options = [{'label': playlist, 'value': playlist} 
                       for playlist in sorted(df['playlist_name'].unique())]
else:
    playlist_options = []

# Get genres if available
if 'playlist_genre' in df.columns:
    genre_options = [{'label': genre, 'value': genre} 
                    for genre in sorted(df['playlist_genre'].unique())]
else:
    genre_options = []

# Create the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Spotify Data Explorer", className="text-center mb-4"),
            html.P("Explore and visualize your Spotify music data interactively", className="text-center")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    html.P("Select Feature for X-axis:"),
                    dcc.Dropdown(
                        id='x-feature',
                        options=[{'label': feature.capitalize(), 'value': feature} for feature in audio_features],
                        value=audio_features[0] if audio_features else None,
                        className="mb-3"
                    ),
                    html.P("Select Feature for Y-axis:"),
                    dcc.Dropdown(
                        id='y-feature',
                        options=[{'label': feature.capitalize(), 'value': feature} for feature in audio_features],
                        value=audio_features[1] if len(audio_features) > 1 else None,
                        className="mb-3"
                    ),
                    html.P("Color by:"),
                    dcc.Dropdown(
                        id='color-by',
                        options=[
                            {'label': 'None', 'value': 'none'},
                            {'label': 'Playlist', 'value': 'playlist_name'} if 'playlist_name' in df.columns else None,
                            {'label': 'Genre', 'value': 'playlist_genre'} if 'playlist_genre' in df.columns else None,
                            {'label': 'Popularity', 'value': 'track_popularity'} if 'track_popularity' in df.columns else None
                        ],
                        value='none',
                        className="mb-3"
                    ),
                    html.P("Filter by Playlist:"),
                    dcc.Dropdown(
                        id='playlist-filter',
                        options=playlist_options,
                        multi=True,
                        className="mb-3"
                    ),
                    html.P("Filter by Genre:"),
                    dcc.Dropdown(
                        id='genre-filter',
                        options=genre_options,
                        multi=True,
                        className="mb-3"
                    ),
                    html.P("Popularity Range:"),
                    dcc.RangeSlider(
                        id='popularity-slider',
                        min=0,
                        max=100,
                        step=5,
                        marks={i: str(i) for i in range(0, 101, 10)},
                        value=[0, 100],
                        className="mb-3"
                    ) if 'track_popularity' in df.columns else None,
                    html.Button('Apply Filters', id='apply-button', className="btn btn-primary w-100")
                ])
            ], className="mb-4")
        ], width=3),
        
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Scatter Plot", tab_id="scatter-tab", children=[
                    dcc.Graph(id='scatter-plot', style={'height': '70vh'})
                ]),
                dbc.Tab(label="Feature Distribution", tab_id="distribution-tab", children=[
                    dcc.Graph(id='distribution-plot', style={'height': '70vh'})
                ]),
                dbc.Tab(label="Radar Chart", tab_id="radar-tab", children=[
                    html.P("Select tracks to compare (max 5):"),
                    dcc.Dropdown(
                        id='track-selection',
                        options=[],
                        multi=True,
                        className="mb-3"
                    ),
                    dcc.Graph(id='radar-plot', style={'height': '60vh'})
                ]),
                dbc.Tab(label="Correlation Heatmap", tab_id="correlation-tab", children=[
                    dcc.Graph(id='correlation-plot', style={'height': '70vh'})
                ]),
                dbc.Tab(label="Feature Comparison", tab_id="comparison-tab", children=[
                    html.P("Select Feature to Compare:"),
                    dcc.Dropdown(
                        id='compare-feature',
                        options=[{'label': feature.capitalize(), 'value': feature} for feature in audio_features],
                        value=audio_features[0] if audio_features else None,
                        className="mb-3"
                    ),
                    dcc.Graph(id='comparison-plot', style={'height': '60vh'})
                ])
            ], id="tabs", active_tab="scatter-tab")
        ], width=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='selected-track-info', className="mt-3")
        ])
    ]),
    
    dbc.Modal([
        dbc.ModalHeader("Track Details"),
        dbc.ModalBody(id="track-detail-body"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-track-modal", className="ml-auto")
        ),
    ], id="track-modal", size="lg"),
    
], fluid=True, className="p-4")

# Update track dropdown options based on filters
@app.callback(
    Output('track-selection', 'options'),
    [Input('apply-button', 'n_clicks')],
    [State('playlist-filter', 'value'),
     State('genre-filter', 'value'),
     State('popularity-slider', 'value')]
)
def update_track_options(n_clicks, selected_playlists, selected_genres, popularity_range):
    if not n_clicks:
        # Return first 100 tracks on initial load
        if not df.empty:
            filtered_df = df.head(100)
            return [{'label': f"{row['track_name']} - {row['track_artist']}", 
                    'value': i} for i, row in filtered_df.iterrows()]
        return []
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_playlists and 'playlist_name' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['playlist_name'].isin(selected_playlists)]
        
    if selected_genres and 'playlist_genre' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['playlist_genre'].isin(selected_genres)]
        
    if popularity_range and 'track_popularity' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['track_popularity'] >= popularity_range[0]) & 
                                 (filtered_df['track_popularity'] <= popularity_range[1])]
    
    # Limit to 100 tracks to avoid overwhelming the dropdown
    filtered_df = filtered_df.head(100)
    
    return [{'label': f"{row['track_name']} - {row['track_artist']}", 
            'value': i} for i, row in filtered_df.iterrows()]

# Update scatter plot based on selections
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('apply-button', 'n_clicks')],
    [State('x-feature', 'value'),
     State('y-feature', 'value'),
     State('color-by', 'value'),
     State('playlist-filter', 'value'),
     State('genre-filter', 'value'),
     State('popularity-slider', 'value')]
)
def update_scatter_plot(n_clicks, x_feature, y_feature, color_by, selected_playlists, selected_genres, popularity_range):
    if df.empty or not x_feature or not y_feature:
        return go.Figure()
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_playlists and 'playlist_name' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['playlist_name'].isin(selected_playlists)]
        
    if selected_genres and 'playlist_genre' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['playlist_genre'].isin(selected_genres)]
        
    if popularity_range and 'track_popularity' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['track_popularity'] >= popularity_range[0]) & 
                                 (filtered_df['track_popularity'] <= popularity_range[1])]
    
    # Create scatter plot
    if color_by != 'none' and color_by in filtered_df.columns:
        fig = px.scatter(
            filtered_df, 
            x=x_feature, 
            y=y_feature,
            color=color_by,
            hover_name='track_name',
            hover_data=['track_artist', 'track_popularity'] if 'track_popularity' in filtered_df.columns else ['track_artist'],
            title=f"{y_feature.capitalize()} vs {x_feature.capitalize()}",
            color_continuous_scale=px.colors.sequential.Viridis if color_by == 'track_popularity' else None,
            template="plotly_dark"
        )
    else:
        fig = px.scatter(
            filtered_df, 
            x=x_feature, 
            y=y_feature,
            hover_name='track_name',
            hover_data=['track_artist', 'track_popularity'] if 'track_popularity' in filtered_df.columns else ['track_artist'],
            title=f"{y_feature.capitalize()} vs {x_feature.capitalize()}",
            template="plotly_dark"
        )
    
    fig.update_layout(
        xaxis_title=x_feature.capitalize(),
        yaxis_title=y_feature.capitalize(),
        clickmode='event+select'
    )
    
    return fig

# Update distribution plot
@app.callback(
    Output('distribution-plot', 'figure'),
    [Input('apply-button', 'n_clicks')],
    [State('playlist-filter', 'value'),
     State('genre-filter', 'value'),
     State('popularity-slider', 'value')]
)
def update_distribution_plot(n_clicks, selected_playlists, selected_genres, popularity_range):
    if df.empty:
        return go.Figure()
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_playlists and 'playlist_name' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['playlist_name'].isin(selected_playlists)]
        
    if selected_genres and 'playlist_genre' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['playlist_genre'].isin(selected_genres)]
        
    if popularity_range and 'track_popularity' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['track_popularity'] >= popularity_range[0]) & 
                                 (filtered_df['track_popularity'] <= popularity_range[1])]
    
    # Create distribution plot for all audio features
    fig = make_subplots(rows=len(audio_features), cols=1, 
                        subplot_titles=[f.capitalize() for f in audio_features],
                        vertical_spacing=0.05)
    
    for i, feature in enumerate(audio_features):
        if feature in filtered_df.columns:
            fig.add_trace(
                go.Histogram(x=filtered_df[feature], name=feature, nbinsx=30,
                           marker=dict(color='rgba(50, 171, 96, 0.6)')),
                row=i+1, col=1
            )
    
    fig.update_layout(
        height=200 * len(audio_features),
        title_text="Audio Feature Distributions",
        showlegend=False,
        template="plotly_dark"
    )
    
    return fig

# Update radar chart for selected tracks
@app.callback(
    Output('radar-plot', 'figure'),
    [Input('track-selection', 'value')]
)
def update_radar_chart(selected_tracks):
    if not selected_tracks or df.empty:
        # Create empty radar chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="Select tracks to compare",
            showarrow=False,
            font=dict(size=20),
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        fig.update_layout(template="plotly_dark")
        return fig
    
    # Limit to 5 tracks for clarity
    selected_tracks = selected_tracks[:5]
    
    # Get selected tracks data
    selected_data = df.loc[selected_tracks]
    
    # Features for radar chart
    radar_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence']
    
    # Filter to features that exist in the DataFrame
    radar_features = [f for f in radar_features if f in df.columns]
    
    # Create radar chart
    fig = go.Figure()
    
    for i, (idx, track) in enumerate(selected_data.iterrows()):
        values = [track[feature] for feature in radar_features]
        values.append(values[0])  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_features + [radar_features[0]],  # Close the loop
            fill='toself',
            name=f"{track['track_name']} - {track['track_artist']}",
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Track Feature Comparison",
        template="plotly_dark"
    )
    
    return fig

# Update correlation heatmap
@app.callback(
    Output('correlation-plot', 'figure'),
    [Input('apply-button', 'n_clicks')],
    [State('playlist-filter', 'value'),
     State('genre-filter', 'value'),
     State('popularity-slider', 'value')]
)
def update_correlation_plot(n_clicks, selected_playlists, selected_genres, popularity_range):
    if df.empty:
        return go.Figure()
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_playlists and 'playlist_name' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['playlist_name'].isin(selected_playlists)]
        
    if selected_genres and 'playlist_genre' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['playlist_genre'].isin(selected_genres)]
        
    if popularity_range and 'track_popularity' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['track_popularity'] >= popularity_range[0]) & 
                                 (filtered_df['track_popularity'] <= popularity_range[1])]
    
    # Calculate correlation
    corr = filtered_df[audio_features].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr,
        labels=dict(x="Feature", y="Feature", color="Correlation"),
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale="RdBu_r",
        template="plotly_dark"
    )
    
    # Add correlation values as text
    for i, row in enumerate(corr.values):
        for j, val in enumerate(row):
            fig.add_annotation(
                text=f"{val:.2f}",
                x=j, y=i,
                showarrow=False,
                font=dict(color="white" if abs(val) < 0.5 else "black")
            )
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=700
    )
    
    return fig

# Update feature comparison plot
@app.callback(
    Output('comparison-plot', 'figure'),
    [Input('apply-button', 'n_clicks'),
     Input('compare-feature', 'value')],
    [State('playlist-filter', 'value'),
     State('genre-filter', 'value'),
     State('popularity-slider', 'value')]
)
def update_comparison_plot(n_clicks, feature, selected_playlists, selected_genres, popularity_range):
    if df.empty or not feature:
        return go.Figure()
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_playlists and 'playlist_name' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['playlist_name'].isin(selected_playlists)]
        
    if selected_genres and 'playlist_genre' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['playlist_genre'].isin(selected_genres)]
        
    if popularity_range and 'track_popularity' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['track_popularity'] >= popularity_range[0]) & 
                                 (filtered_df['track_popularity'] <= popularity_range[1])]
    
    # Create comparison plot
    fig = go.Figure()
    
    # Compare by playlist if available
    if 'playlist_name' in filtered_df.columns:
        playlist_means = filtered_df.groupby('playlist_name')[feature].mean().sort_values(ascending=False)
        
        fig.add_trace(go.Bar(
            x=playlist_means.index,
            y=playlist_means.values,
            name=feature,
            marker_color='rgba(50, 171, 96, 0.6)'
        ))
        
        fig.update_layout(
            title=f"Average {feature.capitalize()} by Playlist",
            xaxis_title="Playlist",
            yaxis_title=feature.capitalize(),
            template="plotly_dark",
            xaxis={'categoryorder':'total descending'}
        )
        
    # Otherwise compare by genre if available
    elif 'playlist_genre' in filtered_df.columns:
        genre_means = filtered_df.groupby('playlist_genre')[feature].mean().sort_values(ascending=False)
        
        fig.add_trace(go.Bar(
            x=genre_means.index,
            y=genre_means.values,
            name=feature,
            marker_color='rgba(50, 171, 96, 0.6)'
        ))
        
        fig.update_layout(
            title=f"Average {feature.capitalize()} by Genre",
            xaxis_title="Genre",
            yaxis_title=feature.capitalize(),
            template="plotly_dark",
            xaxis={'categoryorder':'total descending'}
        )
        
    # Otherwise just show distribution
    else:
        fig.add_trace(go.Histogram(
            x=filtered_df[feature],
            name=feature,
            nbinsx=30,
            marker_color='rgba(50, 171, 96, 0.6)'
        ))
        
        fig.update_layout(
            title=f"Distribution of {feature.capitalize()}",
            xaxis_title=feature.capitalize(),
            yaxis_title="Count",
            template="plotly_dark"
        )
    
    return fig

# Update track modal with details when a point is clicked
@app.callback(
    [Output('track-modal', 'is_open'),
     Output('track-detail-body', 'children')],
    [Input('scatter-plot', 'clickData')],
    [State('track-modal', 'is_open')]
)
def display_track_info(clickData, is_open):
    if clickData is None:
        return False, ""
    
    # Get the index of the clicked point
    curve_number = clickData['points'][0]['curveNumber']
    point_index = clickData['points'][0]['pointIndex']
    
    # Get the track data
    track = df.iloc[point_index]
    
    # Create track details layout
    track_details = [
        html.H4(f"{track['track_name']}"),
        html.H5(f"by {track['track_artist']}"),
        html.Hr(),
        
        html.Div([
            html.Div([
                html.H6("Track Information"),
                html.P(f"Album: {track['track_album_name']}"),
                html.P(f"Release Date: {track['track_album_release_date']}") if 'track_album_release_date' in track else None,
                html.P(f"Popularity: {track['track_popularity']}") if 'track_popularity' in track else None,
                html.P(f"Duration: {int(track['duration_ms']/1000//60)}:{int(track['duration_ms']/1000%60):02d}") if 'duration_ms' in track else None,
            ], className="mb-4"),
            
            html.Div([
                html.H6("Audio Features"),
                dbc.Row([
                    dbc.Col([
                        html.P(f"{feature.capitalize()}: {track[feature]:.3f}")
                    ]) for feature in audio_features if feature in track and feature != 'track_popularity'
                ])
            ]),
            
            html.Div([
                html.H6("Track URI"),
                html.P(track['uri'] if 'uri' in track else "Not available"),
                html.A("Open in Spotify", 
                      href=f"https://open.spotify.com/track/{track['track_id']}" if 'track_id' in track else "#",
                      target="_blank",
                      className="btn btn-success mt-2")
            ], className="mt-4")
        ])
    ]
    
    return True, track_details

# Close the modal
@app.callback(
    Output('track-modal', 'is_open', allow_duplicate=True),
    [Input('close-track-modal', 'n_clicks')],
    [State('track-modal', 'is_open')],
    prevent_initial_call=True
)
def close_modal(n_clicks, is_open):
    if n_clicks:
        return False
    return is_open

# Run the app
if __name__ == '__main__':
    if not df.empty:
        print(f"Loaded data with {len(df)} tracks")
        print("Starting Spotify Data Explorer dashboard...")
        print("Dashboard running at http://127.0.0.1:8050/")
        app.run(debug=True)
    else:
        print("No data available. Please load Spotify data first.")
