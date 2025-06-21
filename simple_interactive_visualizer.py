import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from spotify_scraper import SpotifyDataScraper
import numpy as np

def main():
    print("Loading Spotify data...")
    # Create scraper instance to load data
    scraper = SpotifyDataScraper()
    
    # Check if data is loaded
    if scraper.combined_df is not None:
        df = scraper.combined_df
    elif scraper.high_popularity_df is not None:
        df = scraper.high_popularity_df
    elif scraper.low_popularity_df is not None:
        df = scraper.low_popularity_df
    else:
        print("No data available for visualization")
        return
    
    print(f"Loaded {len(df)} tracks.")
    
    # Get audio features for visualization
    audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness', 'valence']
    
    # Filter to features that exist in the DataFrame
    audio_features = [f for f in audio_features if f in df.columns]
    
    # Add popularity if available
    if 'track_popularity' in df.columns:
        audio_features.append('track_popularity')
    
    while True:
        print("\n===== Interactive Spotify Data Visualization =====")
        print("1. Interactive scatter plot")
        print("2. Feature distributions")
        print("3. Correlation heatmap")
        print("4. Track comparison (radar chart)")
        print("5. Playlist comparison")
        print("6. Top tracks by feature")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ")
        
        if choice == '0':
            print("Exiting...")
            break
            
        elif choice == '1':
            # Interactive scatter plot
            print("\nAvailable features:")
            for i, feature in enumerate(audio_features):
                print(f"{i+1}. {feature}")
                
            x_idx = int(input("\nSelect feature for X-axis (number): ")) - 1
            y_idx = int(input("Select feature for Y-axis (number): ")) - 1
            
            if 0 <= x_idx < len(audio_features) and 0 <= y_idx < len(audio_features):
                x_feature = audio_features[x_idx]
                y_feature = audio_features[y_idx]
                
                # Color options
                print("\nColor by:")
                print("1. None")
                if 'playlist_name' in df.columns:
                    print("2. Playlist")
                if 'playlist_genre' in df.columns:
                    print("3. Genre")
                if 'track_popularity' in df.columns:
                    print("4. Popularity")
                
                color_choice = input("\nSelect color option (number): ")
                
                if color_choice == '2' and 'playlist_name' in df.columns:
                    color_by = 'playlist_name'
                elif color_choice == '3' and 'playlist_genre' in df.columns:
                    color_by = 'playlist_genre'
                elif color_choice == '4' and 'track_popularity' in df.columns:
                    color_by = 'track_popularity'
                else:
                    color_by = None
                
                # Create interactive scatter plot
                if color_by:
                    fig = px.scatter(
                        df, 
                        x=x_feature, 
                        y=y_feature,
                        color=color_by,
                        hover_name='track_name',
                        hover_data=['track_artist', 'track_popularity'] if 'track_popularity' in df.columns else ['track_artist'],
                        title=f"{y_feature.capitalize()} vs {x_feature.capitalize()}",
                        color_continuous_scale=px.colors.sequential.Viridis if color_by == 'track_popularity' else None,
                    )
                else:
                    fig = px.scatter(
                        df, 
                        x=x_feature, 
                        y=y_feature,
                        hover_name='track_name',
                        hover_data=['track_artist', 'track_popularity'] if 'track_popularity' in df.columns else ['track_artist'],
                        title=f"{y_feature.capitalize()} vs {x_feature.capitalize()}",
                    )
                
                fig.update_layout(
                    xaxis_title=x_feature.capitalize(),
                    yaxis_title=y_feature.capitalize(),
                )
                
                fig.show()
            else:
                print("Invalid feature selection")
                
        elif choice == '2':
            # Feature distributions
            # Create distribution plot for all audio features
            rows = len(audio_features)
            fig = make_subplots(rows=rows, cols=1, 
                            subplot_titles=[f.capitalize() for f in audio_features],
                            vertical_spacing=0.05)
            
            for i, feature in enumerate(audio_features):
                if feature in df.columns:
                    fig.add_trace(
                        go.Histogram(x=df[feature], name=feature, nbinsx=30),
                        row=i+1, col=1
                    )
            
            fig.update_layout(
                height=300 * rows,
                title_text="Audio Feature Distributions",
                showlegend=False,
            )
            
            fig.show()
                
        elif choice == '3':
            # Correlation heatmap
            features_to_correlate = [f for f in audio_features if f in df.columns]
            corr = df[features_to_correlate].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr,
                labels=dict(x="Feature", y="Feature", color="Correlation"),
                x=corr.columns,
                y=corr.columns,
                color_continuous_scale="RdBu_r",
                text_auto='.2f'
            )
            
            fig.update_layout(
                title="Feature Correlation Heatmap",
                height=700
            )
            
            fig.show()
                
        elif choice == '4':
            # Track comparison (radar chart)
            # First let user search for tracks
            search_term = input("\nEnter a search term to find tracks: ")
            matches = df[df['track_name'].str.contains(search_term, case=False) | 
                        df['track_artist'].str.contains(search_term, case=False)]
            
            if len(matches) == 0:
                print("No matching tracks found")
                continue
            
            # Show matching tracks
            print(f"\nFound {len(matches)} matching tracks:")
            max_display = min(10, len(matches))
            for i in range(max_display):
                track = matches.iloc[i]
                print(f"{i+1}. {track['track_name']} by {track['track_artist']}")
            
            # Let user select tracks to compare
            selected_indices = input("\nEnter track numbers to compare (comma-separated, max 5): ")
            try:
                indices = [int(idx.strip()) - 1 for idx in selected_indices.split(",")]
                selected_tracks = [matches.iloc[idx] for idx in indices if 0 <= idx < len(matches)]
                
                if not selected_tracks or len(selected_tracks) > 5:
                    print("Invalid selection. Please select between 1 and 5 tracks.")
                    continue
                
                # Features for radar chart
                radar_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                              'instrumentalness', 'liveness', 'valence']
                
                # Filter to features that exist in the DataFrame
                radar_features = [f for f in radar_features if f in df.columns]
                
                # Create radar chart
                fig = go.Figure()
                
                for i, track in enumerate(selected_tracks):
                    values = [track[feature] for feature in radar_features]
                    values.append(values[0])  # Close the loop
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=radar_features + [radar_features[0]],  # Close the loop
                        fill='toself',
                        name=f"{track['track_name']} - {track['track_artist']}",
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Track Feature Comparison",
                )
                
                fig.show()
                
            except (ValueError, IndexError):
                print("Invalid selection")
                
        elif choice == '5':
            # Playlist comparison
            if 'playlist_name' not in df.columns:
                print("No playlist information available in the data")
                continue
            
            # Get unique playlists
            playlists = df['playlist_name'].unique()
            
            print("\nAvailable playlists:")
            for i, playlist in enumerate(playlists):
                print(f"{i+1}. {playlist}")
            
            # Let user select playlists to compare
            selected_indices = input("\nEnter playlist numbers to compare (comma-separated): ")
            try:
                indices = [int(idx.strip()) - 1 for idx in selected_indices.split(",")]
                selected_playlists = [playlists[idx] for idx in indices if 0 <= idx < len(playlists)]
                
                if not selected_playlists:
                    print("No valid playlists selected")
                    continue
                
                # Feature to compare
                print("\nSelect feature to compare:")
                for i, feature in enumerate(audio_features):
                    print(f"{i+1}. {feature}")
                
                feature_idx = int(input("\nEnter feature number: ")) - 1
                
                if 0 <= feature_idx < len(audio_features):
                    feature = audio_features[feature_idx]
                    
                    # Filter to selected playlists
                    filtered_df = df[df['playlist_name'].isin(selected_playlists)]
                    
                    # Calculate average for each playlist
                    playlist_means = filtered_df.groupby('playlist_name')[feature].mean().reindex(selected_playlists)
                    
                    # Create bar chart
                    fig = go.Figure(data=[
                        go.Bar(x=playlist_means.index, y=playlist_means.values)
                    ])
                    
                    fig.update_layout(
                        title=f"Average {feature.capitalize()} by Playlist",
                        xaxis_title="Playlist",
                        yaxis_title=feature.capitalize(),
                    )
                    
                    fig.show()
                    
                    # Also create radar chart comparing all features across playlists
                    radar_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                                  'instrumentalness', 'liveness', 'valence']
                    
                    # Filter to features that exist in the DataFrame
                    radar_features = [f for f in radar_features if f in df.columns]
                    
                    # Calculate means for each playlist
                    playlist_feature_means = filtered_df.groupby('playlist_name')[radar_features].mean().reindex(selected_playlists)
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    for i, (playlist, row) in enumerate(playlist_feature_means.iterrows()):
                        values = row.values.tolist()
                        values.append(values[0])  # Close the loop
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=radar_features + [radar_features[0]],  # Close the loop
                            fill='toself',
                            name=playlist,
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        title="Playlist Feature Comparison",
                    )
                    
                    fig.show()
                else:
                    print("Invalid feature selection")
                
            except (ValueError, IndexError):
                print("Invalid selection")
                
        elif choice == '6':
            # Top tracks by feature
            print("\nSelect feature to rank tracks by:")
            for i, feature in enumerate(audio_features):
                print(f"{i+1}. {feature}")
            
            feature_idx = int(input("\nEnter feature number: ")) - 1
            
            if 0 <= feature_idx < len(audio_features):
                feature = audio_features[feature_idx]
                
                # Ask for ascending or descending
                order = input("Sort ascending? (y/n, default: n): ").lower()
                ascending = order == 'y'
                
                # Number of tracks to show
                n_input = input("Number of tracks to show (default: 10): ")
                top_n = 10
                if n_input:
                    try:
                        top_n = int(n_input)
                    except ValueError:
                        print("Invalid input, using default (10)")
                
                # Sort and get top tracks
                sorted_df = df.sort_values(by=feature, ascending=ascending)
                top_tracks = sorted_df.head(top_n)[['track_name', 'track_artist', feature, 'track_popularity']]
                
                # Create interactive bar chart
                fig = px.bar(
                    top_tracks,
                    x='track_name',
                    y=feature,
                    hover_data=['track_artist', 'track_popularity'],
                    color=feature,
                    title=f"{'Bottom' if ascending else 'Top'} {top_n} Tracks by {feature.capitalize()}",
                )
                
                fig.update_layout(
                    xaxis_title="Track",
                    yaxis_title=feature.capitalize(),
                    xaxis={'categoryorder':'total ascending' if ascending else 'total descending'}
                )
                
                fig.show()
                
                # Also print the tracks
                print(f"\n{'Bottom' if ascending else 'Top'} {top_n} tracks by {feature}:")
                for i, (_, track) in enumerate(top_tracks.iterrows()):
                    print(f"{i+1}. {track['track_name']} by {track['track_artist']} ({feature}: {track[feature]:.3f})")
            else:
                print("Invalid feature selection")
                
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
