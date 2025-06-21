import os
from dotenv import load_dotenv
from spotify_scraper import SpotifyDataScraper
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

def main():
    # Create the scraper
    scraper = SpotifyDataScraper()
    
    # Example operations menu
    while True:
        print("\n===== Spotify Data Analysis Tool =====")
        print("1. Load and analyze existing data")
        print("2. Visualize audio features")
        print("3. Perform clustering analysis")
        print("4. Compare playlists")
        print("5. Genre analysis")
        print("6. Get top tracks by feature")
        print("7. Search for tracks")
        print("8. Get recommendations")
        print("9. Fetch a new playlist (requires API key)")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-9): ")
        
        if choice == '0':
            print("Exiting...")
            break
            
        elif choice == '1':
            # Analyze existing data
            analysis = scraper.analyze_track_features()
            if analysis:
                print("\nStatistics:")
                print(analysis['statistics'])
                
                print("\nCorrelations:")
                print(analysis['correlations'])
                
                # Plot correlation heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(analysis['correlations'], annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Correlation between Audio Features')
                plt.tight_layout()
                plt.show()
                
        elif choice == '2':
            # Visualize audio features
            features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence']
            scraper.visualize_features(features=features)
            
        elif choice == '3':
            # Perform clustering analysis
            n_clusters = 4
            try:
                n_input = input("Enter number of clusters (default: 4): ")
                if n_input:
                    n_clusters = int(n_input)
            except ValueError:
                print("Invalid input, using default (4)")
                
            clusters = scraper.cluster_tracks(n_clusters=n_clusters)
            if clusters:
                scraper.visualize_clusters(clusters)
                
                print("\nRepresentative tracks for each cluster:")
                for track in clusters['representative_tracks']:
                    print(f"Cluster {track['cluster']}: {track['track_name']} by {track['track_artist']} (Popularity: {track['track_popularity']})")
                    
        elif choice == '4':
            # Compare playlists
            if scraper.combined_df is not None:
                available_playlists = scraper.combined_df['playlist_name'].unique()
            elif scraper.high_popularity_df is not None:
                available_playlists = scraper.high_popularity_df['playlist_name'].unique()
            elif scraper.low_popularity_df is not None:
                available_playlists = scraper.low_popularity_df['playlist_name'].unique()
            else:
                print("No playlist data available")
                continue
                
            print("\nAvailable playlists:")
            for i, playlist in enumerate(available_playlists):
                print(f"{i+1}. {playlist}")
                
            selected = input("\nEnter playlist numbers to compare (comma-separated, e.g., 1,2,3): ")
            try:
                indices = [int(idx.strip()) - 1 for idx in selected.split(",")]
                selected_playlists = [available_playlists[idx] for idx in indices if 0 <= idx < len(available_playlists)]
                if selected_playlists:
                    scraper.compare_playlists(selected_playlists)
                else:
                    print("No valid playlists selected")
            except ValueError:
                print("Invalid input")
                
        elif choice == '5':
            # Genre analysis
            scraper.genre_analysis()
            
        elif choice == '6':
            # Get top tracks by feature
            features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence', 'tempo', 'track_popularity']
            
            # Check which features are available in the data
            if scraper.combined_df is not None:
                df = scraper.combined_df
            elif scraper.high_popularity_df is not None:
                df = scraper.high_popularity_df
            elif scraper.low_popularity_df is not None:
                df = scraper.low_popularity_df
            else:
                print("No data available")
                continue
                
            available_features = [f for f in features if f in df.columns]
            
            print("\nAvailable features:")
            for i, feature in enumerate(available_features):
                print(f"{i+1}. {feature}")
                
            selected = input("\nEnter feature number: ")
            try:
                idx = int(selected.strip()) - 1
                if 0 <= idx < len(available_features):
                    feature = available_features[idx]
                    
                    # Ask for ascending or descending
                    order = input("Sort ascending? (y/n, default: n): ").lower()
                    ascending = order == 'y'
                    
                    # Ask for number of tracks
                    n_input = input("Number of tracks to show (default: 10): ")
                    top_n = 10
                    if n_input:
                        try:
                            top_n = int(n_input)
                        except ValueError:
                            print("Invalid input, using default (10)")
                            
                    top_tracks = scraper.get_top_tracks_by_feature(feature, top_n=top_n, ascending=ascending)
                    if not top_tracks.empty:
                        if ascending:
                            print(f"\nBottom {top_n} tracks by {feature}:")
                        else:
                            print(f"\nTop {top_n} tracks by {feature}:")
                        print(top_tracks)
                else:
                    print("Invalid feature selection")
            except ValueError:
                print("Invalid input")
                
        elif choice == '7':
            # Search for tracks (requires API)
            if scraper.sp is None:
                print("Spotify API credentials not set. Cannot search for tracks.")
                continue
                
            query = input("Enter search query: ")
            limit = 5
            limit_input = input("Number of results (default: 5): ")
            if limit_input:
                try:
                    limit = int(limit_input)
                except ValueError:
                    print("Invalid input, using default (5)")
                    
            results = scraper.search_and_get_tracks(query, limit=limit)
            if not results.empty:
                print("\nSearch results:")
                for i, (_, track) in enumerate(results.iterrows()):
                    print(f"{i+1}. {track['track_name']} by {track['track_artist']} (Popularity: {track['track_popularity']})")
                    
                # Option to analyze a specific track
                track_idx = input("\nEnter track number to analyze (or press Enter to skip): ")
                if track_idx:
                    try:
                        idx = int(track_idx) - 1
                        if 0 <= idx < len(results):
                            track = results.iloc[idx]
                            print(f"\nAnalyzing: {track['track_name']} by {track['track_artist']}")
                            
                            # Display track features
                            features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                                      'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness']
                            
                            print("\nAudio Features:")
                            for feature in features:
                                if feature in track:
                                    print(f"{feature}: {track[feature]}")
                                    
                            # Plot radar chart for track
                            # Filter features to normalize between 0 and 1
                            norm_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                                          'instrumentalness', 'liveness', 'valence']
                            
                            # Create radar chart
                            import numpy as np
                            
                            # Number of variables
                            N = len(norm_features)
                            
                            # What will be the angle of each axis in the plot
                            angles = [n / float(N) * 2 * np.pi for n in range(N)]
                            angles += angles[:1]  # Close the loop
                            
                            # Create figure
                            fig = plt.figure(figsize=(8, 8))
                            
                            # Create the subplot
                            ax = plt.subplot(111, polar=True)
                            
                            # Draw one axis per variable and add labels
                            plt.xticks(angles[:-1], norm_features, size=12)
                            
                            # Draw ylabels
                            ax.set_rlabel_position(0)
                            plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
                            plt.ylim(0, 1)
                            
                            # Get values
                            values = [track[feature] for feature in norm_features]
                            values += values[:1]  # Close the loop
                            
                            # Plot values
                            ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
                            ax.fill(angles, values, color='blue', alpha=0.1)
                            
                            plt.title(f"Audio Features: {track['track_name']}", size=15, y=1.1)
                            plt.tight_layout()
                            plt.show()
                    except ValueError:
                        print("Invalid input")
                        
        elif choice == '8':
            # Get recommendations (requires API)
            if scraper.sp is None:
                print("Spotify API credentials not set. Cannot get recommendations.")
                continue
                
            # Option to use search first
            use_search = input("Search for seed tracks first? (y/n, default: n): ").lower() == 'y'
            
            seed_tracks = []
            
            if use_search:
                query = input("Enter search query: ")
                results = scraper.search_and_get_tracks(query, limit=5)
                
                if not results.empty:
                    print("\nSearch results:")
                    for i, (_, track) in enumerate(results.iterrows()):
                        print(f"{i+1}. {track['track_name']} by {track['track_artist']} (Popularity: {track['track_popularity']})")
                        
                    # Select seed tracks
                    selected = input("\nEnter track numbers to use as seeds (comma-separated, e.g., 1,2): ")
                    try:
                        indices = [int(idx.strip()) - 1 for idx in selected.split(",")]
                        seed_tracks = [results.iloc[idx]['track_id'] for idx in indices if 0 <= idx < len(results)]
                    except ValueError:
                        print("Invalid input, no seed tracks selected")
            else:
                # Manual entry of track IDs
                track_ids = input("Enter Spotify track IDs (comma-separated): ")
                seed_tracks = [tid.strip() for tid in track_ids.split(",") if tid.strip()]
                
            if not seed_tracks:
                print("No seed tracks provided")
                continue
                
            # Get number of recommendations
            limit = 10
            limit_input = input("Number of recommendations (default: 10): ")
            if limit_input:
                try:
                    limit = int(limit_input)
                except ValueError:
                    print("Invalid input, using default (10)")
                    
            # Get recommendations
            recommendations = scraper.get_recommendations(seed_tracks=seed_tracks, limit=limit)
            
            if not recommendations.empty:
                print("\nRecommendations:")
                for i, (_, track) in enumerate(recommendations.iterrows()):
                    print(f"{i+1}. {track['track_name']} by {track['track_artist']} (Popularity: {track['track_popularity']})")
                    
                # Option to analyze recommendations
                analyze_recs = input("\nAnalyze recommendations? (y/n, default: n): ").lower() == 'y'
                
                if analyze_recs:
                    # Analyze audio features of recommendations
                    analysis = scraper.analyze_track_features(recommendations)
                    
                    if analysis:
                        print("\nRecommendation Statistics:")
                        print(analysis['statistics'])
                        
                        # Visualize recommendation features
                        scraper.visualize_features(recommendations)
                        
        elif choice == '9':
            # Fetch a new playlist (requires API)
            if scraper.sp is None:
                print("Spotify API credentials not set. Cannot fetch playlists.")
                continue
                
            playlist_id = input("Enter Spotify playlist ID: ")
            if not playlist_id:
                print("No playlist ID provided")
                continue
                
            playlist_name = input("Enter playlist name (or leave empty to fetch from API): ")
            playlist_genre = input("Enter playlist genre (optional): ")
            playlist_subgenre = input("Enter playlist subgenre (optional): ")
            
            filename = input("Enter output CSV filename (default: playlist_data.csv): ")
            if not filename:
                filename = "playlist_data.csv"
                
            success = scraper.fetch_and_save_playlist(
                playlist_id, 
                filename,
                playlist_name,
                playlist_genre if playlist_genre else None,
                playlist_subgenre if playlist_subgenre else None
            )
            
            if success:
                print(f"Playlist data saved to {filename}")
                reload = input("Reload data to include new playlist? (y/n, default: y): ").lower() != 'n'
                if reload:
                    scraper.try_load_csv_data()
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
