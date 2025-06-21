import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_spotify_data():
    """
    Simple example analysis script that demonstrates how to analyze the Spotify data
    without requiring additional libraries
    """
    # Load the data
    try:
        high_pop_df = pd.read_csv("high_popularity_spotify_data.csv")
        low_pop_df = pd.read_csv("low_popularity_spotify_data.csv")
        
        # Combine datasets
        df = pd.concat([high_pop_df, low_pop_df], ignore_index=True)
        print(f"Loaded {len(df)} tracks in total")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Display basic information
    print("\nData overview:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Drop rows with missing values
    df = df.dropna()
    print(f"\nRemaining rows after dropping missing values: {len(df)}")
    
    # Basic statistics
    print("\nBasic statistics:")
    audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness']
    print(df[audio_features].describe())
    
    # Correlation analysis
    print("\nCorrelation between features:")
    correlation = df[audio_features].corr()
    print(correlation)
    
    # Plot correlation heatmap using matplotlib
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(audio_features)), audio_features, rotation=45)
    plt.yticks(range(len(audio_features)), audio_features)
    
    # Add correlation values as text
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            plt.text(j, i, f'{correlation.iloc[i, j]:.2f}', 
                     ha='center', va='center', color='black')
    
    plt.title('Correlation between Audio Features')
    plt.tight_layout()
    plt.show()
    
    # Distribution of audio features
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(audio_features, 1):
        plt.subplot(3, 3, i)
        plt.hist(df[feature], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {feature}')
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Popularity distribution
    if 'track_popularity' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['track_popularity'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Track Popularity')
        plt.xlabel('Popularity')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Genre analysis
    if 'playlist_genre' in df.columns:
        genre_counts = df['playlist_genre'].value_counts()
        
        plt.figure(figsize=(12, 6))
        plt.bar(genre_counts.index, genre_counts.values, color='skyblue')
        plt.title('Number of Tracks in Each Genre')
        plt.xlabel('Genre')
        plt.ylabel('Number of Tracks')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Audio features by genre
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(audio_features[:6], 1):  # Take only first 6 features
            plt.subplot(2, 3, i)
            
            # Create boxplot manually using matplotlib
            genres = df['playlist_genre'].unique()
            data = [df[df['playlist_genre'] == genre][feature] for genre in genres]
            
            plt.boxplot(data, labels=genres)
            plt.title(f'{feature} by Genre')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Scatter plots
    plt.figure(figsize=(15, 10))
    
    # Energy vs Danceability
    plt.subplot(2, 2, 1)
    plt.scatter(df['energy'], df['danceability'], alpha=0.6, color='blue')
    plt.title('Energy vs Danceability')
    plt.xlabel('Energy')
    plt.ylabel('Danceability')
    plt.grid(alpha=0.3)
    
    # Valence vs Energy
    plt.subplot(2, 2, 2)
    plt.scatter(df['valence'], df['energy'], alpha=0.6, color='green')
    plt.title('Valence vs Energy')
    plt.xlabel('Valence')
    plt.ylabel('Energy')
    plt.grid(alpha=0.3)
    
    # Acousticness vs Energy
    plt.subplot(2, 2, 3)
    plt.scatter(df['acousticness'], df['energy'], alpha=0.6, color='red')
    plt.title('Acousticness vs Energy')
    plt.xlabel('Acousticness')
    plt.ylabel('Energy')
    plt.grid(alpha=0.3)
    
    # Tempo vs Energy
    plt.subplot(2, 2, 4)
    plt.scatter(df['tempo'], df['energy'], alpha=0.6, color='purple')
    plt.title('Tempo vs Energy')
    plt.xlabel('Tempo')
    plt.ylabel('Energy')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # If popularity data is available, show relationship between features and popularity
    if 'track_popularity' in df.columns:
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(audio_features[:6], 1):  # Take only first 6 features
            plt.subplot(2, 3, i)
            plt.scatter(df[feature], df['track_popularity'], alpha=0.6, color='blue')
            plt.title(f'{feature} vs Popularity')
            plt.xlabel(feature)
            plt.ylabel('Track Popularity')
            plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Find most danceable songs
    print("\nMost danceable songs:")
    print(df.sort_values('danceability', ascending=False)[['track_name', 'track_artist', 'danceability']].head(10))
    
    # Find most energetic songs
    print("\nMost energetic songs:")
    print(df.sort_values('energy', ascending=False)[['track_name', 'track_artist', 'energy']].head(10))
    
    # Find most popular songs
    if 'track_popularity' in df.columns:
        print("\nMost popular songs:")
        print(df.sort_values('track_popularity', ascending=False)[['track_name', 'track_artist', 'track_popularity']].head(10))
    
    # Feature combinations - Find songs that are both danceable and energetic
    print("\nSongs that are both danceable and energetic:")
    danceable_energetic = df[(df['danceability'] > 0.8) & (df['energy'] > 0.8)]
    print(danceable_energetic[['track_name', 'track_artist', 'danceability', 'energy']].head(10))
    
    # Create a simple music recommendation function
    def recommend_similar_songs(track_name, n=5):
        """Find songs similar to the given track based on audio features"""
        if track_name not in df['track_name'].values:
            print(f"Track '{track_name}' not found in the dataset")
            return None
        
        # Get the audio features of the input track
        track = df[df['track_name'] == track_name].iloc[0]
        
        # Calculate Euclidean distance for each track
        distances = []
        for _, row in df.iterrows():
            if row['track_name'] == track_name:  # Skip the input track
                continue
                
            # Calculate distance based on audio features
            dist = 0
            for feature in audio_features:
                dist += (track[feature] - row[feature]) ** 2
            distances.append((row['track_name'], row['track_artist'], np.sqrt(dist)))
        
        # Sort by distance and get the top n recommendations
        recommendations = sorted(distances, key=lambda x: x[2])[:n]
        
        return recommendations
    
    # Example: Get recommendations for a popular track
    if len(df) > 0:
        popular_track = df.sort_values('track_popularity', ascending=False).iloc[0]['track_name']
        print(f"\nSimilar tracks to '{popular_track}':")
        recommendations = recommend_similar_songs(popular_track)
        if recommendations:
            for i, (name, artist, _) in enumerate(recommendations, 1):
                print(f"{i}. {name} by {artist}")
    
    # Genre distribution visualization
    if 'playlist_genre' in df.columns:
        genre_counts = df['playlist_genre'].value_counts()
        plt.figure(figsize=(12, 8))
        plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', 
                startangle=140, shadow=True)
        plt.title('Genre Distribution')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.tight_layout()
        plt.show()
    
    # Create a plot comparing average audio features across different popularity ranges
    if 'track_popularity' in df.columns:
        # Create popularity bins
        df['popularity_bin'] = pd.cut(df['track_popularity'], 
                                    bins=[0, 25, 50, 75, 100], 
                                    labels=['0-25', '26-50', '51-75', '76-100'])
        
        # Calculate mean of features for each bin
        pop_means = df.groupby('popularity_bin')[audio_features].mean().reset_index()
        
        # Plot
        plt.figure(figsize=(15, 8))
        
        x = np.arange(len(pop_means['popularity_bin']))
        width = 0.1
        offsets = np.linspace(-0.4, 0.4, len(audio_features))
        
        for i, feature in enumerate(audio_features):
            plt.bar(x + offsets[i], pop_means[feature], width, label=feature)
        
        plt.xlabel('Popularity Range')
        plt.ylabel('Average Value')
        plt.title('Audio Features by Popularity Range')
        plt.xticks(x, pop_means['popularity_bin'])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyze_spotify_data()
