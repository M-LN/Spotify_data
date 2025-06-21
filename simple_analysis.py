import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_spotify_data():
    """
    Simple example analysis script that demonstrates how to analyze the Spotify data
    without using the full scraper library
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
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation between Audio Features')
    plt.tight_layout()
    plt.show()
    
    # Distribution of audio features
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(audio_features, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()
    
    # Popularity distribution
    if 'track_popularity' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['track_popularity'], bins=20, kde=True)
        plt.title('Distribution of Track Popularity')
        plt.xlabel('Popularity')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    # Genre analysis
    if 'playlist_genre' in df.columns:
        genre_counts = df['playlist_genre'].value_counts()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=genre_counts.index, y=genre_counts.values)
        plt.title('Number of Tracks in Each Genre')
        plt.xlabel('Genre')
        plt.ylabel('Number of Tracks')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Audio features by genre
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(audio_features[:6], 1):  # Take only first 6 features
            plt.subplot(2, 3, i)
            sns.boxplot(x='playlist_genre', y=feature, data=df)
            plt.title(f'{feature} by Genre')
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    # Scatter plots
    plt.figure(figsize=(15, 10))
    
    # Energy vs Danceability
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='energy', y='danceability', data=df, alpha=0.6)
    plt.title('Energy vs Danceability')
    
    # Valence vs Energy
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='valence', y='energy', data=df, alpha=0.6)
    plt.title('Valence vs Energy')
    
    # Acousticness vs Energy
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='acousticness', y='energy', data=df, alpha=0.6)
    plt.title('Acousticness vs Energy')
    
    # Tempo vs Energy
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='tempo', y='energy', data=df, alpha=0.6)
    plt.title('Tempo vs Energy')
    
    plt.tight_layout()
    plt.show()
    
    # If popularity data is available, show relationship between features and popularity
    if 'track_popularity' in df.columns:
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(audio_features[:6], 1):  # Take only first 6 features
            plt.subplot(2, 3, i)
            sns.scatterplot(x=feature, y='track_popularity', data=df, alpha=0.6)
            plt.title(f'{feature} vs Popularity')
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

if __name__ == "__main__":
    analyze_spotify_data()
