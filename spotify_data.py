import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv(r"C:\Users\mlund\OneDrive\Skrivebord\Scripts\Spotify_data\high_popularity_spotify_data.csv")

# Display the first few rows of the data
df.head(10)
print(df)

df.info()

# nan values
df.isnull().sum()

# Drop nan values
df = df.dropna()


# Plot the data
plt.scatter(df['instrumentalness'], df['playlist_name'])
plt.xlabel('instrunmentalness')
plt.ylabel('Playlist Name')
plt.title('instru vs Playlist Name')
plt.show()

# Plot the data with a logarithmic scale
plt.scatter(df['instrumentalness'], df['playlist_name'])
plt.xlabel('instrunmentalness')
plt.ylabel('Playlist Name')
plt.title('instru vs Playlist Name')
plt.xscale('log')
plt.show()

# Plot the data with a bar chart
plt.bar(df['playlist_name'], df['instrumentalness'])
plt.xlabel('Playlist Name')
plt.ylabel('instrunmentalness')
plt.title('Playlist Name vs instru')
plt.xticks(rotation=90)
plt.show()

#plot the data with comparison
plt.figure(figsize=(10,6))
plt.scatter(df['playlist_name'], df['instrumentalness'], c='red', label='instru')
plt.scatter(df['playlist_name'], df['danceability'], c='blue', label='dance')
plt.scatter(df['playlist_name'], df['energy'], c='green', label='energy')
plt.scatter(df['playlist_name'], df['valence'], c='yellow', label='valence')
plt.xlabel('Playlist Name')
plt.ylabel('Values')
plt.title('Playlist Name vs Values')
plt.xticks(rotation=90)
plt.legend()
plt.show()








