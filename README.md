# Spotify Data Analysis Tools

This repository contains tools for analyzing and visualizing Spotify music data. It includes both scripts for working with existing data and tools for fetching new data from the Spotify Web API.

## Components

1. **spotify_analyzer.py**: The most comprehensive all-in-one tool that combines CSV analysis and live API data with advanced visualizations, clustering, and recommendations.

2. **simple_spotify_analysis.py**: A script for analyzing and visualizing Spotify data from CSV files without requiring additional complex libraries.

3. **simple_spotify_scraper.py**: A lightweight script for fetching data from the Spotify Web API using only basic dependencies.

4. **spotify_scraper.py**: A more comprehensive library with advanced analysis and visualization features (requires additional packages).

5. **spotify_live_scraper.py**: A tool focused specifically on real-time data scraping from the Spotify Web API.

6. **run_spotify_analysis.py**: An interactive command-line interface for the comprehensive scraper.

7. **simple_interactive_visualizer.py**: A terminal-based interactive visualization tool using Plotly.

8. **interactive_spotify_dashboard.py**: A web-based dashboard for visualizing Spotify data.

## Getting Started

### Basic Setup (Minimal Dependencies)

1. Ensure you have Python installed
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
4. Install minimal required packages:
   ```
   pip install pandas matplotlib numpy
   ```
5. Run the basic analysis script:
   ```
   python simple_spotify_analysis.py
   ```

### With Spotify API (Additional Dependencies)

To use the Spotify API features, you'll need:

1. Create a Spotify Developer account at https://developer.spotify.com/dashboard/
2. Create a new application to get your Client ID and Secret
3. Create a `.env` file with your credentials:
   ```
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   ```
4. Install additional packages:
   ```
   pip install python-dotenv requests spotipy
   ```
5. Run the simple scraper:   ```
   python simple_spotify_scraper.py
   ```

### Full Setup (All Features)

For all features including advanced clustering and recommendations:

1. Install all required packages:
   ```
   pip install -r requirements.txt
   ```
   Note: On Windows, if you have trouble installing scikit-learn, you can use the simplified tools instead.
   
2. Run the comprehensive analyzer (recommended):
   ```
   python spotify_analyzer.py
   ```
   
3. Or run the analysis tool:
   ```
   python run_spotify_analysis.py
   ```

## Features

### Comprehensive Analyzer (spotify_analyzer.py)

- All-in-one tool combining the best features from all other scripts
- Works with both CSV data and live Spotify API data
- Interactive command-line interface with intuitive menus
- Advanced visualizations using Plotly
- Machine learning capabilities (clustering, outlier detection)
- Mood-based playlist creation
- Feature distribution analysis
- Track comparison and recommendations

### Simple Analysis (simple_spotify_analysis.py)

- Load and analyze existing Spotify data CSV files
- Generate basic statistics and correlations
- Create visualizations of audio features
- Find top tracks by different metrics
- Compare tracks across genres
- Simple track recommendation system
### Simple Scraper (simple_spotify_scraper.py)

- Search for tracks on Spotify
- Fetch audio features for tracks
- Download playlist data to CSV files
- Get track recommendations
- No complex dependencies required

### Full Scraper Library (spotify_scraper.py)

- Everything in the simple versions plus:
- Advanced clustering to find similar tracks
- Detailed playlist and genre comparison
- Comprehensive visualization options
- Custom recommendation algorithms

### Live Scraper (spotify_live_scraper.py)

- Focused on real-time data retrieval from Spotify Web API
- Get new releases, related artists, and recommendations
- Fetch complete track audio features
- Explore playlists and artist catalogs

### Interactive Visualizer (simple_interactive_visualizer.py)

- Terminal-based interactive visualization using Plotly
- Feature distributions and correlations
- Track comparison with radar charts
- Playlist comparison
- Top tracks by feature

## Data Structure

These tools work with Spotify track data that includes:

- Track information: track_id, track_name, track_artist, etc.
- Audio features: danceability, energy, speechiness, acousticness, etc.
- Playlist information: playlist_name, playlist_genre, etc.

## Examples

### Analyzing Both CSV and API Data (Recommended)

```python
# Using the comprehensive analyzer
python spotify_analyzer.py
```

### Analyzing CSV Data

```python
# Using simple_spotify_analysis.py
python simple_spotify_analysis.py
```

### Fetching Playlist Data

```python
# Using simple_spotify_scraper.py
python simple_spotify_scraper.py
# Then choose option 3 and enter a Spotify playlist ID
```

### Interactive Analysis

```python
# Using run_spotify_analysis.py
python run_spotify_analysis.py
# Follow the menu options for different analyses
```

### Visualizing Data

```python
# Using simple_interactive_visualizer.py
python simple_interactive_visualizer.py
```

## Troubleshooting

- If you encounter 403 errors with certain Spotify API endpoints, check that your API credentials are correct and that your app has the necessary permissions
- For large datasets, some visualizations might take longer to render
- If audio features are missing, the API might be rate-limited - try again with fewer tracks
- If you encounter package installation issues, try using the simple scripts that require fewer dependencies
- For Windows users with scikit-learn installation problems, try using the simpler tools that don't require scikit-learn
