from spotify_scraper import SpotifyDataScraper

# Create a scraper instance
scraper = SpotifyDataScraper()

# Check if API is working by searching for a track
if scraper.sp is not None:
    print("Spotify API is properly configured!")
    
    # Try a simple search
    print("\nSearching for 'Bohemian Rhapsody'...")
    results = scraper.search_and_get_tracks("Bohemian Rhapsody", limit=3)
    
    if not results.empty:
        print("\nResults:")
        for i, (_, track) in enumerate(results.iterrows()):
            print(f"{i+1}. {track['track_name']} by {track['track_artist']} (Popularity: {track['track_popularity']})")
    else:
        print("No results found.")
else:
    print("Spotify API is not properly configured.")
    print("Please check your credentials in the .env file.")
