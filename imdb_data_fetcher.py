import os
import csv
import json
import time
import requests
import pandas as pd
from typing import Dict, List, Any, Optional, Union

class IMDbDataFetcher:
    """
    Fetch movie data from OMDb API and save to CSV files to create a local database.
    
    This allows you to build a local database of IMDb information for use in your
    recommendation system without relying on API calls during runtime.
    """
    
    def __init__(self, api_key: str, data_dir: str = './data'):
        """
        Initialize the IMDb data fetcher.
        
        Args:
            api_key: OMDb API key
            data_dir: Directory to store CSV files
        """
        self.api_key = api_key
        self.data_dir = data_dir
        self.base_url = "http://www.omdbapi.com/"
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Define file paths
        self.movies_file = os.path.join(data_dir, 'movies.csv')
        self.ratings_file = os.path.join(data_dir, 'ratings.csv')
        self.genres_file = os.path.join(data_dir, 'genres.csv')
        self.cast_file = os.path.join(data_dir, 'cast.csv')
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self) -> None:
        """Initialize CSV files with headers if they don't exist."""
        
        # Movies file
        if not os.path.exists(self.movies_file):
            with open(self.movies_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'imdb_id', 'title', 'year', 'rated', 'released', 'runtime',
                    'director', 'writer', 'plot', 'language', 'country',
                    'awards', 'poster', 'metascore', 'imdb_rating', 'imdb_votes',
                    'type', 'dvd', 'box_office', 'production', 'website'
                ])
        
        # Ratings file
        if not os.path.exists(self.ratings_file):
            with open(self.ratings_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['imdb_id', 'source', 'value'])
        
        # Genres file
        if not os.path.exists(self.genres_file):
            with open(self.genres_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['imdb_id', 'genre'])
        
        # Cast file
        if not os.path.exists(self.cast_file):
            with open(self.cast_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['imdb_id', 'actor', 'character'])
    
    def fetch_movie_by_imdb_id(self, imdb_id: str, force_update: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetch movie data from OMDb API by IMDb ID and save to CSV files.
        
        Args:
            imdb_id: IMDb ID (with or without 'tt' prefix)
            force_update: Whether to force update even if movie exists in database
            
        Returns:
            Movie data dictionary or None if not found
        """
        # Add 'tt' prefix if not already present
        if imdb_id and not imdb_id.startswith('tt'):
            imdb_id = f"tt{imdb_id}"
        
        # Check if movie already exists in our database
        if not force_update and self._movie_exists(imdb_id):
            return self._get_movie_from_db(imdb_id)
        
        # Fetch movie data from OMDb API
        params = {
            'i': imdb_id,
            'apikey': self.api_key,
            'plot': 'full'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('Response') == 'False':
                print(f"Movie not found: {imdb_id}, Error: {data.get('Error')}")
                return None
            
            # Save to database
            self._save_movie_to_db(data)
            
            return data
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching movie {imdb_id}: {str(e)}")
            return None
    
    def fetch_movies_by_imdb_ids(self, imdb_ids: List[str], delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Fetch multiple movies by IMDb IDs with a delay between requests.
        
        Args:
            imdb_ids: List of IMDb IDs
            delay: Delay between requests in seconds (to avoid rate limiting)
            
        Returns:
            List of movie data dictionaries
        """
        results = []
        
        for imdb_id in imdb_ids:
            movie = self.fetch_movie_by_imdb_id(imdb_id)
            if movie:
                results.append(movie)
            
            # Delay to avoid rate limiting
            time.sleep(delay)
        
        return results
    
    def _movie_exists(self, imdb_id: str) -> bool:
        """Check if movie exists in the local database."""
        if not os.path.exists(self.movies_file):
            return False
        
        df = pd.read_csv(self.movies_file)
        return imdb_id in df['imdb_id'].values
    
    def _get_movie_from_db(self, imdb_id: str) -> Dict[str, Any]:
        """Get movie data from local database."""
        # Load movie basic info
        movies_df = pd.read_csv(self.movies_file)
        movie_row = movies_df[movies_df['imdb_id'] == imdb_id].iloc[0]
        
        # Convert to dictionary
        movie_data = movie_row.to_dict()
        
        # Load genres
        genres_df = pd.read_csv(self.genres_file)
        genres = genres_df[genres_df['imdb_id'] == imdb_id]['genre'].tolist()
        movie_data['Genre'] = ', '.join(genres)
        
        # Load ratings
        ratings_df = pd.read_csv(self.ratings_file)
        ratings = ratings_df[ratings_df['imdb_id'] == imdb_id]
        
        # Format ratings like OMDb API
        ratings_list = []
        for _, row in ratings.iterrows():
            ratings_list.append({
                'Source': row['source'],
                'Value': row['value']
            })
        movie_data['Ratings'] = ratings_list
        
        # Load cast
        cast_df = pd.read_csv(self.cast_file)
        cast = cast_df[cast_df['imdb_id'] == imdb_id]
        
        # Format actors
        actors = []
        for _, row in cast.iterrows():
            actors.append(row['actor'])
        movie_data['Actors'] = ', '.join(actors)
        
        return movie_data
    
    def _save_movie_to_db(self, movie_data: Dict[str, Any]) -> None:
        """Save movie data to CSV files."""
        imdb_id = movie_data.get('imdbID')
        
        # Save to movies.csv
        with open(self.movies_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                imdb_id,
                movie_data.get('Title', ''),
                movie_data.get('Year', ''),
                movie_data.get('Rated', ''),
                movie_data.get('Released', ''),
                movie_data.get('Runtime', ''),
                movie_data.get('Director', ''),
                movie_data.get('Writer', ''),
                movie_data.get('Plot', ''),
                movie_data.get('Language', ''),
                movie_data.get('Country', ''),
                movie_data.get('Awards', ''),
                movie_data.get('Poster', ''),
                movie_data.get('Metascore', ''),
                movie_data.get('imdbRating', ''),
                movie_data.get('imdbVotes', ''),
                movie_data.get('Type', ''),
                movie_data.get('DVD', ''),
                movie_data.get('BoxOffice', ''),
                movie_data.get('Production', ''),
                movie_data.get('Website', '')
            ])
        
        # Save genres
        genres = movie_data.get('Genre', '').split(', ')
        with open(self.genres_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for genre in genres:
                if genre:
                    writer.writerow([imdb_id, genre])
        
        # Save ratings
        ratings = movie_data.get('Ratings', [])
        with open(self.ratings_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for rating in ratings:
                writer.writerow([
                    imdb_id,
                    rating.get('Source', ''),
                    rating.get('Value', '')
                ])
        
        # Save cast
        actors = movie_data.get('Actors', '').split(', ')
        with open(self.cast_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for actor in actors:
                if actor:
                    writer.writerow([imdb_id, actor, ''])  # We don't have character names

    def search_movies(self, search_term: str, search_type: str = 'title') -> List[Dict[str, Any]]:
        """
        Search for movies in the local database.
        
        Args:
            search_term: Search term
            search_type: Type of search ('title', 'genre', 'director', 'actor')
            
        Returns:
            List of matching movie data dictionaries
        """
        if not os.path.exists(self.movies_file):
            return []
        
        movies_df = pd.read_csv(self.movies_file)
        
        if search_type == 'title':
            # Search by title
            matches = movies_df[movies_df['title'].str.contains(search_term, case=False, na=False)]
            imdb_ids = matches['imdb_id'].tolist()
        
        elif search_type == 'genre':
            # Search by genre
            genres_df = pd.read_csv(self.genres_file)
            genre_matches = genres_df[genres_df['genre'].str.contains(search_term, case=False, na=False)]
            imdb_ids = genre_matches['imdb_id'].tolist()
        
        elif search_type == 'director':
            # Search by director
            matches = movies_df[movies_df['director'].str.contains(search_term, case=False, na=False)]
            imdb_ids = matches['imdb_id'].tolist()
        
        elif search_type == 'actor':
            # Search by actor
            cast_df = pd.read_csv(self.cast_file)
            actor_matches = cast_df[cast_df['actor'].str.contains(search_term, case=False, na=False)]
            imdb_ids = actor_matches['imdb_id'].tolist()
        
        else:
            return []
        
        # Return full movie data for each matching ID
        return [self._get_movie_from_db(imdb_id) for imdb_id in imdb_ids]
    
    def export_for_recommendation_system(self) -> Dict[str, pd.DataFrame]:
        """
        Export data in a format suitable for the recommendation system.
        
        Returns:
            Dictionary of DataFrames ready for the recommendation system
        """
        if not os.path.exists(self.movies_file):
            raise FileNotFoundError("Movie database files not found")
        
        # Load all data
        movies_df = pd.read_csv(self.movies_file)
        genres_df = pd.read_csv(self.genres_file)
        cast_df = pd.read_csv(self.cast_file)
        
        # Create movie metadata DataFrame
        metadata = movies_df[['imdb_id', 'title', 'year', 'plot', 'imdb_rating']]
        metadata = metadata.rename(columns={
            'imdb_id': 'imdb_id',
            'title': 'title',
            'year': 'year',
            'plot': 'overview',
            'imdb_rating': 'vote_average'
        })
        
        # Create genres DataFrame in format expected by recommendation system
        genres_pivot = pd.pivot_table(
            genres_df, 
            index='imdb_id',
            columns='genre', 
            aggfunc=lambda x: 1, 
            fill_value=0
        )
        
        # Create cast DataFrame
        cast_pivot = pd.pivot_table(
            cast_df,
            index='imdb_id',
            columns='actor',
            aggfunc=lambda x: 1,
            fill_value=0
        )
        
        return {
            'metadata': metadata,
            'genres': genres_pivot,
            'cast': cast_pivot
        }

# Example usage
if __name__ == "__main__":
    # Initialize fetcher with your OMDb API key
    fetcher = IMDbDataFetcher(api_key="your_api_key_here")
    
    # Fetch a movie
    toy_story = fetcher.fetch_movie_by_imdb_id("tt0114709")
    
    if toy_story:
        print(f"Successfully fetched and saved: {toy_story['Title']}")
    
    # Fetch multiple movies
    movie_ids = ["tt0068646", "tt0468569", "tt0111161"]  # Godfather, Dark Knight, Shawshank
    fetcher.fetch_movies_by_imdb_ids(movie_ids)
    
    # Search for movies
    batman_movies = fetcher.search_movies("batman", "title")
    print(f"Found {len(batman_movies)} Batman movies")
    
    # Export data for recommendation system
    recommendation_data = fetcher.export_for_recommendation_system()
    print(f"Exported {len(recommendation_data['metadata'])} movies for recommendation system")