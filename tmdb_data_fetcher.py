import os
import csv
import json
import time
import requests
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from utils.gcloud import init_credentials, get_storage_client
import io

class TMDbDataFetcher:
    """
    Fetch movie data from TMDb API and save to CSV files to create a local database.
    
    This allows you to build a local database of TMDb information for use in your
    recommendation system without relying on API calls during runtime.
    """
    
    def __init__(self, api_key: str = None, bucket_name: str = None, data_folder: str = 'data'):
        """
        Initialize the TMDb data fetcher.
        
        Args:
            api_key: TMDb API key (can be set via env var TMDB_API_KEY)
            bucket_name: Name of the Google Cloud Storage bucket (can be set via env var GCS_BUCKET_NAME)
            data_folder: Folder within the bucket to store data
        """
        # Get API key from env var if not provided
        self.api_key = api_key or os.environ.get("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError("TMDb API key must be provided or TMDB_API_KEY environment variable must be set")
            
        # Get bucket name from env var if not provided
        self.bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("Bucket name must be provided or GCS_BUCKET_NAME environment variable must be set")
            
        self.data_folder = data_folder
        
        # Initialize Google Cloud credentials
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable must be set")
            
        try:
            init_credentials(credentials_path)
            self.storage_client = get_storage_client()
            self.bucket = self.storage_client.bucket(self.bucket_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Cloud Storage: {e}")
        
        # Base URL for TMDb API
        self.base_url = "https://api.themoviedb.org/3"
        
        # Define file paths
        self.movies_file = os.path.join(data_folder, 'movies.csv')
        self.genres_file = os.path.join(data_folder, 'genres.csv')
        self.cast_file = os.path.join(data_folder, 'cast.csv')
        self.crew_file = os.path.join(data_folder, 'crew.csv')
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self) -> None:
        """Initialize CSV files with headers if they don't exist."""
        
        # Movies file
        if not os.path.exists(self.movies_file):
            with open(self.movies_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'tmdb_id', 'imdb_id', 'title', 'original_title', 'release_date', 'runtime',
                    'overview', 'tagline', 'status', 'original_language', 'budget', 
                    'revenue', 'popularity', 'vote_average', 'vote_count',
                    'poster_path', 'backdrop_path', 'adult'
                ])
        
        # Genres file
        if not os.path.exists(self.genres_file):
            with open(self.genres_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['tmdb_id', 'genre_id', 'genre_name'])
        
        # Cast file
        if not os.path.exists(self.cast_file):
            with open(self.cast_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['tmdb_id', 'cast_id', 'person_id', 'character', 'name', 'profile_path', 'order'])
        
        # Crew file
        if not os.path.exists(self.crew_file):
            with open(self.crew_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['tmdb_id', 'person_id', 'name', 'job', 'department', 'profile_path'])
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Make a request to the TMDb API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response from the API
        """
        if params is None:
            params = {}
            
        # Add API key to parameters
        params['api_key'] = self.api_key
        
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {endpoint}: {str(e)}")
            return None
            
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict[str, Any]]:
        """
        Get movie details from TMDb API.
        
        Args:
            tmdb_id: TMDb movie ID
            
        Returns:
            Movie details as a dictionary
        """
        endpoint = f"/movie/{tmdb_id}"
        return self._make_request(endpoint)
        
    def get_movie_credits(self, tmdb_id: int) -> Optional[Dict[str, Any]]:
        """
        Get movie credits from TMDb API.
        
        Args:
            tmdb_id: TMDb movie ID
            
        Returns:
            Movie credits as a dictionary
        """
        endpoint = f"/movie/{tmdb_id}/credits"
        return self._make_request(endpoint)
        
    def search_movies(self, query: str, page: int = 1) -> Optional[Dict[str, Any]]:
        """
        Search for movies on TMDb.
        
        Args:
            query: Search query
            page: Page number
            
        Returns:
            Search results as a dictionary
        """
        endpoint = "/search/movie"
        params = {
            'query': query,
            'page': page
        }
        return self._make_request(endpoint, params)
        
    def get_popular_movies(self, page: int = 1) -> Optional[Dict[str, Any]]:
        """
        Get popular movies from TMDb.
        
        Args:
            page: Page number
            
        Returns:
            Popular movies as a dictionary
        """
        endpoint = "/movie/popular"
        params = {'page': page}
        return self._make_request(endpoint, params)
        
    def get_top_rated_movies(self, page: int = 1) -> Optional[Dict[str, Any]]:
        """
        Get top rated movies from TMDb.
        
        Args:
            page: Page number
            
        Returns:
            Top rated movies as a dictionary
        """
        endpoint = "/movie/top_rated"
        params = {'page': page}
        return self._make_request(endpoint, params)
        
    def save_movie_data(self, movie_data: Dict[str, Any], tmdb_id: int) -> bool:
        """
        Save movie data to Google Cloud Storage.
        
        Args:
            movie_data: Movie data to save
            tmdb_id: TMDb movie ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([movie_data])
            
            # Save to GCS
            path = f"{self.data_folder}/movies/{tmdb_id}.csv"
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            blob = self.bucket.blob(path)
            blob.upload_from_string(csv_content, content_type="text/csv")
            
            return True
        except Exception as e:
            print(f"Error saving movie data for {tmdb_id}: {str(e)}")
            return False
            
    def load_movie_data(self, tmdb_id: int) -> Optional[Dict[str, Any]]:
        """
        Load movie data from Google Cloud Storage.
        
        Args:
            tmdb_id: TMDb movie ID
            
        Returns:
            Movie data as a dictionary
        """
        try:
            path = f"{self.data_folder}/movies/{tmdb_id}.csv"
            blob = self.bucket.blob(path)
            
            if not blob.exists():
                return None
                
            content = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(content))
            
            return df.iloc[0].to_dict()
        except Exception as e:
            print(f"Error loading movie data for {tmdb_id}: {str(e)}")
            return None
    
    def fetch_movie_by_tmdb_id(self, tmdb_id: int, force_update: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetch movie data from TMDb API by TMDb ID and save to CSV files.
        
        Args:
            tmdb_id: TMDb ID
            force_update: Whether to force update even if movie exists in database
            
        Returns:
            Movie data dictionary or None if not found
        """
        # Convert to int if it's a string
        if isinstance(tmdb_id, str):
            try:
                tmdb_id = int(tmdb_id)
            except ValueError:
                print(f"Invalid TMDb ID: {tmdb_id}")
                return None
        
        # Check if movie already exists in our database
        if not force_update and self._movie_exists(tmdb_id):
            return self._get_movie_from_db(tmdb_id)
        
        # Fetch movie data from TMDb API
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # Fetch basic movie info
            movie_url = f"{self.base_url}/movie/{tmdb_id}?language=en-US&append_to_response=credits"
            response = requests.get(movie_url, headers=headers)
            response.raise_for_status()
            movie_data = response.json()
            
            # Save to database
            self._save_movie_to_db(movie_data)
            
            return movie_data
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching movie {tmdb_id}: {str(e)}")
            return None
    
    def fetch_movies_by_tmdb_ids(self, tmdb_ids: List[int], delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Fetch multiple movies by TMDb IDs with a delay between requests.
        
        Args:
            tmdb_ids: List of TMDb IDs
            delay: Delay between requests in seconds (to avoid rate limiting)
            
        Returns:
            List of movie data dictionaries
        """
        results = []
        
        for tmdb_id in tmdb_ids:
            movie = self.fetch_movie_by_tmdb_id(tmdb_id)
            if movie:
                results.append(movie)
            
            # Delay to avoid rate limiting
            time.sleep(delay)
        
        return results
    
    def _movie_exists(self, tmdb_id: int) -> bool:
        """Check if movie exists in the local database."""
        if not os.path.exists(self.movies_file):
            return False
        
        df = pd.read_csv(self.movies_file)
        return tmdb_id in df['tmdb_id'].values
    
    def _get_movie_from_db(self, tmdb_id: int) -> Dict[str, Any]:
        """Get movie data from local database."""
        # Load movie basic info
        movies_df = pd.read_csv(self.movies_file)
        movie_row = movies_df[movies_df['tmdb_id'] == tmdb_id].iloc[0]
        
        # Convert to dictionary
        movie_data = movie_row.to_dict()
        
        # Load genres
        genres_df = pd.read_csv(self.genres_file)
        genres = genres_df[genres_df['tmdb_id'] == tmdb_id]
        
        # Format genres
        genre_list = []
        for _, row in genres.iterrows():
            genre_list.append({
                'id': row['genre_id'],
                'name': row['genre_name']
            })
        movie_data['genres'] = genre_list
        
        # Load cast
        cast_df = pd.read_csv(self.cast_file)
        cast = cast_df[cast_df['tmdb_id'] == tmdb_id]
        
        # Format cast
        cast_list = []
        for _, row in cast.iterrows():
            cast_list.append({
                'id': row['person_id'],
                'name': row['name'],
                'character': row['character'],
                'profile_path': row['profile_path'],
                'order': row['order']
            })
        
        # Load crew
        crew_df = pd.read_csv(self.crew_file)
        crew = crew_df[crew_df['tmdb_id'] == tmdb_id]
        
        # Format crew
        crew_list = []
        for _, row in crew.iterrows():
            crew_list.append({
                'id': row['person_id'],
                'name': row['name'],
                'job': row['job'],
                'department': row['department'],
                'profile_path': row['profile_path']
            })
        
        # Add credits
        movie_data['credits'] = {
            'cast': cast_list,
            'crew': crew_list
        }
        
        return movie_data
    
    def _save_movie_to_db(self, movie_data: Dict[str, Any]) -> None:
        """Save movie data to CSV files."""
        tmdb_id = movie_data.get('id')
        
        # Save to movies.csv
        with open(self.movies_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                tmdb_id,
                movie_data.get('imdb_id', ''),
                movie_data.get('title', ''),
                movie_data.get('original_title', ''),
                movie_data.get('release_date', ''),
                movie_data.get('runtime', ''),
                movie_data.get('overview', ''),
                movie_data.get('tagline', ''),
                movie_data.get('status', ''),
                movie_data.get('original_language', ''),
                movie_data.get('budget', 0),
                movie_data.get('revenue', 0),
                movie_data.get('popularity', 0),
                movie_data.get('vote_average', 0),
                movie_data.get('vote_count', 0),
                movie_data.get('poster_path', ''),
                movie_data.get('backdrop_path', ''),
                movie_data.get('adult', False)
            ])
        
        # Save genres
        genres = movie_data.get('genres', [])
        with open(self.genres_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for genre in genres:
                writer.writerow([
                    tmdb_id,
                    genre.get('id', ''),
                    genre.get('name', '')
                ])
        
        # Save cast and crew
        credits = movie_data.get('credits', {})
        
        # Save cast
        cast = credits.get('cast', [])
        with open(self.cast_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for cast_member in cast:
                writer.writerow([
                    tmdb_id,
                    cast_member.get('cast_id', ''),
                    cast_member.get('id', ''),
                    cast_member.get('character', ''),
                    cast_member.get('name', ''),
                    cast_member.get('profile_path', ''),
                    cast_member.get('order', 0)
                ])
        
        # Save crew
        crew = credits.get('crew', [])
        with open(self.crew_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for crew_member in crew:
                writer.writerow([
                    tmdb_id,
                    crew_member.get('id', ''),
                    crew_member.get('name', ''),
                    crew_member.get('job', ''),
                    crew_member.get('department', ''),
                    crew_member.get('profile_path', '')
                ])

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
        crew_df = pd.read_csv(self.crew_file)
        
        # Create movie metadata DataFrame
        metadata = movies_df[['tmdb_id', 'imdb_id', 'title', 'release_date', 'overview', 'vote_average', 'popularity']]
        metadata = metadata.rename(columns={
            'release_date': 'year',
            'overview': 'overview',
            'vote_average': 'vote_average'
        })
        
        # Create genres DataFrame in format expected by recommendation system
        genres_pivot = pd.pivot_table(
            genres_df, 
            index='tmdb_id',
            columns='genre_name', 
            aggfunc=lambda x: 1, 
            fill_value=0
        )
        
        # Create cast DataFrame
        cast_pivot = pd.pivot_table(
            cast_df,
            index='tmdb_id',
            columns='name',
            aggfunc=lambda x: 1,
            fill_value=0
        )
        
        # Create crew DataFrame with directors
        directors = crew_df[crew_df['job'] == 'Director']
        directors_pivot = pd.pivot_table(
            directors,
            index='tmdb_id',
            columns='name',
            aggfunc=lambda x: 1,
            fill_value=0
        )
        
        return {
            'metadata': metadata,
            'genres': genres_pivot,
            'cast': cast_pivot,
            'directors': directors_pivot
        }

# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.environ.get("TMDB_API_KEY")
    
    if not api_key:
        raise ValueError("TMDB_API_KEY environment variable not set")
    
    # Initialize fetcher with your TMDb API key
    fetcher = TMDbDataFetcher(api_key=api_key)
    
    # Fetch a movie
    toy_story = fetcher.fetch_movie_by_tmdb_id(862)  # Toy Story
    
    if toy_story:
        print(f"Successfully fetched and saved: {toy_story['title']}")
    
    # Search for movies
    batman_movies = fetcher.search_movies("batman")
    print(f"Found {len(batman_movies)} Batman movies")