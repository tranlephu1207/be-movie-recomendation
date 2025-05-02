import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender
from utils.gcloud import init_credentials, get_storage_client
import io

class IMDBIntegration:
    """
    Integration with IMDB data stored in Google Cloud Storage.
    """
    
    def __init__(self, bucket_name: str = None, data_folder: str = 'data'):
        """
        Initialize the IMDB integration.
        
        Args:
            bucket_name: Name of the Google Cloud Storage bucket (can be set via env var GCS_BUCKET_NAME)
            data_folder: Folder within the bucket containing CSV files
        """
        # Get bucket name from env var if not provided
        self.bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("Bucket name must be provided or GCS_BUCKET_NAME environment variable must be set")
            
        self.data_folder = data_folder
        
        # CSV file paths in the bucket
        self.movies_file = f"{data_folder}/imdb_movies.csv"
        self.ratings_file = f"{data_folder}/imdb_ratings.csv"
        self.links_file = f"{data_folder}/imdb_links.csv"
        
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
        
        # Check if necessary files exist
        self._check_files()
        
        # Load data from GCS
        self.movies_df = self._read_csv_from_bucket(self.movies_file)
        self.ratings_df = self._read_csv_from_bucket(self.ratings_file)
        self.links_df = self._read_csv_from_bucket(self.links_file)
        
    def _read_csv_from_bucket(self, file_path: str) -> pd.DataFrame:
        """
        Read a CSV file from Google Cloud Storage bucket.
        
        Args:
            file_path: Path to CSV file in the bucket
            
        Returns:
            Pandas DataFrame with the CSV data
        """
        blob = self.bucket.blob(file_path)
        if not blob.exists():
            raise FileNotFoundError(f"File not found in bucket: {file_path}")
            
        # Download as bytes and convert to DataFrame
        content = blob.download_as_bytes()
        return pd.read_csv(io.BytesIO(content))
        
    def _write_csv_to_bucket(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Write a DataFrame as CSV to Google Cloud Storage bucket.
        
        Args:
            df: Pandas DataFrame to write
            file_path: Path in the bucket where to save the CSV
        """
        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Upload to bucket
        blob = self.bucket.blob(file_path)
        blob.upload_from_string(csv_content, content_type="text/csv")
        
    def _check_files(self) -> None:
        """Check if necessary files exist in the bucket."""
        required_files = [self.movies_file, self.ratings_file, self.links_file]
        missing_files = []
        
        for file_path in required_files:
            blob = self.bucket.blob(file_path)
            if not blob.exists():
                missing_files.append(file_path)
        
        if missing_files:
            raise FileNotFoundError(f"Required files not found in bucket: {', '.join(missing_files)}")
            
    def get_movie_by_imdb_id(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        """
        Get movie details by IMDB ID.
        
        Args:
            imdb_id: IMDB ID (e.g., 'tt0111161')
            
        Returns:
            Movie details as a dictionary
        """
        movie = self.movies_df[self.movies_df['imdb_id'] == imdb_id]
        if len(movie) == 0:
            return None
            
        return movie.iloc[0].to_dict()
        
    def get_movie_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get movie details by title.
        
        Args:
            title: Movie title
            
        Returns:
            Movie details as a dictionary
        """
        # Case-insensitive search
        movie = self.movies_df[self.movies_df['title'].str.lower() == title.lower()]
        if len(movie) == 0:
            return None
            
        return movie.iloc[0].to_dict()
        
    def get_movie_ratings(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        """
        Get movie ratings by IMDB ID.
        
        Args:
            imdb_id: IMDB ID (e.g., 'tt0111161')
            
        Returns:
            Movie ratings as a dictionary
        """
        ratings = self.ratings_df[self.ratings_df['imdb_id'] == imdb_id]
        if len(ratings) == 0:
            return None
            
        return ratings.iloc[0].to_dict()
        
    def get_movie_links(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        """
        Get movie links by IMDB ID.
        
        Args:
            imdb_id: IMDB ID (e.g., 'tt0111161')
            
        Returns:
            Movie links as a dictionary
        """
        links = self.links_df[self.links_df['imdb_id'] == imdb_id]
        if len(links) == 0:
            return None
            
        return links.iloc[0].to_dict()
        
    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for movies by title.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of movie details dictionaries
        """
        # Case-insensitive search
        results = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(query.lower())
        ].head(limit)
        
        return results.to_dict('records')
        
    def get_top_rated_movies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top rated movies.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of movie details dictionaries
        """
        # Merge movies and ratings
        merged = pd.merge(
            self.movies_df,
            self.ratings_df,
            on='imdb_id',
            how='inner'
        )
        
        # Sort by rating and get top movies
        top_movies = merged.sort_values('rating', ascending=False).head(limit)
        
        return top_movies.to_dict('records')
        
    def update_movie_data(self, movie_data: Dict[str, Any]) -> bool:
        """
        Update movie data in the bucket.
        
        Args:
            movie_data: Dictionary with movie data (must include imdb_id)
            
        Returns:
            True if successful, False otherwise
        """
        if 'imdb_id' not in movie_data:
            return False
            
        imdb_id = movie_data['imdb_id']
        
        # Update movies dataframe
        movie_idx = self.movies_df[self.movies_df['imdb_id'] == imdb_id].index
        if len(movie_idx) == 0:
            return False
            
        # Update relevant fields in movies dataframe
        for field in ['title', 'year', 'genres', 'director', 'actors']:
            if field in movie_data:
                self.movies_df.loc[movie_idx, field] = movie_data[field]
        
        # Write updated dataframe to bucket
        self._write_csv_to_bucket(self.movies_df, self.movies_file)
        
        return True

# Example usage
if __name__ == "__main__":
    # Initialize IMDB integration
    imdb_integration = IMDBIntegration()
    
    # Get movie details
    movie = imdb_integration.get_movie_by_imdb_id("tt0114709")  # Toy Story
    print(f"Movie: {movie['title']} ({movie['year']})")
    print(f"Genres: {', '.join(movie['genres'])}")
    print(f"Rating: {movie['imdb_rating']}")
    print(f"Plot: {movie['plot'][:100]}...")
    
    # Get recommendations based on a movie
    recommendations = imdb_integration.get_recommendations_by_imdb_id("tt0114709", top_n=5)
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} - {rec['vote_average']}")
    
    # Add a user rating
    imdb_integration.add_user_rating(user_id=1, imdb_id="tt0114709", rating=5.0)
    
    # Get personalized recommendations
    personalized_recs = imdb_integration.get_recommendations_by_imdb_id("tt0114709", user_id=1, top_n=5)
    print("\nPersonalized recommendations:")
    for i, rec in enumerate(personalized_recs, 1):
        print(f"{i}. {rec['title']} - {rec['vote_average']}")
    
    # Save user ratings
    imdb_integration.save_user_ratings()