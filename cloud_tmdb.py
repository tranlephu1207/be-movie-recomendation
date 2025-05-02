import os
import pandas as pd
import numpy as np
import io
from typing import Dict, List, Any, Optional, Union
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender
from utils.gcloud import init_credentials, get_storage_client
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

class CloudTMDbRecommendationSystem:
    """
    Movie recommendation system using TMDb data stored in Google Cloud Storage bucket.
    This is a drop-in replacement for the local file-based TMDbRecommendationSystem.
    """
    
    def __init__(self, bucket_name: str = None, data_folder: str = 'data'):
        """
        Initialize the recommendation system.
        
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
        self.movies_file = f"{data_folder}/movies.csv"
        self.genres_file = f"{data_folder}/genres.csv"
        self.cast_file = f"{data_folder}/cast.csv"
        self.crew_file = f"{data_folder}/crew.csv"
        self.mapping_file = f"{data_folder}/id_mapping.csv" 
        self.ratings_file = f"{data_folder}/ratings_small.csv"
        
        # Initialize Google Cloud credentials
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable must be set")
            
        try:
            init_credentials(credentials_path)
            self.storage_client = get_storage_client()
            self.bucket = self.storage_client.bucket(self.bucket_name)
            logger.info(f"Successfully connected to bucket: {self.bucket_name}")
        except Exception as e:
            error_msg = f"Failed to initialize Google Cloud Storage: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Check if necessary files exist in bucket
        self._check_files()
        
        # Load data directly from Cloud Storage
        logger.info("Loading data from Cloud Storage...")
        self.movies_df = self._read_csv_from_bucket(self.movies_file)
        self.genres_df = self._read_csv_from_bucket(self.genres_file)
        self.cast_df = self._read_csv_from_bucket(self.cast_file)
        self.crew_df = self._read_csv_from_bucket(self.crew_file)
        
        # Try to load mapping file, but continue if it doesn't exist
        try:
            self.mapping_df = self._read_csv_from_bucket(self.mapping_file)
        except Exception as e:
            logger.warning(f"Mapping file not found: {e}")
            self.mapping_df = None
            
        # Try to load existing ratings
        try:
            self.user_ratings_df = self._read_csv_from_bucket(self.ratings_file)
        except Exception as e:
            logger.warning(f"Ratings file not found, creating empty ratings dataframe: {e}")
            # Create empty ratings dataframe if file doesn't exist
            self.user_ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
        
        # Prepare data for recommendation system
        self._prepare_data()
        
        # Initialize recommenders
        logger.info("Initializing recommender models...")
        self.content_recommender = self._initialize_content_recommender()
        self.collaborative_recommender = self._initialize_collaborative_recommender()
        self.hybrid_recommender = HybridRecommender(
            content_recommender=self.content_recommender,
            collab_recommender=self.collaborative_recommender
        )
        logger.info("Recommendation system initialization complete")
    
    def _read_csv_from_bucket(self, file_path: str) -> pd.DataFrame:
        """
        Read a CSV file directly from Google Cloud Storage bucket.
        
        Args:
            file_path: Path to CSV file in the bucket
            
        Returns:
            Pandas DataFrame with the CSV data
            
        Raises:
            Exception: If file doesn't exist or can't be read
        """
        blob = self.bucket.blob(file_path)
        
        if not blob.exists():
            raise FileNotFoundError(f"File not found in bucket: {file_path}")
        
        # Download as bytes and convert to DataFrame
        content = blob.download_as_bytes()
        return pd.read_csv(io.BytesIO(content))
    
    def _write_csv_to_bucket(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Write a DataFrame as CSV directly to Google Cloud Storage bucket.
        
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
        logger.info(f"Successfully wrote to {file_path} in bucket {self.bucket_name}")
    
    def _check_files(self) -> None:
        """Check if necessary files exist in the bucket."""
        required_files = [self.movies_file, self.genres_file, self.cast_file, self.crew_file]
        missing_files = []
        
        for file_path in required_files:
            blob = self.bucket.blob(file_path)
            if not blob.exists():
                missing_files.append(file_path)
        
        if missing_files:
            raise FileNotFoundError(f"Required files not found in bucket: {', '.join(missing_files)}")
    
    def _prepare_data(self) -> None:
        """Prepare data for recommendation system."""
        # Use movieId from mapping if available, otherwise use tmdb_id
        if self.mapping_df is not None:
            # Merge with movies_df on tmdb_id
            self.movies_df = pd.merge(
                self.movies_df,
                self.mapping_df,
                left_on='tmdb_id',
                right_on='tmdbId',
                how='left'
            )
            
            # Use movieId as id if available, otherwise use tmdb_id
            self.movies_df['id'] = self.movies_df['movieId'].fillna(self.movies_df['tmdb_id']).astype(int)
        else:
            # Use tmdb_id as id
            self.movies_df['id'] = self.movies_df['tmdb_id']
        
        # Create metadata dataframe for content-based filtering
        self.metadata_df = self.movies_df[['id', 'tmdb_id', 'imdb_id', 'title', 'vote_average', 'overview', 'release_date']].copy()
        
        # Extract year from release_date
        self.metadata_df['year'] = self.metadata_df['release_date'].str[:4] if 'release_date' in self.metadata_df.columns else ''
        
        # Add genres as a list for each movie
        genres_grouped = self.genres_df.groupby('tmdb_id')['genre_name'].apply(list).reset_index()
        self.metadata_df = pd.merge(
            self.metadata_df,
            genres_grouped,
            left_on='tmdb_id',
            right_on='tmdb_id',
            how='left'
        )
        self.metadata_df = self.metadata_df.rename(columns={'genre_name': 'genres'})
        
        # Add cast as a list for each movie
        cast_grouped = self.cast_df.groupby('tmdb_id')['name'].apply(list).reset_index()
        self.metadata_df = pd.merge(
            self.metadata_df,
            cast_grouped,
            left_on='tmdb_id',
            right_on='tmdb_id',
            how='left'
        )
        self.metadata_df = self.metadata_df.rename(columns={'name': 'cast'})
        
        # Add director from crew
        directors = self.crew_df[self.crew_df['job'] == 'Director']
        directors_grouped = directors.groupby('tmdb_id')['name'].first().reset_index()
        self.metadata_df = pd.merge(
            self.metadata_df,
            directors_grouped,
            left_on='tmdb_id',
            right_on='tmdb_id',
            how='left'
        )
        self.metadata_df = self.metadata_df.rename(columns={'name': 'director'})
        
        # Fill NaN values
        self.metadata_df['genres'] = self.metadata_df['genres'].apply(lambda x: [] if isinstance(x, float) and np.isnan(x) else x)
        self.metadata_df['cast'] = self.metadata_df['cast'].apply(lambda x: [] if isinstance(x, float) and np.isnan(x) else x)
        self.metadata_df['overview'] = self.metadata_df['overview'].fillna('')
        self.metadata_df['director'] = self.metadata_df['director'].fillna('')
    
    def _initialize_content_recommender(self) -> ContentBasedRecommender:
        """Initialize content-based recommender with processed data."""
        # Create a custom initializer that doesn't need file paths
        class TMDbContentRecommender(ContentBasedRecommender):
            def __init__(self, metadata_df):
                self.metadata_df = metadata_df
                self.nlp = None  # Initialize when needed
                self.vectorizer = None
                self.tfidf_matrix = None
                
                # Process data
                self.process_data()
                self.create_profiles()
                self.build_model()
                
            def load_data(self, *args, **kwargs):
                # Override to skip loading from files
                pass
        
        # Initialize and return the recommender
        return TMDbContentRecommender(self.metadata_df)
    
    def _initialize_collaborative_recommender(self) -> CollaborativeRecommender:
        """Initialize collaborative filtering recommender."""
        # Create a custom initializer that doesn't need file paths
        recommender = CollaborativeRecommender()
        recommender.ratings_df = self.user_ratings_df
        
        # Only build model if we have ratings
        if len(self.user_ratings_df) > 0:
            recommender.build_model()
            
        return recommender
    
    def get_movie_by_tmdb_id(self, tmdb_id: int) -> Optional[Dict[str, Any]]:
        """
        Get movie details by TMDb ID.
        
        Args:
            tmdb_id: TMDb ID
            
        Returns:
            Movie details dictionary or None if not found
        """
        # Convert to int if it's a string
        if isinstance(tmdb_id, str):
            try:
                tmdb_id = int(tmdb_id)
            except ValueError:
                return None
                
        # Find movie in dataframe
        movie = self.movies_df[self.movies_df['tmdb_id'] == tmdb_id]
        if len(movie) == 0:
            return None
            
        movie_data = movie.iloc[0].to_dict()
        
        # Add genres
        genres = self.genres_df[self.genres_df['tmdb_id'] == tmdb_id]['genre_name'].tolist()
        movie_data['genres'] = genres
        
        # Add cast
        cast = self.cast_df[self.cast_df['tmdb_id'] == tmdb_id]['name'].tolist()
        movie_data['cast'] = cast
        
        # Add director
        director = self.crew_df[
            (self.crew_df['tmdb_id'] == tmdb_id) & 
            (self.crew_df['job'] == 'Director')
        ]['name'].values
        
        movie_data['director'] = director[0] if len(director) > 0 else ''
        
        return movie_data
    
    def get_movie_by_id(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """
        Get movie details by internal movie ID.
        
        Args:
            movie_id: Movie ID (from your links.csv)
            
        Returns:
            Movie details dictionary or None if not found
        """
        # If we have a mapping, use it to find the tmdb_id
        if self.mapping_df is not None:
            mapping = self.mapping_df[self.mapping_df['movieId'] == movie_id]
            if len(mapping) > 0:
                tmdb_id = mapping.iloc[0]['tmdbId']
                return self.get_movie_by_tmdb_id(tmdb_id)
        
        # Otherwise, try to find it directly in the movies dataframe
        movie = self.movies_df[self.movies_df['id'] == movie_id]
        if len(movie) == 0:
            return None
            
        tmdb_id = movie.iloc[0]['tmdb_id']
        return self.get_movie_by_tmdb_id(tmdb_id)
    
    def get_recommendations(self, query: str, user_id: Optional[int] = None, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get movie recommendations based on a text query.
        
        Args:
            query: Text query for content-based filtering
            user_id: Optional user ID for collaborative filtering
            top_n: Number of recommendations to return
            
        Returns:
            List of movie recommendations
        """
        if user_id is not None and len(self.user_ratings_df) > 0:
            # Use hybrid recommendations
            recommendations = self.hybrid_recommender.get_recommendations(
                user_id=user_id,
                query=query,
                top_n=top_n
            )
        else:
            # Use content-based only
            recommendations = self.content_recommender.get_recommendations(
                query=query,
                top_n=top_n
            )

        # Add detailed movie data for each recommendation
        movie_data = []
        for movie_id in recommendations['id']:
            movie_info = self.get_movie_by_id(movie_id)
            movie_data.append(movie_info if movie_info is not None else {})
        
        recommendations['movie_data'] = movie_data
            
        # Convert to list of dictionaries
        return recommendations.to_dict(orient='records')
    
    def get_recommendations_by_tmdb_id(self, tmdb_id: int, user_id: Optional[int] = None, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get movie recommendations based on a similar movie.
        
        Args:
            tmdb_id: TMDb ID of the reference movie
            user_id: Optional user ID for collaborative filtering
            top_n: Number of recommendations to return
            
        Returns:
            List of movie recommendations
        """
        # Get movie details
        movie = self.get_movie_by_tmdb_id(tmdb_id)
        if movie is None:
            return []
            
        # Create query from movie details
        query = f"{movie['title']} {' '.join(movie['genres']) if 'genres' in movie else ''} {movie['overview'] if 'overview' in movie else ''}"
        
        # Get recommendations
        return self.get_recommendations(query, user_id, top_n)
    
    def get_recommendations_by_id(self, movie_id: int, user_id: Optional[int] = None, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get movie recommendations based on a similar movie using internal movie ID.
        
        Args:
            movie_id: Movie ID (from your links.csv)
            user_id: Optional user ID for collaborative filtering
            top_n: Number of recommendations to return
            
        Returns:
            List of movie recommendations
        """
        # If we have a mapping, use it to find the tmdb_id
        if self.mapping_df is not None:
            mapping = self.mapping_df[self.mapping_df['movieId'] == movie_id]
            if len(mapping) > 0:
                tmdb_id = mapping.iloc[0]['tmdbId']
                return self.get_recommendations_by_tmdb_id(tmdb_id, user_id, top_n)
        
        # Otherwise, try to find it directly in the movies dataframe
        movie = self.movies_df[self.movies_df['id'] == movie_id]
        if len(movie) == 0:
            return []
            
        tmdb_id = movie.iloc[0]['tmdb_id']
        return self.get_recommendations_by_tmdb_id(tmdb_id, user_id, top_n)
        
    def add_user_rating(self, user_id: int, movie_id: int, rating: float) -> bool:
        """
        Add a user rating using internal movie ID.
        
        Args:
            user_id: User ID
            movie_id: Movie ID (from your links.csv)
            rating: Rating (0.5-5.0)
            
        Returns:
            True if successful, False otherwise
        """
        # Validate rating
        if rating < 0.5 or rating > 5.0:
            return False
            
        # Add rating
        new_rating = pd.DataFrame({
            'userId': [user_id],
            'movieId': [movie_id],
            'rating': [rating],
            'timestamp': [int(time.time())]
        })
        
        # Remove existing rating if present
        self.user_ratings_df = self.user_ratings_df[
            ~((self.user_ratings_df['userId'] == user_id) & 
              (self.user_ratings_df['movieId'] == movie_id))
        ]
        
        # Add new rating
        self.user_ratings_df = pd.concat([self.user_ratings_df, new_rating], ignore_index=True)
        
        # Write updated ratings to bucket immediately
        self._write_csv_to_bucket(self.user_ratings_df, self.ratings_file)
        
        # Rebuild collaborative model
        self.collaborative_recommender.ratings_df = self.user_ratings_df
        self.collaborative_recommender.build_model()
        
        return True
    
    def add_user_rating_by_tmdb_id(self, user_id: int, tmdb_id: int, rating: float) -> bool:
        """
        Add a user rating using TMDb ID.
        
        Args:
            user_id: User ID
            tmdb_id: TMDb ID
            rating: Rating (0.5-5.0)
            
        Returns:
            True if successful, False otherwise
        """
        # Find internal movie ID
        movie = self.movies_df[self.movies_df['tmdb_id'] == tmdb_id]
        if len(movie) == 0:
            return False
            
        # Get the internal movie ID
        movie_id = movie.iloc[0]['id']
        
        # Add rating using internal movie ID
        return self.add_user_rating(user_id, movie_id, rating)
    
    def save_user_ratings(self, file_path: Optional[str] = None) -> None:
        """
        Save user ratings to CSV file in the cloud bucket.
        
        Args:
            file_path: Path to save ratings (default: ratings_file class attribute)
        """
        if file_path is None:
            file_path = self.ratings_file
            
        # Add timestamp column if it doesn't exist
        if 'timestamp' not in self.user_ratings_df.columns:
            self.user_ratings_df['timestamp'] = int(time.time())
            
        # Ensure columns are in the correct order
        self.user_ratings_df = self.user_ratings_df[['userId', 'movieId', 'rating', 'timestamp']]
        
        # Write to cloud
        self._write_csv_to_bucket(self.user_ratings_df, file_path)
        logger.info(f"User ratings saved to {file_path}")
    
    def load_user_ratings(self, file_path: Optional[str] = None) -> None:
        """
        Load user ratings from CSV file in the cloud bucket.
        
        Args:
            file_path: Path to load ratings from (default: ratings_file class attribute)
        """
        if file_path is None:
            file_path = self.ratings_file
            
        try:
            self.user_ratings_df = self._read_csv_from_bucket(file_path)
            
            # Ensure required columns exist
            required_columns = ['userId', 'movieId', 'rating', 'timestamp']
            if not all(col in self.user_ratings_df.columns for col in required_columns):
                raise ValueError(f"Ratings file must contain columns: {required_columns}")
            
            # Rebuild collaborative model
            self.collaborative_recommender.ratings_df = self.user_ratings_df
            self.collaborative_recommender.build_model()
            logger.info(f"User ratings loaded from {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load ratings from {file_path}: {e}")
            # Create empty ratings dataframe
            self.user_ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

    def get_user_rated_movies(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all movies rated by a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of dictionaries containing movie details and user ratings
        """
        # Get all ratings for this user
        user_ratings = self.user_ratings_df[self.user_ratings_df['userId'] == user_id]
        if len(user_ratings) == 0:
            return []
        
        # Get movie details for each rated movie
        rated_movies = []
        for _, row in user_ratings.iterrows():
            movieId = int(row['movieId'])
            movie_info = self.get_movie_by_id(movieId)
            if movie_info is not None:
                # Add user's rating to movie info
                movie_info['user_rating'] = row['rating']
                movie_info['rating_timestamp'] = row['timestamp']
                rated_movies.append(movie_info)
        
        # Sort by rating timestamp (most recent first)
        rated_movies.sort(key=lambda x: x['rating_timestamp'], reverse=True)
        
        return rated_movies

    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for movies by title, genre, director, or actor.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching movies with their details
        """
        query = query.lower()
        
        # Search in metadata_df
        matches = []
        
        for _, movie in self.metadata_df.iterrows():
            score = 0
            
            # Check title (highest weight)
            if query in movie['title'].lower():
                score += 10
                
            # Check genres
            if isinstance(movie['genres'], list):
                for genre in movie['genres']:
                    if query in genre.lower():
                        score += 5
                        
            # Check director
            if query in movie['director'].lower():
                score += 5
                
            # Check cast
            if isinstance(movie['cast'], list):
                for actor in movie['cast']:
                    if query in actor.lower():
                        score += 3
                        
            # Check overview
            if query in movie['overview'].lower():
                score += 2
                
            if score > 0:
                movie_data = self.get_movie_by_id(movie['id'])
                if movie_data:
                    movie_data['search_score'] = score
                    matches.append(movie_data)
        
        # Sort by search score and limit results
        matches.sort(key=lambda x: x['search_score'], reverse=True)
        return matches[:limit]

    def update_movie_data(self, updated_movie: Dict[str, Any]) -> bool:
        """
        Update movie data in the bucket.
        
        Args:
            updated_movie: Dictionary with movie data (must include tmdb_id)
            
        Returns:
            True if successful, False otherwise
        """
        if 'tmdb_id' not in updated_movie:
            return False
            
        tmdb_id = updated_movie['tmdb_id']
        
        # Update movies dataframe
        movie_idx = self.movies_df[self.movies_df['tmdb_id'] == tmdb_id].index
        if len(movie_idx) == 0:
            return False
            
        # Update relevant fields in movies dataframe
        for field in ['title', 'overview', 'vote_average', 'release_date']:
            if field in updated_movie:
                self.movies_df.loc[movie_idx, field] = updated_movie[field]
        
        # Update genres if provided
        if 'genres' in updated_movie:
            # Remove existing genres
            self.genres_df = self.genres_df[self.genres_df['tmdb_id'] != tmdb_id]
            
            # Add new genres
            new_genres = []
            for genre in updated_movie['genres']:
                new_genres.append({'tmdb_id': tmdb_id, 'genre_name': genre})
                
            if new_genres:
                self.genres_df = pd.concat([self.genres_df, pd.DataFrame(new_genres)], ignore_index=True)
        
        # Update cast if provided
        if 'cast' in updated_movie:
            # Remove existing cast
            self.cast_df = self.cast_df[self.cast_df['tmdb_id'] != tmdb_id]
            
            # Add new cast
            new_cast = []
            for actor in updated_movie['cast']:
                new_cast.append({'tmdb_id': tmdb_id, 'name': actor})
                
            if new_cast:
                self.cast_df = pd.concat([self.cast_df, pd.DataFrame(new_cast)], ignore_index=True)
        
        # Update director if provided
        if 'director' in updated_movie:
            # Remove existing director
            self.crew_df = self.crew_df[~((self.crew_df['tmdb_id'] == tmdb_id) & (self.crew_df['job'] == 'Director'))]
            
            # Add new director
            new_director = {'tmdb_id': tmdb_id, 'name': updated_movie['director'], 'job': 'Director'}
            self.crew_df = pd.concat([self.crew_df, pd.DataFrame([new_director])], ignore_index=True)
        
        # Write updated dataframes to bucket
        self._write_csv_to_bucket(self.movies_df, self.movies_file)
        self._write_csv_to_bucket(self.genres_df, self.genres_file)
        self._write_csv_to_bucket(self.cast_df, self.cast_file)
        self._write_csv_to_bucket(self.crew_df, self.crew_file)
        
        # Update metadata dataframe
        self._prepare_data()
        
        # Rebuild content recommender
        self.content_recommender = self._initialize_content_recommender()
        self.hybrid_recommender.content_recommender = self.content_recommender
        
        return True
