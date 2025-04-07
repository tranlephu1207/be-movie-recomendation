import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender

class IMDbRecommendationSystem:
    """
    Movie recommendation system using locally stored IMDb data (from CSV files).
    """
    
    def __init__(self, data_dir: str = './data'):
        """
        Initialize the recommendation system.
        
        Args:
            data_dir: Directory with CSV files containing IMDb data
        """
        self.data_dir = data_dir
        self.movies_file = os.path.join(data_dir, 'movies.csv')
        self.genres_file = os.path.join(data_dir, 'genres.csv')
        self.ratings_file = os.path.join(data_dir, 'ratings.csv')
        self.cast_file = os.path.join(data_dir, 'cast.csv')
        
        # Check if necessary files exist
        self._check_files()
        
        # Load data
        self.movies_df = pd.read_csv(self.movies_file)
        self.genres_df = pd.read_csv(self.genres_file)
        self.cast_df = pd.read_csv(self.cast_file)
        
        # Prepare data for recommendation system
        self._prepare_data()
        
        # Initialize recommenders
        self.content_recommender = self._initialize_content_recommender()
        self.collaborative_recommender = self._initialize_collaborative_recommender()
        self.hybrid_recommender = HybridRecommender(
            content_recommender=self.content_recommender,
            collab_recommender=self.collaborative_recommender
        )
    
    def _check_files(self) -> None:
        """Check if necessary files exist."""
        files = [self.movies_file, self.genres_file, self.cast_file]
        for file in files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")
    
    def _prepare_data(self) -> None:
        """Prepare data for recommendation system."""
        # Preprocess movies dataframe
        self.movies_df['id'] = self.movies_df.index + 1  # Create unique ID
        
        # Create metadata dataframe for content-based filtering
        self.metadata_df = self.movies_df[['id', 'imdb_id', 'title', 'imdb_rating', 'plot', 'director']].copy()
        self.metadata_df = self.metadata_df.rename(columns={
            'imdb_rating': 'vote_average',
            'plot': 'overview'
        })
        
        # Add genres as a list for each movie
        genres_grouped = self.genres_df.groupby('imdb_id')['genre'].apply(list).reset_index()
        self.metadata_df = pd.merge(
            self.metadata_df,
            genres_grouped,
            on='imdb_id',
            how='left'
        )
        
        # Add cast as a list for each movie
        cast_grouped = self.cast_df.groupby('imdb_id')['actor'].apply(list).reset_index()
        self.metadata_df = pd.merge(
            self.metadata_df,
            cast_grouped,
            on='imdb_id',
            how='left',
            suffixes=('', '_cast')
        )
        self.metadata_df = self.metadata_df.rename(columns={'actor': 'cast'})
        
        # Fill NaN values
        self.metadata_df['genre'] = self.metadata_df['genre'].apply(lambda x: [] if isinstance(x, float) and np.isnan(x) else x)
        self.metadata_df['cast'] = self.metadata_df['cast'].apply(lambda x: [] if isinstance(x, float) and np.isnan(x) else x)
        self.metadata_df['overview'] = self.metadata_df['overview'].fillna('')
        
        # Create user ratings dataframe for collaborative filtering
        self.user_ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
    
    def _initialize_content_recommender(self) -> ContentBasedRecommender:
        """Initialize content-based recommender with processed data."""
        # Create a custom initializer that doesn't need file paths
        class IMDbContentRecommender(ContentBasedRecommender):
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
        return IMDbContentRecommender(self.metadata_df)
    
    def _initialize_collaborative_recommender(self) -> CollaborativeRecommender:
        """Initialize collaborative filtering recommender."""
        # Create a custom initializer that doesn't need file paths
        recommender = CollaborativeRecommender()
        recommender.ratings_df = self.user_ratings_df
        
        # Only build model if we have ratings
        if len(self.user_ratings_df) > 0:
            recommender.build_model()
            
        return recommender
    
    def get_movie_by_imdb_id(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        """
        Get movie details by IMDb ID.
        
        Args:
            imdb_id: IMDb ID (with or without 'tt' prefix)
            
        Returns:
            Movie details dictionary or None if not found
        """
        # Add 'tt' prefix if not already present
        if imdb_id and not imdb_id.startswith('tt'):
            imdb_id = f"tt{imdb_id}"
            
        # Find movie in dataframe
        movie = self.movies_df[self.movies_df['imdb_id'] == imdb_id]
        if len(movie) == 0:
            return None
            
        movie_data = movie.iloc[0].to_dict()
        
        # Add genres
        genres = self.genres_df[self.genres_df['imdb_id'] == imdb_id]['genre'].tolist()
        movie_data['genres'] = genres
        
        # Add cast
        cast = self.cast_df[self.cast_df['imdb_id'] == imdb_id]['actor'].tolist()
        movie_data['cast'] = cast
        
        return movie_data
    
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
            
        # Convert to list of dictionaries
        return recommendations.to_dict(orient='records')
    
    def get_recommendations_by_imdb_id(self, imdb_id: str, user_id: Optional[int] = None, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get movie recommendations based on a similar movie.
        
        Args:
            imdb_id: IMDb ID of the reference movie
            user_id: Optional user ID for collaborative filtering
            top_n: Number of recommendations to return
            
        Returns:
            List of movie recommendations
        """
        # Get movie details
        movie = self.get_movie_by_imdb_id(imdb_id)
        if movie is None:
            return []
            
        # Create query from movie details
        query = f"{movie['title']} {' '.join(movie['genres']) if 'genres' in movie else ''} {movie['plot'] if 'plot' in movie else ''}"
        
        # Get recommendations
        return self.get_recommendations(query, user_id, top_n)
    
    def add_user_rating(self, user_id: int, imdb_id: str, rating: float) -> bool:
        """
        Add a user rating.
        
        Args:
            user_id: User ID
            imdb_id: IMDb ID of the movie
            rating: Rating (0.5-5.0)
            
        Returns:
            True if successful, False otherwise
        """
        # Find movie ID
        movie = self.movies_df[self.movies_df['imdb_id'] == imdb_id]
        if len(movie) == 0:
            return False
            
        movie_id = movie.iloc[0]['id']
        
        # Add rating
        new_rating = pd.DataFrame({
            'userId': [user_id],
            'movieId': [movie_id],
            'rating': [rating]
        })
        
        # Remove existing rating if present
        self.user_ratings_df = self.user_ratings_df[
            ~((self.user_ratings_df['userId'] == user_id) & 
              (self.user_ratings_df['movieId'] == movie_id))
        ]
        
        # Add new rating
        self.user_ratings_df = pd.concat([self.user_ratings_df, new_rating], ignore_index=True)
        
        # Rebuild collaborative model
        self.collaborative_recommender.ratings_df = self.user_ratings_df
        self.collaborative_recommender.build_model()
        
        return True
    
    def save_user_ratings(self, file_path: Optional[str] = None) -> None:
        """
        Save user ratings to CSV file.
        
        Args:
            file_path: Path to save ratings (default: data_dir/user_ratings.csv)
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'user_ratings.csv')
            
        self.user_ratings_df.to_csv(file_path, index=False)
    
    def load_user_ratings(self, file_path: Optional[str] = None) -> None:
        """
        Load user ratings from CSV file.
        
        Args:
            file_path: Path to load ratings from (default: data_dir/user_ratings.csv)
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'user_ratings.csv')
            
        if os.path.exists(file_path):
            self.user_ratings_df = pd.read_csv(file_path)
            
            # Rebuild collaborative model
            self.collaborative_recommender.ratings_df = self.user_ratings_df
            self.collaborative_recommender.build_model()

# Example usage
if __name__ == "__main__":
    # Initialize recommendation system
    rec_sys = IMDbRecommendationSystem()
    
    # Get movie details
    movie = rec_sys.get_movie_by_imdb_id("tt0114709")  # Toy Story
    print(f"Movie: {movie['title']} ({movie['year']})")
    print(f"Genres: {', '.join(movie['genres'])}")
    print(f"Rating: {movie['imdb_rating']}")
    print(f"Plot: {movie['plot'][:100]}...")
    
    # Get recommendations based on a movie
    recommendations = rec_sys.get_recommendations_by_imdb_id("tt0114709", top_n=5)
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} - {rec['vote_average']}")
    
    # Add a user rating
    rec_sys.add_user_rating(user_id=1, imdb_id="tt0114709", rating=5.0)
    
    # Get personalized recommendations
    personalized_recs = rec_sys.get_recommendations_by_imdb_id("tt0114709", user_id=1, top_n=5)
    print("\nPersonalized recommendations:")
    for i, rec in enumerate(personalized_recs, 1):
        print(f"{i}. {rec['title']} - {rec['vote_average']}")
    
    # Save user ratings
    rec_sys.save_user_ratings()