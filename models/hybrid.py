import pandas as pd
from typing import List, Dict, Any, Optional

from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender

class HybridRecommender:
    """
    Hybrid recommendation system combining content-based and collaborative filtering.
    """
    
    def __init__(self, content_recommender: ContentBasedRecommender, collab_recommender: CollaborativeRecommender):
        """
        Initialize the hybrid recommender.
        
        Args:
            content_recommender: Content-based recommender instance
            collab_recommender: Collaborative filtering recommender instance
        """
        self.content_recommender = content_recommender
        self.collab_recommender = collab_recommender
    
    def get_recommendations(self, user_id: int, query: str, top_n: int = 5) -> pd.DataFrame:
        """
        Get hybrid recommendations by combining content-based and collaborative filtering.
        
        Args:
            user_id: User identifier
            query: Text query for content-based filtering
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with top recommendations
        """
        # Get content-based recommendations
        cb_recommendations = self.content_recommender.get_recommendations(query, top_n=20)
        cb_movie_ids = cb_recommendations['id'].tolist()
        
        # Predict ratings for content-based recommendations
        predicted_ratings = self.collab_recommender.predict_ratings_for_movies(user_id, cb_movie_ids)
        
        # Get top N movie IDs
        top_movie_ids = [movie_id for movie_id, _ in predicted_ratings[:top_n]]
        
        # Return final recommendations
        metadata_df = self.content_recommender.metadata_df
        final_recommendations = metadata_df[metadata_df['id'].isin(top_movie_ids)][['id', 'title', 'vote_average', 'genres']]
        
        return final_recommendations
    
    def get_content_recommendations(self, query: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get pure content-based recommendations.
        
        Args:
            query: Text query
            top_n: Number of recommendations
            
        Returns:
            DataFrame with recommendations
        """
        return self.content_recommender.get_recommendations(query, top_n)
    
    def get_collaborative_recommendations(self, user_id: int, movie_ids: List[int]) -> List[int]:
        """
        Get collaborative filtering recommendations.
        
        Args:
            user_id: User identifier
            movie_ids: List of movie IDs to predict ratings for
            
        Returns:
            List of movie IDs sorted by predicted rating
        """
        predicted_ratings = self.collab_recommender.predict_ratings_for_movies(user_id, movie_ids)
        return [movie_id for movie_id, _ in predicted_ratings]
    
    def add_user_ratings(self, user_id: int, movie_ratings: List[tuple]) -> None:
        """
        Add new user ratings to the collaborative filtering model.
        
        Args:
            user_id: User identifier
            movie_ratings: List of tuples (movie_id, rating)
        """
        self.collab_recommender.add_user_ratings(user_id, movie_ratings)
        
    def get_recommendations_by_imdb(self, imdb_id: str, user_id: Optional[int] = None, top_n: int = 5) -> pd.DataFrame:
        """
        Get recommendations based on a movie's IMDb ID.
        
        Args:
            imdb_id: IMDb ID of the movie
            user_id: Optional user ID for collaborative filtering
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations
        """
        # Find the movie by IMDb ID
        movie = self.content_recommender.get_movie_by_imdb_id(imdb_id)
        if movie is None:
            raise ValueError(f"Movie with IMDb ID {imdb_id} not found")
        
        # Use the movie's title, genres, and overview as a query
        movie_profile = f"{movie['title']} {' '.join(movie['genres'])} {movie['overview']}"
        
        # If user ID is provided, use hybrid recommendations
        if user_id is not None:
            return self.get_recommendations(user_id, movie_profile, top_n)
        
        # Otherwise, use content-based only
        recommendations = self.content_recommender.get_recommendations(movie_profile, top_n + 1)
        
        # Remove the input movie from recommendations
        recommendations = recommendations[recommendations['id'] != movie['id']]
        
        return recommendations.head(top_n)