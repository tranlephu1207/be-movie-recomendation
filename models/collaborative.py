import os
import pandas as pd
import numpy as np
import joblib
from surprise import Dataset, Reader, SVD
from typing import List, Tuple, Optional, Dict, Any

class CollaborativeRecommender:
    """
    Collaborative filtering recommendation system using SVD.
    """
    
    def __init__(self, ratings_path: Optional[str] = None):
        """
        Initialize the collaborative recommender.
        
        Args:
            ratings_path: Path to ratings CSV file (optional)
        """
        self.model = SVD()
        self.ratings_df = None
        self.reader = Reader(rating_scale=(0.5, 5.0))
        
        if ratings_path:
            self.load_data(ratings_path)
            self.build_model()
    
    def load_data(self, ratings_path: str) -> None:
        """
        Load ratings data from CSV.
        
        Args:
            ratings_path: Path to ratings CSV file
        """
        self.ratings_df = pd.read_csv(ratings_path)
        if 'timestamp' in self.ratings_df.columns:
            self.ratings_df = self.ratings_df.drop(columns=['timestamp'])
    
    def build_model(self) -> None:
        """
        Build and train the SVD model.
        """
        if self.ratings_df is None or len(self.ratings_df) == 0:
            raise ValueError("No ratings data available. Please load data first.")
            
        data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']], 
            self.reader
        )
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
    
    def add_user_ratings(self, user_id: int, movie_ratings: List[Tuple[int, float]]) -> None:
        """
        Add new ratings for a user and rebuild the model.
        
        Args:
            user_id: User identifier
            movie_ratings: List of tuples (movie_id, rating)
        """
        new_ratings = pd.DataFrame([
            {'userId': user_id, 'movieId': movie_id, 'rating': rating}
            for movie_id, rating in movie_ratings
        ])
        
        if self.ratings_df is None:
            self.ratings_df = new_ratings
        else:
            # Remove existing ratings for this user-movie pair before adding new ones
            user_movie_pairs = [(user_id, movie_id) for movie_id, _ in movie_ratings]
            mask = ~self.ratings_df.apply(
                lambda row: (row['userId'], row['movieId']) in user_movie_pairs, 
                axis=1
            )
            self.ratings_df = pd.concat([self.ratings_df[mask], new_ratings], ignore_index=True)
            
        # Rebuild model with new ratings
        self.build_model()
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a specific movie and user.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
            
        Returns:
            Predicted rating
        """
        return self.model.predict(uid=user_id, iid=movie_id).est
    
    def predict_ratings_for_movies(self, user_id: int, movie_ids: List[int]) -> List[Tuple[int, float]]:
        """
        Predict ratings for a list of movies for a specific user.
        
        Args:
            user_id: User identifier
            movie_ids: List of movie identifiers
            
        Returns:
            List of tuples (movie_id, predicted_rating) sorted by rating
        """
        predictions = []
        for movie_id in movie_ids:
            try:
                pred = self.model.predict(uid=user_id, iid=movie_id)
                predictions.append((movie_id, pred.est))
            except Exception as e:
                # Handle case when the model can't make a prediction
                # (e.g., new user or movie not in training data)
                predictions.append((movie_id, 0.0))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions
    
    def get_top_rated_movies_for_user(self, user_id: int, n: int = 10) -> List[int]:
        """
        Get the top rated movies for a specific user.
        
        Args:
            user_id: User identifier
            n: Number of movies to return
            
        Returns:
            List of movie IDs
        """
        if self.ratings_df is None:
            return []
            
        # Get movies this user has already rated
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        if len(user_ratings) == 0:
            return []
            
        # Sort by rating and return top n movie IDs
        top_movies = user_ratings.sort_values('rating', ascending=False)['movieId'].tolist()[:n]
        return top_movies
    
    def save_model(self, path: str) -> None:
        """
        Save the model and data to disk.
        
        Args:
            path: Directory path to save model
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, 'svd_model.pkl'))
        if self.ratings_df is not None:
            self.ratings_df.to_pickle(os.path.join(path, 'ratings_df.pkl'))
    
    @classmethod
    def load_model(cls, path: str) -> 'CollaborativeRecommender':
        """
        Load the model and data from disk.
        
        Args:
            path: Directory path to load model from
            
        Returns:
            Loaded recommender instance
        """
        instance = cls.__new__(cls)
        instance.model = joblib.load(os.path.join(path, 'svd_model.pkl'))
        instance.reader = Reader(rating_scale=(0.5, 5.0))
        
        ratings_path = os.path.join(path, 'ratings_df.pkl')
        if os.path.exists(ratings_path):
            instance.ratings_df = pd.read_pickle(ratings_path)
        else:
            instance.ratings_df = None
            
        return instance