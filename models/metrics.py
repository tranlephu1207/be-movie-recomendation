import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import csv
import io
from utils.gcloud import get_storage_client, init_credentials
from google.cloud import storage

class RecommendationMetrics:
    """
    Class to track and analyze recommendation system performance metrics using Google Cloud Storage.
    """
    
    def __init__(self, bucket_name: str = None, metrics_file: str = "metrics/interactions.csv"):
        """
        Initialize the metrics tracker.
        
        Args:
            bucket_name: Name of the Google Cloud Storage bucket
            metrics_file: Path within bucket to store metrics data
        """
        # Initialize GCS client
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable must be set")
            
        try:
            init_credentials(credentials_path)
            self.storage_client = get_storage_client()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Cloud Storage: {e}")
      
        # Get bucket name from env var if not provided
        self.bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("Bucket name must be provided or GCS_BUCKET_NAME environment variable must be set")
            
        self.metrics_file = metrics_file
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
        # Create metrics file if it doesn't exist
        if not self._file_exists():
            self._create_metrics_file()
    
    def _file_exists(self) -> bool:
        """
        Check if metrics file exists in bucket.
        
        Returns:
            True if file exists, False otherwise
        """
        blob = self.bucket.blob(self.metrics_file)
        return blob.exists()
    
    def _create_metrics_file(self) -> None:
        """
        Create a new metrics CSV file with headers.
        """
        headers = [
            "user_id", "movie_id", "recommendation_id", "interaction_type", 
            "experiment_id", "timestamp"
        ]
        
        csv_data = io.StringIO()
        writer = csv.writer(csv_data)
        writer.writerow(headers)
        
        blob = self.bucket.blob(self.metrics_file)
        blob.upload_from_string(csv_data.getvalue(), content_type="text/csv")
    
    def _read_interactions(self) -> pd.DataFrame:
        """
        Read interactions from CSV file in bucket.
        
        Returns:
            DataFrame containing all interactions
        """
        blob = self.bucket.blob(self.metrics_file)
        
        if not blob.exists():
            return pd.DataFrame(columns=[
                "user_id", "movie_id", "recommendation_id", "interaction_type", 
                "experiment_id", "timestamp"
            ])
        
        content = blob.download_as_string()
        return pd.read_csv(io.BytesIO(content))
    
    def track_interaction(
        self,
        user_id: int,
        movie_id: int,
        recommendation_id: str,
        interaction_type: str,
        experiment_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Track a user interaction with a recommended movie.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
            recommendation_id: ID of the recommendation that led to the interaction
            interaction_type: Type of interaction (e.g., 'click', 'watch', 'rate')
            experiment_id: Optional experiment identifier for AB testing
            timestamp: Optional timestamp of the interaction
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Create new row for the interaction
        new_row = {
            "user_id": user_id,
            "movie_id": movie_id,
            "recommendation_id": recommendation_id,
            "interaction_type": interaction_type,
            "experiment_id": experiment_id if experiment_id else "",
            "timestamp": timestamp.isoformat()
        }
        
        # Append to CSV in bucket
        blob = self.bucket.blob(self.metrics_file)
        
        if blob.exists():
            # Download existing content
            content = blob.download_as_string()
            df = pd.read_csv(io.BytesIO(content))
            
            # Append new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Create new DataFrame with this row
            df = pd.DataFrame([new_row])
        
        # Upload updated CSV
        csv_data = io.StringIO()
        df.to_csv(csv_data, index=False)
        blob.upload_from_string(csv_data.getvalue(), content_type="text/csv")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.
        
        Returns:
            Dictionary containing summary statistics
        """
        df = self._read_interactions()
        
        if df.empty:
            return {
                "total_recommendations": 0,
                "total_interactions": 0,
                "hit_rate": 0.0
            }
        
        total_interactions = len(df)
        unique_recommendations = df["recommendation_id"].nunique()
        hit_rate = total_interactions / max(1, unique_recommendations)
        
        return {
            "total_recommendations": unique_recommendations,
            "total_interactions": total_interactions,
            "hit_rate": hit_rate
        }
    
    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary for a specific experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary containing experiment summary or None if not found
        """
        df = self._read_interactions()
        
        if df.empty or experiment_id not in df["experiment_id"].values:
            return None
        
        exp_df = df[df["experiment_id"] == experiment_id]
        
        total_interactions = len(exp_df)
        unique_recommendations = exp_df["recommendation_id"].nunique()
        hit_rate = total_interactions / max(1, unique_recommendations)
        
        return {
            "total_recommendations": unique_recommendations,
            "total_interactions": total_interactions,
            "hit_rate": hit_rate,
            "interaction_types": exp_df["interaction_type"].value_counts().to_dict()
        }
    
    def get_interactions(
        self,
        user_id: Optional[int] = None,
        movie_id: Optional[int] = None,
        experiment_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get filtered interactions based on criteria.
        
        Args:
            user_id: Filter by user ID
            movie_id: Filter by movie ID
            experiment_id: Filter by experiment ID
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of matching interactions
        """
        df = self._read_interactions()
        
        if df.empty:
            return []
        
        # Apply filters
        if user_id is not None:
            df = df[df["user_id"] == user_id]
            
        if movie_id is not None:
            df = df[df["movie_id"] == movie_id]
            
        if experiment_id is not None:
            df = df[df["experiment_id"] == experiment_id]
            
        if start_time is not None:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp_dt"] >= start_time]
            df = df.drop(columns=["timestamp_dt"])
            
        if end_time is not None:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp_dt"] <= end_time]
            df = df.drop(columns=["timestamp_dt"])
            
        # Convert to list of dictionaries
        return df.to_dict(orient="records")