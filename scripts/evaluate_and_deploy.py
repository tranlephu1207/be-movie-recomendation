import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
from typing import Dict, Any, Optional
import requests
from models.metrics import RecommendationMetrics
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender
from google.cloud import storage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Class to evaluate recommendation models based on hit rate metrics.
    """
    
    def __init__(self, metrics: RecommendationMetrics, api_base_url: str = "http://localhost:8000", 
                 bucket_name: str = "recommendation-models"):
        """
        Initialize the model evaluator.
        
        Args:
            metrics: RecommendationMetrics instance
            api_base_url: Base URL of the recommendation API
            bucket_name: Google Cloud Storage bucket name
        """
        self.metrics = metrics
        self.api_base_url = api_base_url
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.current_model_version = self._get_current_model_version()
        
    def _get_current_model_version(self) -> str:
        """
        Get the current deployed model version from Google Cloud Storage.
        
        Returns:
            Current model version
        """
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob("model_version.json")
            
            if not blob.exists():
                return "v1"
                
            content = blob.download_as_text()
            return json.loads(content).get("version", "v1")
        except Exception as e:
            logger.error(f"Error getting model version: {str(e)}")
            return "v1"
    
    def _save_model_version(self, version: str) -> None:
        """
        Save the current model version to Google Cloud Storage.
        
        Args:
            version: Model version to save
        """
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob("model_version.json")
            blob.upload_from_string(json.dumps({"version": version}))
            logger.info(f"Saved model version {version} to GCS")
        except Exception as e:
            logger.error(f"Error saving model version: {str(e)}")
    
    def evaluate_model_performance(
        self,
        experiment_id: str,
        time_window: int = 7  # days
    ) -> Dict[str, Any]:
        """
        Evaluate model performance based on hit rate metrics.
        
        Args:
            experiment_id: Experiment identifier
            time_window: Time window in days to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get interactions for the experiment within the time window
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_window)
        
        interactions = self.metrics.get_interactions(
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if not interactions:
            return {
                "hit_rate": 0.0,
                "total_interactions": 0,
                "total_recommendations": 0,
                "success": False,
                "message": "No interactions found in the specified time window"
            }
        
        # Calculate metrics
        unique_recommendations = len(set(i["recommendation_id"] for i in interactions))
        total_interactions = len(interactions)
        
        hit_rate = total_interactions / max(1, unique_recommendations)
        
        # Calculate interaction types distribution
        interaction_types = {}
        for interaction in interactions:
            itype = interaction["interaction_type"]
            interaction_types[itype] = interaction_types.get(itype, 0) + 1
        
        return {
            "hit_rate": hit_rate,
            "total_interactions": total_interactions,
            "total_recommendations": unique_recommendations,
            "interaction_types": interaction_types,
            "success": True,
            "message": "Evaluation completed successfully"
        }
    
    def compare_models(
        self,
        experiment_ids: list,
        time_window: int = 7
    ) -> Dict[str, Any]:
        """
        Compare performance of multiple models.
        
        Args:
            experiment_ids: List of experiment identifiers
            time_window: Time window in days to evaluate
            
        Returns:
            Dictionary containing comparison results
        """
        results = {}
        for exp_id in experiment_ids:
            results[exp_id] = self.evaluate_model_performance(
                experiment_id=exp_id,
                time_window=time_window
            )
        
        # Find best performing model
        best_model = max(
            results.items(),
            key=lambda x: x[1]["hit_rate"] if x[1]["success"] else 0
        )
        
        return {
            "results": results,
            "best_model": {
                "experiment_id": best_model[0],
                "hit_rate": best_model[1]["hit_rate"]
            }
        }
    
    def deploy_model(self, experiment_id: str) -> bool:
        """
        Deploy a new model version.
        
        Args:
            experiment_id: Experiment identifier to deploy
            
        Returns:
            True if deployment was successful
        """
        try:
            # Update model version
            new_version = f"v{int(self.current_model_version[1:]) + 1}"
            self._save_model_version(new_version)
            
            # Call API to update model
            response = requests.post(
                f"{self.api_base_url}/movies/deploy-model",
                json={"experiment_id": experiment_id}
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully deployed model from experiment {experiment_id}")
                return True
            else:
                logger.error(f"Failed to deploy model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            return False

def main():
    # Initialize metrics
    metrics = RecommendationMetrics()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(metrics)
    
    # Define experiments to evaluate
    experiments = [
        "content_based_v1",
        "collaborative_v1",
        "hybrid_v1"
    ]
    
    # Compare models
    comparison = evaluator.compare_models(experiments)
    
    # Get best model
    best_model = comparison["best_model"]
    logger.info(f"Best performing model: {best_model['experiment_id']} with hit rate {best_model['hit_rate']:.3f}")
    
    # Deploy best model if it's better than current
    current_performance = evaluator.evaluate_model_performance(evaluator.current_model_version)
    if best_model["hit_rate"] > current_performance["hit_rate"]:
        logger.info(f"Deploying new model: {best_model['experiment_id']}")
        success = evaluator.deploy_model(best_model["experiment_id"])
        if success:
            logger.info("Deployment successful")
        else:
            logger.error("Deployment failed")
    else:
        logger.info("Current model performs better, no deployment needed")

if __name__ == "__main__":
    main() 