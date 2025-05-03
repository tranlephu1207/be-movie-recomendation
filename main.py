from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, FastAPI
from typing import List, Optional, Dict, Any
import os
import logging
from auth_api import router as auth_router
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime
import uuid

# Load environment variables from .env file
load_dotenv()

# Import the CloudTMDbRecommendationSystem instead of the local file-based one
from cloud_tmdb import CloudTMDbRecommendationSystem
from models.metrics import RecommendationMetrics

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="API for movie recommendations using TMDb data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
      "http://localhost:3000",
      "https://movie-mood-app68.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define router
router = APIRouter(
    prefix="/movies",
    tags=["Movies"],
    responses={404: {"description": "Not found"}},
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instance of recommendation system
recommendation_system = None

# Initialize metrics tracker
metrics = RecommendationMetrics()

def get_recommendation_system():
    """Get or initialize recommendation system."""
    global recommendation_system
    
    if recommendation_system is None:
        # Get bucket name from environment variable
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable must be set")
        
        # Get data folder from environment variable
        data_folder = os.environ.get("GCS_DATA_FOLDER", "data")
        
        try:
            # Initialize the cloud-based recommendation system
            recommendation_system = CloudTMDbRecommendationSystem(
                bucket_name=bucket_name,
                data_folder=data_folder
            )
            logger.info(f"Cloud recommendation system initialized with bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize cloud recommendation system: {e}")
            raise
        
    return recommendation_system

@router.get("/")
async def get_all_movies(
    page: int = 1,
    page_size: int = 20,
    rec_sys = Depends(get_recommendation_system)
):
    """
    Get all movies with pagination.
    
    Args:
        page: Page number (starting from 1)
        page_size: Number of items per page
        
    Returns:
        Dictionary containing:
        - items: List of movies for the current page
        - total: Total number of movies
        - page: Current page number
        - total_pages: Total number of pages
    """
    # Calculate start and end indices
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    # Get total number of movies
    total_movies = len(rec_sys.movies_df)
    total_pages = (total_movies + page_size - 1) // page_size
    
    # Get movies for current page
    movies = rec_sys.movies_df.iloc[start_idx:end_idx]
    
    # Convert movies to list of dictionaries with all necessary information
    movie_list = []
    for _, movie in movies.iterrows():
        movie_data = rec_sys.get_movie_by_tmdb_id(movie['tmdb_id'])
        if movie_data:
            movie_list.append({
                "id": movie_data.get('id', 0),
                "tmdb_id": movie_data.get('tmdb_id', 0),
                "imdb_id": movie_data.get('imdb_id', ''),
                "title": movie_data.get('title', ''),
                "release_date": movie_data.get('release_date', ''),
                "genres": movie_data.get('genres', []),
                "director": movie_data.get('director', ''),
                "cast": movie_data.get('cast', []),
                "overview": movie_data.get('overview', ''),
                "vote_average": float(movie_data.get('vote_average', 0.0))
            })
    
    return {
        "items": movie_list,
        "total": total_movies,
        "page": page,
        "total_pages": total_pages
    }

@router.get("/tmdb/{tmdb_id}")
async def get_movie_by_tmdb_id(
    tmdb_id: int,
    rec_sys = Depends(get_recommendation_system)
):
    """
    Get movie details by TMDb ID.
    
    Requires authentication.
    """
    movie = rec_sys.get_movie_by_tmdb_id(tmdb_id)
    if movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with TMDb ID {tmdb_id} not found")
    
    return {
        "id": movie.get('id', 0),
        "tmdb_id": movie.get('tmdb_id', 0),
        "imdb_id": movie.get('imdb_id', ''),
        "title": movie.get('title', ''),
        "release_date": movie.get('release_date', ''),
        "genres": movie.get('genres', []),
        "director": movie.get('director', ''),
        "cast": movie.get('cast', []),
        "overview": movie.get('overview', ''),
        "vote_average": float(movie.get('vote_average', 0.0))
    }

@router.get("/id/{movie_id}")
async def get_movie_by_id(
    movie_id: int,
    rec_sys = Depends(get_recommendation_system)
):
    """
    Get movie details by internal movie ID (from links.csv).
    
    Requires authentication.
    """
    movie = rec_sys.get_movie_by_id(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    
    return {
        "id": movie.get('id', 0),
        "tmdb_id": movie.get('tmdb_id', 0),
        "imdb_id": movie.get('imdb_id', ''),
        "title": movie.get('title', ''),
        "release_date": movie.get('release_date', ''),
        "genres": movie.get('genres', []),
        "director": movie.get('director', ''),
        "cast": movie.get('cast', []),
        "overview": movie.get('overview', ''),
        "vote_average": float(movie.get('vote_average', 0.0))
    }

@router.post("/recommend")
async def get_recommendations(
    query: str,
    top_n: int = 10,
    rec_sys = Depends(get_recommendation_system),
    user_id: int = None,
    experiment_id: Optional[str] = None
):
    """
    Get movie recommendations based on a text query.
    
    Args:
        query: Text query for content-based filtering
        top_n: Number of recommendations to return
        user_id: Optional user ID for collaborative filtering
        experiment_id: Optional experiment ID for AB testing
        
    Returns:
        List of movie recommendations with tracking IDs
    """
    try:
        recommendations = rec_sys.get_recommendations(
            query=query,
            user_id=user_id,
            top_n=top_n
        )
        
        if not recommendations:
            return []
            
        # Generate a unique recommendation ID for tracking
        recommendation_id = str(uuid.uuid4())
        
        # Convert recommendations to native Python types and add tracking ID
        converted_recs = []
        for rec in recommendations:
            if isinstance(rec, dict):
                converted_rec = {
                    "id": int(rec.get('id', 0)),
                    "tmdb_id": int(rec.get('tmdb_id', 0)),
                    "imdb_id": str(rec.get('imdb_id', '')),
                    "title": str(rec.get('title', '')),
                    "release_date": str(rec.get('release_date', '')),
                    "genres": [str(g) for g in rec.get('genres', [])],
                    "director": str(rec.get('director', '')),
                    "cast": [str(c) for c in rec.get('cast', [])],
                    "overview": str(rec.get('overview', '')),
                    "vote_average": float(rec.get('vote_average', 0.0)),
                    "recommendation_id": recommendation_id  # Add tracking ID
                }
                converted_recs.append(converted_rec)
            else:
                # If rec is not a dict, it might be just an ID
                movie = rec_sys.get_movie_by_id(int(rec))
                if movie:
                    converted_recs.append({
                        "id": int(movie.get('id', 0)),
                        "tmdb_id": int(movie.get('tmdb_id', 0)),
                        "imdb_id": str(movie.get('imdb_id', '')),
                        "title": str(movie.get('title', '')),
                        "release_date": str(movie.get('release_date', '')),
                        "genres": [str(g) for g in movie.get('genres', [])],
                        "director": str(movie.get('director', '')),
                        "cast": [str(c) for c in movie.get('cast', [])],
                        "overview": str(movie.get('overview', '')),
                        "vote_average": float(movie.get('vote_average', 0.0)),
                        "recommendation_id": recommendation_id  # Add tracking ID
                    })
        return converted_recs
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend/tmdb/{tmdb_id}")
async def get_recommendations_by_tmdb_id(
    tmdb_id: int,
    top_n: int = 10,
    rec_sys = Depends(get_recommendation_system),
    user_id: int = None,
    experiment_id: Optional[str] = None
):
    """
    Get movie recommendations based on a movie's TMDb ID.
    
    Args:
        tmdb_id: TMDb ID of the reference movie
        top_n: Number of recommendations to return
        user_id: Optional user ID for collaborative filtering
        experiment_id: Optional experiment ID for AB testing
        
    Returns:
        List of movie recommendations with tracking IDs
    """
    try:
        recommendations = rec_sys.get_recommendations_by_tmdb_id(
            tmdb_id=tmdb_id,
            user_id=user_id,
            top_n=top_n
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"Movie with TMDb ID {tmdb_id} not found or no recommendations available")
        
        # Generate a unique recommendation ID for tracking
        recommendation_id = str(uuid.uuid4())
        
        # Convert recommendations to native Python types and add tracking ID
        converted_recs = []
        for rec in recommendations:
            if isinstance(rec, dict):
                converted_rec = {
                    "id": int(rec.get('id', 0)),
                    "tmdb_id": int(rec.get('tmdb_id', 0)),
                    "imdb_id": str(rec.get('imdb_id', '')),
                    "title": str(rec.get('title', '')),
                    "release_date": str(rec.get('release_date', '')),
                    "genres": [str(g) for g in rec.get('genres', [])],
                    "director": str(rec.get('director', '')),
                    "cast": [str(c) for c in rec.get('cast', [])],
                    "overview": str(rec.get('overview', '')),
                    "vote_average": float(rec.get('vote_average', 0.0)),
                    "recommendation_id": recommendation_id  # Add tracking ID
                }
                # Add similarity score if present
                if 'similarity' in rec:
                    converted_rec["similarity_score"] = float(rec['similarity'])
                converted_recs.append(converted_rec)
            
        return converted_recs
            
    except Exception as e:
        logger.error(f"Error getting recommendations by TMDb ID: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend/id/{movie_id}")
async def get_recommendations_by_id(
    movie_id: int,
    top_n: int = 10,
    rec_sys = Depends(get_recommendation_system),
    user_id: int = None
):
    """
    Get movie recommendations based on a movie's internal ID (from links.csv).
    
    Requires authentication.
    """
    try:
        recommendations = rec_sys.get_recommendations_by_id(
            movie_id=movie_id,
            user_id=user_id,
            top_n=top_n
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found or no recommendations available")
            
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations by movie ID: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
@router.post("/rating/tmdb/{tmdb_id}")
async def add_user_rating_by_tmdb_id(
    tmdb_id: int,
    rating: float,
    background_tasks: BackgroundTasks,
    rec_sys = Depends(get_recommendation_system),
    user_id: int = None
):
    """
    Add a user rating for a movie by TMDb ID.
    
    Requires authentication.
    Rating should be between 0.5 and 5.0.
    """
    # Validate rating
    if rating < 0.5 or rating > 5.0:
        raise HTTPException(status_code=400, detail="Rating must be between 0.5 and 5.0")
    
    # Ensure user_id is provided
    if user_id is None:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    success = rec_sys.add_user_rating_by_tmdb_id(
        user_id=user_id,
        tmdb_id=tmdb_id,
        rating=rating
    )
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Movie with TMDb ID {tmdb_id} not found")
    
    # Note: With the cloud-based implementation, ratings are saved immediately
    # No need for background task, but keeping it for API compatibility
    
    return {"success": True, "message": f"Rating {rating} added for movie {tmdb_id}"}

@router.post("/rating/id/{movie_id}")
async def add_user_rating(
    movie_id: int,
    rating: float,
    background_tasks: BackgroundTasks,
    rec_sys = Depends(get_recommendation_system),
    user_id: int = None
):
    """
    Add a user rating for a movie by internal ID (from links.csv).
    
    Requires authentication.
    Rating should be between 0.5 and 5.0.
    """
    # Validate rating
    if rating < 0.5 or rating > 5.0:
        raise HTTPException(status_code=400, detail="Rating must be between 0.5 and 5.0")
    
    # Ensure user_id is provided
    if user_id is None:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    success = rec_sys.add_user_rating(
        user_id=user_id,
        movie_id=movie_id,
        rating=rating
    )
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    
    # Note: With the cloud-based implementation, ratings are saved immediately
    # No need for background task, but keeping it for API compatibility
    
    return {"success": True, "message": f"Rating {rating} added for movie {movie_id}"}

@router.get("/search")
async def search_movies(
    query: str,
    limit: int = 10,
    rec_sys = Depends(get_recommendation_system)
):
    """
    Search for movies by title, genre, director, or actor.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        
    Returns:
        List of matching movies with their details
    """
    try:
        results = rec_sys.search_movies(query=query, limit=limit)
        
        # Convert results to consistent types
        converted_results = []
        for movie in results:
            converted_movie = {
                "id": int(movie.get('id', 0)),
                "tmdb_id": int(movie.get('tmdb_id', 0)),
                "imdb_id": str(movie.get('imdb_id', '')),
                "title": str(movie.get('title', '')),
                "release_date": str(movie.get('release_date', '')),
                "genres": [str(g) for g in movie.get('genres', [])],
                "director": str(movie.get('director', '')),
                "cast": [str(c) for c in movie.get('cast', [])],
                "overview": str(movie.get('overview', '')),
                "vote_average": float(movie.get('vote_average', 0.0)),
                "search_score": float(movie.get('search_score', 0.0))
            }
            converted_results.append(converted_movie)
            
        return converted_results
        
    except Exception as e:
        logger.error(f"Error searching movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}/ratings")
async def get_user_ratings(
    user_id: int,
    rec_sys = Depends(get_recommendation_system)
):
    """
    Get all ratings by a specific user.
    
    Requires authentication.
    """
    # Filter user ratings for this user
    if rec_sys.user_ratings_df is not None and len(rec_sys.user_ratings_df) > 0:
        user_ratings = rec_sys.user_ratings_df[rec_sys.user_ratings_df['userId'] == user_id]
        
        if len(user_ratings) > 0:
            # Convert to list of dictionaries
            ratings_list = user_ratings.to_dict(orient='records')
            
            # Add movie details to each rating
            result = []
            for rating in ratings_list:
                movie = rec_sys.get_movie_by_id(rating['movieId'])
                if movie:
                    result.append({
                        'userId': rating['userId'],
                        'movieId': rating['movieId'],
                        'tmdb_id': movie.get('tmdb_id', None),
                        'title': movie.get('title', 'Unknown'),
                        'rating': rating['rating']
                    })
                else:
                    result.append({
                        'userId': rating['userId'],
                        'movieId': rating['movieId'],
                        'tmdb_id': None,
                        'title': 'Unknown',
                        'rating': rating['rating']
                    })
            
            return result
    
    return []

@router.get("/user/{user_id}/rated-movies")
async def get_user_rated_movies_endpoint(
    user_id: int,
    rec_sys = Depends(get_recommendation_system)
):
    """
    Get all movies rated by a specific user with full movie details.
    """
    try:
        rated_movies = rec_sys.get_user_rated_movies(user_id)
        
        if not rated_movies:
            return []
            
        # Convert to list of dictionaries with consistent types
        converted_movies = []
        for movie in rated_movies:
            # Safe conversion functions
            def safe_int(value, default=0):
                try:
                    if pd.isna(value):  # Check for NaN
                        return default
                    return int(value)
                except (ValueError, TypeError):
                    return default
                    
            def safe_float(value, default=0.0):
                try:
                    if pd.isna(value):  # Check for NaN
                        return default
                    return float(value)
                except (ValueError, TypeError):
                    return default

            converted_movie = {
                "id": safe_int(movie.get('id', 0)),
                "tmdb_id": safe_int(movie.get('tmdb_id', 0)),
                "imdb_id": str(movie.get('imdb_id', '')),
                "title": str(movie.get('title', '')),
                "release_date": str(movie.get('release_date', '')),
                "genres": [str(g) for g in movie.get('genres', [])],
                "director": str(movie.get('director', '')),
                "cast": [str(c) for c in movie.get('cast', [])],
                "overview": str(movie.get('overview', '')),
                "vote_average": safe_float(movie.get('vote_average', 0.0)),
                "user_rating": safe_float(movie.get('user_rating', 0.0)),
                "rating_timestamp": safe_int(movie.get('rating_timestamp', 0))
            }
            converted_movies.append(converted_movie)
            
        return converted_movies
        
    except Exception as e:
        logger.error(f"Error getting user rated movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update-movie/{tmdb_id}")
async def update_movie_data(
    tmdb_id: int,
    movie_data: dict,
    rec_sys = Depends(get_recommendation_system)
):
    """
    Update movie data in the cloud storage.
    
    Requires authentication.
    """
    # Ensure tmdb_id is in the request body
    if 'tmdb_id' not in movie_data:
        movie_data['tmdb_id'] = tmdb_id
    elif movie_data['tmdb_id'] != tmdb_id:
        raise HTTPException(status_code=400, detail="TMDb ID in path must match TMDb ID in request body")
    
    try:
        success = rec_sys.update_movie_data(movie_data)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Movie with TMDb ID {tmdb_id} not found")
        
        return {"success": True, "message": f"Movie data updated for TMDb ID {tmdb_id}"}
        
    except Exception as e:
        logger.error(f"Error updating movie data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics/track")
async def track_interaction(
    user_id: int,
    movie_id: int,
    recommendation_id: str,
    interaction_type: str,
    experiment_id: Optional[str] = None
):
    """
    Track a user interaction with a recommended movie.
    
    Args:
        user_id: User identifier
        movie_id: Movie identifier
        recommendation_id: ID of the recommendation that led to the interaction
        interaction_type: Type of interaction (e.g., 'click', 'watch', 'rate')
        experiment_id: Optional experiment identifier for AB testing
    """
    try:
        metrics.track_interaction(
            user_id=user_id,
            movie_id=movie_id,
            recommendation_id=recommendation_id,
            interaction_type=interaction_type,
            experiment_id=experiment_id
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error tracking interaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/summary")
async def get_metrics_summary():
    """
    Get summary of all recommendation metrics.
    
    Returns:
        Dictionary containing summary statistics
    """
    try:
        return metrics.get_summary()
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/experiment/{experiment_id}")
async def get_experiment_metrics(experiment_id: str):
    """
    Get metrics for a specific experiment.
    
    Args:
        experiment_id: Experiment identifier
        
    Returns:
        Dictionary containing experiment metrics
    """
    try:
        experiment_metrics = metrics.get_experiment_summary(experiment_id)
        if experiment_metrics is None:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        return experiment_metrics
    except Exception as e:
        logger.error(f"Error getting experiment metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/interactions")
async def get_interactions(
    user_id: Optional[int] = None,
    movie_id: Optional[int] = None,
    experiment_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
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
    try:
        return metrics.get_interactions(
            user_id=user_id,
            movie_id=movie_id,
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=end_time
        )
    except Exception as e:
        logger.error(f"Error getting interactions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include routers
app.include_router(router)
app.include_router(auth_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Create FastAPI instance if this file is run directly
# (Your app.py already includes this router, so this part would only be used if main.py is run directly)
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=port)