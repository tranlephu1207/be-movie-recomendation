from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, FastAPI
from typing import List, Optional, Dict, Any
import os
import logging
from auth_api import router as auth_router

# Import TMDb recommendation system
from tmdb_integration import TMDbRecommendationSystem

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

def get_recommendation_system():
    """Get or initialize recommendation system."""
    global recommendation_system
    
    if recommendation_system is None:
        data_dir = os.environ.get("DATA_DIR", "./data")
        recommendation_system = TMDbRecommendationSystem(data_dir=data_dir)
        # Load existing user ratings if available
        recommendation_system.load_user_ratings()
        
    return recommendation_system

@router.get("/tmdb/{tmdb_id}")
async def get_movie_by_tmdb_id(
    tmdb_id: int,
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system)
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
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system)
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
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system),
    user_id: int = None
):
    """
    Get movie recommendations based on a text query.
    
    Requires authentication.
    """
    try:
        recommendations = rec_sys.get_recommendations(
            query=query,
            user_id=user_id,
            top_n=top_n
        )
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend/tmdb/{tmdb_id}")
async def get_recommendations_by_tmdb_id(
    tmdb_id: int,
    top_n: int = 10,
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system),
    user_id: int = None
):
    """
    Get movie recommendations based on a movie's TMDb ID.
    
    Requires authentication.
    """
    try:
        recommendations = rec_sys.get_recommendations_by_tmdb_id(
            tmdb_id=tmdb_id,
            user_id=user_id,
            top_n=top_n
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"Movie with TMDb ID {tmdb_id} not found or no recommendations available")
            
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations by TMDb ID: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend/id/{movie_id}")
async def get_recommendations_by_id(
    movie_id: int,
    top_n: int = 10,
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system),
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
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system),
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
    
    # Save ratings in background
    background_tasks.add_task(rec_sys.save_user_ratings)
    
    return {"success": True, "message": f"Rating {rating} added for movie {tmdb_id}"}

@router.post("/rating/id/{movie_id}")
async def add_user_rating(
    movie_id: int,
    rating: float,
    background_tasks: BackgroundTasks,
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system),
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
    
    # Save ratings in background
    background_tasks.add_task(rec_sys.save_user_ratings)
    
    return {"success": True, "message": f"Rating {rating} added for movie {movie_id}"}

@router.get("/search")
async def search_movies(
    query: str,
    limit: int = 10,
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system)
):
    """
    Search for movies by title, genre, director, or actor.
    
    Requires authentication.
    """
    # This would require adding a search method to the TMDbRecommendationSystem class
    # For now, return a placeholder response
    return {"message": "Search functionality not yet implemented", "query": query}

@router.get("/user/{user_id}/ratings")
async def get_user_ratings(
    user_id: int,
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system)
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

# Create FastAPI instance
app = FastAPI()

# Include routers
app.include_router(auth_router)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)