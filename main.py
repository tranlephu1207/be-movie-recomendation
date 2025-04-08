from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, FastAPI
from typing import List, Optional, Dict, Any
import os
import logging
from auth_api import router as auth_router
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

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

@router.get("/")
async def get_all_movies(
    page: int = 1,
    page_size: int = 20,
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system)
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
        
        # Convert recommendations to native Python types
        if isinstance(recommendations, list):
            converted_recs = []
            for rec in recommendations:
                if isinstance(rec, dict):
                    converted_rec = {
                        "id": int(rec.get('movie_data', {}).get('id', 0)),
                        "title": str(rec.get('movie_data', {}).get('title', '')),
                        "vote_average": float(rec.get('movie_data', {}).get('vote_average', 0.0)),
                        "genres": [str(g) for g in rec.get('movie_data', {}).get('genres', [])],
                        "tmdb_id": int(rec.get('movie_data', {}).get('tmdb_id', 0)),
                        "imdb_id": str(rec.get('movie_data', {}).get('imdb_id', '')),
                        "release_date": str(rec.get('movie_data', {}).get('release_date', '')),
                        "director": str(rec.get('movie_data', {}).get('director', '')),
                        "cast": [str(c) for c in rec.get('movie_data', {}).get('cast', [])],
                        "overview": str(rec.get('movie_data', {}).get('overview', ''))
                    }
                    # Add similarity score if present
                    if 'similarity' in rec:
                        converted_rec["similarity_score"] = float(rec['similarity'])
                    converted_recs.append(converted_rec)
                else:
                    # If rec is not a dict, it might be just an ID
                    movie = rec_sys.get_movie_by_id(int(rec))
                    print(movie)
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
                            "vote_average": float(movie.get('vote_average', 0.0))
                        })
            return converted_recs
        return []
        
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
        
        # Convert recommendations to native Python types
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
                    "vote_average": float(rec.get('vote_average', 0.0))
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

@router.get("/user/{user_id}/rated-movies")
async def get_user_rated_movies_endpoint(
    user_id: int,
    rec_sys: TMDbRecommendationSystem = Depends(get_recommendation_system)
):
    """
    Get all movies rated by a specific user with full movie details.
    
    Args:
        user_id: User ID
        
    Returns:
        List of movies with their details and user ratings
    """
    try:
        rated_movies = rec_sys.get_user_rated_movies(user_id)
        
        if not rated_movies:
            return []
            
        # Convert to list of dictionaries with consistent types
        converted_movies = []
        for movie in rated_movies:
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
                "user_rating": float(movie.get('user_rating', 0.0)),
                "rating_timestamp": int(movie.get('rating_timestamp', 0))
            }
            converted_movies.append(converted_movie)
            
        return converted_movies
        
    except Exception as e:
        logger.error(f"Error getting user rated movies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Create FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
      "http://localhost:3000",
      "https://localhost:3000",
      "https://movie-mood-app68.vercel.app",
      "https://be-movie-recomendation.onrender.com"  # Add your backend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)