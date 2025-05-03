from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Import routers
from auth_api import router as auth_router, get_current_user
from main import router as movie_router  # This imports the router from main.py

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TMDb Movie Recommendation API with Cloud Storage",
    description="API for movie recommendations with user authentication using TMDb data stored in Google Cloud Storage",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
      "http://localhost:3000",
      "https://movie-mood-app68.vercel.app/",
      "https://localhost:3000",
      "https://be-movie-recomendation.onrender.com/"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cloud_storage": os.environ.get("GCS_BUCKET_NAME", "Not configured")
    }

# Include routers
app.include_router(auth_router)
app.include_router(movie_router, dependencies=[Depends(get_current_user)])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "TMDb Movie Recommendation API with Cloud Storage",
        "docs": "/docs",
        "auth": "/auth endpoints for authentication",
        "movies": "/movies endpoints for recommendations",
        "cloud_storage": os.environ.get("GCS_BUCKET_NAME", "Not configured")
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the app
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)