import os
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from tmdb_data_fetcher import TMDbDataFetcher
from utils.gcloud import init_credentials, get_storage_client
import io

def extract_tmdb_ids_from_links(links_path: str, bucket_name: str = None) -> list:
    """
    Extract TMDb IDs from links.csv in Google Cloud Storage.
    
    Args:
        links_path: Path to links.csv file in the bucket
        bucket_name: Name of the Google Cloud Storage bucket
    
    Returns:
        List of TMDb IDs
    """
    try:
        # Initialize GCS client
        bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("Bucket name must be provided or GCS_BUCKET_NAME environment variable must be set")
            
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        
        # Load links.csv from GCS
        blob = bucket.blob(links_path)
        if not blob.exists():
            raise FileNotFoundError(f"File not found in bucket: {links_path}")
            
        content = blob.download_as_bytes()
        links_df = pd.read_csv(io.BytesIO(content))
        
        # Ensure tmdbId column exists
        if 'tmdbId' not in links_df.columns:
            raise ValueError(f"File {links_path} does not contain a 'tmdbId' column")
        
        # Extract TMDb IDs
        tmdb_ids = links_df['tmdbId'].dropna().astype(int).tolist()
        
        return tmdb_ids
    
    except Exception as e:
        print(f"Error extracting TMDb IDs from {links_path}: {str(e)}")
        return []

def create_id_mapping(links_path: str, output_path: str, bucket_name: str = None) -> None:
    """
    Create a mapping between movieId and TMDb IDs and save to Google Cloud Storage.
    
    Args:
        links_path: Path to links.csv file in the bucket
        output_path: Path in the bucket where to save the mapping
        bucket_name: Name of the Google Cloud Storage bucket
    """
    try:
        # Initialize GCS client
        bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("Bucket name must be provided or GCS_BUCKET_NAME environment variable must be set")
            
        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        
        # Load links.csv from GCS
        blob = bucket.blob(links_path)
        if not blob.exists():
            raise FileNotFoundError(f"File not found in bucket: {links_path}")
            
        content = blob.download_as_bytes()
        links_df = pd.read_csv(io.BytesIO(content))
        
        # Ensure required columns exist
        required_columns = ['movieId', 'tmdbId']
        for col in required_columns:
            if col not in links_df.columns:
                raise ValueError(f"File {links_path} does not contain required column: {col}")
        
        # Create mapping
        mapping_df = links_df[['movieId', 'tmdbId']].copy()
        mapping_df = mapping_df.dropna()
        mapping_df['tmdbId'] = mapping_df['tmdbId'].astype(int)
        
        # Save to GCS
        csv_buffer = io.StringIO()
        mapping_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        blob = bucket.blob(output_path)
        blob.upload_from_string(csv_content, content_type="text/csv")
        
        print(f"Successfully created and saved ID mapping to {output_path}")
        
    except Exception as e:
        print(f"Error creating ID mapping: {str(e)}")

def build_tmdb_database(bucket_name: str = None, data_folder: str = 'data'):
    """
    Build TMDb database and save to Google Cloud Storage.
    
    Args:
        bucket_name: Name of the Google Cloud Storage bucket
        data_folder: Folder within the bucket to store data
    """
    # Initialize GCS client
    bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("Bucket name must be provided or GCS_BUCKET_NAME environment variable must be set")
        
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    
    # Initialize TMDb data fetcher
    fetcher = TMDbDataFetcher()
    
    # Define file paths in the bucket
    links_path = f"{data_folder}/links.csv"
    movies_path = f"{data_folder}/movies.csv"
    genres_path = f"{data_folder}/genres.csv"
    cast_path = f"{data_folder}/cast.csv"
    crew_path = f"{data_folder}/crew.csv"
    mapping_path = f"{data_folder}/id_mapping.csv"
    
    try:
        # Extract TMDb IDs
        print("Extracting TMDb IDs...")
        tmdb_ids = extract_tmdb_ids_from_links(links_path, bucket_name)
        if not tmdb_ids:
            raise ValueError("No TMDb IDs found")
            
        # Create ID mapping
        print("Creating ID mapping...")
        create_id_mapping(links_path, mapping_path, bucket_name)
        
        # Fetch and save movie data
        print("Fetching movie data...")
        movies_data = []
        genres_data = []
        cast_data = []
        crew_data = []
        
        for tmdb_id in tmdb_ids:
            try:
                # Fetch movie details
                movie = fetcher.get_movie_details(tmdb_id)
                if movie:
                    movies_data.append(movie)
                    
                    # Extract genres
                    for genre in movie.get('genres', []):
                        genres_data.append({
                            'tmdb_id': tmdb_id,
                            'genre_name': genre
                        })
                    
                    # Fetch credits
                    credits = fetcher.get_movie_credits(tmdb_id)
                    if credits:
                        # Extract cast
                        for actor in credits.get('cast', [])[:10]:  # Top 10 cast members
                            cast_data.append({
                                'tmdb_id': tmdb_id,
                                'name': actor.get('name'),
                                'character': actor.get('character')
                            })
                        
                        # Extract crew
                        for member in credits.get('crew', []):
                            if member.get('job') == 'Director':
                                crew_data.append({
                                    'tmdb_id': tmdb_id,
                                    'name': member.get('name'),
                                    'job': 'Director'
                                })
                
                # Sleep to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing movie {tmdb_id}: {str(e)}")
                continue
        
        # Convert to DataFrames
        movies_df = pd.DataFrame(movies_data)
        genres_df = pd.DataFrame(genres_data)
        cast_df = pd.DataFrame(cast_data)
        crew_df = pd.DataFrame(crew_data)
        
        # Save to GCS
        print("Saving data to Google Cloud Storage...")
        
        def save_df_to_gcs(df, path):
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            blob = bucket.blob(path)
            blob.upload_from_string(csv_content, content_type="text/csv")
        
        save_df_to_gcs(movies_df, movies_path)
        save_df_to_gcs(genres_df, genres_path)
        save_df_to_gcs(cast_df, cast_path)
        save_df_to_gcs(crew_df, crew_path)
        
        print("Database build complete!")
        
    except Exception as e:
        print(f"Error building database: {str(e)}")
        raise

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Build TMDb database and save to Google Cloud Storage")
    parser.add_argument("--bucket", help="Google Cloud Storage bucket name")
    parser.add_argument("--data-folder", default="data", help="Folder within bucket to store data")
    args = parser.parse_args()
    
    # Build database
    build_tmdb_database(args.bucket, args.data_folder)