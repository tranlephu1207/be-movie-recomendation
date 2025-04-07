import os
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from tmdb_data_fetcher import TMDbDataFetcher

def extract_tmdb_ids_from_links(links_path):
    """
    Extract TMDb IDs from links.csv.
    
    Args:
        links_path: Path to links.csv file
    
    Returns:
        List of TMDb IDs
    """
    try:
        # Load links.csv
        links_df = pd.read_csv(links_path)
        
        # Ensure tmdbId column exists
        if 'tmdbId' not in links_df.columns:
            raise ValueError(f"File {links_path} does not contain a 'tmdbId' column")
        
        # Extract TMDb IDs
        tmdb_ids = links_df['tmdbId'].dropna().astype(int).tolist()
        
        return tmdb_ids
    
    except Exception as e:
        print(f"Error extracting TMDb IDs from {links_path}: {str(e)}")
        return []

def create_id_mapping(links_path, output_path):
    """
    Create a mapping between movieId and TMDb IDs.
    
    Args:
        links_path: Path to links.csv file
        output_path: Path to save the mapping
    """
    try:
        # Load links.csv
        links_df = pd.read_csv(links_path)
        
        # Ensure required columns exist
        required_columns = ['movieId', 'tmdbId']
        for col in required_columns:
            if col not in links_df.columns:
                raise ValueError(f"File {links_path} does not contain a '{col}' column")
        
        # Select only movieId and tmdbId columns
        mapping_df = links_df[required_columns].copy()
        
        # Remove rows with NaN tmdbId
        mapping_df = mapping_df.dropna(subset=['tmdbId'])
        
        # Convert tmdbId to integer
        mapping_df['tmdbId'] = mapping_df['tmdbId'].astype(int)
        
        # Save the mapping
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mapping_df.to_csv(output_path, index=False)
        print(f"Created ID mapping at {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Error creating ID mapping: {str(e)}")
        return False

def build_tmdb_database():
    """
    Build a local TMDb database from scratch using the TMDb API.
    Uses environment variables for configuration.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get environment variables
    api_key = os.environ.get('TMDB_API_KEY')
    links_path = os.environ.get('FULL_MOVIE_LINKS', './data/recommendation_data/links.csv')
    data_dir = os.environ.get('DATA_DIR', './data')
    delay = float(os.environ.get('API_DELAY', '0.5'))
    batch_size = os.environ.get('BATCH_SIZE')
    
    # Convert batch_size to int if specified
    if batch_size:
        try:
            batch_size = int(batch_size)
        except ValueError:
            print(f"Warning: Invalid BATCH_SIZE '{batch_size}'. Using default of 100.")
            batch_size = 100
    else:
        # Default batch size if not specified
        batch_size = 100
    
    # Verify API key
    if not api_key:
        raise ValueError("TMDB_API_KEY not found in environment variables. Please add it to your .env file.")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  TMDb API Key: {'*' * 8}{api_key[-4:] if api_key else 'Not set'}")
    print(f"  Links CSV Path: {links_path}")
    print(f"  Data Directory: {data_dir}")
    print(f"  API Delay: {delay} seconds")
    print(f"  Batch Size: {batch_size}")
    
    # Initialize TMDb data fetcher
    fetcher = TMDbDataFetcher(api_key=api_key, data_dir=data_dir)
    
    # Collect TMDb IDs to fetch
    all_ids_to_fetch = []
    
    # Use links.csv if it exists
    if os.path.exists(links_path):
        print(f"Extracting TMDb IDs from {links_path}...")
        extracted_ids = extract_tmdb_ids_from_links(links_path)
        if extracted_ids:
            all_ids_to_fetch = extracted_ids
            print(f"Found {len(all_ids_to_fetch)} TMDb IDs in links.csv")
            
            # Create ID mapping
            mapping_path = os.path.join(data_dir, 'id_mapping.csv')
            create_id_mapping(links_path, mapping_path)
    else:
        print(f"Warning: Links file not found at {links_path}")
        # Use a default sample list if no links.csv
        print("Using default sample list of popular movies...")
        all_ids_to_fetch = [
            # Popular movies (TMDb IDs)
            862,    # Toy Story
            278,    # The Shawshank Redemption
            238,    # The Godfather
            240,    # The Godfather: Part II
            155,    # The Dark Knight
            424,    # Schindler's List
            122,    # The Lord of the Rings: The Return of the King
            680,    # Pulp Fiction
            429,    # The Good, the Bad and the Ugly
            13,     # Forrest Gump
            120,    # The Lord of the Rings: The Fellowship of the Ring
            27205,  # Inception
            603,    # The Matrix
            1891,   # Star Wars: Episode V - The Empire Strikes Back
            769,    # Goodfellas
            11216,  # The Lord of the Rings: The Two Towers
            11,     # Star Wars: Episode IV - A New Hope
            497,    # The Green Mile
        ]
    
    # Remove duplicates while preserving order
    all_ids_to_fetch = list(dict.fromkeys(all_ids_to_fetch))
    total_ids = len(all_ids_to_fetch)
    
    # Track overall statistics
    overall_successful = 0
    overall_failed = 0
    
    # Check if there are any fetched movies already
    processed_ids = set()
    if os.path.exists(os.path.join(data_dir, 'movies.csv')):
        try:
            # Read already processed IDs from movies.csv
            movies_df = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
            if 'tmdb_id' in movies_df.columns:
                processed_ids = set(movies_df['tmdb_id'].tolist())
                print(f"Found {len(processed_ids)} movies already fetched. Will skip these.")
        except Exception as e:
            print(f"Error reading existing movie data: {str(e)}")
    
    # Filter out already processed IDs
    ids_to_fetch = [id for id in all_ids_to_fetch if id not in processed_ids]
    remaining_ids = len(ids_to_fetch)
    
    print(f"Total movies to fetch: {remaining_ids} (out of {total_ids} total)")
    
    # Process in batches
    batch_number = 1
    for start_idx in range(0, len(ids_to_fetch), batch_size):
        # Get current batch of IDs
        end_idx = min(start_idx + batch_size, len(ids_to_fetch))
        batch_ids = ids_to_fetch[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_number} ({start_idx+1}-{end_idx} of {remaining_ids} remaining movies)")
        print(f"Starting to fetch {len(batch_ids)} movies...")
        
        successful = 0
        failed = 0
        
        for i, tmdb_id in enumerate(batch_ids, 1):
            try:
                print(f"[{i}/{len(batch_ids)}] Fetching movie {tmdb_id}...")
                movie = fetcher.fetch_movie_by_tmdb_id(tmdb_id)
                
                if movie:
                    print(f"✓ Successfully fetched: {movie.get('title', 'Unknown')} ({movie.get('release_date', 'N/A')})")
                    successful += 1
                    overall_successful += 1
                    processed_ids.add(tmdb_id)
                else:
                    print(f"✗ Failed to fetch movie with ID {tmdb_id}")
                    failed += 1
                    overall_failed += 1
                    
                # Add a delay to avoid hitting rate limits
                if i < len(batch_ids):  # Don't sleep after the last request
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error fetching movie {tmdb_id}: {str(e)}")
                print(f"✗ Failed to fetch movie with ID {tmdb_id}")
                failed += 1
                overall_failed += 1
                # Continue with the next movie
        
        print(f"\nBatch {batch_number} complete:")
        print(f"  Attempted:         {len(batch_ids)}")
        print(f"  Successfully:      {successful}")
        print(f"  Failed:            {failed}")
        
        batch_number += 1
        
        # Optional: Ask user if they want to continue to the next batch
        if end_idx < len(ids_to_fetch):
            try:
                response = input(f"\nContinue to the next batch? (y/n) [y]: ").strip().lower()
                if response and response != 'y':
                    print("Database build paused by user.")
                    break
            except KeyboardInterrupt:
                print("\nDatabase build interrupted by user.")
                break
    
    print("\nDatabase build complete!")
    print(f"CSV files are stored in: {data_dir}")
    print(f"\nSummary:")
    print(f"  Total movies in dataset: {total_ids}")
    print(f"  Previously processed:    {len(processed_ids) - overall_successful}")
    print(f"  Newly processed:         {overall_successful + overall_failed}")
    print(f"  Successfully fetched:    {overall_successful}")
    print(f"  Failed to fetch:         {overall_failed}")
    print(f"  Total movies in database: {len(processed_ids)}")
    print(f"\nFolder structure:")
    print(f"  {data_dir}/")
    print(f"  ├── movies.csv       # Main movie information")
    print(f"  ├── genres.csv       # Movie genres")
    print(f"  ├── cast.csv         # Movie cast members")
    print(f"  ├── crew.csv         # Movie crew members")
    if os.path.exists(links_path):
        print(f"  └── id_mapping.csv   # Mapping between movieId and TMDb IDs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a local TMDb database from TMDb API using environment variables")
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    
    args = parser.parse_args()
    
    # Specify .env file path if provided
    if args.env_file:
        load_dotenv(dotenv_path=args.env_file)
    
    try:
        build_tmdb_database()
    except Exception as e:
        print(f"Error building TMDb database: {str(e)}")