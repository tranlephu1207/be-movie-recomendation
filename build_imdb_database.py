import os
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from imdb_data_fetcher import IMDbDataFetcher

def extract_imdb_ids_from_links(links_path):
    """
    Extract IMDb IDs from links.csv.
    
    Args:
        links_path: Path to links.csv file
    
    Returns:
        List of IMDb IDs
    """
    try:
        # Load links.csv
        links_df = pd.read_csv(links_path)
        
        # Ensure imdbId column exists
        if 'imdbId' not in links_df.columns:
            raise ValueError(f"File {links_path} does not contain an 'imdbId' column")
        
        # Extract IMDb IDs
        imdb_ids = links_df['imdbId'].dropna().astype(str).tolist()
        
        # Format IMDb IDs with 'tt' prefix if needed
        formatted_ids = []
        for imdb_id in imdb_ids:
            # Add leading zeros if needed to make it 7 digits
            imdb_id = imdb_id.zfill(7)
            
            # Add 'tt' prefix if not already present
            if not imdb_id.startswith('tt'):
                imdb_id = f"tt{imdb_id}"
            
            formatted_ids.append(imdb_id)
        
        return formatted_ids
    
    except Exception as e:
        print(f"Error extracting IMDb IDs from {links_path}: {str(e)}")
        return []

def create_id_mapping(links_path, output_path):
    """
    Create a mapping between movieId and IMDb IDs.
    
    Args:
        links_path: Path to links.csv file
        output_path: Path to save the mapping
    """
    try:
        # Load links.csv
        links_df = pd.read_csv(links_path)
        
        # Ensure required columns exist
        required_columns = ['movieId', 'imdbId']
        for col in required_columns:
            if col not in links_df.columns:
                raise ValueError(f"File {links_path} does not contain a '{col}' column")
        
        # Select only movieId and imdbId columns
        mapping_df = links_df[required_columns].copy()
        
        # Format IMDb IDs with 'tt' prefix
        mapping_df['imdb_id'] = mapping_df['imdbId'].astype(str).apply(
            lambda x: f"tt{x.zfill(7)}" if not str(x).startswith('tt') else x
        )
        
        # Save the mapping
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mapping_df.to_csv(output_path, index=False)
        print(f"Created ID mapping at {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Error creating ID mapping: {str(e)}")
        return False

def build_imdb_database(api_key, links_path=None, movie_list_file=None, imdb_ids=None, data_dir='./data', delay=1.0, batch_size=None):
    """
    Build a local IMDb database from scratch using the OMDb API.
    
    Args:
        api_key: OMDb API key
        links_path: Path to links.csv file
        movie_list_file: File containing IMDb IDs (one per line)
        imdb_ids: List of IMDb IDs
        data_dir: Directory to store the database
        delay: Delay between API requests in seconds
        batch_size: Number of movies to process in one batch (None = all)
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize IMDb data fetcher
    fetcher = IMDbDataFetcher(api_key=api_key, data_dir=data_dir)
    
    # Collect IMDb IDs to fetch
    ids_to_fetch = []
    
    # Priority 1: Use links.csv if provided
    if links_path and os.path.exists(links_path):
        print(f"Extracting IMDb IDs from {links_path}...")
        extracted_ids = extract_imdb_ids_from_links(links_path)
        if extracted_ids:
            ids_to_fetch = extracted_ids
            print(f"Found {len(ids_to_fetch)} IMDb IDs in links.csv")
            
            # Create ID mapping
            mapping_path = os.path.join(data_dir, 'id_mapping.csv')
            create_id_mapping(links_path, mapping_path)
    
    # Priority 2: Load from file if provided and links.csv wasn't used
    elif movie_list_file and os.path.exists(movie_list_file):
        print(f"Loading IMDb IDs from {movie_list_file}...")
        with open(movie_list_file, 'r') as f:
            ids_to_fetch = [line.strip() for line in f.readlines()]
    
    # Priority 3: Use provided list if available and previous methods weren't used
    elif imdb_ids:
        ids_to_fetch = imdb_ids
    
    # Priority 4: If no IDs provided, use a default sample list
    else:
        print("No IMDb IDs provided, using default sample list...")
        ids_to_fetch = [
            # Popular movies from different genres
            "tt0111161",  # The Shawshank Redemption
            "tt0068646",  # The Godfather
            "tt0071562",  # The Godfather: Part II
            "tt0468569",  # The Dark Knight
            "tt0050083",  # 12 Angry Men
            "tt0108052",  # Schindler's List
            "tt0167260",  # The Lord of the Rings: The Return of the King
            "tt0110912",  # Pulp Fiction
            "tt0060196",  # The Good, the Bad and the Ugly
            "tt0109830",  # Forrest Gump
            "tt0114709",  # Toy Story
            "tt0120737",  # The Lord of the Rings: The Fellowship of the Ring
            "tt1375666",  # Inception
            "tt0133093",  # The Matrix
            "tt0080684",  # Star Wars: Episode V - The Empire Strikes Back
            "tt0099685",  # Goodfellas
            "tt0073486",  # One Flew Over the Cuckoo's Nest
            "tt0167261",  # The Lord of the Rings: The Two Towers
            "tt0076759",  # Star Wars: Episode IV - A New Hope
            "tt0120689",  # The Green Mile
        ]
    
    # Apply batch size if specified
    if batch_size and batch_size > 0 and batch_size < len(ids_to_fetch):
        print(f"Processing first {batch_size} movies (out of {len(ids_to_fetch)} total)")
        ids_to_fetch = ids_to_fetch[:batch_size]
    
    # Remove duplicates while preserving order
    ids_to_fetch = list(dict.fromkeys(ids_to_fetch))
    
    # Fetch movies
    total_ids = len(ids_to_fetch)
    print(f"Starting to fetch {total_ids} movies...")
    
    successful = 0
    failed = 0
    
    for i, imdb_id in enumerate(ids_to_fetch, 1):
        try:
            print(f"[{i}/{total_ids}] Fetching movie {imdb_id}...")
            movie = fetcher.fetch_movie_by_imdb_id(imdb_id)
            
            if movie:
                print(f"✓ Successfully fetched: {movie.get('Title', 'Unknown')} ({movie.get('Year', 'N/A')})")
                successful += 1
            else:
                print(f"✗ Failed to fetch movie with ID {imdb_id}")
                failed += 1
                
            # Add a delay to avoid hitting rate limits
            if i < total_ids:  # Don't sleep after the last request
                time.sleep(delay)
                
        except Exception as e:
            print(f"✗ Error fetching movie {imdb_id}: {str(e)}")
            failed += 1
            # Continue with the next movie
    
    print("\nDatabase build complete!")
    print(f"CSV files are stored in: {data_dir}")
    print(f"\nSummary:")
    print(f"  Total movies attempted: {total_ids}")
    print(f"  Successfully fetched:   {successful}")
    print(f"  Failed to fetch:        {failed}")
    print(f"\nFolder structure:")
    print(f"  {data_dir}/")
    print(f"  ├── movies.csv       # Main movie information")
    print(f"  ├── genres.csv       # Movie genres")
    print(f"  ├── cast.csv         # Movie cast members")
    print(f"  ├── ratings.csv      # External ratings")
    if links_path:
        print(f"  └── id_mapping.csv   # Mapping between movieId and IMDb IDs")

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Get environment variables
    default_links_path = os.environ.get('FULL_MOVIE_LINKS', './data/recommendation_data/links.csv')
    
    parser = argparse.ArgumentParser(description="Build a local IMDb database from OMDb API")
    parser.add_argument("--api-key", required=True, help="OMDb API key")
    parser.add_argument("--links", default=default_links_path, help=f"Path to links.csv (default: {default_links_path})")
    parser.add_argument("--file", help="File containing IMDb IDs (one per line)")
    parser.add_argument("--ids", nargs="+", help="List of IMDb IDs to fetch")
    parser.add_argument("--data-dir", default="./data", help="Directory to store the database (default: ./data)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API requests in seconds (default: 1.0)")
    parser.add_argument("--batch-size", type=int, help="Number of movies to process in one batch")
    
    args = parser.parse_args()
    
    build_imdb_database(
        api_key=args.api_key,
        links_path=args.links,
        movie_list_file=args.file,
        imdb_ids=args.ids,
        data_dir=args.data_dir,
        delay=args.delay,
        batch_size=args.batch_size
    )