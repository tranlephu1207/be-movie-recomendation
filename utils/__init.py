import os
from typing import Dict, Any, Optional

def get_file_paths() -> Dict[str, str]:
    """
    Get paths to data files from environment variables or use defaults.
    
    Returns:
        Dictionary with file paths
    """
    base_dir = os.environ.get('DATA_DIR', './data')
    
    return {
        'metadata': os.environ.get('METADATA_PATH', f'{base_dir}/movies_metadata.csv'),
        'links': os.environ.get('LINKS_PATH', f'{base_dir}/links_small.csv'),
        'credits': os.environ.get('CREDITS_PATH', f'{base_dir}/credits.csv'),
        'keywords': os.environ.get('KEYWORDS_PATH', f'{base_dir}/keywords.csv'),
        'ratings': os.environ.get('RATINGS_PATH', f'{base_dir}/ratings_small.csv'),
        'models': os.environ.get('MODELS_DIR', './models_saved')
    }

def create_example_rating_data() -> None:
    """
    Create example rating data if it doesn't exist.
    """
    import pandas as pd
    
    file_paths = get_file_paths()
    ratings_path = file_paths['ratings']
    
    if os.path.exists(ratings_path):
        return
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(ratings_path), exist_ok=True)
    
    # Create sample ratings data
    ratings_data = [
        {'userId': 1, 'movieId': 1, 'rating': 4.5},
        {'userId': 1, 'movieId': 2, 'rating': 3.0},
        {'userId': 1, 'movieId': 3, 'rating': 5.0},
        {'userId': 2, 'movieId': 1, 'rating': 3.5},
        {'userId': 2, 'movieId': 4, 'rating': 4.0},
        {'userId': 3, 'movieId': 2, 'rating': 2.5},
        {'userId': 3, 'movieId': 3, 'rating': 4.5},
    ]
    
    pd.DataFrame(ratings_data).to_csv(ratings_path, index=False)
    print(f"Created example ratings data at {ratings_path}")