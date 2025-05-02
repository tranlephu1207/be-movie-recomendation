import pandas as pd
import numpy as np
import re
import joblib
import os
from ast import literal_eval
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Union, Optional
from utils.gcloud import init_credentials, get_storage_client, download_blob, upload_blob

class ContentBasedRecommender:
    """
    Content-based recommendation system using TF-IDF and cosine similarity.
    """
    
    def __init__(self, metadata_path: str, links_path: str, credits_path: str, keywords_path: str, bucket_name: str = None):
        """
        Initialize the content-based recommender.
        
        Args:
            metadata_path: Path to movies metadata CSV
            links_path: Path to links CSV
            credits_path: Path to credits CSV
            keywords_path: Path to keywords CSV
            bucket_name: Name of the Google Cloud Storage bucket
        """
        try:
            # Initialize spaCy with fallback
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"Warning: Could not load spaCy model: {str(e)}")
                print("Attempting to download spaCy model...")
                try:
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception as download_error:
                    print(f"Could not download spaCy model: {str(download_error)}")
                    self.nlp = None

            self.metadata_df = None
            self.tfidf_matrix = None
            self.vectorizer = None
            self.bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
            if not self.bucket_name:
                raise ValueError("Bucket name must be provided or GCS_BUCKET_NAME environment variable must be set")
            
            # Initialize GCS client
            credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable must be set")
            
            try:
                init_credentials(credentials_path)
                self.storage_client = get_storage_client()
                self.bucket = self.storage_client.bucket(self.bucket_name)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Google Cloud Storage: {e}")
            
            self.load_data(metadata_path, links_path, credits_path, keywords_path)
            self.process_data()
            self.create_profiles()
            self.build_model()
        except Exception as e:
            print(f"Error initializing recommender: {str(e)}")
            raise
    
    def load_data(self, metadata_path: str, links_path: str, credits_path: str, keywords_path: str) -> None:
        """
        Load and prepare movie metadata.
        Handle missing credits file gracefully.
        """
        try:
            # Load required data
            metadata = pd.read_csv(metadata_path)
            links_small = pd.read_csv(links_path)
            
            # Process IDs
            links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
            metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce', downcast='integer')
            metadata = metadata.dropna(subset=['id']).astype({'id': 'int'})
            metadata = metadata[metadata['id'].isin(links_small)]
            
            # Try to load optional data
            try:
                credits = pd.read_csv(credits_path)
                credits['id'] = credits['id'].astype('int')
                metadata = metadata.merge(credits, on='id', how='left')
            except Exception as e:
                print(f"Warning: Could not load credits data: {str(e)}")
            
            try:
                keywords = pd.read_csv(keywords_path)
                keywords['id'] = keywords['id'].astype('int')
                metadata = metadata.merge(keywords, on='id', how='left')
            except Exception as e:
                print(f"Warning: Could not load keywords data: {str(e)}")
            
            # Select relevant columns that exist
            required_columns = ['genres', 'id', 'imdb_id', 'overview', 'popularity',
                              'production_companies', 'spoken_languages', 'title',
                              'vote_average', 'vote_count']
            optional_columns = ['cast', 'crew', 'keywords']
            
            # Get all available columns
            available_columns = [col for col in required_columns + optional_columns 
                               if col in metadata.columns]
            
            self.metadata_df = metadata[available_columns]
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def process_data(self) -> None:
        """
        Process the metadata to extract relevant features.
        Handles missing columns gracefully.
        """
        try:
            # Parse string data to lists with error handling
            features = ['cast', 'keywords', 'genres', 'production_companies', 'spoken_languages']
            
            # Handle crew separately since it's optional
            if 'crew' in self.metadata_df.columns:
                self.metadata_df['crew'] = self.metadata_df['crew'].apply(
                    lambda x: literal_eval(x) if isinstance(x, str) and pd.notna(x) else []
                )
                # Extract director from crew
                self.metadata_df['director'] = self.metadata_df['crew'].apply(self._get_director)
                # Drop crew column as we've extracted directors
                self.metadata_df = self.metadata_df.drop(columns='crew')
            else:
                # If no crew data, set default director
                self.metadata_df['director'] = 'Unknown Director'
            
            # Process other features
            for feature in features:
                if feature in self.metadata_df.columns:
                    self.metadata_df[feature] = self.metadata_df[feature].apply(
                        lambda x: literal_eval(x) if isinstance(x, str) and pd.notna(x) else []
                    )
                else:
                    print(f"Warning: {feature} column not found. Setting empty lists.")
                    self.metadata_df[feature] = [[]] * len(self.metadata_df)
            
            # Get lists of names for certain features
            for feature in features:
                self.metadata_df[feature] = self.metadata_df[feature].apply(self._get_list)
            
            # Handle rating calculation with missing columns
            if 'vote_average' not in self.metadata_df.columns:
                print("Warning: vote_average column not found. Setting default values.")
                self.metadata_df['vote_average'] = 0.0
            
            if 'popularity' not in self.metadata_df.columns:
                print("Warning: popularity column not found. Setting default values.")
                self.metadata_df['popularity'] = 1.0  # neutral popularity
            
            # Create rating column for weighting with safe defaults
            vote_average = self.metadata_df['vote_average'].fillna(0)
            popularity = self.metadata_df['popularity'].fillna(1)  # use 1 as neutral popularity
            mean_popularity = popularity.mean() if len(popularity) > 0 else 1
            
            # Avoid division by zero
            self.metadata_df['rating'] = (vote_average * popularity) / (mean_popularity if mean_popularity != 0 else 1)
            
            # Create rating class with safe handling
            try:
                self.metadata_df['rating_class'] = pd.qcut(
                    self.metadata_df['rating'].fillna(0), 
                    q=[0, 0.3, 0.6, 1.0], 
                    labels=['low', 'medium', 'high']
                )
            except ValueError as e:
                print("Warning: Could not create rating classes. Setting default class.")
                self.metadata_df['rating_class'] = 'medium'  # default rating class
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            raise
    
    def _get_director(self, crew: List[Dict[str, Any]]) -> Union[str, float]:
        """
        Extract director name from crew list.
        Returns 'Unknown Director' if no crew data or no director found.
        """
        if not isinstance(crew, list) or not crew:
            return 'Unknown Director'
            
        for person in crew:
            if isinstance(person, dict) and person.get('job') == 'Director':
                return person.get('name', 'Unknown Director')
        return 'Unknown Director'
    
    def _get_list(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        Extract names from a list of dictionaries.
        """
        if not isinstance(items, list):
            return []
            
        names = [item.get('name', '') for item in items if 'name' in item]
        if len(names) > 5:
            names = names[:5]
        return names
    
    def create_profiles(self) -> None:
        """
        Create movie profiles by combining features.
        """
        try:
            self.metadata_df['movie_profile'] = self.metadata_df.apply(self._create_movie_profile, axis=1)
            
            # Clean text with error handling
            cleaned_profiles = []
            for text in self.metadata_df['movie_profile']:
                try:
                    cleaned_text = self._clean_text(text)
                    cleaned_profiles.append(cleaned_text)
                except Exception as e:
                    print(f"Warning: Error cleaning text: {str(e)}")
                    cleaned_profiles.append("")
                
            self.metadata_df['movie_profile'] = cleaned_profiles
            
        except Exception as e:
            print(f"Error creating profiles: {str(e)}")
            # Set empty profiles as fallback
            self.metadata_df['movie_profile'] = [""] * len(self.metadata_df)
    
    def _create_movie_profile(self, row: pd.Series) -> str:
        """
        Create a text profile for a movie by combining different features.
        """
        genres = ' '.join(row['genres']) if isinstance(row['genres'], list) else ''
        keywords = ' '.join(row['keywords']) if isinstance(row['keywords'], list) else ''
        overview = row['overview'] if isinstance(row['overview'], str) else ''
        cast = ' '.join(row['cast']) if isinstance(row['cast'], list) else ''
        production_companies = ' '.join(row['production_companies']) if isinstance(row['production_companies'], list) else ''
        spoken_languages = ' '.join(row['spoken_languages']) if isinstance(row['spoken_languages'], list) else ''
        director = row['director'] if isinstance(row['director'], str) else ''
        rating_class = row['rating_class'] if isinstance(row['rating_class'], str) else ''
        return f"{genres} {overview} {keywords} {cast} {production_companies} {spoken_languages} {director} {rating_class}"
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for TF-IDF vectorization.
        Falls back to basic cleaning if spaCy is not available.
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters & punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        if self.nlp is not None:
            try:
                # Apply spaCy pipeline
                doc = self.nlp(text)
                
                # Lemmatize + remove stopwords + remove spaces
                tokens = [
                    token.lemma_
                    for token in doc
                    if token.lemma_ not in ENGLISH_STOP_WORDS and not token.is_space
                ]
                return ' '.join(tokens)
            except Exception as e:
                print(f"Warning: SpaCy processing failed: {str(e)}. Using basic cleaning.")
        
        # Fallback: basic cleaning without spaCy
        words = text.split()
        # Remove common English stop words
        words = [w for w in words if w.lower() not in ENGLISH_STOP_WORDS]
        return ' '.join(words)
    
    def build_model(self) -> None:
        """
        Build the TF-IDF model.
        """
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.metadata_df['movie_profile'])
    
    def get_recommendations(self, query: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get movie recommendations based on a text query.
        
        Args:
            query: Text query to find similar movies
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with top recommendations
        """
        # Clean and vectorize query
        query = self._clean_text(query)
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarity
        cos_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top recommendations
        top_indices = cos_sim.argsort()[-top_n:][::-1]
        
        # Return movie info
        return self.metadata_df.iloc[top_indices][['id', 'title', 'vote_average', 'genres']]
    
    def get_movie_by_id(self, movie_id: int) -> Optional[pd.Series]:
        """
        Get movie information by ID.
        
        Args:
            movie_id: Movie ID
            
        Returns:
            Movie information or None if not found
        """
        movie = self.metadata_df[self.metadata_df['id'] == movie_id]
        if len(movie) == 0:
            return None
        return movie.iloc[0]
        
    def get_movie_by_imdb_id(self, imdb_id: str) -> Optional[pd.Series]:
        """
        Get movie information by IMDb ID.
        
        Args:
            imdb_id: IMDb ID (e.g., 'tt0114709')
            
        Returns:
            Movie information or None if not found
        """
        # Add 'tt' prefix if not already present
        if imdb_id and not imdb_id.startswith('tt'):
            imdb_id = f"tt{imdb_id}"
            
        movie = self.metadata_df[self.metadata_df['imdb_id'] == imdb_id]
        if len(movie) == 0:
            return None
        return movie.iloc[0]
    
    def save_model(self, path: str) -> None:
        """
        Save the recommender model to Google Cloud Storage.
        
        Args:
            path: Path in the bucket where to save the model
        """
        try:
            # Create temporary directory for model files
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save model components to temporary files
                vectorizer_path = os.path.join(temp_dir, 'vectorizer.pkl')
                tfidf_path = os.path.join(temp_dir, 'tfidf_matrix.pkl')
                metadata_path = os.path.join(temp_dir, 'metadata_df.pkl')
                
                joblib.dump(self.vectorizer, vectorizer_path)
                joblib.dump(self.tfidf_matrix, tfidf_path)
                self.metadata_df.to_pickle(metadata_path)
                
                # Upload files to GCS
                upload_blob(self.bucket_name, vectorizer_path, f"{path}/vectorizer.pkl")
                upload_blob(self.bucket_name, tfidf_path, f"{path}/tfidf_matrix.pkl")
                upload_blob(self.bucket_name, metadata_path, f"{path}/metadata_df.pkl")
                
        except Exception as e:
            print(f"Error saving model to GCS: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, path: str, bucket_name: str = None) -> 'ContentBasedRecommender':
        """
        Load the recommender model from Google Cloud Storage.
        
        Args:
            path: Path in the bucket where the model is stored
            bucket_name: Name of the Google Cloud Storage bucket
            
        Returns:
            Loaded recommender instance
        """
        try:
            # Get bucket name
            bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
            if not bucket_name:
                raise ValueError("Bucket name must be provided or GCS_BUCKET_NAME environment variable must be set")
            
            # Create temporary directory for downloaded files
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download model components
                vectorizer_path = os.path.join(temp_dir, 'vectorizer.pkl')
                tfidf_path = os.path.join(temp_dir, 'tfidf_matrix.pkl')
                metadata_path = os.path.join(temp_dir, 'metadata_df.pkl')
                
                download_blob(bucket_name, f"{path}/vectorizer.pkl", vectorizer_path)
                download_blob(bucket_name, f"{path}/tfidf_matrix.pkl", tfidf_path)
                download_blob(bucket_name, f"{path}/metadata_df.pkl", metadata_path)
                
                # Create instance and load model components
                instance = cls.__new__(cls)
                instance.vectorizer = joblib.load(vectorizer_path)
                instance.tfidf_matrix = joblib.load(tfidf_path)
                instance.metadata_df = pd.read_pickle(metadata_path)
                instance.nlp = spacy.load("en_core_web_sm")
                instance.bucket_name = bucket_name
                
                # Initialize GCS client
                credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                if not credentials_path:
                    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable must be set")
                
                init_credentials(credentials_path)
                instance.storage_client = get_storage_client()
                instance.bucket = instance.storage_client.bucket(bucket_name)
                
                return instance
                
        except Exception as e:
            print(f"Error loading model from GCS: {str(e)}")
            raise