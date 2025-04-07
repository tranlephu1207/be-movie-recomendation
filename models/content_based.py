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

class ContentBasedRecommender:
    """
    Content-based recommendation system using TF-IDF and cosine similarity.
    """
    
    def __init__(self, metadata_path: str, links_path: str, credits_path: str, keywords_path: str):
        """
        Initialize the content-based recommender.
        
        Args:
            metadata_path: Path to movies metadata CSV
            links_path: Path to links CSV
            credits_path: Path to credits CSV
            keywords_path: Path to keywords CSV
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.metadata_df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.load_data(metadata_path, links_path, credits_path, keywords_path)
        self.process_data()
        self.create_profiles()
        self.build_model()
    
    def load_data(self, metadata_path: str, links_path: str, credits_path: str, keywords_path: str) -> None:
        """
        Load and prepare movie metadata.
        """
        # Load data
        metadata = pd.read_csv(metadata_path)
        links_small = pd.read_csv(links_path)
        credits = pd.read_csv(credits_path)
        keywords = pd.read_csv(keywords_path)
        
        # Process IDs
        links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
        metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce', downcast='integer')
        metadata = metadata.dropna(subset=['id']).astype({'id': 'int'})
        metadata = metadata[metadata['id'].isin(links_small)]
        
        # Merge datasets
        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        metadata['id'] = metadata['id'].astype('int')
        metadata = metadata.merge(credits, on='id')
        metadata = metadata.merge(keywords, on='id')
        
        # Select relevant columns
        self.metadata_df = metadata[['genres', 'id', 'imdb_id', 'overview', 'popularity',
                               'production_companies', 'spoken_languages', 'title',
                               'vote_average', 'vote_count', 'cast', 'crew', 'keywords']]
    
    def process_data(self) -> None:
        """
        Process the metadata to extract relevant features.
        """
        # Parse string data to lists
        features = ['crew', 'cast', 'keywords', 'genres', 'production_companies', 'spoken_languages']
        for feature in features:
            self.metadata_df[feature] = self.metadata_df[feature].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
        
        # Extract director
        self.metadata_df['director'] = self.metadata_df['crew'].apply(self._get_director)
        
        # Get lists of names for certain features
        features = ['cast', 'keywords', 'genres', 'production_companies', 'spoken_languages']
        for feature in features:
            self.metadata_df[feature] = self.metadata_df[feature].apply(self._get_list)
        
        # Drop crew column as we've extracted directors
        self.metadata_df = self.metadata_df.drop(columns='crew')
        
        # Create rating column for weighting
        self.metadata_df['rating'] = (self.metadata_df['vote_average'] * self.metadata_df['popularity']) / self.metadata_df['popularity'].mean()
        
        # Create rating class
        self.metadata_df['rating_class'] = pd.qcut(self.metadata_df['rating'], q=[0, 0.3, 0.6, 1.0], labels=['low', 'medium', 'high'])
    
    def _get_director(self, crew: List[Dict[str, Any]]) -> Union[str, float]:
        """
        Extract director name from crew list.
        """
        if not isinstance(crew, list):
            return np.nan
            
        for person in crew:
            if person.get('job') == 'Director':
                return person.get('name', np.nan)
        return np.nan
    
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
        self.metadata_df['movie_profile'] = self.metadata_df.apply(self._create_movie_profile, axis=1)
        self.metadata_df['movie_profile'] = self.metadata_df['movie_profile'].apply(self._clean_text)
    
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
        """
        if not isinstance(text, str):
            return ""
            
        # Lowercase
        text = text.lower()
        
        # Remove special characters & punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Apply spaCy pipeline
        doc = self.nlp(text)
        
        # Lemmatize + remove stopwords + remove spaces
        tokens = [
            token.lemma_
            for token in doc
            if token.lemma_ not in ENGLISH_STOP_WORDS and not token.is_space
        ]
        
        return ' '.join(tokens)
    
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
        Save the recommender model to disk.
        
        Args:
            path: Directory path to save model
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(path, 'vectorizer.pkl'))
        joblib.dump(self.tfidf_matrix, os.path.join(path, 'tfidf_matrix.pkl'))
        self.metadata_df.to_pickle(os.path.join(path, 'metadata_df.pkl'))
    
    @classmethod
    def load_model(cls, path: str) -> 'ContentBasedRecommender':
        """
        Load the recommender model from disk.
        
        Args:
            path: Directory path to load model from
            
        Returns:
            Loaded recommender instance
        """
        instance = cls.__new__(cls)
        instance.vectorizer = joblib.load(os.path.join(path, 'vectorizer.pkl'))
        instance.tfidf_matrix = joblib.load(os.path.join(path, 'tfidf_matrix.pkl'))
        instance.metadata_df = pd.read_pickle(os.path.join(path, 'metadata_df.pkl'))
        instance.nlp = spacy.load("en_core_web_sm")
        return instance