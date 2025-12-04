import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """
    Content-based movie recommender using TF-IDF + cosine similarity.
    """

    def __init__(self, csv_path: str):
        """
        Initialize recommender with path to movies CSV.
        CSV should contain at least: title, overview.
        Optional: genres, keywords.
        """
        self.csv_path = csv_path
        self.df = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.title_to_index = {}
        self.vectorizer = None

    def load_data(self):
        """Load movies dataset from CSV and prepare basic columns."""
        self.df = pd.read_csv(self.csv_path)

        required_cols = ["title", "overview"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")

        for col in ["genres", "keywords"]:
            if col not in self.df.columns:
                self.df[col] = ""

        self.df[["title", "overview", "genres", "keywords"]] = self.df[
            ["title", "overview", "genres", "keywords"]
        ].fillna("")

    def build_corpus(self):
        """
        Create a combined text 'corpus' per movie using overview + genres + keywords.
        This is what we will feed into TF-IDF.
        """
        self.df["corpus"] = (
            self.df["overview"].astype(str)
            + " " + self.df["genres"].astype(str)
            + " " + self.df["keywords"].astype(str)
        )

    def vectorize(self, max_features: int = 5000):
        """Convert text corpus into TF-IDF feature vectors."""
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["corpus"])

    def compute_similarity(self):
        """Compute cosine similarity between all movies."""
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Build title → index mapping (case-insensitive)
        self.title_to_index = {
            str(title).lower(): idx for idx, title in enumerate(self.df["title"])
        }

    def fit(self):
        """
        Run full pipeline:
        1) Load data
        2) Build text corpus
        3) Vectorize with TF-IDF
        4) Compute cosine similarity matrix
        """
        print("[INFO] Loading data...")
        self.load_data()
        print("[INFO] Building text corpus...")
        self.build_corpus()
        print("[INFO] Vectorizing with TF-IDF...")
        self.vectorize()
        print("[INFO] Computing similarity matrix...")
        self.compute_similarity()
        print("[INFO] Recommender is ready! ✅")

    def recommend(self, movie_title: str, top_n: int = 10):
        """
        Recommend top_n similar movies for the given movie_title.
        Returns a list of (title, similarity_score).
        """
        if self.similarity_matrix is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        title_key = movie_title.lower().strip()
        if title_key not in self.title_to_index:
            raise ValueError(f"Movie '{movie_title}' not found in dataset.")

        idx = self.title_to_index[title_key]

        # Get similarity scores for this movie
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort movies based on similarity score (high → low)
        similarity_scores = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True
        )

        # Skip first (itself), then take top_n
        top_scores = similarity_scores[1 : top_n + 1]

        recommendations = []
        for movie_idx, score in top_scores:
            title = self.df.iloc[movie_idx]["title"]
            recommendations.append((title, float(score)))

        return recommendations
