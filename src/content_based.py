"""
Content-Based Filtering Engine.

Uses TF-IDF vectorization on movie genres and user-generated tags
to compute item-item similarity via cosine similarity.

Approach:
1. Combine genres + tags into a single text feature per movie
2. Apply TF-IDF vectorization
3. Compute cosine similarity matrix
4. Recommend similar movies or personalized recommendations for users
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Content-based movie recommender using TF-IDF + cosine similarity."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movies = None
        self.movie_idx = {}  # movieId → matrix index
        self.idx_movie = {}  # matrix index → movieId
        self._is_fitted = False

    def fit(self, movies_df, tags_df=None):
        """
        Build the TF-IDF model and compute similarity matrix.

        Args:
            movies_df: DataFrame with columns [movieId, title, genres, genre_list]
            tags_df: Optional DataFrame with columns [movieId, tag]
        """
        self.movies = movies_df.copy()

        # Build text features: genres + tags combined
        self.movies["text_features"] = self.movies["genre_list"].apply(
            lambda x: " ".join(x)
        )

        # Aggregate tags per movie and append to text features
        if tags_df is not None and len(tags_df) > 0:
            tag_agg = (
                tags_df.groupby("movieId")["tag"]
                .apply(lambda x: " ".join(x))
                .reset_index()
            )
            tag_agg.columns = ["movieId", "tags_text"]
            self.movies = self.movies.merge(tag_agg, on="movieId", how="left")
            self.movies["tags_text"] = self.movies["tags_text"].fillna("")
            self.movies["text_features"] = (
                self.movies["text_features"] + " " + self.movies["tags_text"]
            )

        # Build index mappings
        self.movies = self.movies.reset_index(drop=True)
        self.movie_idx = {
            mid: idx for idx, mid in enumerate(self.movies["movieId"])
        }
        self.idx_movie = {
            idx: mid for mid, idx in self.movie_idx.items()
        }

        # TF-IDF vectorization
        self.tfidf_matrix = self.tfidf.fit_transform(
            self.movies["text_features"]
        )

        # Compute cosine similarity (this is the most memory-intensive step)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        self._is_fitted = True
        return self

    def recommend_similar(self, movie_id, n=10):
        """
        Get top-N most similar movies to a given movie.

        Args:
            movie_id: The movieId to find similar movies for
            n: Number of recommendations to return

        Returns:
            pd.DataFrame with columns [movieId, title, genres, similarity_score]
        """
        self._check_fitted()

        if movie_id not in self.movie_idx:
            return pd.DataFrame(columns=["movieId", "title", "genres", "similarity_score"])

        idx = self.movie_idx[movie_id]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity (descending), exclude the movie itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:n]

        # Build result DataFrame
        results = []
        for i, score in sim_scores:
            mid = self.idx_movie[i]
            movie_row = self.movies[self.movies["movieId"] == mid].iloc[0]
            results.append({
                "movieId": mid,
                "title": movie_row["title"],
                "genres": movie_row["genres"],
                "similarity_score": round(score, 4),
            })

        return pd.DataFrame(results)

    def recommend_for_user(self, user_id, ratings_df, n=10):
        """
        Generate personalized recommendations for a user based on their
        rating history. Aggregates similarity scores from highly-rated movies.

        Args:
            user_id: The userId to generate recommendations for
            ratings_df: DataFrame with columns [userId, movieId, rating]
            n: Number of recommendations to return

        Returns:
            pd.DataFrame with columns [movieId, title, genres, content_score]
        """
        self._check_fitted()

        # Get user's rated movies (only those rated >= 3.5)
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if len(user_ratings) == 0:
            return pd.DataFrame(columns=["movieId", "title", "genres", "content_score"])

        liked_movies = user_ratings[user_ratings["rating"] >= 3.5]
        if len(liked_movies) == 0:
            # Fall back to all rated movies if none rated >= 3.5
            liked_movies = user_ratings

        # Compute weighted average similarity scores
        all_movie_ids = set(self.movies["movieId"])
        rated_movie_ids = set(user_ratings["movieId"])
        candidate_ids = all_movie_ids - rated_movie_ids

        scores = {}
        for _, row in liked_movies.iterrows():
            mid = row["movieId"]
            if mid not in self.movie_idx:
                continue
            idx = self.movie_idx[mid]
            weight = row["rating"] / 5.0  # Normalize rating as weight

            for cand_id in candidate_ids:
                if cand_id not in self.movie_idx:
                    continue
                cand_idx = self.movie_idx[cand_id]
                sim = self.similarity_matrix[idx][cand_idx]
                scores[cand_id] = scores.get(cand_id, 0) + sim * weight

        # Normalize scores to [0, 1]
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        # Sort and return top-N
        top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

        results = []
        for mid, score in top_movies:
            movie_row = self.movies[self.movies["movieId"] == mid].iloc[0]
            results.append({
                "movieId": mid,
                "title": movie_row["title"],
                "genres": movie_row["genres"],
                "content_score": round(score, 4),
            })

        return pd.DataFrame(results)

    def get_movie_features(self, movie_id):
        """Get the TF-IDF feature names and weights for a movie."""
        self._check_fitted()

        if movie_id not in self.movie_idx:
            return {}

        idx = self.movie_idx[movie_id]
        feature_names = self.tfidf.get_feature_names_out()
        tfidf_scores = self.tfidf_matrix[idx].toarray().flatten()

        # Return top features with non-zero weights
        features = {
            feature_names[i]: round(tfidf_scores[i], 4)
            for i in tfidf_scores.argsort()[::-1]
            if tfidf_scores[i] > 0
        }
        return dict(list(features.items())[:20])  # Top 20 features

    def _check_fitted(self):
        """Ensure the model has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call fit(movies_df, tags_df) first."
            )
