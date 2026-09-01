"""
Hybrid Recommendation Engine.

Combines content-based and collaborative filtering into a single
recommendation pipeline using a weighted hybrid strategy:

    hybrid_score = α × content_score + (1 - α) × collaborative_score

Key features:
- Normalizes scores from both engines to [0, 1] before combining
- Falls back to content-based for cold-start users (few ratings)
- Provides explainability for each recommendation
"""

import numpy as np
import pandas as pd
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender


class HybridRecommender:
    """Weighted hybrid recommender combining content-based and collaborative filtering."""

    def __init__(self, alpha=0.3, collab_algo="svd"):
        """
        Initialize the hybrid recommender.

        Args:
            alpha: Weight for content-based scores (0-1).
                   Higher α = more content-based influence.
                   Default 0.3 (collaborative filtering dominates).
            collab_algo: Collaborative filtering algorithm ('knn', 'svd', 'nmf')
        """
        self.alpha = alpha
        self.content_engine = ContentBasedRecommender()
        self.collab_engine = CollaborativeRecommender(collab_algo)
        self.movies = None
        self.ratings = None
        self.tags = None
        self._is_fitted = False

    def fit(self, ratings_df, movies_df, tags_df=None):
        """
        Train both recommendation engines.

        Args:
            ratings_df: DataFrame with [userId, movieId, rating]
            movies_df: DataFrame with [movieId, title, genres, genre_list]
            tags_df: Optional DataFrame with [movieId, tag]
        """
        self.movies = movies_df
        self.ratings = ratings_df
        self.tags = tags_df

        print("Training content-based engine...")
        self.content_engine.fit(movies_df, tags_df)

        print("Training collaborative filtering engine...")
        self.collab_engine.fit(ratings_df)

        self._is_fitted = True
        print("[OK] Hybrid recommender ready!")
        return self

    def recommend(self, user_id, n=10, alpha=None):
        """
        Generate hybrid recommendations for a user.

        Args:
            user_id: The userId
            n: Number of recommendations
            alpha: Override the default alpha (optional)

        Returns:
            pd.DataFrame with [movieId, title, genres, content_score,
                                collab_score, hybrid_score]
        """
        self._check_fitted()
        alpha = alpha if alpha is not None else self.alpha

        # Check if user has enough ratings for collaborative filtering
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        is_cold_start = len(user_ratings) < 5

        if is_cold_start:
            # Cold start: rely mostly on content-based
            alpha = 0.9
            print(f"  Cold-start user ({len(user_ratings)} ratings) — "
                  f"using α={alpha} (content-heavy)")

        # Get content-based scores
        content_recs = self.content_engine.recommend_for_user(
            user_id, self.ratings, n=n * 3  # Get extra candidates
        )

        # Get collaborative filtering scores
        collab_recs = self.collab_engine.recommend_for_user(
            user_id, self.ratings, self.movies, n=n * 3
        )

        # Merge scores on movieId
        if len(content_recs) == 0 and len(collab_recs) == 0:
            return pd.DataFrame(columns=[
                "movieId", "title", "genres",
                "content_score", "collab_score", "hybrid_score"
            ])

        # Normalize scores to [0, 1]
        if len(content_recs) > 0 and content_recs["content_score"].max() > 0:
            content_recs["content_score_norm"] = (
                content_recs["content_score"] / content_recs["content_score"].max()
            )
        else:
            content_recs["content_score_norm"] = 0

        if len(collab_recs) > 0:
            # Normalize predicted ratings from [0.5, 5] to [0, 1]
            collab_recs["collab_score_norm"] = (
                (collab_recs["predicted_rating"] - 0.5) / 4.5
            )
        else:
            collab_recs["collab_score_norm"] = 0

        # Combine using full outer join
        content_scores = content_recs[["movieId", "content_score_norm"]].copy()
        collab_scores = collab_recs[["movieId", "collab_score_norm"]].copy()

        combined = pd.merge(
            content_scores, collab_scores,
            on="movieId", how="outer"
        )
        combined["content_score_norm"] = combined["content_score_norm"].fillna(0)
        combined["collab_score_norm"] = combined["collab_score_norm"].fillna(0)

        # Weighted hybrid score
        combined["hybrid_score"] = (
            alpha * combined["content_score_norm"] +
            (1 - alpha) * combined["collab_score_norm"]
        )

        # Sort by hybrid score and get top-N
        combined = combined.sort_values("hybrid_score", ascending=False).head(n)

        # Enrich with movie metadata
        result = combined.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId", how="left"
        )

        result = result.rename(columns={
            "content_score_norm": "content_score",
            "collab_score_norm": "collab_score",
        })

        result = result[["movieId", "title", "genres",
                         "content_score", "collab_score", "hybrid_score"]]

        # Round scores
        for col in ["content_score", "collab_score", "hybrid_score"]:
            result[col] = result[col].round(4)

        return result.reset_index(drop=True)

    def explain(self, user_id, movie_id):
        """
        Explain why a movie was recommended for a user.

        Args:
            user_id: The userId
            movie_id: The movieId

        Returns:
            dict with explanation details
        """
        self._check_fitted()

        movie_match = self.movies[self.movies["movieId"] == movie_id]
        if len(movie_match) == 0:
            return {"error": f"Movie {movie_id} not found"}

        movie = movie_match.iloc[0]

        # Content-based: similar to which of the user's liked movies?
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= 3.5].merge(
            self.movies[["movieId", "title"]], on="movieId"
        )

        # Find most similar liked movie
        similar_liked = []
        if movie_id in self.content_engine.movie_idx:
            target_idx = self.content_engine.movie_idx[movie_id]
            for _, row in liked.iterrows():
                mid = row["movieId"]
                if mid in self.content_engine.movie_idx:
                    src_idx = self.content_engine.movie_idx[mid]
                    sim = self.content_engine.similarity_matrix[target_idx][src_idx]
                    if sim > 0.1:
                        similar_liked.append({
                            "title": row["title"],
                            "similarity": round(sim, 4),
                        })
            similar_liked.sort(key=lambda x: x["similarity"], reverse=True)

        # Collaborative: predicted rating
        predicted_rating = self.collab_engine.predict(user_id, movie_id)

        # Content features
        features = self.content_engine.get_movie_features(movie_id)

        return {
            "movie": movie["title"],
            "genres": movie["genres"],
            "predicted_rating": round(predicted_rating, 2),
            "similar_to_liked_movies": similar_liked[:5],
            "key_features": dict(list(features.items())[:10]),
            "explanation": (
                f"Recommended because: predicted rating {predicted_rating:.1f}/5 "
                f"and similar to {len(similar_liked)} movies you liked."
            ),
        }

    def _check_fitted(self):
        """Ensure the model has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call fit(ratings_df, movies_df, tags_df) first."
            )
