"""
Collaborative Filtering Engine.

Implements and compares multiple collaborative filtering algorithms
using the `surprise` library:
- KNN Basic (memory-based, user-user / item-item)
- SVD (Singular Value Decomposition — matrix factorization)
- NMF (Non-negative Matrix Factorization)

Each algorithm predicts ratings for unseen user-movie pairs, then
recommends the top-N highest predicted-rating movies.
"""

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD, NMF
from surprise.model_selection import cross_validate as surprise_cv
from collections import defaultdict


# Available algorithms
ALGORITHMS = {
    "knn": {
        "name": "KNN Basic",
        "class": KNNBasic,
        "params": {
            "k": 40,
            "sim_options": {"name": "cosine", "user_based": True},
            "verbose": False,
        },
    },
    "svd": {
        "name": "SVD",
        "class": SVD,
        "params": {
            "n_factors": 100,
            "n_epochs": 20,
            "lr_all": 0.005,
            "reg_all": 0.02,
            "verbose": False,
        },
    },
    "nmf": {
        "name": "NMF",
        "class": NMF,
        "params": {
            "n_factors": 15,
            "n_epochs": 50,
            "verbose": False,
        },
    },
}


class CollaborativeRecommender:
    """Collaborative filtering recommender using the surprise library."""

    def __init__(self, algo_name="svd"):
        """
        Initialize with a specific algorithm.

        Args:
            algo_name: One of 'knn', 'svd', 'nmf'
        """
        if algo_name not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algo_name}'. "
                f"Choose from: {list(ALGORITHMS.keys())}"
            )

        self.algo_name = algo_name
        self.algo_config = ALGORITHMS[algo_name]
        self.model = self.algo_config["class"](**self.algo_config["params"])
        self.trainset = None
        self._is_fitted = False

    def fit(self, ratings_df):
        """
        Train the collaborative filtering model.

        Args:
            ratings_df: DataFrame with columns [userId, movieId, rating]
        """
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings_df[["userId", "movieId", "rating"]], reader
        )
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)
        self._is_fitted = True
        return self

    def predict(self, user_id, movie_id):
        """
        Predict a rating for a user-movie pair.

        Args:
            user_id: The userId
            movie_id: The movieId

        Returns:
            float — predicted rating
        """
        self._check_fitted()
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est

    def recommend_for_user(self, user_id, ratings_df, movies_df, n=10):
        """
        Generate top-N recommendations for a user.

        Predicts ratings for all movies the user hasn't rated,
        then returns the top-N highest predicted.

        Args:
            user_id: The userId
            ratings_df: Full ratings DataFrame
            movies_df: Movies DataFrame (for titles/genres)
            n: Number of recommendations

        Returns:
            pd.DataFrame with [movieId, title, genres, predicted_rating]
        """
        self._check_fitted()

        # Get movies the user has already rated
        rated_movies = set(
            ratings_df[ratings_df["userId"] == user_id]["movieId"]
        )

        # Get all movie IDs
        all_movies = set(movies_df["movieId"])
        candidates = all_movies - rated_movies

        # Predict ratings for all candidates
        predictions = []
        for mid in candidates:
            pred = self.model.predict(user_id, mid)
            predictions.append((mid, pred.est))

        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_preds = predictions[:n]

        # Build result DataFrame
        results = []
        for mid, pred_rating in top_preds:
            movie_match = movies_df[movies_df["movieId"] == mid]
            if len(movie_match) > 0:
                movie_row = movie_match.iloc[0]
                results.append({
                    "movieId": mid,
                    "title": movie_row["title"],
                    "genres": movie_row["genres"],
                    "predicted_rating": round(pred_rating, 4),
                })

        return pd.DataFrame(results)

    def cross_validate(self, ratings_df, cv=5):
        """
        Perform k-fold cross-validation and return metrics.

        Args:
            ratings_df: DataFrame with [userId, movieId, rating]
            cv: Number of folds (default 5)

        Returns:
            dict with 'rmse' and 'mae' (mean ± std)
        """
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings_df[["userId", "movieId", "rating"]], reader
        )

        # Create a fresh model instance for CV
        model = self.algo_config["class"](**self.algo_config["params"])
        results = surprise_cv(model, data, measures=["RMSE", "MAE"], cv=cv, verbose=False)

        return {
            "algorithm": self.algo_config["name"],
            "rmse_mean": round(np.mean(results["test_rmse"]), 4),
            "rmse_std": round(np.std(results["test_rmse"]), 4),
            "mae_mean": round(np.mean(results["test_mae"]), 4),
            "mae_std": round(np.std(results["test_mae"]), 4),
        }

    def _check_fitted(self):
        """Ensure the model has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call fit(ratings_df) first."
            )


def compare_algorithms(ratings_df, cv=5):
    """
    Compare all available collaborative filtering algorithms.

    Args:
        ratings_df: DataFrame with [userId, movieId, rating]
        cv: Number of cross-validation folds

    Returns:
        pd.DataFrame with comparison results
    """
    results = []
    for algo_name in ALGORITHMS:
        print(f"  Cross-validating {ALGORITHMS[algo_name]['name']}...")
        rec = CollaborativeRecommender(algo_name)
        cv_result = rec.cross_validate(ratings_df, cv=cv)
        results.append(cv_result)

    return pd.DataFrame(results)
