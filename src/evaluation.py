"""
Evaluation Module.

Provides comprehensive metrics for comparing recommendation algorithms:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Precision@K
- Recall@K
- Coverage (catalog coverage)
- Algorithm comparison tables
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import KFold


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Compute Precision@K and Recall@K for each user.

    A relevant item is one where the true rating >= threshold.

    Args:
        predictions: list of surprise Prediction objects
        k: Number of top recommendations to consider
        threshold: Minimum rating to consider an item relevant

    Returns:
        (mean_precision, mean_recall)
    """
    # Group predictions by user
    user_est_true = defaultdict(list)
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    precisions = []
    recalls = []

    for uid, user_preds in user_est_true.items():
        # Sort by estimated rating (descending)
        user_preds.sort(key=lambda x: x[0], reverse=True)

        # Top-K predictions
        top_k = user_preds[:k]

        # Number of relevant items in top-K
        n_relevant_in_k = sum(1 for (est, true) in top_k if true >= threshold)

        # Total number of relevant items for this user
        n_relevant_total = sum(1 for (est, true) in user_preds if true >= threshold)

        # Precision@K = relevant in top-K / K
        precisions.append(n_relevant_in_k / k)

        # Recall@K = relevant in top-K / total relevant
        if n_relevant_total > 0:
            recalls.append(n_relevant_in_k / n_relevant_total)
        else:
            recalls.append(0)

    return np.mean(precisions), np.mean(recalls)


def compute_coverage(predictions, all_movie_ids, k=10):
    """
    Compute catalog coverage: fraction of items that appear in
    at least one user's top-K recommendations.

    Args:
        predictions: list of surprise Prediction objects
        all_movie_ids: set of all movieIds in the catalog
        k: Number of top recommendations per user

    Returns:
        float — coverage percentage (0-100)
    """
    # Group predictions by user
    user_preds = defaultdict(list)
    for pred in predictions:
        user_preds[pred.uid].append((pred.iid, pred.est))

    recommended_items = set()
    for uid, preds in user_preds.items():
        preds.sort(key=lambda x: x[1], reverse=True)
        for iid, _ in preds[:k]:
            recommended_items.add(iid)

    coverage = len(recommended_items) / len(all_movie_ids) * 100
    return round(coverage, 2)


def evaluate_collaborative_model(model, ratings_df, k=10, n_folds=5):
    """
    Full evaluation of a collaborative filtering model.

    Performs k-fold cross-validation and computes:
    - RMSE, MAE
    - Precision@K, Recall@K
    - Coverage

    Args:
        model: A surprise algorithm instance
        ratings_df: DataFrame with [userId, movieId, rating]
        k: K for Precision@K and Recall@K
        n_folds: Number of CV folds

    Returns:
        dict with all metrics
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings_df[["userId", "movieId", "rating"]], reader
    )

    kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)

    rmse_scores = []
    mae_scores = []
    precision_scores = []
    recall_scores = []

    all_movie_ids = set(ratings_df["movieId"])
    all_predictions = []

    for trainset, testset in kf.split(data):
        model.fit(trainset)
        predictions = model.test(testset)
        all_predictions.extend(predictions)

        rmse_scores.append(accuracy.rmse(predictions, verbose=False))
        mae_scores.append(accuracy.mae(predictions, verbose=False))

        prec, rec = precision_recall_at_k(predictions, k=k)
        precision_scores.append(prec)
        recall_scores.append(rec)

    coverage = compute_coverage(all_predictions, all_movie_ids, k=k)

    return {
        "rmse": round(np.mean(rmse_scores), 4),
        "rmse_std": round(np.std(rmse_scores), 4),
        "mae": round(np.mean(mae_scores), 4),
        "mae_std": round(np.std(mae_scores), 4),
        f"precision@{k}": round(np.mean(precision_scores), 4),
        f"recall@{k}": round(np.mean(recall_scores), 4),
        f"coverage@{k}": coverage,
    }


def full_comparison(ratings_df, k=10, n_folds=5):
    """
    Compare all collaborative filtering algorithms with full metrics.

    Args:
        ratings_df: DataFrame with [userId, movieId, rating]
        k: K for Precision/Recall/Coverage
        n_folds: Number of CV folds

    Returns:
        pd.DataFrame with comparison results
    """
    from surprise import KNNBasic, SVD, NMF
    from src.collaborative import ALGORITHMS

    results = []

    for algo_name, config in ALGORITHMS.items():
        print(f"  Evaluating {config['name']}...")
        model = config["class"](**config["params"])
        metrics = evaluate_collaborative_model(
            model, ratings_df, k=k, n_folds=n_folds
        )
        metrics["algorithm"] = config["name"]
        results.append(metrics)

    df = pd.DataFrame(results)
    # Reorder columns
    cols = ["algorithm"] + [c for c in df.columns if c != "algorithm"]
    return df[cols]
