"""
Data loading and preprocessing module.

Handles:
- Loading CSV files into pandas DataFrames
- Data cleaning (missing values, type conversion)
- Genre parsing (pipe-separated → list)
- User-item interaction matrix creation
- Train/test splitting
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Resolve paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ml-latest-small")


def load_movies(data_dir=None):
    """
    Load and clean the movies dataset.

    Returns:
        pd.DataFrame with columns: [movieId, title, genres, genre_list, year]
    """
    data_dir = data_dir or DATA_DIR
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))

    # Parse genres from pipe-separated string to list
    movies["genre_list"] = movies["genres"].apply(
        lambda x: x.split("|") if x != "(no genres listed)" else []
    )

    # Extract year from title (e.g., "Toy Story (1995)" → 1995)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")

    # Clean title (remove year suffix)
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

    return movies


def load_ratings(data_dir=None):
    """
    Load and clean the ratings dataset.

    Returns:
        pd.DataFrame with columns: [userId, movieId, rating, timestamp]
    """
    data_dir = data_dir or DATA_DIR
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))

    # Convert timestamp to datetime
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

    return ratings


def load_tags(data_dir=None):
    """
    Load and clean the tags dataset.

    Returns:
        pd.DataFrame with columns: [userId, movieId, tag, timestamp]
    """
    data_dir = data_dir or DATA_DIR
    tags = pd.read_csv(os.path.join(data_dir, "tags.csv"))

    # Lowercase and strip tags for consistency
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()

    # Remove empty/null tags
    tags = tags[tags["tag"].notna() & (tags["tag"] != "")]

    return tags


def load_all_data(data_dir=None):
    """
    Load all datasets and return as a dictionary.

    Returns:
        dict with keys: 'movies', 'ratings', 'tags'
    """
    data_dir = data_dir or DATA_DIR
    return {
        "movies": load_movies(data_dir),
        "ratings": load_ratings(data_dir),
        "tags": load_tags(data_dir),
    }


def create_user_item_matrix(ratings):
    """
    Create a user-item interaction matrix (pivot table).

    Args:
        ratings: DataFrame with columns [userId, movieId, rating]

    Returns:
        pd.DataFrame — rows=users, columns=movies, values=ratings (NaN for unrated)
    """
    matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )
    return matrix


def get_sparsity(user_item_matrix):
    """Calculate the sparsity of the user-item matrix as a percentage."""
    total = user_item_matrix.shape[0] * user_item_matrix.shape[1]
    filled = user_item_matrix.notna().sum().sum()
    sparsity = 1.0 - (filled / total)
    return sparsity * 100


def split_ratings(ratings, test_size=0.2, random_state=42):
    """
    Split ratings into train and test sets.

    Uses stratified split by user to ensure each user has ratings in both sets.

    Args:
        ratings: DataFrame with columns [userId, movieId, rating]
        test_size: Fraction of data for testing (default 0.2)
        random_state: Random seed for reproducibility

    Returns:
        (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=ratings["userId"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_dataset_stats(movies, ratings, tags):
    """
    Compute summary statistics for the dataset.

    Returns:
        dict with dataset statistics
    """
    stats = {
        "n_users": ratings["userId"].nunique(),
        "n_movies": movies["movieId"].nunique(),
        "n_ratings": len(ratings),
        "n_tags": len(tags),
        "rating_range": (ratings["rating"].min(), ratings["rating"].max()),
        "avg_rating": ratings["rating"].mean(),
        "median_rating": ratings["rating"].median(),
        "avg_ratings_per_user": ratings.groupby("userId").size().mean(),
        "avg_ratings_per_movie": ratings.groupby("movieId").size().mean(),
        "n_genres": len(set(g for gl in movies["genre_list"] for g in gl)),
        "year_range": (
            int(movies["year"].min()) if movies["year"].notna().any() else None,
            int(movies["year"].max()) if movies["year"].notna().any() else None,
        ),
    }
    return stats
