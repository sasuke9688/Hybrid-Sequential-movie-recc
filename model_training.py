"""
Model training pipeline.
Handles collaborative filtering (TruncatedSVD), genre encoding (MultiLabelBinarizer),
and genre-to-latent mapping (Ridge Regression).
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Ridge
import joblib

from config import LATENT_DIM, RIDGE_ALPHA, MODEL_BUNDLE_FILENAME, MODEL_DIR
from data_preprocessing import (
    load_movielens_ratings, load_movielens_movies,
    load_tmdb_dataset, build_movielens_genre_lists
)


def build_rating_matrix(ratings_df, movies_df):
    """
    Build the user-movie rating matrix.
    Returns: sparse matrix, user_id_map, movie_id_map
    """
    user_ids = sorted(ratings_df["UserID"].unique())
    movie_ids = sorted(movies_df["MovieID"].unique())

    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}

    # Filter ratings to only include known movies
    valid_ratings = ratings_df[ratings_df["MovieID"].isin(movie_id_map)]

    rows = valid_ratings["UserID"].map(user_id_map).values
    cols = valid_ratings["MovieID"].map(movie_id_map).values
    vals = valid_ratings["Rating"].values.astype(np.float32)

    n_users = len(user_ids)
    n_movies = len(movie_ids)

    rating_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))

    return rating_matrix, user_id_map, movie_id_map


def train_collaborative_filter(rating_matrix, n_components=LATENT_DIM):
    """
    Train TruncatedSVD on the rating matrix.
    R ≈ U × V
    Returns: user_factors (U), movie_factors (V^T), svd model
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(rating_matrix)  # (n_users, latent_dim)
    movie_factors = svd.components_.T  # (n_movies, latent_dim)

    print(f"  SVD explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}")
    print(f"  User factors shape: {user_factors.shape}")
    print(f"  Movie factors shape: {movie_factors.shape}")

    return user_factors, movie_factors, svd


def train_genre_encoder(movies_df, movie_factors, movie_id_map):
    """
    Train MultiLabelBinarizer on genres and Ridge Regression
    to map genre vectors to latent movie vectors.

    Returns: mlb, ridge_model
    """
    movies_with_genres = build_movielens_genre_lists(movies_df)

    # Build aligned genre and latent matrices
    genre_lists = []
    latent_vectors = []

    for _, row in movies_with_genres.iterrows():
        mid = row["MovieID"]
        if mid in movie_id_map:
            idx = movie_id_map[mid]
            genre_lists.append(row["genre_list"])
            latent_vectors.append(movie_factors[idx])

    # Fit MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genre_lists)  # (n_movies, n_genres)
    latent_matrix = np.array(latent_vectors)  # (n_movies, latent_dim)

    print(f"  Genre matrix shape: {genre_matrix.shape}")
    print(f"  Genre classes: {list(mlb.classes_)}")

    # Train Ridge Regression: genre_vector -> latent_vector
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(genre_matrix, latent_matrix)

    # Training score
    train_score = ridge.score(genre_matrix, latent_matrix)
    print(f"  Ridge R² score: {train_score:.4f}")

    return mlb, ridge


def project_tmdb_to_latent(tmdb_df, mlb, ridge):
    """
    Project TMDB movies into the learned latent space
    using genre encoding + ridge regression.

    Returns: tmdb_latent matrix (n_tmdb_movies, latent_dim)
    """
    # Encode TMDB genres using the fitted MLB
    tmdb_genres = tmdb_df["genres"].tolist()

    # Transform using MLB (unknown genres will be ignored)
    tmdb_genre_matrix = mlb.transform(tmdb_genres)

    # Project to latent space
    tmdb_latent = ridge.predict(tmdb_genre_matrix)

    print(f"  TMDB latent vectors shape: {tmdb_latent.shape}")

    return tmdb_latent


def save_models(user_factors, movie_factors, tmdb_latent, mlb, ridge,
                movie_id_map, user_id_map, tmdb_df, model_dir=MODEL_DIR):
    """Save all trained model artifacts, including a single-file bundle."""
    os.makedirs(model_dir, exist_ok=True)

    artifacts = {
        "user_factors.pkl": user_factors,
        "movie_factors.pkl": movie_factors,
        "tmdb_latent.pkl": tmdb_latent,
        "mlb.pkl": mlb,
        "ridge_model.pkl": ridge,
        "movie_id_map.pkl": movie_id_map,
        "user_id_map.pkl": user_id_map,
        "tmdb_dataset.pkl": tmdb_df,
    }

    for filename, obj in artifacts.items():
        path = os.path.join(model_dir, filename)
        joblib.dump(obj, path)
        print(f"  Saved: {path}")

    bundle = {
        "user_factors": user_factors,
        "movie_factors": movie_factors,
        "tmdb_latent": tmdb_latent,
        "mlb": mlb,
        "ridge_model": ridge,
        "movie_id_map": movie_id_map,
        "user_id_map": user_id_map,
        "tmdb_dataset": tmdb_df,
    }
    bundle_path = os.path.join(model_dir, MODEL_BUNDLE_FILENAME)
    joblib.dump(bundle, bundle_path)
    print(f"  Saved bundle: {bundle_path}")


def load_models(model_dir=MODEL_DIR):
    """Load all trained model artifacts from the bundle when available."""
    bundle_path = os.path.join(model_dir, MODEL_BUNDLE_FILENAME)
    if os.path.exists(bundle_path):
        return joblib.load(bundle_path)

    artifacts = {}
    files = [
        "user_factors.pkl", "movie_factors.pkl", "tmdb_latent.pkl",
        "mlb.pkl", "ridge_model.pkl", "movie_id_map.pkl",
        "user_id_map.pkl", "tmdb_dataset.pkl"
    ]

    for filename in files:
        path = os.path.join(model_dir, filename)
        key = filename.replace(".pkl", "")
        artifacts[key] = joblib.load(path)

    return artifacts


def run_training_pipeline():
    """Execute the full training pipeline."""
    print("=" * 60)
    print("HYBRID MOVIE RECOMMENDATION SYSTEM - Training Pipeline")
    print("=" * 60)

    # Step 1: Load data
    print("\n[Step 1] Loading MovieLens data...")
    ratings = load_movielens_ratings()
    movies = load_movielens_movies()
    print(f"  Ratings: {ratings.shape[0]} entries")
    print(f"  Movies: {movies.shape[0]} entries")

    # Step 2: Load TMDB data
    print("\n[Step 2] Loading TMDB data...")
    tmdb = load_tmdb_dataset()
    print(f"  TMDB movies after filtering: {tmdb.shape[0]}")

    # Step 3: Build rating matrix
    print("\n[Step 3] Building rating matrix...")
    rating_matrix, user_id_map, movie_id_map = build_rating_matrix(ratings, movies)
    print(f"  Rating matrix shape: {rating_matrix.shape}")
    print(f"  Non-zero entries: {rating_matrix.nnz}")

    # Step 4: Collaborative Filtering (TruncatedSVD)
    print(f"\n[Step 4] Training TruncatedSVD (latent_dim={LATENT_DIM})...")
    user_factors, movie_factors, svd = train_collaborative_filter(rating_matrix)

    # Step 5: Genre Encoding + Ridge Regression
    print("\n[Step 5] Training genre encoder and Ridge Regression...")
    mlb, ridge = train_genre_encoder(movies, movie_factors, movie_id_map)

    # Step 6: Project TMDB movies to latent space
    print("\n[Step 6] Projecting TMDB movies to latent space...")
    tmdb_latent = project_tmdb_to_latent(tmdb, mlb, ridge)

    # Step 7: Save models
    print("\n[Step 7] Saving model artifacts...")
    save_models(user_factors, movie_factors, tmdb_latent, mlb, ridge,
                movie_id_map, user_id_map, tmdb, MODEL_DIR)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return {
        "user_factors": user_factors,
        "movie_factors": movie_factors,
        "tmdb_latent": tmdb_latent,
        "mlb": mlb,
        "ridge": ridge,
        "movie_id_map": movie_id_map,
        "user_id_map": user_id_map,
        "tmdb_df": tmdb,
    }


if __name__ == "__main__":
    run_training_pipeline()
