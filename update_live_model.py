"""
Standalone fast-update script.
Ingests live Supabase ratings, fuses them with the baseline MovieLens dataset,
and recalculates the coupled latent space without triggering the heavy Kaggle ETL pipeline.
"""

import os
import pandas as pd
from supabase import create_client

from config import MODEL_DIR, LATENT_DIM
from data_preprocessing import load_movielens_ratings, load_movielens_movies
from model_training import (
    load_models, save_models, build_rating_matrix,
    train_collaborative_filter, train_genre_encoder, project_tmdb_to_latent
)

def fetch_live_supabase_ratings():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("WARNING: Supabase credentials missing.")
        return pd.DataFrame()

    try:
        supabase = create_client(url, key)
        res = supabase.table("watch_history").select("username, movie_index, rating").not_is("rating", "null").execute()
        
        if not res.data:
            return pd.DataFrame()

        df_live = pd.DataFrame(res.data)
        df_live = df_live.rename(columns={
            "username": "UserID", 
            "movie_index": "MovieID", 
            "rating": "Rating"
        })
        return df_live
    except Exception as e:
        print(f"Supabase error: {e}")
        return pd.DataFrame()

def run_fast_update():
    print("=" * 60)
    print("INITIATING FAST LIVE-DATA FUSION")
    print("=" * 60)

    # 1. Load existing artifacts to avoid re-downloading
    print("Loading existing model artifacts...")
    artifacts = load_models(MODEL_DIR)
    tmdb_df = artifacts["tmdb_dataset"]

    # 2. Load and fuse rating data
    print("Fetching baseline and live ratings...")
    baseline_ratings = load_movielens_ratings()
    live_ratings = fetch_live_supabase_ratings()
    
    if not live_ratings.empty:
        max_baseline_uid = baseline_ratings["UserID"].max()
        live_ratings["UserID"] = pd.factorize(live_ratings["UserID"])[0] + max_baseline_uid + 1
        ratings = pd.concat([baseline_ratings, live_ratings], ignore_index=True)
        print(f"Fused {len(live_ratings)} live user ratings.")
    else:
        ratings = baseline_ratings
        print("No live ratings found. Proceeding with baseline.")

    movies = load_movielens_movies()

    # 3. Rebuild Matrix
    print("Rebuilding user-item matrix...")
    rating_matrix, user_id_map, movie_id_map = build_rating_matrix(ratings, movies)

    # 4. Retrain Collaborative Model
    print(f"Executing TruncatedSVD (latent_dim={LATENT_DIM})...")
    user_factors, movie_factors, svd = train_collaborative_filter(rating_matrix)

    # 5. Re-align Content Model (CRITICAL STEP)
    print("Re-aligning Ridge Regression to new latent geometry...")
    mlb, ridge = train_genre_encoder(movies, movie_factors, movie_id_map)

    # 6. Re-project TMDB dataset
    print("Projecting TMDB catalog...")
    tmdb_latent = project_tmdb_to_latent(tmdb_df, mlb, ridge)

    # 7. Overwrite Model Artifacts
    print("Saving updated artifacts...")
    save_models(user_factors, movie_factors, tmdb_latent, mlb, ridge,
                movie_id_map, user_id_map, tmdb_df, MODEL_DIR)

    print("Fast update complete. Server is ready to serve updated vectors.")

if __name__ == "__main__":
    run_fast_update()