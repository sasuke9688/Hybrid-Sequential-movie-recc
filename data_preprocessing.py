"""
Data preprocessing module.
Handles loading and cleaning of MovieLens 1M and TMDB datasets.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from config import (
    MOVIELENS_RATINGS, MOVIELENS_MOVIES, MOVIELENS_USERS,
    TMDB_MOVIES, MIN_VOTE_AVERAGE, MIN_POPULARITY
)

# Language display names for common ISO 639-1 codes
LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "pt": "Portuguese", "hi": "Hindi", "ru": "Russian", "ar": "Arabic",
    "sv": "Swedish", "da": "Danish", "nl": "Dutch", "pl": "Polish",
    "tr": "Turkish", "th": "Thai", "id": "Indonesian", "te": "Telugu",
    "ta": "Tamil", "ml": "Malayalam", "bn": "Bengali", "fa": "Persian",
    "uk": "Ukrainian", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    "fi": "Finnish", "nb": "Norwegian", "he": "Hebrew", "vi": "Vietnamese",
    "cn": "Cantonese",
}


def get_language_label(code):
    """Return human-readable label for an ISO language code."""
    if not code or code == "unknown":
        return "Unknown"
    return LANGUAGE_NAMES.get(code.lower(), code.upper())


def load_movielens_ratings(path=MOVIELENS_RATINGS):
    """Load MovieLens 1M ratings data."""
    ratings = pd.read_csv(
        path, sep="::", header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python", encoding="latin-1"
    )
    return ratings


def load_movielens_movies(path=MOVIELENS_MOVIES):
    """Load MovieLens 1M movies data."""
    movies = pd.read_csv(
        path, sep="::", header=None,
        names=["MovieID", "Title", "Genres"],
        engine="python", encoding="latin-1"
    )
    return movies


def load_movielens_users(path=MOVIELENS_USERS):
    """Load MovieLens 1M users data."""
    users = pd.read_csv(
        path, sep="::", header=None,
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        engine="python", encoding="latin-1"
    )
    return users


def load_tmdb_dataset(path=TMDB_MOVIES):
    """Load and clean the TMDB movie dataset."""
    tmdb = pd.read_csv(path, low_memory=False)

    # Standardize column names to lowercase
    tmdb.columns = [c.strip().lower() for c in tmdb.columns]

    # Ensure required columns exist
    required = ["title", "genres", "release_date", "vote_average", "popularity"]
    for col in required:
        if col not in tmdb.columns:
            alt_map = {
                "release_date": ["release_year"],
                "vote_average": ["vote_avg", "rating"],
                "popularity": ["pop"],
            }
            found = False
            for alt in alt_map.get(col, []):
                if alt in tmdb.columns:
                    tmdb.rename(columns={alt: col}, inplace=True)
                    found = True
                    break
            if not found and col not in tmdb.columns:
                raise ValueError(f"Required column '{col}' not found in TMDB dataset. "
                                 f"Available: {list(tmdb.columns)}")

    # Parse release_date to extract release_year
    if "release_year" not in tmdb.columns:
        tmdb["release_date"] = pd.to_datetime(tmdb["release_date"], errors="coerce")
        tmdb["release_year"] = tmdb["release_date"].dt.year

    # Drop rows with missing critical data
    tmdb = tmdb.dropna(subset=["title", "genres", "release_year"])

    # Convert numeric columns
    tmdb["vote_average"] = pd.to_numeric(tmdb["vote_average"], errors="coerce").fillna(0)
    tmdb["popularity"] = pd.to_numeric(tmdb["popularity"], errors="coerce").fillna(0)
    tmdb["release_year"] = tmdb["release_year"].astype(int)

    # Apply quality filters
    tmdb = tmdb[tmdb["vote_average"] >= MIN_VOTE_AVERAGE].copy()
    tmdb = tmdb[tmdb["popularity"] >= MIN_POPULARITY].copy()

    # Clean genres
    tmdb["genres"] = tmdb["genres"].apply(_parse_genres)
    tmdb = tmdb[tmdb["genres"].apply(len) > 0].copy()

    # ── NEW: Preserve and clean original_language column ──────────────
    if "original_language" in tmdb.columns:
        tmdb["original_language"] = (
            tmdb["original_language"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        # Fallback: try "language" column, else mark unknown
        if "language" in tmdb.columns:
            tmdb["original_language"] = (
                tmdb["language"].fillna("unknown").astype(str).str.strip().str.lower()
            )
        else:
            tmdb["original_language"] = "unknown"

    # Replace empty strings with "unknown"
    tmdb["original_language"] = tmdb["original_language"].replace("", "unknown")
    # ──────────────────────────────────────────────────────────────────

    # Reset index
    tmdb = tmdb.reset_index(drop=True)

    return tmdb


def _parse_genres(genre_val):
    """Parse genre field into a list of genre strings."""
    if pd.isna(genre_val) or genre_val == "" or genre_val == "[]":
        return []

    genre_str = str(genre_val)

    if "name" in genre_str:
        import json
        try:
            genre_list = json.loads(genre_str.replace("'", '"'))
            return [g["name"] for g in genre_list if "name" in g]
        except (json.JSONDecodeError, TypeError):
            pass

    if "|" in genre_str:
        return [g.strip() for g in genre_str.split("|") if g.strip()]

    if "," in genre_str:
        return [g.strip() for g in genre_str.split(",") if g.strip()]

    if genre_str.strip():
        return [genre_str.strip()]

    return []


def extract_movielens_genres(movies_df):
    """Extract unique genres from MovieLens movies."""
    all_genres = set()
    for genres_str in movies_df["Genres"]:
        for g in genres_str.split("|"):
            all_genres.add(g.strip())
    return sorted(all_genres)


def build_movielens_genre_lists(movies_df):
    """Convert MovieLens genre strings to lists."""
    movies_df = movies_df.copy()
    movies_df["genre_list"] = movies_df["Genres"].apply(
        lambda x: [g.strip() for g in x.split("|")]
    )
    return movies_df


def get_current_year():
    """Return current year."""
    return datetime.now().year


if __name__ == "__main__":
    print("Testing data loading...")

    if os.path.exists(MOVIELENS_RATINGS):
        ratings = load_movielens_ratings()
        print(f"Ratings: {ratings.shape}")

    if os.path.exists(MOVIELENS_MOVIES):
        movies = load_movielens_movies()
        print(f"Movies: {movies.shape}")

    if os.path.exists(TMDB_MOVIES):
        tmdb = load_tmdb_dataset()
        print(f"TMDB: {tmdb.shape}")
        if "original_language" in tmdb.columns:
            top_langs = tmdb["original_language"].value_counts().head(10)
            print("Top languages:", top_langs.to_dict())
