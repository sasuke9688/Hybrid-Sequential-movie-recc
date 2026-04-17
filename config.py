# Configuration
import os

DATA_DIR  = os.getenv("DATA_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_BUNDLE_FILENAME = os.getenv("MODEL_BUNDLE_FILENAME", "recommender_bundle.joblib")
EVALUATION_DIR = os.getenv("EVALUATION_DIR", "evaluation_outputs")

# MovieLens files
MOVIELENS_RATINGS = f"{DATA_DIR}/ml-1m/ratings.dat"
MOVIELENS_MOVIES  = f"{DATA_DIR}/ml-1m/movies.dat"
MOVIELENS_USERS   = f"{DATA_DIR}/ml-1m/users.dat"

# TMDB file
TMDB_MOVIES = f"{DATA_DIR}/tmdb_movies.csv"

# Model hyperparameters
LATENT_DIM  = 50
RIDGE_ALPHA = 1.0

# Hybrid scoring weights (base defaults — engine overrides these dynamically)
ALPHA_COLLAB      = 0.50   # collaborative filtering
BETA_CONTENT      = 0.20   # content-based filtering
GAMMA_SEQUENTIAL  = 0.20   # sequential preference
DELTA_POPULARITY  = 0.10   # popularity prior

# Sequential preference
DECAY_RATE     = 0.05
HISTORY_LENGTH = 10

# Temporal filtering
MAX_AGE_YEARS      = 20
BEFORE_NEWEST_YEARS = 5

# Recommendation
TOP_K = 10

# Ratings
RATING_SCALE_MAX = 5

# TMDB filtering thresholds
MIN_VOTE_AVERAGE = 5.0
MIN_POPULARITY   = 1.0

# Language filter — minimum number of movies a language must have to appear in the
# language selector dropdown returned by /api/languages
MIN_LANGUAGE_COUNT = 5
MIN_GENRE_COUNT = 1

# Flask
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
FLASK_SECRET_KEY = os.getenv(
    "FLASK_SECRET_KEY",
    "change-this-to-a-random-secret-in-production",
)
