"""
Recommendation engine.
Implements hybrid scoring with collaborative filtering, content-based filtering,
sequential preference modeling, and temporal filtering.

Dynamic weight rules (refined):
  - 0 movies       : pure content + popularity (no user signal at all)
  - 1–2 movies     : heavy content, tiny collab
  - 3–6 movies     : cold-start — content dominates, collab catching up
  - 7–14 movies    : warming — collab takes the lead
  - 15+ movies     : warm — collab leads; sequential boosted if recent burst
  - recent burst   : sequential takes over when user is actively watching
"""

import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    ALPHA_COLLAB, BETA_CONTENT, GAMMA_SEQUENTIAL, DELTA_POPULARITY,
    DECAY_RATE, HISTORY_LENGTH, MAX_AGE_YEARS, BEFORE_NEWEST_YEARS,
    TOP_K, MIN_GENRE_COUNT
)

# Window in seconds for "recent burst" detection (~60 days)
RECENT_WINDOW_SECONDS = 60 * 86400


class RecommendationEngine:
    """Hybrid movie recommendation engine with dynamic weight adjustment."""

    def __init__(self, tmdb_df, tmdb_latent, mlb, ridge,
                 user_factors=None, movie_factors=None):
        self.tmdb_df = tmdb_df.copy()
        self.tmdb_latent = tmdb_latent
        self.mlb = mlb
        self.ridge = ridge
        self.user_factors = user_factors
        self.movie_factors = movie_factors

        # Normalize popularity scores to [0, 1]
        max_pop = self.tmdb_df["popularity"].max()
        if max_pop > 0:
            self.tmdb_df["popularity_norm"] = self.tmdb_df["popularity"] / max_pop
        else:
            self.tmdb_df["popularity_norm"] = 0.0

        # Precompute norms for latent vectors
        norms = np.linalg.norm(self.tmdb_latent, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tmdb_latent_normed = self.tmdb_latent / norms

    # ──────────── Language helpers ────────────

    def get_available_languages(self, min_count=5):
        """
        Return a sorted list of languages present in the TMDB catalog.
        """
        if "original_language" not in self.tmdb_df.columns:
            return []

        from data_preprocessing import get_language_label
        counts = self.tmdb_df["original_language"].value_counts()
        langs = []
        for code, cnt in counts.items():
            if cnt >= min_count and code and code != "unknown":
                langs.append({
                    "code": code,
                    "label": get_language_label(code),
                    "count": int(cnt),
                })
        langs.sort(key=lambda x: (-x["count"], x["label"]))
        return langs

    def get_available_genres(self, min_count=MIN_GENRE_COUNT):
        """
        Return a sorted list of genres present in the TMDB catalog.
        """
        genre_counts = {}
        for genres in self.tmdb_df.get("genres", []):
            if not isinstance(genres, list):
                continue
            for genre in genres:
                genre = str(genre).strip()
                if not genre:
                    continue
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        results = [
            {"name": genre, "count": count}
            for genre, count in genre_counts.items()
            if count >= min_count
        ]
        results.sort(key=lambda item: (-item["count"], item["name"]))
        return results

    # ──────────── Dynamic weight computation ────────────

    @staticmethod
    def compute_dynamic_weights(interaction_history, current_selection_count=0):
        """
        Compute dynamic alpha/beta/gamma/delta based on the user's current signal volume.
        """
        now = datetime.now().timestamp()
        total = len(interaction_history) if interaction_history else 0

        # Count movies watched within the last 2 months
        recent_count = 0
        if interaction_history:
            for m in interaction_history:
                ts = m.get("timestamp", 0)
                if not ts or (now - ts) <= RECENT_WINDOW_SECONDS:
                    recent_count += 1

        decay = DECAY_RATE

        # ── Regime selection ──────────────────────────────────────────
        if total == 0:
            alpha = 0.00
            beta  = 0.60
            gamma = 0.00
            delta = 0.40
            decay = 0.01
            regime = "new_user"

        elif total <= 2:
            alpha = 0.10
            beta  = 0.55
            gamma = 0.05
            delta = 0.30
            decay = 0.02
            regime = "very_cold"

        elif total <= 6:
            alpha = 0.20
            beta  = 0.45
            gamma = 0.10
            delta = 0.25
            decay = 0.02
            regime = "cold_start"

        elif total <= 14:
            alpha = 0.35
            beta  = 0.30
            gamma = 0.20
            delta = 0.15
            decay = 0.04
            regime = "warming"

        else:
            if recent_count >= 5:
                alpha = 0.35
                beta  = 0.10
                gamma = 0.45
                delta = 0.10
                decay = 0.10
                regime = "recent_burst"
            elif recent_count >= 3:
                alpha = 0.40
                beta  = 0.15
                gamma = 0.32
                delta = 0.13
                decay = 0.06
                regime = "moderate_recent"
            else:
                alpha = 0.52
                beta  = 0.23
                gamma = 0.10
                delta = 0.15
                decay = 0.03
                regime = "warm_sparse"

        info = {
            "regime": regime,
            "total_watched": total,
            "recent_2mo": recent_count,
            "current_selection_count": current_selection_count,
            "alpha": round(alpha, 2),
            "beta":  round(beta,  2),
            "gamma": round(gamma, 2),
            "delta": round(delta, 2),
            "decay_rate": round(decay, 3),
            "regime_description": _regime_description(regime),
        }

        return alpha, beta, gamma, delta, decay, info

    # ──────────── User vector building ────────────

    def build_user_vector_from_movies(self, selected_movies):
        if not selected_movies:
            return np.zeros(self.tmdb_latent.shape[1])

        indices, weights = [], []
        for m in selected_movies:
            idx = m["index"]
            if 0 <= idx < len(self.tmdb_latent):
                indices.append(idx)
                rating = m.get("rating")
                weights.append(float(rating) if rating and rating > 0 else 1.0)

        if not indices:
            return np.zeros(self.tmdb_latent.shape[1])

        vectors = self.tmdb_latent[indices]
        weights = np.array(weights)
        weights /= weights.sum()
        return np.average(vectors, axis=0, weights=weights)

    def sequential_preference_vector(self, selected_movies, decay_rate=None):
        if decay_rate is None:
            decay_rate = DECAY_RATE

        if not selected_movies:
            return np.zeros(self.tmdb_latent.shape[1])

        sorted_movies = sorted(selected_movies, key=lambda m: m.get("timestamp", 0))
        recent = sorted_movies[-HISTORY_LENGTH:]

        now = datetime.now().timestamp()
        weights, vectors = [], []

        for i, movie in enumerate(recent):
            idx = movie["index"]
            if idx < 0 or idx >= len(self.tmdb_latent):
                continue
            vectors.append(self.tmdb_latent[idx])

            if "timestamp" in movie and movie["timestamp"]:
                age_days = (now - movie["timestamp"]) / 86400.0
            else:
                age_days = (len(recent) - 1 - i) * 30

            time_weight = np.exp(-decay_rate * age_days)

            rating = movie.get("rating")
            rating_boost = float(rating) / 5.0 if rating and rating > 0 else 1.0
            weights.append(time_weight * rating_boost)

        if not vectors:
            return np.zeros(self.tmdb_latent.shape[1])

        weights = np.array(weights)
        total_w = weights.sum()
        if total_w > 0:
            weights /= total_w
        return np.average(np.array(vectors), axis=0, weights=weights)

    # ──────────── Score computation ────────────

    def compute_collaborative_scores(self, user_vector):
        return self.tmdb_latent @ user_vector

    def compute_content_scores(self, user_vector):
        user_norm = np.linalg.norm(user_vector)
        if user_norm == 0:
            return np.zeros(len(self.tmdb_df))
        return self.tmdb_latent_normed @ (user_vector / user_norm)

    def compute_sequential_scores(self, seq_vector):
        return self.tmdb_latent @ seq_vector

    def compute_popularity_scores(self):
        return self.tmdb_df["popularity_norm"].values

    def apply_temporal_filter(self, selected_movies):
        current_year = datetime.now().year
        if selected_movies:
            years = [
                self.tmdb_df.iloc[m["index"]]["release_year"]
                for m in selected_movies
                if 0 <= m["index"] < len(self.tmdb_df)
            ]
            newest_year = max(years) if years else current_year
        else:
            newest_year = current_year

        min_year = max(newest_year - BEFORE_NEWEST_YEARS, current_year - MAX_AGE_YEARS)
        mask = self.tmdb_df["release_year"].values >= min_year
        return mask, min_year

    # ──────────── Main recommendation method ────────────

    def recommend(self, selected_movies, top_k=TOP_K,
                  alpha=None, beta=None, gamma=None, delta=None,
                  watch_history=None, language_filter=None, genre_filters=None,
                  candidate_indices=None, apply_temporal_filter=True):
        
        all_movies = list(selected_movies)
        history_indices = set()
        if watch_history:
            history_indices = {m["index"] for m in watch_history}
            sel_set = {m["index"] for m in selected_movies}
            for h in watch_history:
                if h["index"] not in sel_set:
                    all_movies.append(h)

        if alpha is None or beta is None or gamma is None or delta is None:
            d_alpha, d_beta, d_gamma, d_delta, d_decay, weight_info = \
                self.compute_dynamic_weights(
                    all_movies,
                    current_selection_count=len(selected_movies),
                )
            alpha  = alpha  if alpha  is not None else d_alpha
            beta   = beta   if beta   is not None else d_beta
            gamma  = gamma  if gamma  is not None else d_gamma
            delta  = delta  if delta  is not None else d_delta
            decay_rate = d_decay
        else:
            decay_rate = DECAY_RATE
            weight_info = {
                "regime": "manual",
                "total_watched": len(watch_history) if watch_history else 0,
                "recent_2mo": 0,
                "alpha": alpha, "beta": beta,
                "gamma": gamma, "delta": delta,
                "decay_rate": decay_rate,
                "regime_description": "Manually specified weights.",
            }

        user_vector = self.build_user_vector_from_movies(all_movies)
        seq_vector  = self.sequential_preference_vector(all_movies, decay_rate=decay_rate)

        collab_scores = _normalize_scores(self.compute_collaborative_scores(user_vector))
        content_scores = _normalize_scores(self.compute_content_scores(user_vector))
        seq_scores    = _normalize_scores(self.compute_sequential_scores(seq_vector))
        pop_scores    = self.compute_popularity_scores()

        final_scores = (
            alpha * collab_scores
            + beta  * content_scores
            + gamma * seq_scores
            + delta * pop_scores
        )

        # ─── OPTIMAL MERGE: REGIME-AWARE DIVERSITY PENALTY ───
        # Apply the popularity penalty ONLY to users with enough history (warm users).
        # This breaks the filter bubble for active users while protecting the Hit Rate 
        # for brand new (cold-start) users who need safe, popular recommendations.
        current_regime = weight_info.get("regime", "manual")
        warm_regimes = ["warming", "warm_sparse", "moderate_recent", "recent_burst"]
        
        if current_regime in warm_regimes:
            popularity_array = self.tmdb_df["popularity"].values
            penalty_factor = np.log(2.0 + popularity_array)
            final_scores = final_scores / penalty_factor

        if apply_temporal_filter:
            temporal_mask, _ = self.apply_temporal_filter(all_movies)
            final_scores[~temporal_mask] = -np.inf

        if language_filter and language_filter.lower() not in ("all", ""):
            if "original_language" in self.tmdb_df.columns:
                lang_mask = (
                    self.tmdb_df["original_language"] == language_filter.lower()
                )
                final_scores[~lang_mask.values] = -np.inf

        normalized_genres = _normalize_genre_filters(genre_filters)
        if normalized_genres:
            genre_mask = self.tmdb_df["genres"].apply(
                lambda genres: any(
                    str(genre).strip().lower() in normalized_genres
                    for genre in (genres if isinstance(genres, list) else [genres])
                )
            )
            final_scores[~genre_mask.values] = -np.inf

        if candidate_indices is not None:
            candidate_mask = np.zeros(len(final_scores), dtype=bool)
            valid_candidate_indices = [
                int(idx) for idx in candidate_indices
                if 0 <= int(idx) < len(final_scores)
            ]
            candidate_mask[valid_candidate_indices] = True
            final_scores[~candidate_mask] = -np.inf

        exclude = {m["index"] for m in selected_movies} | history_indices
        for idx in exclude:
            if 0 <= idx < len(final_scores):
                final_scores[idx] = -np.inf

        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            if final_scores[idx] == -np.inf:
                break
            row = self.tmdb_df.iloc[idx]
            results.append({
                "rank":         rank,
                "index":        int(idx),
                "title":        row["title"],
                "genres":       row["genres"],
                "release_year": int(row["release_year"]),
                "vote_average": round(float(row["vote_average"]), 1),
                "popularity":   round(float(row["popularity"]), 1),
                "score":        round(float(final_scores[idx]), 4),
                "language":     row.get("original_language", "unknown"),
            })

        return pd.DataFrame(results), weight_info

    # ──────────── Search ────────────

    def search_movies(self, query, limit=20, language_filter=None, genre_filters=None):
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        mask = self.tmdb_df["title"].str.lower().str.contains(query_lower, na=False)

        if language_filter and language_filter.lower() not in ("all", ""):
            if "original_language" in self.tmdb_df.columns:
                lang_mask = (
                    self.tmdb_df["original_language"] == language_filter.lower()
                )
                mask = mask & lang_mask

        normalized_genres = _normalize_genre_filters(genre_filters)
        if normalized_genres:
            genre_mask = self.tmdb_df["genres"].apply(
                lambda genres: any(
                    str(genre).strip().lower() in normalized_genres
                    for genre in (genres if isinstance(genres, list) else [genres])
                )
            )
            mask = mask & genre_mask

        matches = self.tmdb_df[mask].head(limit)

        results = []
        for idx, row in matches.iterrows():
            results.append({
                "index":        idx,
                "title":        row["title"],
                "release_year": int(row["release_year"]),
                "genres":       row["genres"],
                "vote_average": round(float(row["vote_average"]), 1),
                "popularity":   round(float(row["popularity"]), 1),
                "language":     row.get("original_language", "unknown"),
            })

        return results


def _normalize_scores(scores):
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val == 0:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def _normalize_genre_filters(genre_filters):
    if not genre_filters:
        return set()

    if isinstance(genre_filters, str):
        genre_filters = [part.strip() for part in genre_filters.split(",")]

    return {
        str(genre).strip().lower()
        for genre in genre_filters
        if str(genre).strip()
    }


def _regime_description(regime):
    descriptions = {
        "new_user":        "No history yet — using content similarity and popularity.",
        "very_cold":       "Very few movies (1–2) — content-based filtering dominates.",
        "cold_start":      "Few movies (3–6) — content leads, collaborative filtering catching up.",
        "warming":         "Building history (7–14) — collaborative taking the lead.",
        "warm_sparse":     "Good history, mostly older — collaborative filtering leads.",
        "moderate_recent": "Active recently — sequential preferences boosted.",
        "recent_burst":    "High recent activity — sequential preferences dominate.",
        "manual":          "Manually specified weights.",
    }
    return descriptions.get(regime, regime)
