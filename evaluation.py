"""
Offline evaluation utilities for the recommendation system.

Builds a MovieLens-to-TMDB alignment, evaluates held-out recommendations,
and saves both metrics and graphs for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EVALUATION_DIR
from data_preprocessing import (
    load_movielens_movies,
    load_movielens_ratings,
)


DEFAULT_K_VALUES = (10, 12, 14, 16, 18, 20, 22, 25)
DEFAULT_HOLDOUT_SIZE = 2
DEFAULT_MIN_HISTORY = 5
DEFAULT_POSITIVE_RATING = 4
DEFAULT_MAX_USERS = 250


def precision_at_k(recommended, relevant, k=10):
    """Precision@K: fraction of the top-k recommendations that are relevant."""
    if k <= 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k=10):
    """Recall@K: fraction of relevant items that appear in the top-k list."""
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant)


def dcg_at_k(relevance_scores, k=10):
    """Discounted cumulative gain at K."""
    relevance_scores = np.asarray(relevance_scores[:k], dtype=np.float64)
    if relevance_scores.size == 0:
        return 0.0
    positions = np.arange(1, relevance_scores.size + 1)
    discounts = np.log2(positions + 1)
    return float(np.sum(relevance_scores / discounts))


def ndcg_at_k(recommended, relevant, k=10):
    """Normalized discounted cumulative gain at K."""
    if not relevant:
        return 0.0

    relevance = [1.0 if item in relevant else 0.0 for item in recommended[:k]]
    dcg = dcg_at_k(relevance, k)

    ideal_length = min(len(relevant), k)
    ideal_scores = [1.0] * ideal_length + [0.0] * max(0, k - ideal_length)
    idcg = dcg_at_k(ideal_scores, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(recommended, relevant, k=10):
    """HitRate@K: whether at least one relevant item appears in the top-k list."""
    return float(bool(set(recommended[:k]) & set(relevant)))


def reciprocal_rank_at_k(recommended, relevant, k=10):
    """Reciprocal rank at K using the first relevant recommendation."""
    relevant = set(relevant)
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def _normalize_title(title):
    """Normalize titles so MovieLens and TMDB names can be compared."""
    title = re.sub(r"\s*\(\d{4}\)$", "", str(title)).strip().lower()
    title = title.replace("&", "and")
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def _extract_movielens_year(title):
    """Extract the release year from a MovieLens title like 'Toy Story (1995)'."""
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else None


def _prepare_tmdb_lookup(tmdb_df):
    """Create lookup tables from normalized TMDB titles."""
    exact_lookup = {}
    title_lookup = {}

    tmdb_sorted = tmdb_df.copy()
    tmdb_sorted["norm_title"] = tmdb_sorted["title"].apply(_normalize_title)
    tmdb_sorted = tmdb_sorted.sort_values(
        by=["norm_title", "release_year", "popularity", "vote_average"],
        ascending=[True, True, False, False],
    )

    for idx, row in tmdb_sorted.iterrows():
        year = int(row["release_year"])
        norm_title = row["norm_title"]
        exact_lookup.setdefault((norm_title, year), []).append(int(idx))
        title_lookup.setdefault(norm_title, []).append((int(idx), year))

    return exact_lookup, title_lookup


def build_catalog_alignment(movielens_movies_df, tmdb_df):
    """Map MovieLens MovieIDs to TMDB row indices using normalized title/year matching."""
    exact_lookup, title_lookup = _prepare_tmdb_lookup(tmdb_df)

    movie_to_tmdb = {}
    stats = {
        "movielens_movies": int(len(movielens_movies_df)),
        "matched_movies": 0,
        "exact_matches": 0,
        "near_year_matches": 0,
        "title_only_unique_matches": 0,
        "unmatched_movies": 0,
    }

    for _, row in movielens_movies_df.iterrows():
        movie_id = int(row["MovieID"])
        title = row["Title"]
        norm_title = _normalize_title(title)
        year = _extract_movielens_year(title)

        chosen_idx = None
        match_type = None

        if year is not None:
            exact_matches = exact_lookup.get((norm_title, year), [])
            if exact_matches:
                chosen_idx = exact_matches[0]
                match_type = "exact_matches"

        if chosen_idx is None:
            title_matches = title_lookup.get(norm_title, [])
            if year is not None:
                close_matches = [
                    idx for idx, candidate_year in title_matches
                    if abs(candidate_year - year) <= 1
                ]
                if len(close_matches) == 1:
                    chosen_idx = close_matches[0]
                    match_type = "near_year_matches"

        if chosen_idx is None:
            title_matches = title_lookup.get(norm_title, [])
            unique_indices = sorted({idx for idx, _ in title_matches})
            if len(unique_indices) == 1:
                chosen_idx = unique_indices[0]
                match_type = "title_only_unique_matches"

        if chosen_idx is None:
            stats["unmatched_movies"] += 1
            continue

        movie_to_tmdb[movie_id] = chosen_idx
        stats["matched_movies"] += 1
        stats[match_type] += 1

    if stats["movielens_movies"]:
        stats["alignment_rate"] = round(
            stats["matched_movies"] / stats["movielens_movies"], 4
        )
    else:
        stats["alignment_rate"] = 0.0

    return movie_to_tmdb, stats


def build_test_users(
    ratings_df,
    movielens_movies_df,
    tmdb_df,
    positive_rating=DEFAULT_POSITIVE_RATING,
    min_history=DEFAULT_MIN_HISTORY,
    holdout_size=DEFAULT_HOLDOUT_SIZE,
    max_users=DEFAULT_MAX_USERS,
    random_state=42,
):
    """Construct per-user leave-last-n-out evaluation cases."""
    movie_to_tmdb, alignment_stats = build_catalog_alignment(movielens_movies_df, tmdb_df)
    candidate_indices = sorted({int(idx) for idx in movie_to_tmdb.values()})

    positive_ratings = ratings_df[ratings_df["Rating"] >= positive_rating].copy()
    positive_ratings["tmdb_index"] = positive_ratings["MovieID"].map(movie_to_tmdb)
    positive_ratings = positive_ratings.dropna(subset=["tmdb_index"]).copy()
    positive_ratings["tmdb_index"] = positive_ratings["tmdb_index"].astype(int)
    positive_ratings = positive_ratings.sort_values(["UserID", "Timestamp"])

    test_users = []

    for user_id, group in positive_ratings.groupby("UserID"):
        deduped = group.drop_duplicates(subset=["tmdb_index"], keep="last").copy()
        if len(deduped) < min_history:
            continue

        holdout_n = min(holdout_size, len(deduped) - 2)
        if holdout_n <= 0:
            continue

        selected_part = deduped.iloc[:-holdout_n]
        relevant_part = deduped.iloc[-holdout_n:]
        if len(selected_part) < 2:
            continue

        selected_movies = []
        for _, item in selected_part.iterrows():
            tmdb_index = int(item["tmdb_index"])
            tmdb_row = tmdb_df.iloc[tmdb_index]
            selected_movies.append({
                "index": tmdb_index,
                "title": tmdb_row["title"],
                "release_year": int(tmdb_row["release_year"]),
                "rating": float(item["Rating"]),
                "timestamp": float(item["Timestamp"]),
            })

        relevant_indices = [int(idx) for idx in relevant_part["tmdb_index"].tolist()]

        test_users.append({
            "user_id": int(user_id),
            "selected": selected_movies,
            "relevant_indices": relevant_indices,
            "selected_count": int(len(selected_movies)),
            "relevant_count": int(len(relevant_indices)),
        })

    total_eligible = len(test_users)

    if max_users and total_eligible > max_users:
        rng = np.random.default_rng(random_state)
        chosen = np.sort(rng.choice(total_eligible, size=max_users, replace=False))
        test_users = [test_users[idx] for idx in chosen]

    dataset_stats = {
        "positive_rating_threshold": positive_rating,
        "min_history": min_history,
        "holdout_size": holdout_size,
        "mapped_positive_ratings": int(len(positive_ratings)),
        "eligible_users": int(total_eligible),
        "sampled_users": int(len(test_users)),
        "avg_selected_count": round(
            float(np.mean([user["selected_count"] for user in test_users])), 2
        ) if test_users else 0.0,
        "avg_relevant_count": round(
            float(np.mean([user["relevant_count"] for user in test_users])), 2
        ) if test_users else 0.0,
    }
    dataset_stats.update(alignment_stats)

    dataset_stats["aligned_tmdb_candidates"] = int(len(candidate_indices))
    return test_users, dataset_stats, candidate_indices


def _history_bucket(selected_count):
    """Group users into readable history-size buckets."""
    if selected_count <= 4:
        return "2-4 seed movies"
    if selected_count <= 8:
        return "5-8 seed movies"
    return "9+ seed movies"


def _resolve_focal_k(k_values):
    """Prefer K=10 when available, else the largest requested cutoff."""
    if 10 in k_values:
        return 10
    return max(k_values)


def evaluate_recommendations(engine, test_users, k_values=DEFAULT_K_VALUES, candidate_indices=None):
    """Evaluate recommendations across multiple K cutoffs."""
    if not test_users:
        raise ValueError("No eligible test users were built for evaluation.")

    k_values = tuple(sorted(set(int(k) for k in k_values if int(k) > 0)))
    if not k_values:
        raise ValueError("At least one positive K value is required.")

    max_k = max(k_values)
    focal_k = _resolve_focal_k(k_values)

    metric_lists = {
        k: {
            "precision": [],
            "recall": [],
            "ndcg": [],
            "hit_rate": [],
            "mrr": [],
        }
        for k in k_values
    }
    per_user_rows = []
    regime_rows = []
    first_hit_ranks = []
    recommended_catalog = set()

    for user_data in test_users:
        recs_df, weight_info = engine.recommend(
            user_data["selected"],
            top_k=max_k,
            candidate_indices=candidate_indices,
            apply_temporal_filter=False,
        )
        rec_indices = (
            recs_df["index"].astype(int).tolist()
            if (not recs_df.empty and "index" in recs_df.columns)
            else []
        )
        recommended_catalog.update(rec_indices)

        relevant_indices = user_data["relevant_indices"]
        first_hit_rank = None
        for rank, movie_idx in enumerate(rec_indices, start=1):
            if movie_idx in relevant_indices:
                first_hit_rank = rank
                first_hit_ranks.append(rank)
                break

        row = {
            "user_id": user_data["user_id"],
            "selected_count": user_data["selected_count"],
            "relevant_count": user_data["relevant_count"],
            "history_bucket": _history_bucket(user_data["selected_count"]),
            "regime": weight_info.get("regime", "unknown"),
            "first_hit_rank": first_hit_rank if first_hit_rank is not None else "",
        }

        for k in k_values:
            precision = precision_at_k(rec_indices, relevant_indices, k)
            recall = recall_at_k(rec_indices, relevant_indices, k)
            ndcg = ndcg_at_k(rec_indices, relevant_indices, k)
            hit_rate = hit_rate_at_k(rec_indices, relevant_indices, k)
            mrr = reciprocal_rank_at_k(rec_indices, relevant_indices, k)

            metric_lists[k]["precision"].append(precision)
            metric_lists[k]["recall"].append(recall)
            metric_lists[k]["ndcg"].append(ndcg)
            metric_lists[k]["hit_rate"].append(hit_rate)
            metric_lists[k]["mrr"].append(mrr)

            row[f"precision_at_{k}"] = round(precision, 6)
            row[f"recall_at_{k}"] = round(recall, 6)
            row[f"ndcg_at_{k}"] = round(ndcg, 6)
            row[f"hit_rate_at_{k}"] = round(hit_rate, 6)
            row[f"mrr_at_{k}"] = round(mrr, 6)

        per_user_rows.append(row)
        regime_rows.append({
            "regime": row["regime"],
            "history_bucket": row["history_bucket"],
            "selected_count": row["selected_count"],
            f"hit_rate_at_{focal_k}": row[f"hit_rate_at_{focal_k}"],
            f"ndcg_at_{focal_k}": row[f"ndcg_at_{focal_k}"],
            f"mrr_at_{focal_k}": row[f"mrr_at_{focal_k}"],
        })

    metrics_by_k = []
    for k in k_values:
        metrics_by_k.append({
            "k": k,
            "precision": round(float(np.mean(metric_lists[k]["precision"])), 4),
            "recall": round(float(np.mean(metric_lists[k]["recall"])), 4),
            "ndcg": round(float(np.mean(metric_lists[k]["ndcg"])), 4),
            "hit_rate": round(float(np.mean(metric_lists[k]["hit_rate"])), 4),
            "mrr": round(float(np.mean(metric_lists[k]["mrr"])), 4),
        })

    metrics_by_k_df = pd.DataFrame(metrics_by_k)
    per_user_df = pd.DataFrame(per_user_rows)
    regime_df = pd.DataFrame(regime_rows)

    regime_metrics_df = (
        regime_df.groupby("regime", as_index=False)
        .agg(
            users=("regime", "size"),
            selected_count_mean=("selected_count", "mean"),
            hit_rate=(f"hit_rate_at_{focal_k}", "mean"),
            ndcg=(f"ndcg_at_{focal_k}", "mean"),
            mrr=(f"mrr_at_{focal_k}", "mean"),
        )
        .sort_values("users", ascending=False)
    )
    if not regime_metrics_df.empty:
        regime_metrics_df["selected_count_mean"] = regime_metrics_df["selected_count_mean"].round(2)
        regime_metrics_df["hit_rate"] = regime_metrics_df["hit_rate"].round(4)
        regime_metrics_df["ndcg"] = regime_metrics_df["ndcg"].round(4)
        regime_metrics_df["mrr"] = regime_metrics_df["mrr"].round(4)

    overall = {
        "evaluated_users": int(len(test_users)),
        "focal_k": int(focal_k),
        "catalog_coverage_at_max_k": round(
            len(recommended_catalog) / len(engine.tmdb_df), 4
        ) if len(engine.tmdb_df) else 0.0,
        "mean_first_hit_rank": round(float(np.mean(first_hit_ranks)), 2) if first_hit_ranks else None,
        "users_with_any_hit_at_max_k": int(sum(1 for rank in first_hit_ranks if rank <= max_k)),
    }

    return {
        "overall": overall,
        "metrics_by_k": metrics_by_k_df,
        "per_user": per_user_df,
        "regime_metrics": regime_metrics_df,
        "first_hit_ranks": first_hit_ranks,
    }


def _plot_metrics_vs_k(metrics_by_k_df, output_path):
    """Save a line chart of all ranking metrics versus K."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    metric_columns = ["precision", "recall", "ndcg", "hit_rate", "mrr"]
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "ndcg": "NDCG",
        "hit_rate": "Hit Rate",
        "mrr": "MRR",
    }

    for metric_name in metric_columns:
        ax.plot(
            metrics_by_k_df["k"],
            metrics_by_k_df[metric_name],
            marker="o",
            linewidth=2,
            label=labels[metric_name],
        )

    ax.set_title("Evaluation Metrics Across Recommendation Cutoffs")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_hit_rate_by_regime(regime_metrics_df, focal_k, output_path):
    """Save a bar chart of hit rate by dynamic-weight regime."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if regime_metrics_df.empty:
        ax.text(0.5, 0.5, "No regime data available", ha="center", va="center")
        ax.set_xticks([])
    else:
        plot_df = regime_metrics_df.sort_values("hit_rate", ascending=False)
        ax.bar(plot_df["regime"], plot_df["hit_rate"], color="#2e7d32")
        ax.tick_params(axis="x", rotation=20)

    ax.set_title(f"Hit Rate by Dynamic Regime (K={focal_k})")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Hit Rate")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_first_hit_rank_distribution(first_hit_ranks, max_k, output_path):
    """Save a histogram of the first relevant hit rank."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if first_hit_ranks:
        bins = np.arange(1, max_k + 2) - 0.5
        ax.hist(first_hit_ranks, bins=bins, color="#1565c0", edgecolor="white")
        ax.set_xticks(range(1, max_k + 1))
    else:
        ax.text(0.5, 0.5, "No hits found within evaluated cutoffs", ha="center", va="center")
        ax.set_xticks([])

    ax.set_title("Distribution of First Relevant Hit Rank")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Users")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_evaluation_artifacts(results, dataset_stats, output_dir=EVALUATION_DIR):
    """Write metrics tables and graph files to disk."""
    os.makedirs(output_dir, exist_ok=True)

    metrics_by_k_path = os.path.join(output_dir, "metrics_by_k.csv")
    per_user_path = os.path.join(output_dir, "per_user_results.csv")
    regime_metrics_path = os.path.join(output_dir, "regime_metrics.csv")
    summary_path = os.path.join(output_dir, "evaluation_summary.json")

    metrics_plot_path = os.path.join(output_dir, "metrics_vs_k.png")
    regime_plot_path = os.path.join(output_dir, "hit_rate_by_regime.png")
    hit_rank_plot_path = os.path.join(output_dir, "first_hit_rank_distribution.png")

    results["metrics_by_k"].to_csv(metrics_by_k_path, index=False)
    results["per_user"].to_csv(per_user_path, index=False)
    results["regime_metrics"].to_csv(regime_metrics_path, index=False)

    _plot_metrics_vs_k(results["metrics_by_k"], metrics_plot_path)
    _plot_hit_rate_by_regime(
        results["regime_metrics"],
        results["overall"]["focal_k"],
        regime_plot_path,
    )
    _plot_first_hit_rank_distribution(
        results["first_hit_ranks"],
        int(results["metrics_by_k"]["k"].max()),
        hit_rank_plot_path,
    )

    focal_row = results["metrics_by_k"].loc[
        results["metrics_by_k"]["k"] == results["overall"]["focal_k"]
    ]
    focal_metrics = focal_row.iloc[0].to_dict() if not focal_row.empty else {}

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_stats,
        "overall": results["overall"],
        "focal_metrics": focal_metrics,
        "artifact_paths": {
            "metrics_by_k_csv": metrics_by_k_path,
            "per_user_csv": per_user_path,
            "regime_metrics_csv": regime_metrics_path,
            "summary_json": summary_path,
            "metrics_plot": metrics_plot_path,
            "regime_plot": regime_plot_path,
            "first_hit_rank_plot": hit_rank_plot_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    return summary_payload


def run_offline_evaluation(
    artifacts,
    output_dir=EVALUATION_DIR,
    k_values=DEFAULT_K_VALUES,
    positive_rating=DEFAULT_POSITIVE_RATING,
    min_history=DEFAULT_MIN_HISTORY,
    holdout_size=DEFAULT_HOLDOUT_SIZE,
    max_users=DEFAULT_MAX_USERS,
    random_state=42,
):
    """End-to-end offline evaluation using the trained recommender artifacts."""
    from recommendation_engine import RecommendationEngine

    ratings_df = load_movielens_ratings()
    movielens_movies_df = load_movielens_movies()
    tmdb_df = artifacts["tmdb_df"]

    test_users, dataset_stats, candidate_indices = build_test_users(
        ratings_df=ratings_df,
        movielens_movies_df=movielens_movies_df,
        tmdb_df=tmdb_df,
        positive_rating=positive_rating,
        min_history=min_history,
        holdout_size=holdout_size,
        max_users=max_users,
        random_state=random_state,
    )

    engine = RecommendationEngine(
        tmdb_df=tmdb_df,
        tmdb_latent=artifacts["tmdb_latent"],
        mlb=artifacts["mlb"],
        ridge=artifacts["ridge"],
        user_factors=artifacts.get("user_factors"),
        movie_factors=artifacts.get("movie_factors"),
    )

    results = evaluate_recommendations(
        engine,
        test_users,
        k_values=k_values,
        candidate_indices=candidate_indices,
    )
    summary_payload = save_evaluation_artifacts(results, dataset_stats, output_dir=output_dir)
    return {
        "dataset_stats": dataset_stats,
        "overall": results["overall"],
        "metrics_by_k": results["metrics_by_k"],
        "per_user": results["per_user"],
        "regime_metrics": results["regime_metrics"],
        "artifact_paths": summary_payload["artifact_paths"],
        "summary": summary_payload,
    }


def print_evaluation_results(results):
    """Print a concise evaluation summary."""
    dataset = results["dataset_stats"]
    overall = results["overall"]
    metrics_df = results["metrics_by_k"]

    focal_row = metrics_df.loc[metrics_df["k"] == overall["focal_k"]]
    focal_metrics = focal_row.iloc[0].to_dict() if not focal_row.empty else {}

    print(f"\n{'=' * 52}")
    print("  Offline Evaluation Summary")
    print(f"{'=' * 52}")
    print(f"  Evaluated users: {overall['evaluated_users']}")
    print(f"  MovieLens -> TMDB alignment rate: {dataset['alignment_rate']:.2%}")
    print(f"  Catalog coverage@max_k: {overall['catalog_coverage_at_max_k']:.2%}")
    if overall["mean_first_hit_rank"] is not None:
        print(f"  Mean first-hit rank: {overall['mean_first_hit_rank']:.2f}")
    print(f"  Focal metrics (K={overall['focal_k']}):")
    print(f"    Precision: {focal_metrics.get('precision', 0.0):.4f}")
    print(f"    Recall:    {focal_metrics.get('recall', 0.0):.4f}")
    print(f"    NDCG:      {focal_metrics.get('ndcg', 0.0):.4f}")
    print(f"    Hit Rate:  {focal_metrics.get('hit_rate', 0.0):.4f}")
    print(f"    MRR:       {focal_metrics.get('mrr', 0.0):.4f}")
    print(f"{'=' * 52}")


def _load_artifacts_from_disk():
    """Load model artifacts when the evaluator is run as a standalone script."""
    from model_training import load_models

    loaded = load_models()
    return {
        "tmdb_df": loaded["tmdb_dataset"],
        "tmdb_latent": loaded["tmdb_latent"],
        "mlb": loaded["mlb"],
        "ridge": loaded["ridge_model"],
        "user_factors": loaded.get("user_factors"),
        "movie_factors": loaded.get("movie_factors"),
    }


def main():
    """CLI entry point for standalone evaluation runs."""
    parser = argparse.ArgumentParser(description="Run offline recommendation evaluation.")
    parser.add_argument("--output-dir", default=EVALUATION_DIR, help="Directory for metrics and graphs.")
    parser.add_argument("--max-users", type=int, default=DEFAULT_MAX_USERS, help="Maximum users to evaluate.")
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY, help="Minimum positive history per user.")
    parser.add_argument("--holdout-size", type=int, default=DEFAULT_HOLDOUT_SIZE, help="Relevant items held out per user.")
    parser.add_argument("--positive-rating", type=int, default=DEFAULT_POSITIVE_RATING, help="Minimum rating treated as positive.")
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="Ranking cutoffs to evaluate, for example: --k-values 5 10 20",
    )
    args = parser.parse_args()

    artifacts = _load_artifacts_from_disk()
    results = run_offline_evaluation(
        artifacts=artifacts,
        output_dir=args.output_dir,
        k_values=tuple(args.k_values),
        positive_rating=args.positive_rating,
        min_history=args.min_history,
        holdout_size=args.holdout_size,
        max_users=args.max_users,
    )
    print_evaluation_results(results)

    print("Saved evaluation artifacts:")
    for label, path in results["artifact_paths"].items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
    import os
    import pandas as pd
    metrics_path = "evaluation_outputs/metrics_by_k.csv"
    if os.path.exists(metrics_path):
        print("\n=== FULL K METRICS ===")
        print(pd.read_csv(metrics_path))
        print(pd.read_csv("evaluation_outputs/regime_metrics.csv"))
