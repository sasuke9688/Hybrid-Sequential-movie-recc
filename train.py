"""
Main training script.
Downloads datasets if needed, runs the full training pipeline, and evaluates the model.

The TMDB dataset is fetched from Kaggle (updated daily) to ensure the latest movies
are always included in the recommendation catalog.
"""

import os
import sys
import glob
import zipfile
import subprocess
import requests

from config import (
    DATA_DIR, EVALUATION_DIR, MODEL_BUNDLE_FILENAME, MODEL_DIR, MOVIELENS_RATINGS, TMDB_MOVIES
)


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
KAGGLE_TMDB_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"


def download_movielens(data_dir=DATA_DIR):
    """Download and extract MovieLens 1M dataset if not present."""
    ml_dir = os.path.join(data_dir, "ml-1m")
    if os.path.exists(os.path.join(ml_dir, "ratings.dat")):
        print("MovieLens 1M dataset already present.")
        return

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-1m.zip")

    print("Downloading MovieLens 1M dataset...")
    response = requests.get(MOVIELENS_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Downloaded: {pct:.1f}%", end="", flush=True)

    print("\n  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    os.remove(zip_path)
    print("  MovieLens 1M dataset ready.")


def download_tmdb_from_kaggle(data_dir=DATA_DIR):
    """
    Download the latest TMDB dataset from Kaggle using the kaggle CLI.
    This dataset is updated daily, so it always contains the newest movies.

    Requires:
      - pip install kaggle
      - ~/.kaggle/kaggle.json with valid API credentials
        (get from https://www.kaggle.com/settings -> "Create New Token")
    """
    os.makedirs(data_dir, exist_ok=True)

    print(f"  Downloading latest TMDB dataset from Kaggle...")
    print(f"  Dataset: {KAGGLE_TMDB_DATASET}")

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_TMDB_DATASET,
             "-p", data_dir, "--unzip", "--force"],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "kaggle.json" in stderr or "Could not find" in stderr or "403" in stderr:
                print("\n  ERROR: Kaggle API credentials not configured.")
                print("  To set up Kaggle API credentials:")
                print("    1. Go to https://www.kaggle.com/settings")
                print("    2. Click 'Create New Token' to download kaggle.json")
                print("    3. Place kaggle.json in ~/.kaggle/ (Linux/Mac)")
                print("       or C:\\Users\\<username>\\.kaggle\\ (Windows)")
                return False
            print(f"\n  Kaggle CLI error: {stderr}")
            return False

        print(f"  {result.stdout.strip()}")

        # Find the downloaded CSV file and rename to expected path
        csv_candidates = glob.glob(os.path.join(data_dir, "*.csv"))
        tmdb_candidates = [f for f in csv_candidates
                           if "tmdb" in os.path.basename(f).lower()
                           or "movie" in os.path.basename(f).lower()]

        target_path = TMDB_MOVIES
        if tmdb_candidates and not os.path.exists(target_path):
            source = tmdb_candidates[0]
            os.rename(source, target_path)
            print(f"  Renamed: {os.path.basename(source)} -> {os.path.basename(target_path)}")
        elif os.path.exists(target_path):
            print(f"  TMDB dataset ready at: {target_path}")
        elif csv_candidates:
            # Use the first CSV found
            source = csv_candidates[0]
            os.rename(source, target_path)
            print(f"  Renamed: {os.path.basename(source)} -> {os.path.basename(target_path)}")
        else:
            print("  WARNING: No CSV file found after download.")
            return False

        return True

    except FileNotFoundError:
        print("\n  ERROR: kaggle CLI not found. Install it with:")
        print("    pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print("\n  ERROR: Download timed out. Try again or download manually.")
        return False


def check_tmdb_dataset():
    """Check if TMDB dataset is present, attempt Kaggle download if not."""
    if os.path.exists(TMDB_MOVIES):
        return True

    print(f"\n  TMDB dataset not found at: {TMDB_MOVIES}")
    print("  Attempting to download from Kaggle (updated daily)...")

    if download_tmdb_from_kaggle():
        return os.path.exists(TMDB_MOVIES)

    print(f"\n  Could not auto-download TMDB dataset.")
    print("  Please download it manually from Kaggle:")
    print("    https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies")
    print(f"  and place the CSV file at: {TMDB_MOVIES}")
    return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("  HYBRID MOVIE RECOMMENDATION SYSTEM")
    print("  Setup and Training")
    print("=" * 60)

    # Step 1: Download MovieLens
    print("\n[1/4] Checking datasets...")
    download_movielens()

    if not check_tmdb_dataset():
        sys.exit(1)

    # Step 2: Run training pipeline
    print("\n[2/4] Running training pipeline...")
    from model_training import run_training_pipeline
    artifacts = run_training_pipeline()

    # Step 3: Offline evaluation
    print("\n[3/4] Running offline evaluation...")
    run_quick_evaluation(artifacts)

    # Step 4: Summary
    print("\n[4/4] Setup complete!")
    print(f"\nModel files saved to: {MODEL_DIR}/")
    print(f"Single-file bundle: {MODEL_DIR}/{MODEL_BUNDLE_FILENAME}")
    print(f"Evaluation outputs saved to: {EVALUATION_DIR}/")
    print("To start the web application, run:")
    print("  python app.py")
    print(f"\nThen open http://localhost:5000 in your browser.")


def run_quick_evaluation(artifacts):
    """Run offline evaluation and save graphs/metrics to disk."""
    from evaluation import print_evaluation_results, run_offline_evaluation

    results = run_offline_evaluation(
        artifacts=artifacts,
        output_dir=EVALUATION_DIR,
        k_values=(5, 10, 20),
        positive_rating=4,
        min_history=5,
        holdout_size=2,
        max_users=250,
    )
    print_evaluation_results(results)

    print("  Saved evaluation artifacts:")
    for label, path in results["artifact_paths"].items():
        print(f"    {label}: {path}")


if __name__ == "__main__":
    main()
