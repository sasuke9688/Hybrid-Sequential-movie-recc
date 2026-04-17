---
title: Hybrid Movie Recommendation Engine
emoji: 🎥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Hybrid Sequential Movie Recommendation Engine

This repository contains a production-grade movie recommendation system utilizing a Lambda architecture. It is designed to provide high-precision collaborative filtering while maintaining real-time responsiveness to user interactions.

## Architecture Overview

The system is deployed on Hugging Face Spaces (16GB RAM) utilizing a custom Docker container, supported by an automated CI/CD pipeline via GitHub Actions.

* **Speed Layer (Fast Sync):** A lightweight GitHub Actions workflow triggers every 4 hours to extract the latest user interaction data from the Supabase PostgreSQL database, applying real-time updates to the latent user vectors.
* **Batch Layer (Global Retraining):** A heavy workflow executes weekly to download the updated TMDB dataset (131,000+ movies) via the Kaggle API, recalculating the global item-space matrix using Truncated SVD to resolve item cold-start issues.
* **Data Persistence:** User credentials and watch histories are securely managed via Supabase, utilizing `PBKDF2` for cryptographic password hashing.

## Technology Stack

* **Backend:** Python 3.11, Flask, Gunicorn
* **Machine Learning:** Scikit-learn (Truncated SVD, Ridge Regression), NumPy, Pandas
* **Database:** Supabase (PostgreSQL)
* **Infrastructure:** Hugging Face Spaces (Docker SDK), GitHub Actions, Git LFS

## Deployment Protocol

This application utilizes a "Clean Snapshot" deployment methodology to bypass Git LFS history restrictions. Pushes to the `main` branch automatically trigger a GitHub Action that isolates the core assets and forces a synchronized deployment to the Hugging Face container registry.
