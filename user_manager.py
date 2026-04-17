"""
User management module.
Handles user registration, login, and watch history storage using JSON files.
Each user's data is stored in data/users/<username>.json.
"""

import os
import json
import hashlib
import secrets
import time

from config import DATA_DIR, RATING_SCALE_MAX

USERS_DIR = os.path.join(DATA_DIR, "users")


def _users_dir():
    os.makedirs(USERS_DIR, exist_ok=True)
    return USERS_DIR


def _user_path(username):
    safe = "".join(c for c in username if c.isalnum() or c in "_-")
    return os.path.join(_users_dir(), f"{safe}.json")


def _hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return salt, hashed


def register_user(username, password):
    """
    Register a new user. Returns (success, message).
    """
    username = username.strip().lower()
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    path = _user_path(username)
    if os.path.exists(path):
        return False, "Username already exists."

    salt, hashed = _hash_password(password)
    user_data = {
        "username": username,
        "password_hash": hashed,
        "password_salt": salt,
        "created_at": time.time(),
        "watch_history": [],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2)

    return True, "Registration successful."


def authenticate_user(username, password):
    """
    Authenticate a user. Returns (success, message).
    """
    username = username.strip().lower()
    path = _user_path(username)

    if not os.path.exists(path):
        return False, "User not found."

    with open(path, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    salt = user_data["password_salt"]
    _, hashed = _hash_password(password, salt)

    if hashed != user_data["password_hash"]:
        return False, "Incorrect password."

    return True, "Login successful."


def _load_user(username):
    username = username.strip().lower()
    path = _user_path(username)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    if _migrate_history_ratings(user_data):
        _save_user(username, user_data)

    return user_data


def _save_user(username, user_data):
    username = username.strip().lower()
    path = _user_path(username)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(user_data, f, indent=2)


def _normalize_rating(rating):
    """Normalize legacy or incoming ratings to the active star scale."""
    if rating is None:
        return None

    value = float(rating)
    if value > RATING_SCALE_MAX:
        value /= 2.0

    value = int(round(value))
    return max(1, min(RATING_SCALE_MAX, value))


def _migrate_history_ratings(user_data):
    """Keep stored history ratings consistent with the current 5-star UI."""
    changed = False
    for entry in user_data.get("watch_history", []):
        if entry.get("rating") is None:
            continue
        normalized = _normalize_rating(entry["rating"])
        if entry["rating"] != normalized:
            entry["rating"] = normalized
            changed = True
    return changed


def add_to_watch_history(username, movie_index, movie_title, release_year, rating=None):
    """
    Add a movie to the user's watch history with current timestamp and optional rating.
    Avoids duplicates (by index). If already present, updates the timestamp and rating.
    """
    user_data = _load_user(username)
    if user_data is None:
        return False, "User not found."

    history = user_data.get("watch_history", [])

    # Check for duplicate by index, update timestamp and rating if found
    for entry in history:
        if entry["index"] == movie_index:
            entry["timestamp"] = time.time()
            if rating is not None:
                entry["rating"] = _normalize_rating(rating)
            user_data["watch_history"] = history
            _save_user(username, user_data)
            return True, "Watch timestamp updated."

    entry = {
        "index": movie_index,
        "title": movie_title,
        "release_year": release_year,
        "timestamp": time.time(),
    }
    if rating is not None:
        entry["rating"] = _normalize_rating(rating)

    history.append(entry)
    user_data["watch_history"] = history
    _save_user(username, user_data)
    return True, "Movie added to watch history."


def update_rating(username, movie_index, rating):
    """Update the rating for a movie in the user's watch history."""
    user_data = _load_user(username)
    if user_data is None:
        return False, "User not found."

    for entry in user_data.get("watch_history", []):
        if entry["index"] == movie_index:
            entry["rating"] = _normalize_rating(rating)
            _save_user(username, user_data)
            return True, "Rating updated."

    return False, "Movie not found in history."


def remove_from_watch_history(username, movie_index):
    """Remove a movie from user's watch history."""
    user_data = _load_user(username)
    if user_data is None:
        return False, "User not found."

    history = user_data.get("watch_history", [])
    user_data["watch_history"] = [e for e in history if e["index"] != movie_index]
    _save_user(username, user_data)
    return True, "Movie removed from history."


def get_watch_history(username):
    """
    Get user's full watch history sorted by timestamp (newest first).
    Returns list of dicts with index, title, release_year, timestamp.
    """
    user_data = _load_user(username)
    if user_data is None:
        return []

    history = user_data.get("watch_history", [])
    history.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return history
