"""
User management and authentication module.
Interfaces directly with the Supabase PostgreSQL database.
Handles password hashing and CRUD operations for user watch history.
"""

import os
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client

# Initialize Supabase Client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
else:
    print("WARNING: Supabase credentials missing. User management will fail.")
    supabase = None


def register_user(username: str, password: str) -> tuple[bool, str]:
    """Hash password and create a new user record in Supabase."""
    if not supabase:
        return False, "Database connection error."
    
    username = username.lower().strip()
    
    if not username or not password:
        return False, "Username and password are required."

    # Check for existing user
    res = supabase.table("app_users").select("username").eq("username", username).execute()
    if len(res.data) > 0:
        return False, "Username already exists."

    hashed_pw = generate_password_hash(password)
    
    try:
        supabase.table("app_users").insert({
            "username": username,
            "password_hash": hashed_pw
        }).execute()
        return True, "Registration successful."
    except Exception as e:
        return False, f"Database error: {str(e)}"


def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    """Verify user credentials against stored hash."""
    if not supabase:
        return False, "Database connection error."
    
    username = username.lower().strip()

    res = supabase.table("app_users").select("password_hash").eq("username", username).execute()
    if len(res.data) == 0:
        return False, "Invalid username or password."

    stored_hash = res.data[0]["password_hash"]
    
    if check_password_hash(stored_hash, password):
        return True, "Login successful."
    
    return False, "Invalid username or password."


def add_to_watch_history(username: str, movie_index: int, title: str, release_year: int, rating: float = None) -> tuple[bool, str]:
    """Upsert a movie record into the user's watch history."""
    if not supabase:
        return False, "Database connection error."
    
    username = username.lower().strip()

    data = {
        "username": username,
        "movie_index": movie_index,
        "title": title,
        "release_year": str(release_year),
        "rating": rating
    }

    try:
        # Check if movie already exists in history
        res = supabase.table("watch_history").select("id").eq("username", username).eq("movie_index", movie_index).execute()
        
        if len(res.data) > 0:
            # Update existing record
            supabase.table("watch_history").update(data).eq("username", username).eq("movie_index", movie_index).execute()
        else:
            # Insert new record
            supabase.table("watch_history").insert(data).execute()
            
        return True, "Watch history updated."
    except Exception as e:
        return False, f"Database error: {str(e)}"


def update_rating(username: str, movie_index: int, rating: float) -> tuple[bool, str]:
    """Update the explicit rating for a specific movie in the history."""
    if not supabase:
        return False, "Database connection error."
    
    username = username.lower().strip()
    
    try:
        supabase.table("watch_history").update({"rating": rating}).eq("username", username).eq("movie_index", movie_index).execute()
        return True, "Rating updated successfully."
    except Exception as e:
        return False, f"Database error: {str(e)}"


def remove_from_watch_history(username: str, movie_index: int) -> tuple[bool, str]:
    """Delete a movie record from the user's watch history."""
    if not supabase:
        return False, "Database connection error."
    
    username = username.lower().strip()
    
    try:
        supabase.table("watch_history").delete().eq("username", username).eq("movie_index", movie_index).execute()
        return True, "Movie removed from history."
    except Exception as e:
        return False, f"Database error: {str(e)}"


def get_watch_history(username: str) -> list:
    """Retrieve the complete watch history for a specific user."""
    if not supabase:
        return []
    
    username = username.lower().strip()
    
    try:
        res = supabase.table("watch_history").select("*").eq("username", username).execute()
        
        history = []
        for row in res.data:
            history.append({
                "index": row["movie_index"],
                "title": row["title"],
                "release_year": row["release_year"],
                "rating": row["rating"]
            })
        return history
    except Exception:
        return []
