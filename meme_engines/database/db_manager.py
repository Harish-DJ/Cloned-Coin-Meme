import sqlite3
import os
from datetime import datetime

# Central DB Path
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "meme_data.sqlite")

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    """Initializes the database schema for the Immutable Raw Data Layer."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Raw Posts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_posts (
        id TEXT PRIMARY KEY,
        coin_ticker TEXT NOT NULL,
        platform TEXT NOT NULL,
        text TEXT NOT NULL,
        created_at DATETIME NOT NULL,
        favorite_count INTEGER DEFAULT 0,
        retweet_count INTEGER DEFAULT 0,
        reply_count INTEGER DEFAULT 0,
        author_karma INTEGER DEFAULT 0,
        is_verified BOOLEAN DEFAULT 0
    )
    """)
    
    # 2. Add an index for fetching by ticker/time quickly
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_coin_time 
    ON raw_posts (coin_ticker, created_at)
    """)

    conn.commit()
    conn.close()
    print(f"[DB Manager] Initialized database at {DB_PATH}")

def fetch_recent_posts(ticker: str, limit: int = 100):
    """Utility to pull the latest posts for a coin from the DB."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT * FROM raw_posts
    WHERE coin_ticker = ?
    ORDER BY created_at DESC
    LIMIT ?
    """, (ticker.upper(), limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert rows to dicts
    return [dict(r) for r in rows]

if __name__ == "__main__":
    create_tables()
