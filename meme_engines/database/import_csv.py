import csv
import sqlite3
import uuid
from datetime import datetime
from db_manager import get_connection, create_tables

def import_csv_to_db(csv_file_path: str, default_coin: str = None):
    """
    Reads a CSV dataset and imports it into the raw_posts table.
    Expects standard columns but provides fallbacks if some are missing.
    """
    create_tables()  # Ensure tables exist
    
    conn = get_connection()
    cursor = conn.cursor()
    
    success_count = 0
    error_count = 0

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            try:
                # Map CSV columns (handle flexible naming conventions)
                # Fallback to UUID if no ID is present
                post_id = row.get("id") or row.get("post_id") or str(uuid.uuid4())
                
                # Ticker can be provided per row or defaulted
                ticker = row.get("coin_ticker") or row.get("ticker") or row.get("coin") or default_coin
                if not ticker:
                    continue  # Skip row if we don't know what coin this belongs to
                     
                text = row.get("text") or row.get("content") or row.get("tweet") or ""
                platform = row.get("platform") or row.get("source") or "unknown"
                
                # Try to parse timestamps; default to now if missing
                created_at = row.get("created_at") or row.get("timestamp") or row.get("date")
                if not created_at:
                    created_at = datetime.utcnow().isoformat() + "Z"
                    
                favorites = int(row.get("favorite_count") or row.get("likes") or row.get("upvotes") or 0)
                retweets = int(row.get("retweet_count") or row.get("retweets") or row.get("shares") or 0)
                replies = int(row.get("reply_count") or row.get("replies") or row.get("comments") or 0)
                
                karma = int(row.get("author_karma") or row.get("karma") or row.get("followers") or 0)
                
                # Boolean logic for verified
                verified_str = str(row.get("is_verified") or row.get("verified") or "False").lower()
                is_verified = 1 if verified_str in ["true", "1", "yes"] else 0
                
                cursor.execute("""
                    INSERT OR IGNORE INTO raw_posts 
                    (id, coin_ticker, platform, text, created_at, favorite_count, retweet_count, reply_count, author_karma, is_verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post_id, ticker.upper(), platform.lower(), text, created_at, 
                    favorites, retweets, replies, karma, is_verified
                ))
                
                success_count += 1
            except Exception as e:
                print(f"[Import Error] Failed to parse row: {e}")
                error_count += 1

    conn.commit()
    conn.close()
    
    print(f"\n--- Import Complete ---")
    print(f"Data File: {csv_file_path}")
    print(f"Successfully Inserted: {success_count} posts")
    if error_count > 0:
        print(f"Errors/Skipped: {error_count} posts")
    print("-----------------------\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import CSV dataset into the Meme Database.")
    parser.add_argument("csv_path", help="Path to the CSV file.")
    parser.add_argument("--coin", "-c", help="Default coin ticker if the CSV doesn't specify one (e.g., DOGE).", default=None)
    
    args = parser.parse_args()
    
    try:
        import_csv_to_db(args.csv_path, args.coin)
    except FileNotFoundError:
        print(f"Error: Could not find file at '{args.csv_path}'")
