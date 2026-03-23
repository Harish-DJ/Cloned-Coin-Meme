import json
import os
import sys
from datetime import datetime, timezone

# Add parent dir so we can import from engines/models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.feature_fusion import FeatureFusionLayer
from models.trend_prediction_engine import TrendPredictionEngine

def parse_twitter(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    posts = []
    for item in raw.get('data', []):
        metrics = item.get('public_metrics', {})
        posts.append({
            "id": str(item.get('id')),
            "text": item.get('text', ''),
            "platform": "twitter",
            "created_at": item.get('created_at'),
            "favorite_count": metrics.get('like_count', 0),
            "retweet_count": metrics.get('retweet_count', 0),
            "reply_count": metrics.get('reply_count', 0),
            "author_metadata": {"author_id": item.get('author_id')}
        })
    return posts

def parse_reddit(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    posts = []
    children = raw.get('data', {}).get('children', [])
    for child in children:
        data = child.get('data', {})
        # Combine title and selftext for better analysis
        full_text = f"{data.get('title', '')}\n\n{data.get('selftext', '')}".strip()
        
        created_at = datetime.fromtimestamp(data.get('created_utc', 0), tz=timezone.utc).isoformat()
        
        posts.append({
            "id": data.get('name'),
            "text": full_text,
            "platform": "reddit",
            "created_at": created_at,
            "upvotes": data.get('ups', 0),
            "num_comments": data.get('num_comments', 0),
            "author_metadata": {"author": data.get('author')}
        })
    return posts

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    twitter_path = os.path.join(root_dir, "data", "mock_twitter_500.json")
    reddit_path = os.path.join(root_dir, "data", "CryptoMoonShots.json")
    
    print(f"[DataProcessor] Loading Twitter data from {twitter_path}")
    twitter_posts = parse_twitter(twitter_path)
    
    print(f"[DataProcessor] Loading Reddit data from {reddit_path}")
    reddit_posts = parse_reddit(reddit_path)
    
    all_posts = twitter_posts + reddit_posts
    print(f"[DataProcessor] Total posts to analyze: {len(all_posts)}")
    
    # Initialize engines
    fusion = FeatureFusionLayer()
    predictor = TrendPredictionEngine(use_ml=True)
    
    # Define coins to track
    coins = ["DOGE", "PEPE", "SHIB", "FLOKI", "BONK", "WIF"]
    
    print("[Pipeline] Running Feature Fusion & Sentiment Analysis...")
    fused = fusion.fuse(all_posts, coins)
    
    print("[Pipeline] Running Trend Prediction Model...")
    predictions = predictor.predict(fused)
    
    # Format final result
    output = {
        "metadata": {
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_posts_processed": len(all_posts),
            "engines_active": [
                "TextCleaningEngine",
                "SentimentEngine (Twitter-RoBERTa)",
                "EngagementMetadataEngine",
                "TrendSpikeEngine",
                "ContextualTopicEngine"
            ],
            "model_version": "1.0.0 (RandomForest Ensemble)"
        },
        "coins": predictions
    }
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mock_analysis_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
    
    print(f"[Success] Updated analysis result saved to {output_path}")

if __name__ == "__main__":
    main()
