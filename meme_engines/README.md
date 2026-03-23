# Meme Coin Trend Prediction — Backend

Multi-engine social signal analyzer that predicts meme coin trends
from Reddit, Twitter, and Discord posts.

---

## Project Structure

```
meme_coin_backend/
│
├── engines/
│   ├── text_cleaning_engine.py      # Part 1 — clean raw posts
│   ├── sentiment_engine.py          # Part 1 — VADER + influencer weighting
│   ├── engagement_engine.py         # Part 2 — cross-platform engagement scores
│   ├── trend_spike_engine.py        # Part 2 — hourly mentions + spike detection
│   └── contextual_topic_engine.py   # Part 3 — bullish/bearish keyword signals
│
├── models/
│   ├── feature_fusion.py            # Part 3 — fuses all engine outputs
│   ├── trend_prediction_engine.py   # Part 4 — Random Forest + confidence scores
│   └── trend_model.pkl              # saved after first train (auto-generated)
│
├── api/
│   └── routes.py                    # Part 5 — all FastAPI endpoints
│
├── main.py                          # Part 5 — app entry point
└── requirements.txt                 # Part 5 — dependencies
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the API
uvicorn main:app --reload --port 8000

# 3. Open Swagger docs
http://localhost:8000/docs
```

---

## API Endpoints

| Method | Path                    | Description                              |
|--------|-------------------------|------------------------------------------|
| GET    | /api/v1/health          | Health check + model status              |
| POST   | /api/v1/train           | Train the ML model (run once)            |
| POST   | /api/v1/analyze         | Full pipeline — posts → predictions      |
| POST   | /api/v1/coin/{ticker}   | Predict trend for a single coin          |
| POST   | /api/v1/batch-coins     | All coins ranked by meme viral score     |
| GET    | /api/v1/trending        | Top trending coins (after /analyze)      |

---

## Sample Request — /api/v1/analyze

```json
POST /api/v1/analyze
{
  "posts": [
    {
      "id": "1",
      "text": "DOGE to the moon! Massive pump incoming #DOGE",
      "platform": "twitter",
      "created_at": "2024-06-01T15:00:00Z",
      "favorite_count": 1200,
      "retweet_count": 450,
      "reply_count": 80,
      "author_metadata": { "karma": 50000, "is_verified": true }
    }
  ],
  "coins": ["DOGE", "PEPE", "SHIB"]
}
```

## Sample Response

```json
{
  "predictions": {
    "DOGE": {
      "trend_label": "upward",
      "confidence": 0.82,
      "probabilities": { "upward": 0.82, "downward": 0.12, "neutral": 0.06 },
      "meme_viral_score": 74.3,
      "hype_state_score": 0.71,
      "spike_flag": true,
      "momentum_score": 18.5,
      "top_keywords": ["moon", "pump", "hodl"],
      "predictor": "random_forest"
    }
  },
  "top_trending": [...],
  "posts_analyzed": 1
}
```

---

## Engine Pipeline

```
Raw Posts
   │
   ▼
TextCleaningEngine       → clean_text, is_spam
   │
   ▼
SentimentEngine          → sentiment_score, author_weight
   │
   ▼
EngagementEngine         → engagement_score (cross-platform normalized)
   │
   ▼
ContextualTopicEngine    → keyword_score, dominant_signal, topic_flags
   │
   ▼
TrendSpikeEngine         → spike_flag, momentum_score, anomaly_pct
   │
   ▼
FeatureFusionLayer       → 15-feature vector + hype_state_score per coin
   │
   ▼
TrendPredictionEngine    → trend_label, confidence, meme_viral_score
```
