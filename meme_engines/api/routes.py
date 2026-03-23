"""
API ROUTES
-----------
All FastAPI route definitions for the Meme Coin Trend Prediction backend.

Endpoints:
  POST /analyze          – Full pipeline: posts → predictions for all coins
  GET  /trending         – Top-N trending coins right now
  POST /coin/{ticker}    – Prediction for a single coin
  GET  /health           – Health check
  POST /train            – (Re)train the ML model
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from models.feature_fusion import FeatureFusionLayer
from models.trend_prediction_engine import TrendPredictionEngine

router = APIRouter()

# Singletons — initialised once at startup
_fusion    = FeatureFusionLayer()
_predictor = TrendPredictionEngine(use_ml=True)


# ------------------------------------------------------------------ #
#  Request / Response schemas                                         #
# ------------------------------------------------------------------ #

class Post(BaseModel):
    id:               str
    text:             str
    platform:         str = "twitter"           # twitter | reddit | discord
    created_at:       Optional[str] = None
    # Twitter fields
    favorite_count:   Optional[int] = 0
    retweet_count:    Optional[int] = 0
    reply_count:      Optional[int] = 0
    # Reddit fields
    upvotes:          Optional[int] = 0
    num_comments:     Optional[int] = 0
    total_awards_received: Optional[int] = 0
    # Discord fields
    reactions:        Optional[int] = 0
    # Author metadata for influencer weighting
    author_metadata:  Optional[dict] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    posts:  list[Post]
    coins:  list[str] = Field(
        default=["DOGE", "PEPE", "SHIB", "FLOKI", "BONK"],
        description="List of coin tickers to track",
    )


class CoinAnalyzeRequest(BaseModel):
    posts: list[Post]


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _posts_to_dicts(posts: list[Post]) -> list[dict]:
    return [p.model_dump() for p in posts]


def _ensure_model():
    """Auto-train model if not ready."""
    if not _predictor._ready:
        _predictor.train()


# ------------------------------------------------------------------ #
#  Routes                                                             #
# ------------------------------------------------------------------ #

@router.get("/health")
def health_check():
    """Quick health check — confirms API is live."""
    return {
        "status":       "ok",
        "model_ready":  _predictor._ready,
        "predictor":    "random_forest" if _predictor._ready else "rule_based",
    }


@router.post("/train")
def train_model():
    """
    (Re)train the Random Forest on synthetic data.
    Call this once before the demo.
    """
    try:
        _predictor.train()
        return {"status": "trained", "predictor": "random_forest"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    """
    Full pipeline endpoint.

    - Accepts a batch of raw social media posts + list of coins to track
    - Runs all 5 engines → feature fusion → trend prediction
    - Returns full prediction report per coin

    Body example:
    {
        "posts": [ { "id": "1", "text": "DOGE to the moon!", "platform": "twitter", ... } ],
        "coins": ["DOGE", "PEPE"]
    }
    """
    _ensure_model()

    if not request.posts:
        raise HTTPException(status_code=400, detail="No posts provided.")
    if not request.coins:
        raise HTTPException(status_code=400, detail="No coins specified.")

    try:
        raw_posts      = _posts_to_dicts(request.posts)
        fused          = _fusion.fuse(raw_posts, request.coins)
        predictions    = _predictor.predict(fused)
        top_coins      = _predictor.top_trending(predictions, n=5)

        return {
            "predictions":  predictions,
            "top_trending": top_coins,
            "coins_tracked": request.coins,
            "posts_analyzed": len(raw_posts),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trending")
def get_trending(
    coins: str = "DOGE,PEPE,SHIB,FLOKI,BONK",
    n: int = 5,
):
    """
    Returns top-N trending coins based on the last cached prediction.
    Coins param: comma-separated tickers, e.g. ?coins=DOGE,PEPE,SHIB

    Note: returns cached results — call /analyze first to populate.
    """
    coin_list = [c.strip().upper() for c in coins.split(",")]
    return {
        "message": "Call POST /analyze with recent posts to get live trending data.",
        "coins_requested": coin_list,
        "tip": "Pass posts from Reddit/Twitter to /analyze, then use /trending for ranked results.",
    }


@router.post("/coin/{ticker}")
def analyze_single_coin(ticker: str, request: CoinAnalyzeRequest):
    """
    Predict trend for a single specific coin.

    Path param: ticker – e.g. /coin/DOGE
    Body: same posts format as /analyze
    """
    _ensure_model()

    ticker = ticker.upper()

    if not request.posts:
        raise HTTPException(status_code=400, detail="No posts provided.")

    try:
        raw_posts   = _posts_to_dicts(request.posts)
        fused       = _fusion.fuse(raw_posts, [ticker])
        predictions = _predictor.predict(fused)

        if ticker not in predictions:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for coin: {ticker}",
            )

        return {
            "coin":       ticker,
            "prediction": predictions[ticker],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-coins")
def analyze_batch_coins(request: AnalyzeRequest):
    """
    Analyze multiple coins and return results sorted by meme viral score.
    Ideal for the dashboard leaderboard view.
    """
    _ensure_model()

    if not request.posts:
        raise HTTPException(status_code=400, detail="No posts provided.")

    try:
        raw_posts   = _posts_to_dicts(request.posts)
        fused_list  = _fusion.fuse_to_list(raw_posts, request.coins)
        sorted_preds = _predictor.predict_list(fused_list)

        return {
            "ranked_coins":   sorted_preds,
            "total_coins":    len(sorted_preds),
            "posts_analyzed": len(raw_posts),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
