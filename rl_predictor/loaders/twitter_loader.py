"""
twitter_loader.py
─────────────────
Loads mock_twitter_200 (1).json and converts it into the same 10-feature
vector format used by TrendDataSimulator, so the RL agent can train on
real social signal data instead of purely synthetic numbers.

Feature mapping:
  0  sentiment_score     → positive/negative word heuristic on tweet text
  1  mention_count       → # tweets per time window (normalized)
  2  mention_growth_rate → % change vs previous window
  3  engagement_score    → (likes + retweets) normalized
  4  spike_score         → 1 if engagement > 2x rolling mean
  5  cross_platform_score→ impression_count normalized (reach proxy)
  6  momentum_score      → rolling 3-window mean of sentiment
  7  influencer_score    → quote_count normalized (amplification)
  8  keyword_hype_score  → hashtag density per window
  9  volatility_proxy    → reply_count / like_count (controversy ratio)

Labels (actual trend):
  2 = UP      if next-window engagement > 20% above current
  0 = DOWN    if next-window engagement < 10% below current
  1 = NEUTRAL otherwise

Dataset is expanded 4x via Gaussian augmentation so the RL agent has
~400+ steps to train on (standard practice for small datasets).
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

# ── Sentiment lexicon (no heavy NLP deps) ─────────────────────────────────────
POSITIVE_WORDS = {
    "bullish", "pump", "moon", "buy", "buying", "bought", "up", "ath",
    "win", "gains", "profit", "breaking", "spike", "surge", "rising",
    "growth", "strong", "alpha", "listing", "confirmed", "launch", "gem",
    "printing", "green", "rocket", "chart", "trophy",
    "rally", "recovery", "partnership", "announced", "growing", "organic"
}
NEGATIVE_WORDS = {
    "bearish", "dump", "sell", "selling", "sold", "down", "crash", "dip",
    "rekt", "loss", "rug", "scam", "warning", "caution", "risk", "fear",
    "overbought", "top", "liquidation", "panic", "dumped", "bag", "regret",
    "crying", "chart_down", "alert", "wrong", "sad", "pullback", "correction"
}

def compute_sentiment(text: str) -> float:
    """Returns sentiment score in [-1, 1] using keyword heuristic."""
    text_lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    total = pos + neg
    if total == 0:
        return 0.0
    return float(pos - neg) / total


class TwitterDataLoader:
    """
    Loads the mock Twitter JSON and produces the same interface as
    TrendDataSimulator: self.features (n_steps × 10), self.labels (n_steps,).
    """

    def __init__(self, json_path: str = None, augment_factor: int = 4, seed: int = 42):
        np.random.seed(seed)
        if json_path is None:
            # Auto-locate relative to this file (now in data/ directory, two levels up from loaders/)
            base = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(base, "..", "..", "data", "mock_twitter_200 (1).json")

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        tweets = raw["data"]
        self._process(tweets, augment_factor)

    def _process(self, tweets: list, augment_factor: int):
        # ── Parse timestamps ───────────────────────────────────────────────────
        rows = []
        for t in tweets:
            try:
                ts = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
            except Exception:
                continue
            m = t.get("public_metrics", {})
            rows.append({
                "ts": ts,
                "hour": ts.replace(minute=0, second=0, microsecond=0),
                "text": t.get("text", ""),
                "likes": m.get("like_count", 0),
                "retweets": m.get("retweet_count", 0),
                "replies": m.get("reply_count", 0),
                "quotes": m.get("quote_count", 0),
                "impressions": m.get("impression_count", 0),
                "hashtag_count": len(t.get("entities", {}).get("hashtags", [])),
                "sentiment": compute_sentiment(t.get("text", ""))
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("ts").reset_index(drop=True)

        # ── Group by hour window ───────────────────────────────────────────────
        grp = df.groupby("hour").agg(
            tweet_count=("text", "count"),
            avg_sentiment=("sentiment", "mean"),
            total_likes=("likes", "sum"),
            total_retweets=("retweets", "sum"),
            total_replies=("replies", "sum"),
            total_quotes=("quotes", "sum"),
            total_impressions=("impressions", "sum"),
            total_hashtags=("hashtag_count", "sum"),
        ).reset_index().sort_values("hour")

        # ── Derived signals ────────────────────────────────────────────────────
        grp["engagement"] = grp["total_likes"] + grp["total_retweets"]
        grp["mention_growth"] = grp["tweet_count"].pct_change().fillna(0)
        grp["momentum"] = grp["avg_sentiment"].rolling(3, min_periods=1).mean()
        grp["rolling_eng"] = grp["engagement"].rolling(3, min_periods=1).mean()
        grp["spike"] = (grp["engagement"] > grp["rolling_eng"] * 2).astype(float)
        grp["controversy"] = (grp["total_replies"] / (grp["total_likes"] + 1)).clip(0, 1)
        grp["influencer"] = grp["total_quotes"] / (grp["total_quotes"].max() + 1)
        grp["hype"] = grp["total_hashtags"] / (grp["total_hashtags"].max() + 1)

        # ── Labels from next-window engagement change ─────────────────────────
        grp["next_eng"] = grp["engagement"].shift(-1).fillna(grp["engagement"])
        grp["eng_change"] = (grp["next_eng"] - grp["engagement"]) / (grp["engagement"] + 1)
        labels = np.ones(len(grp), dtype=int)  # NEUTRAL
        labels[grp["eng_change"].values > 0.20] = 2   # UP
        labels[grp["eng_change"].values < -0.10] = 0  # DOWN

        # ── Normalize to [0,1] ────────────────────────────────────────────────
        def norm(s):
            mn, mx = s.min(), s.max()
            if mx == mn:
                return pd.Series(np.zeros(len(s)))
            return (s - mn) / (mx - mn)

        feat_df = pd.DataFrame({
            "sentiment_score":      grp["avg_sentiment"].clip(-1, 1),
            "mention_count":        norm(grp["tweet_count"]),
            "mention_growth_rate":  norm(grp["mention_growth"].clip(-2, 2)),
            "engagement_score":     norm(grp["engagement"]),
            "spike_score":          grp["spike"],
            "cross_platform_score": norm(grp["total_impressions"]),
            "momentum_score":       norm(grp["momentum"]),
            "influencer_score":     grp["influencer"],
            "keyword_hype_score":   grp["hype"],
            "volatility_proxy":     grp["controversy"],
        })

        base_features = feat_df.values.astype(np.float32)
        base_labels   = labels

        # ── Augment: repeat with small Gaussian noise ─────────────────────────
        all_features = [base_features]
        all_labels   = [base_labels]
        for _ in range(augment_factor - 1):
            noise = np.random.normal(0, 0.03, base_features.shape).astype(np.float32)
            aug = np.clip(base_features + noise, -1.0, 1.0)
            all_features.append(aug)
            all_labels.append(base_labels)

        self.features = np.concatenate(all_features, axis=0)
        self.labels   = np.concatenate(all_labels,   axis=0)
        self.n_steps  = len(self.features)

        print(f"[TwitterDataLoader] Loaded {len(base_features)} windows "
              f"-> augmented to {self.n_steps} steps.")
        label_counts = {0: int((self.labels==0).sum()),
                        1: int((self.labels==1).sum()),
                        2: int((self.labels==2).sum())}
        print(f"[TwitterDataLoader] Labels: "
              f"DOWN={label_counts[0]}, NEUTRAL={label_counts[1]}, UP={label_counts[2]}")

    def get_step(self, t: int):
        if t >= self.n_steps:
            return None, None
        return self.features[t], self.labels[t]


if __name__ == "__main__":
    loader = TwitterDataLoader()
    feat, label = loader.get_step(0)
    print(f"\nStep 0 features: {feat}")
    print(f"Step 0 label:    {['DOWN','NEUTRAL','UP'][label]}")
    feat2, label2 = loader.get_step(1)
    print(f"\nStep 1 features: {feat2}")
    print(f"Step 1 label:    {['DOWN','NEUTRAL','UP'][label2]}")
