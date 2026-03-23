"""
TREND PREDICTION ENGINE
-------------------------
Takes the fused feature vectors from FeatureFusionLayer and
predicts trend direction per coin with confidence scores.

Model: Random Forest (primary) + XGBoost (optional ensemble)
Output per coin:
  - trend_label      : "upward" | "downward" | "neutral"
  - confidence       : float (0.0 – 1.0)
  - probabilities    : { upward: float, downward: float, neutral: float }
  - meme_viral_score : how likely this coin goes viral in next 24h

No training data? No problem — we include:
  1. A rule-based fallback predictor (works without any training)
  2. A synthetic data generator to train the ML model on the spot
  3. Model save/load so you train once and reuse at demo time
"""

import os
import json
import math
import random
import pickle
from datetime import datetime, timezone


# ------------------------------------------------------------------ #
#  Feature columns (must match FeatureFusionLayer output exactly)    #
# ------------------------------------------------------------------ #

FEATURE_COLUMNS = [
    "sentiment_score",
    "positive_ratio",
    "negative_ratio",
    "avg_engagement",
    "top_post_engagement",
    "mentions_per_hour",
    "spike_flag",
    "spike_intensity",
    "momentum_score",
    "anomaly_pct",
    "keyword_score",
    "bullish_post_ratio",
    "bearish_post_ratio",
    "hype_post_ratio",
    "hype_state_score",
]

LABEL_MAP     = {0: "downward", 1: "neutral", 2: "upward"}
LABEL_MAP_INV = {"downward": 0, "neutral": 1, "upward": 2}
MODEL_PATH    = "models/trend_model.pkl"


# ------------------------------------------------------------------ #
#  Synthetic training data generator                                  #
# ------------------------------------------------------------------ #

def generate_synthetic_data(n_samples: int = 1200) -> tuple:
    """
    Generate labelled training samples using domain-logic rules.
    This lets us train the RF model without real historical data.

    Returns (X: list[list[float]], y: list[int])
    """
    X, y = [], []
    random.seed(42)

    for _ in range(n_samples):
        sentiment      = random.uniform(-1, 1)
        pos_ratio      = random.uniform(0, 1)
        neg_ratio      = 1 - pos_ratio - random.uniform(0, 0.3)
        neg_ratio      = max(0.0, neg_ratio)
        engagement     = random.uniform(0, 1)
        top_eng        = random.uniform(engagement, min(1.0, engagement + 0.3))
        mentions       = random.randint(0, 200)
        spike_flag     = random.choice([0, 0, 0, 1])          # 25% spike rate
        spike_intensity = random.uniform(1, 6) if spike_flag else 1.0
        momentum       = random.uniform(-30, 50)
        anomaly_pct    = random.uniform(0, 100)
        keyword_score  = random.uniform(-1, 1)
        bull_ratio     = random.uniform(0, 1)
        bear_ratio     = 1 - bull_ratio - random.uniform(0, 0.2)
        bear_ratio     = max(0.0, bear_ratio)
        hype_ratio     = random.uniform(0, 0.4)
        hype_state     = (
            0.25 * ((sentiment + 1) / 2) +
            0.20 * engagement             +
            0.25 * max(0, min(1, (momentum + 50) / 100)) +
            0.20 * ((keyword_score + 1) / 2) +
            0.10 * spike_flag
        )

        features = [
            sentiment, pos_ratio, neg_ratio, engagement, top_eng,
            mentions, spike_flag, spike_intensity, momentum, anomaly_pct,
            keyword_score, bull_ratio, bear_ratio, hype_ratio,
            round(hype_state, 4),
        ]

        # Label via domain rules (mirrors what real data would show)
        if (sentiment > 0.3 and momentum > 5 and hype_state > 0.55) or \
           (spike_flag and spike_intensity > 3 and keyword_score > 0.2):
            label = 2   # upward
        elif (sentiment < -0.2 and momentum < -5) or \
             (bear_ratio > 0.5 and keyword_score < -0.2):
            label = 0   # downward
        else:
            label = 1   # neutral

        X.append(features)
        y.append(label)

    return X, y


# ------------------------------------------------------------------ #
#  Rule-based fallback predictor (no sklearn needed)                 #
# ------------------------------------------------------------------ #

class RuleBasedPredictor:
    """
    Fast deterministic fallback when the ML model isn't trained yet.
    Uses the hype_state_score + key signals to produce trend + confidence.
    """

    def predict_one(self, feature_vector: dict) -> dict:
        hype     = feature_vector.get("hype_state_score", 0.5)
        momentum = feature_vector.get("momentum_score", 0.0)
        spike    = feature_vector.get("spike_flag", 0)
        sentiment= feature_vector.get("sentiment_score", 0.0)
        bear_r   = feature_vector.get("bearish_post_ratio", 0.0)
        keyword  = feature_vector.get("keyword_score", 0.0)

        # Score upward evidence
        up_score   = 0.0
        down_score = 0.0

        up_score   += hype * 2
        up_score   += max(0, momentum / 50)
        up_score   += spike * 0.5
        up_score   += max(0, sentiment)
        up_score   += max(0, keyword)

        down_score += bear_r * 2
        down_score += max(0, -sentiment)
        down_score += max(0, -keyword)
        down_score += max(0, -momentum / 50)

        total = up_score + down_score + 0.001

        p_up   = up_score / total
        p_down = down_score / total
        p_neut = max(0.0, 1.0 - p_up - p_down)

        # Normalise
        s = p_up + p_down + p_neut
        p_up, p_down, p_neut = p_up/s, p_down/s, p_neut/s

        if p_up >= p_down and p_up >= p_neut:
            label, conf = "upward",   p_up
        elif p_down >= p_up and p_down >= p_neut:
            label, conf = "downward", p_down
        else:
            label, conf = "neutral",  p_neut

        return {
            "trend_label":   label,
            "confidence":    round(conf, 3),
            "probabilities": {
                "upward":   round(p_up,   3),
                "downward": round(p_down, 3),
                "neutral":  round(p_neut, 3),
            },
            "predictor": "rule_based",
        }


# ------------------------------------------------------------------ #
#  ML Predictor (Random Forest via sklearn)                          #
# ------------------------------------------------------------------ #

class MLPredictor:
    """
    Trains a Random Forest on synthetic (or real) data and predicts
    trend labels with probability-based confidence scores.
    """

    def __init__(self):
        self._model  = None
        self._trained = False

    def _extract_features(self, feature_vector: dict) -> list:
        return [feature_vector.get(col, 0.0) for col in FEATURE_COLUMNS]

    def train(self, X: list = None, y: list = None):
        """
        Train the Random Forest.
        If X/y not provided, generates synthetic training data automatically.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
        except ImportError:
            raise ImportError("pip install scikit-learn")

        if X is None or y is None:
            print("[MLPredictor] Generating synthetic training data...")
            X, y = generate_synthetic_data(n_samples=1500)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self._model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train)
        self._trained = True

        # Quick eval
        y_pred = self._model.predict(X_test)
        print("[MLPredictor] Training complete.")
        print(classification_report(
            y_test, y_pred,
            target_names=["downward", "neutral", "upward"]
        ))

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        print(f"[MLPredictor] Model saved -> {path}")

    def load(self, path: str = MODEL_PATH):
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self._trained = True
        print(f"[MLPredictor] Model loaded <- {path}")

    def predict_one(self, feature_vector: dict) -> dict:
        if not self._trained or self._model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        feats = [self._extract_features(feature_vector)]
        label_idx = int(self._model.predict(feats)[0])
        proba     = self._model.predict_proba(feats)[0]

        label = LABEL_MAP[label_idx]
        conf  = float(proba[label_idx])

        return {
            "trend_label":   label,
            "confidence":    round(conf, 3),
            "probabilities": {
                "upward":   round(float(proba[2]), 3),
                "downward": round(float(proba[0]), 3),
                "neutral":  round(float(proba[1]), 3),
            },
            "predictor": "random_forest",
        }


# ------------------------------------------------------------------ #
#  Meme Viral Score                                                   #
# ------------------------------------------------------------------ #

def compute_meme_viral_score(feature_vector: dict) -> float:
    """
    Predict 'meme potential' — how likely this coin goes viral
    in the next 24h. Judges love this over raw price prediction.

    Combines: momentum, spike intensity, hype ratio, anomaly percentile.
    Returns a 0–100 score.
    """
    momentum    = feature_vector.get("momentum_score", 0.0)
    spike_i     = feature_vector.get("spike_intensity", 1.0)
    hype_ratio  = feature_vector.get("hype_post_ratio", 0.0)
    anomaly_pct = feature_vector.get("anomaly_pct", 0.0)
    hype_state  = feature_vector.get("hype_state_score", 0.0)
    keyword_s   = feature_vector.get("keyword_score", 0.0)

    norm_momentum  = max(0.0, min(1.0, (momentum + 50) / 100))
    norm_spike     = max(0.0, min(1.0, (spike_i - 1) / 5))
    norm_anomaly   = anomaly_pct / 100.0
    norm_keyword   = (keyword_s + 1) / 2.0

    viral_score = (
        0.30 * norm_momentum  +
        0.20 * norm_spike     +
        0.15 * hype_ratio     +
        0.15 * norm_anomaly   +
        0.10 * hype_state     +
        0.10 * norm_keyword
    )

    return round(viral_score * 100, 1)   # 0–100


# ------------------------------------------------------------------ #
#  Main Trend Prediction Engine                                       #
# ------------------------------------------------------------------ #

class TrendPredictionEngine:
    """
    Top-level engine: takes fused feature vectors per coin,
    runs ML or rule-based predictor, and returns full prediction report.

    Usage
    -----
    engine = TrendPredictionEngine()
    engine.train()                         # once — trains RF on synthetic data
    predictions = engine.predict(fused)   # fused = output of FeatureFusionLayer
    """

    def __init__(self, use_ml: bool = True):
        self._use_ml  = use_ml
        self._ml      = MLPredictor()
        self._rules   = RuleBasedPredictor()
        self._ready   = False

    # ---------------------------------------------------------------- #
    #  Training / loading                                               #
    # ---------------------------------------------------------------- #

    def train(self, X=None, y=None):
        """Train the ML model (auto-generates data if X/y not provided)."""
        self._ml.train(X, y)
        self._ml.save()
        self._ready = True

    def load_model(self, path: str = MODEL_PATH):
        """Load a previously saved model."""
        self._ml.load(path)
        self._ready = True

    def ensure_ready(self):
        """Auto-train if model not ready yet."""
        if not self._ready:
            print("[TrendPredictionEngine] Auto-training on synthetic data...")
            self.train()

    # ---------------------------------------------------------------- #
    #  Prediction                                                       #
    # ---------------------------------------------------------------- #

    def _predict_one(self, feature_vector: dict) -> dict:
        if self._use_ml and self._ready:
            try:
                return self._ml.predict_one(feature_vector)
            except Exception as e:
                print(f"[TrendPredictionEngine] ML fallback to rules: {e}")
        return self._rules.predict_one(feature_vector)

    def predict(self, fused_features: dict) -> dict:
        """
        Predict trend for all coins in the fused feature dict.

        Parameters
        ----------
        fused_features : dict  – output of FeatureFusionLayer.fuse()
            { "DOGE": {...features...}, "PEPE": {...features...} }

        Returns
        -------
        {
            "DOGE": {
                "trend_label"      : "upward",
                "confidence"       : 0.82,
                "probabilities"    : {upward: 0.82, downward: 0.12, neutral: 0.06},
                "meme_viral_score" : 74.3,
                "hype_state_score" : 0.71,
                "spike_flag"       : True,
                "momentum_score"   : 18.5,
                "top_keywords"     : ["moon", "pump", "hodl"],
                "predictor"        : "random_forest",
                "predicted_at"     : "2024-06-01T15:00:00Z",
            },
            ...
        }
        """
        self.ensure_ready()
        results = {}
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for coin, features in fused_features.items():
            pred   = self._predict_one(features)
            viral  = compute_meme_viral_score(features)

            results[coin] = {
                # Core prediction
                "trend_label":       pred["trend_label"],
                "confidence":        pred["confidence"],
                "probabilities":     pred["probabilities"],

                # Meme potential
                "meme_viral_score":  viral,

                # Key signals (pass-through for dashboard)
                "hype_state_score":  features.get("hype_state_score", 0.0),
                "spike_flag":        bool(features.get("spike_flag", 0)),
                "spike_intensity":   features.get("spike_intensity", 1.0),
                "momentum_score":    features.get("momentum_score", 0.0),
                "anomaly_pct":       features.get("anomaly_pct", 0.0),
                "sentiment_score":   features.get("sentiment_score", 0.0),
                "mentions_per_hour": features.get("mentions_per_hour", 0),
                "top_keywords":      features.get("top_keywords", []),

                # Meta
                "predictor":         pred["predictor"],
                "predicted_at":      now_iso,
            }

        return results

    def predict_list(self, fused_list: list) -> list:
        """
        Accepts output of FeatureFusionLayer.fuse_to_list().
        Returns list of prediction dicts sorted by meme_viral_score desc.
        """
        fused_dict = {fv["coin"]: fv for fv in fused_list}
        preds      = self.predict(fused_dict)
        sorted_preds = sorted(
            preds.values(),
            key=lambda x: x["meme_viral_score"],
            reverse=True,
        )
        return sorted_preds

    def top_trending(self, predictions: dict, n: int = 5) -> list:
        """
        Return top-N coins most likely to trend upward,
        ranked by confidence * meme_viral_score.
        """
        scored = []
        for coin, pred in predictions.items():
            composite = pred["confidence"] * (pred["meme_viral_score"] / 100)
            scored.append({**pred, "coin": coin, "_rank_score": composite})

        scored.sort(key=lambda x: x["_rank_score"], reverse=True)
        return scored[:n]
