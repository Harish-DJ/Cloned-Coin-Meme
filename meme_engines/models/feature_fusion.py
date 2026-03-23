"""
FEATURE FUSION LAYER
----------------------
Combines outputs from all 4 engines into a single feature vector
per coin per time window. This is the direct input to the
Trend Prediction Engine.

Feature vector per coin:
  - sentiment_score      (Sentiment Engine)
  - positive_ratio       (Sentiment Engine)
  - negative_ratio       (Sentiment Engine)
  - avg_engagement       (Engagement Engine)
  - top_post_engagement  (Engagement Engine)
  - mentions_per_hour    (Trend/Spike Engine)
  - spike_flag           (Trend/Spike Engine)  → int 0/1
  - spike_intensity      (Trend/Spike Engine)
  - momentum_score       (Trend/Spike Engine)
  - anomaly_pct          (Trend/Spike Engine)
  - keyword_score        (Contextual Engine)
  - bullish_post_ratio   (Contextual Engine)
  - bearish_post_ratio   (Contextual Engine)
  - hype_post_ratio      (Contextual Engine)
  - hype_state_score     (Composite — weighted blend of all signals)
"""

from engines.sentiment_engine       import SentimentEngine
from engines.engagement_engine      import EngagementMetadataEngine
from engines.trend_spike_engine     import TrendSpikeEngine
from engines.contextual_topic_engine import ContextualTopicEngine
from engines.text_cleaning_engine   import TextCleaningEngine


# ------------------------------------------------------------------ #
#  Fusion weights for the composite hype_state_score                 #
# ------------------------------------------------------------------ #

HYPE_WEIGHTS = {
    "sentiment":   0.25,
    "engagement":  0.20,
    "momentum":    0.25,
    "keyword":     0.20,
    "spike":       0.10,   # binary bonus
}


class FeatureFusionLayer:
    """
    Orchestrates all engines and fuses their outputs into
    a clean feature dict per coin, ready for the ML model.
    """

    def __init__(self):
        self.cleaner    = TextCleaningEngine()
        self.sentiment  = SentimentEngine()
        self.engagement = EngagementMetadataEngine()
        self.trend      = TrendSpikeEngine()
        self.context    = ContextualTopicEngine()

    # ---------------------------------------------------------------- #
    #  Step 1 — run all engines on the raw posts                       #
    # ---------------------------------------------------------------- #

    def _run_engines(self, raw_posts: list) -> list:
        """
        Full preprocessing pipeline on raw posts.
        Returns enriched post list with all engine outputs attached.
        """
        posts = self.cleaner.clean_batch(raw_posts)

        # Filter spam before feeding NLP engines
        posts = [p for p in posts if not p.get("is_spam", False)]

        posts = self.sentiment.analyze_batch(posts)
        posts = self.engagement.compute_batch(posts)
        posts = self.context.analyze_batch(posts)

        return posts

    # ---------------------------------------------------------------- #
    #  Step 2 — build feature vector per coin                          #
    # ---------------------------------------------------------------- #

    def _build_coin_vector(
        self,
        coin: str,
        posts: list,
        trend_data: dict,
    ) -> dict:
        """
        Merge per-coin aggregations from all engines into one flat dict.
        """
        sent = self.sentiment.aggregate_coin_sentiment(posts, coin)
        eng  = self.engagement.aggregate_coin_engagement(posts, coin)
        ctx  = self.context.aggregate_coin_context(posts, coin)
        tr   = trend_data.get(coin, {})

        spike_flag      = 1 if tr.get("spike_flag", False) else 0
        spike_intensity = tr.get("spike_intensity", 1.0)
        momentum        = tr.get("momentum_score", 0.0)
        anomaly_pct     = tr.get("anomaly_pct", 0.0)
        current_mentions = tr.get("current_mentions", 0)

        # ── Composite hype_state_score ──────────────────────────────
        # Normalise momentum to 0-1 using a soft cap of 50 mentions/hr delta
        norm_momentum = max(0.0, min(1.0, (momentum + 50) / 100))
        # Normalise keyword_score from -1..+1 to 0..1
        norm_keyword  = (ctx.get("avg_keyword_score", 0.0) + 1.0) / 2.0
        # Normalise sentiment from -1..+1 to 0..1
        norm_sentiment = (sent.get("avg_sentiment", 0.0) + 1.0) / 2.0

        hype_score = (
            HYPE_WEIGHTS["sentiment"]  * norm_sentiment          +
            HYPE_WEIGHTS["engagement"] * eng.get("avg_engagement", 0.0) +
            HYPE_WEIGHTS["momentum"]   * norm_momentum            +
            HYPE_WEIGHTS["keyword"]    * norm_keyword             +
            HYPE_WEIGHTS["spike"]      * spike_flag
        )

        return {
            "coin": coin,

            # — Sentiment features —
            "sentiment_score":     sent.get("avg_sentiment", 0.0),
            "positive_ratio":      sent.get("positive_ratio", 0.0),
            "negative_ratio":      sent.get("negative_ratio", 0.0),

            # — Engagement features —
            "avg_engagement":      eng.get("avg_engagement", 0.0),
            "top_post_engagement": eng.get("top_post_engagement", 0.0),

            # — Trend / Spike features —
            "mentions_per_hour":   current_mentions,
            "spike_flag":          spike_flag,
            "spike_intensity":     spike_intensity,
            "momentum_score":      momentum,
            "anomaly_pct":         anomaly_pct,

            # — Contextual features —
            "keyword_score":       ctx.get("avg_keyword_score", 0.0),
            "bullish_post_ratio":  ctx.get("bullish_post_ratio", 0.0),
            "bearish_post_ratio":  ctx.get("bearish_post_ratio", 0.0),
            "hype_post_ratio":     ctx.get("hype_post_ratio", 0.0),
            "top_keywords":        ctx.get("top_keywords", []),

            # — Composite —
            "hype_state_score":    round(hype_score, 4),

            # — Meta —
            "total_posts_analyzed": sent.get("total_posts", 0),
        }

    # ---------------------------------------------------------------- #
    #  Public: fuse everything                                          #
    # ---------------------------------------------------------------- #

    def fuse(self, raw_posts: list, coins: list) -> dict:
        """
        Full pipeline: clean → enrich → trend → fuse per coin.

        Parameters
        ----------
        raw_posts : list of raw post dicts from Reddit/Twitter/Discord
        coins     : list of coin tickers to track, e.g. ["DOGE","PEPE"]

        Returns
        -------
        {
            "DOGE": { ...feature_vector... },
            "PEPE": { ...feature_vector... },
            ...
        }
        """
        enriched_posts = self._run_engines(raw_posts)
        trend_data     = self.trend.compute_all(enriched_posts, coins)

        result = {}
        for coin in coins:
            result[coin] = self._build_coin_vector(coin, enriched_posts, trend_data)

        return result

    def fuse_to_list(self, raw_posts: list, coins: list) -> list:
        """
        Same as fuse() but returns a list of feature vectors
        (convenient for feeding directly into the ML model).
        """
        fused = self.fuse(raw_posts, coins)
        return list(fused.values())
