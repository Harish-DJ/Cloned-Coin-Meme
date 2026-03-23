"""
ENGINE 3: Engagement & Metadata Engine
----------------------------------------
Computes a normalized engagement score for each post across platforms.
- Combines likes, comments, shares/retweets with platform-aware weights
- Normalizes scores across platforms so Reddit upvotes and Twitter
  likes are comparable
- Output: numeric engagement_score per post
"""


# ------------------------------------------------------------------ #
#  Platform-specific field maps & normalization caps                  #
# ------------------------------------------------------------------ #

PLATFORM_CONFIG = {
    "reddit": {
        "like_field":     "upvotes",
        "comment_field":  "num_comments",
        "share_field":    None,           # Reddit has no native retweet
        "award_field":    "total_awards_received",
        "like_weight":    1.0,
        "comment_weight": 1.5,           # comments signal deeper engagement
        "share_weight":   0.0,
        "award_weight":   2.0,
        "norm_cap":       5_000,          # clip before normalizing
    },
    "twitter": {
        "like_field":     "favorite_count",
        "comment_field":  "reply_count",
        "share_field":    "retweet_count",
        "award_field":    None,
        "like_weight":    1.0,
        "comment_weight": 1.2,
        "share_weight":   1.8,           # retweets = viral amplification
        "award_weight":   0.0,
        "norm_cap":       10_000,
    },
    "discord": {
        "like_field":     "reactions",
        "comment_field":  "reply_count",
        "share_field":    None,
        "award_field":    None,
        "like_weight":    1.0,
        "comment_weight": 1.3,
        "share_weight":   0.0,
        "award_weight":   0.0,
        "norm_cap":       2_000,
    },
}

DEFAULT_PLATFORM = "twitter"


# ------------------------------------------------------------------ #
#  Engine                                                             #
# ------------------------------------------------------------------ #

class EngagementMetadataEngine:
    """
    Computes a cross-platform normalized engagement score per post.
    Score range: 0.0 – 1.0  (higher = more engaged)
    """

    def _get_config(self, platform: str) -> dict:
        return PLATFORM_CONFIG.get(platform.lower(), PLATFORM_CONFIG[DEFAULT_PLATFORM])

    def _raw_engagement(self, post: dict, cfg: dict) -> float:
        """Weighted sum of raw engagement signals."""
        likes    = post.get(cfg["like_field"],    0) or 0 if cfg["like_field"]    else 0
        comments = post.get(cfg["comment_field"], 0) or 0 if cfg["comment_field"] else 0
        shares   = post.get(cfg["share_field"],   0) or 0 if cfg["share_field"]   else 0
        awards   = post.get(cfg["award_field"],   0) or 0 if cfg["award_field"]   else 0

        return (
            likes    * cfg["like_weight"]    +
            comments * cfg["comment_weight"] +
            shares   * cfg["share_weight"]   +
            awards   * cfg["award_weight"]
        )

    def compute(self, post: dict) -> dict:
        """
        Compute engagement for a single post.

        Required post keys:
            platform : str   – "reddit" | "twitter" | "discord"
            + platform-specific count fields (see PLATFORM_CONFIG)

        Returns
        -------
        {
            "raw_engagement"  : float,
            "engagement_score": float,   # 0.0 – 1.0 normalized
            "platform"        : str,
        }
        """
        platform = post.get("platform", DEFAULT_PLATFORM)
        cfg      = self._get_config(platform)

        raw   = self._raw_engagement(post, cfg)
        norm  = min(raw / cfg["norm_cap"], 1.0)   # clip then normalize

        return {
            "raw_engagement":   round(raw, 2),
            "engagement_score": round(norm, 4),
            "platform":         platform,
        }

    def compute_batch(self, posts: list) -> list:
        """
        Enrich a list of post dicts with engagement scores.
        Injects: raw_engagement, engagement_score
        """
        results = []
        for post in posts:
            eng = self.compute(post)
            results.append({
                **post,
                "raw_engagement":   eng["raw_engagement"],
                "engagement_score": eng["engagement_score"],
            })
        return results

    def aggregate_coin_engagement(self, posts: list, coin: str) -> dict:
        """
        Average engagement metrics for a specific coin across all posts.

        Returns
        -------
        {
            "coin"              : str,
            "avg_engagement"    : float,
            "total_posts"       : int,
            "top_post_engagement: float,
        }
        """
        coin_posts = [
            p for p in posts
            if coin.lower() in p.get("clean_text", "").lower()
        ]

        if not coin_posts:
            return {
                "coin": coin, "avg_engagement": 0.0,
                "total_posts": 0, "top_post_engagement": 0.0,
            }

        scores = [p.get("engagement_score", 0.0) for p in coin_posts]

        return {
            "coin":               coin,
            "avg_engagement":     round(sum(scores) / len(scores), 4),
            "total_posts":        len(scores),
            "top_post_engagement": round(max(scores), 4),
        }
