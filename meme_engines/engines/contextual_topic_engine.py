"""
ENGINE 5: Contextual / Topic Engine
--------------------------------------
Detects trend-driving language that plain sentiment misses.
- Scans cleaned text for curated keyword categories
- Each category has a directional signal (bullish / bearish / neutral)
- Returns topic_flags, dominant_signal, and a keyword_score per post

Why this matters:
  "to the moon" is neutral in VADER but strongly bullish here.
  "dev rugged" is neutral in VADER but strongly bearish here.
"""


# ------------------------------------------------------------------ #
#  Keyword taxonomy                                                   #
# ------------------------------------------------------------------ #

KEYWORD_GROUPS = {
    "bullish": {
        "signal":   "bullish",
        "weight":   1.0,
        "keywords": [
            "to the moon", "moon", "mooning", "pump", "pumping",
            "buy now", "buy the dip", "dip", "bullish", "breakout",
            "100x", "1000x", "massive gains", "gem", "undervalued",
            "accumulate", "hodl", "hold", "all time high", "ath",
            "next big", "early", "alpha", "send it", "wagmi",
            "we are so back", "back", "rip and run",
        ],
    },
    "bearish": {
        "signal":   "bearish",
        "weight":   1.0,
        "keywords": [
            "sell now", "sell", "dump", "dumping", "crash", "crashing",
            "rug", "rug pull", "rugged", "scam", "exit scam", "honeypot",
            "dead", "dead coin", "avoid", "stay away", "ngmi",
            "not gonna make it", "bearish", "correction", "collapse",
            "down bad", "rekt", "wrecked", "liquidated",
        ],
    },
    "hype_neutral": {
        "signal":   "hype",
        "weight":   0.5,
        "keywords": [
            "trending", "viral", "everyone talking", "hype", "hyped",
            "blowing up", "exploding", "hot", "fire", "on fire",
            "community growing", "new listing", "listed", "partnership",
            "announcement", "big news", "major update",
        ],
    },
    "fud": {
        "signal":   "bearish",
        "weight":   0.8,
        "keywords": [
            "fud", "fear", "uncertainty", "doubt", "suspicious",
            "whale dump", "whale selling", "manipulation", "ponzi",
            "insider selling", "team dumping",
        ],
    },
    "meme_viral": {
        "signal":   "bullish",
        "weight":   0.7,
        "keywords": [
            "meme", "memes", "dank", "based", "this is the way",
            "cope", "seethe", "gigachad", "chad move", "nfa dyor",
            "just vibes", "vibes only", "cult", "army",
        ],
    },
}


# ------------------------------------------------------------------ #
#  Engine                                                             #
# ------------------------------------------------------------------ #

class ContextualTopicEngine:
    """
    Scans cleaned post text for trend-driving keyword categories.
    No ML required — fast regex-based matching for hackathon speed.
    """

    def __init__(self):
        import re
        self._patterns = {}
        for category, config in KEYWORD_GROUPS.items():
            # Build one compiled pattern per category
            escaped = [re.escape(kw) for kw in config["keywords"]]
            pattern = r"\b(" + "|".join(escaped) + r")\b"
            self._patterns[category] = re.compile(pattern, re.IGNORECASE)

    # ---------------------------------------------------------------- #
    #  Single post                                                      #
    # ---------------------------------------------------------------- #

    def analyze(self, clean_text: str) -> dict:
        """
        Analyze a single cleaned post for contextual signals.

        Returns
        -------
        {
            "topic_flags"      : dict,   # {category: [matched_keywords]}
            "keyword_score"    : float,  # net directional score (-1 to +1)
            "dominant_signal"  : str,    # "bullish" | "bearish" | "hype" | "none"
            "matched_count"    : int,    # total keyword hits
        }
        """
        if not clean_text:
            return {
                "topic_flags":    {},
                "keyword_score":  0.0,
                "dominant_signal": "none",
                "matched_count":  0,
            }

        topic_flags   = {}
        bullish_score = 0.0
        bearish_score = 0.0
        hype_score    = 0.0
        total_matches = 0

        for category, pattern in self._patterns.items():
            matches = pattern.findall(clean_text)
            if matches:
                topic_flags[category] = list(set(m.lower() for m in matches))
                total_matches += len(matches)

                cfg    = KEYWORD_GROUPS[category]
                signal = cfg["signal"]
                weight = cfg["weight"]
                count  = len(matches)

                if signal == "bullish":
                    bullish_score += count * weight
                elif signal == "bearish":
                    bearish_score += count * weight
                elif signal == "hype":
                    hype_score    += count * weight

        # Net keyword score: positive = bullish, negative = bearish
        net    = bullish_score - bearish_score
        denom  = bullish_score + bearish_score + hype_score or 1
        norm   = max(-1.0, min(1.0, net / denom))

        # Dominant signal
        if bullish_score == 0 and bearish_score == 0 and hype_score == 0:
            dominant = "none"
        elif bullish_score >= bearish_score and bullish_score >= hype_score:
            dominant = "bullish"
        elif bearish_score > bullish_score:
            dominant = "bearish"
        else:
            dominant = "hype"

        return {
            "topic_flags":     topic_flags,
            "keyword_score":   round(norm, 4),
            "dominant_signal": dominant,
            "matched_count":   total_matches,
        }

    # ---------------------------------------------------------------- #
    #  Batch                                                            #
    # ---------------------------------------------------------------- #

    def analyze_batch(self, posts: list) -> list:
        """
        Enrich a batch of post dicts with contextual features.
        Injects: topic_flags, keyword_score, dominant_signal, matched_count
        """
        results = []
        for post in posts:
            ctx = self.analyze(post.get("clean_text", ""))
            results.append({
                **post,
                "topic_flags":     ctx["topic_flags"],
                "keyword_score":   ctx["keyword_score"],
                "dominant_signal": ctx["dominant_signal"],
                "matched_count":   ctx["matched_count"],
            })
        return results

    # ---------------------------------------------------------------- #
    #  Coin-level aggregation                                          #
    # ---------------------------------------------------------------- #

    def aggregate_coin_context(self, posts: list, coin: str) -> dict:
        """
        Aggregate contextual signals across all posts for a coin.

        Returns
        -------
        {
            "coin"                : str,
            "avg_keyword_score"   : float,
            "bullish_post_ratio"  : float,
            "bearish_post_ratio"  : float,
            "hype_post_ratio"     : float,
            "top_keywords"        : list[str],   # top 10 most-seen keywords
            "total_posts"         : int,
        }
        """
        from collections import Counter

        coin_posts = [
            p for p in posts
            if coin.lower() in p.get("clean_text", "").lower()
        ]

        if not coin_posts:
            return {
                "coin": coin, "avg_keyword_score": 0.0,
                "bullish_post_ratio": 0.0, "bearish_post_ratio": 0.0,
                "hype_post_ratio": 0.0, "top_keywords": [], "total_posts": 0,
            }

        n       = len(coin_posts)
        scores  = [p.get("keyword_score", 0.0) for p in coin_posts]
        signals = [p.get("dominant_signal", "none") for p in coin_posts]

        # Collect all matched keywords across posts
        kw_counter = Counter()
        for p in coin_posts:
            for kw_list in p.get("topic_flags", {}).values():
                kw_counter.update(kw_list)

        return {
            "coin":               coin,
            "avg_keyword_score":  round(sum(scores) / n, 4),
            "bullish_post_ratio": round(signals.count("bullish") / n, 3),
            "bearish_post_ratio": round(signals.count("bearish") / n, 3),
            "hype_post_ratio":    round(signals.count("hype")    / n, 3),
            "top_keywords":       [kw for kw, _ in kw_counter.most_common(10)],
            "total_posts":        n,
        }
