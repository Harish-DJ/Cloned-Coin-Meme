"""
ENGINE 4: Trend / Spike Detection Engine
------------------------------------------
Computes time-series mention frequency features per coin and
detects anomalous spikes vs. historical baseline.

Core outputs per coin:
  - mentions_per_hour   : rolling count
  - spike_flag          : True if current mentions > N x baseline
  - spike_intensity     : how extreme the spike is (e.g. 3.7x baseline)
  - momentum_score      : how fast sentiment + mentions are growing
  - anomaly_percentile  : "faster than X% of historical trends"
"""

from collections import defaultdict
from datetime import datetime, timezone
import statistics


# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

SPIKE_MULTIPLIER   = 3.0    # flag spike if mentions > 3x rolling avg
MOMENTUM_WINDOW    = 3      # hours to compute momentum over
MIN_BASELINE_HOURS = 6      # need at least this many hours of history


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _hour_bucket(ts: str) -> str:
    """
    Convert an ISO-8601 timestamp string to 'YYYY-MM-DD HH' bucket.
    Falls back to current UTC hour if parsing fails.
    """
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d %H")


def _safe_stdev(values: list) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


# ------------------------------------------------------------------ #
#  Engine                                                             #
# ------------------------------------------------------------------ #

class TrendSpikeEngine:
    """
    Builds hourly mention time-series per coin and detects spikes.

    Usage
    -----
    engine = TrendSpikeEngine()
    features = engine.compute_all(posts, coins=["DOGE", "PEPE", "SHIB"])
    """

    # ---------------------------------------------------------------- #
    #  Step 1 – build hourly mention series                            #
    # ---------------------------------------------------------------- #

    def build_mention_series(self, posts: list, coins: list) -> dict:
        """
        Count how many posts mention each coin per hour bucket.

        Returns
        -------
        {
            "DOGE": {"2024-06-01 14": 12, "2024-06-01 15": 45, ...},
            ...
        }
        """
        series = {coin: defaultdict(int) for coin in coins}

        for post in posts:
            text = post.get("clean_text", "").lower()
            ts   = post.get("created_at", "")
            bucket = _hour_bucket(ts)

            for coin in coins:
                if coin.lower() in text:
                    series[coin][bucket] += 1

        # Convert defaultdict → plain dict, sorted by time
        return {
            coin: dict(sorted(buckets.items()))
            for coin, buckets in series.items()
        }

    # ---------------------------------------------------------------- #
    #  Step 2 – spike detection per coin                               #
    # ---------------------------------------------------------------- #

    def detect_spike(self, mention_series: dict) -> dict:
        """
        Given a sorted hour→count dict for one coin, detect spikes.

        Returns
        -------
        {
            "current_mentions" : int,
            "baseline_avg"     : float,
            "spike_flag"       : bool,
            "spike_intensity"  : float,   # current / baseline (1.0 = normal)
            "anomaly_pct"      : float,   # percentile vs. all historical hours
        }
        """
        counts = list(mention_series.values())

        if not counts:
            return {
                "current_mentions": 0, "baseline_avg": 0.0,
                "spike_flag": False, "spike_intensity": 1.0,
                "anomaly_pct": 0.0,
            }

        current   = counts[-1]
        history   = counts[:-1] if len(counts) > 1 else counts

        baseline  = statistics.mean(history) if history else 0.0
        intensity = (current / baseline) if baseline > 0 else 1.0
        spike     = intensity >= SPIKE_MULTIPLIER

        # Anomaly percentile: what % of historical hours had fewer mentions
        pct = (sum(1 for h in history if h < current) / len(history) * 100
               if history else 0.0)

        return {
            "current_mentions": current,
            "baseline_avg":     round(baseline, 2),
            "spike_flag":       spike,
            "spike_intensity":  round(intensity, 3),
            "anomaly_pct":      round(pct, 1),
        }

    # ---------------------------------------------------------------- #
    #  Step 3 – momentum score                                         #
    # ---------------------------------------------------------------- #

    def compute_momentum(self, mention_series: dict,
                          sentiment_series: dict = None) -> float:
        """
        Momentum = average rate of change of mentions over the last
        MOMENTUM_WINDOW hours, optionally boosted by sentiment trend.

        Returns a float: positive = growing, negative = shrinking.
        """
        counts = list(mention_series.values())
        if len(counts) < 2:
            return 0.0

        window = counts[-MOMENTUM_WINDOW:]
        if len(window) < 2:
            return 0.0

        # Simple average delta per hour
        deltas = [window[i] - window[i - 1] for i in range(1, len(window))]
        momentum = statistics.mean(deltas)

        # Optional sentiment boost
        if sentiment_series:
            sent_vals = list(sentiment_series.values())
            if len(sent_vals) >= 2:
                sent_delta = sent_vals[-1] - sent_vals[-2]
                momentum += sent_delta * 10  # scale sentiment contribution

        return round(momentum, 3)

    # ---------------------------------------------------------------- #
    #  Public: compute all trend features for all coins                #
    # ---------------------------------------------------------------- #

    def compute_all(self, posts: list, coins: list) -> dict:
        """
        Full pipeline: build series → detect spikes → compute momentum.

        Returns
        -------
        {
            "DOGE": {
                "mention_series"  : {...},
                "current_mentions": int,
                "baseline_avg"    : float,
                "spike_flag"      : bool,
                "spike_intensity" : float,
                "anomaly_pct"     : float,
                "momentum_score"  : float,
            },
            ...
        }
        """
        mention_series = self.build_mention_series(posts, coins)
        results = {}

        for coin in coins:
            series  = mention_series.get(coin, {})
            spike   = self.detect_spike(series)
            momentum = self.compute_momentum(series)

            results[coin] = {
                "mention_series":   series,
                **spike,
                "momentum_score":   momentum,
            }

        return results
