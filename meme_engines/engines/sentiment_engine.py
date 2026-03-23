"""
ENGINE 2: Sentiment Engine (Upgraded)
----------------------------
Performs sentiment analysis on cleaned post text.
- Uses 'Shoriful025/Crypto-Sentiment-Analyzer' from HuggingFace
- Applies influencer weighting: posts from high-karma/verified users
  get a boosted sentiment contribution
- Output: sentiment_score (float, -1 to +1) per post
"""

import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ------------------------------------------------------------------ #
#  Influencer / author weighting                                      #
# ------------------------------------------------------------------ #

def compute_author_weight(author_metadata: dict) -> float:
    """
    Returns a multiplier (1.0 = normal, up to 3.0 = top influencer).
    """
    weight = 1.0

    karma = author_metadata.get("karma", 0) or 0
    if karma > 100_000:
        weight += 1.5
    elif karma > 10_000:
        weight += 0.8
    elif karma > 1_000:
        weight += 0.3

    if author_metadata.get("is_verified", False):
        weight += 0.5

    if author_metadata.get("is_mod", False):
        weight += 0.2

    # Cap at 3.0 so no single post dominates
    return min(weight, 3.0)


# ------------------------------------------------------------------ #
#  Sentiment Engine                                                   #
# ------------------------------------------------------------------ #

class SentimentEngine:
    """
    Wraps the HuggingFace Crypto Sentiment Model and applies author weighting
    to produce a weighted sentiment score per post.
    """

    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.pipeline = None
        
        if HAS_TRANSFORMERS:
            try:
                print(f"[SentimentEngine] Loading local HuggingFace model: {self.model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
                print(f"[SentimentEngine] Model loaded successfully.")
            except Exception as e:
                print(f"[SentimentEngine] Failed to load model: {e}")
                self.pipeline = None
        else:
            print("[SentimentEngine] Transformers not installed. Run: pip install transformers torch")

    def _map_label_to_score(self, label: str, score: float) -> tuple:
        """
        Maps HF labels (e.g., 'Positive', 'Negative', 'Neutral' or 'LABEL_1')
        to a numeric score between -1.0 and 1.0.
        """
        label_upper = str(label).upper()
        
        # Determine the sign based on label name
        if "POS" in label_upper or label_upper in ["LABEL_2", "1"]:  # Assuming 2 is positive
            numeric_score = score
            normalized_label = "positive"
        elif "NEG" in label_upper or label_upper in ["LABEL_0", "-1"]: # Assuming 0 is negative
            numeric_score = -score
            normalized_label = "negative"
        else:
            numeric_score = 0.0
            normalized_label = "neutral"
            
        return numeric_score, normalized_label

    def analyze(self, clean_text: str, author_metadata: dict = None) -> dict:
        """
        Analyze sentiment for a single cleaned post using the AI model.
        """
        if not clean_text or not self.pipeline:
            return {
                "raw_score":      0.0,
                "weighted_score": 0.0,
                "author_weight":  1.0,
                "label":          "neutral",
            }

        # Run inference
        try:
            # The pipeline usually expects maximum sequence lengths, so we truncate just in case
            result = self.pipeline(clean_text[:512])[0]
            raw_score, label = self._map_label_to_score(str(result['label']), float(result['score']))
        except Exception as e:
            raw_score, label = 0.0, "neutral"

        author_weight = compute_author_weight(author_metadata or {})
        weighted_score = max(-1.0, min(1.0, float(raw_score) * author_weight))

        return {
            "raw_score":      float(round(raw_score, 4)),
            "weighted_score": float(round(weighted_score, 4)),
            "author_weight":  float(round(author_weight, 2)),
            "label":          label,
        }

    def analyze_batch(self, posts: list) -> list:
        """
        Analyze a batch of post dicts. (Can be optimized to pass a list to pipeline).
        """
        if not self.pipeline or not posts:
            # Fallback if no model loaded
            for p in posts:
                p.update({"sentiment_score": 0.0, "sentiment_label": "neutral", "author_weight": 1.0})
            return posts

        # Optimization: Pass all texts to pipeline at once
        texts = [str(p.get("clean_text", ""))[:512] for p in posts]
        results = []
        
        try:
            # Inference in batch
            hf_results = self.pipeline(texts)
            print("HF RAW RESULTS:", hf_results)
            
            for post, hf_res in zip(posts, hf_results):
                raw_score, label = self._map_label_to_score(str(hf_res['label']), float(hf_res['score']))
                
                author_weight = compute_author_weight(post.get("author_metadata") or {})
                weighted_score = max(-1.0, min(1.0, float(raw_score) * author_weight))
                
                post.update({
                    "sentiment_score":  float(round(weighted_score, 4)),
                    "sentiment_label":  label,
                    "author_weight":    float(round(author_weight, 2)),
                })
                results.append(post)
        except Exception as e:
            print(f"[SentimentEngine] Batch inference error: {e}")
            # Fallback
            for p in posts:
                p.update({"sentiment_score": 0.0, "sentiment_label": "neutral", "author_weight": 1.0})
                results.append(p)
                
        return results

    def aggregate_coin_sentiment(self, posts: list, coin: str) -> dict:
        """
        Aggregate sentiment across all posts for a given coin.
        """
        coin_posts = [
            p for p in posts
            if coin.lower() in p.get("clean_text", "").lower()
        ]

        if not coin_posts:
            return {
                "coin": coin, "avg_sentiment": 0.0,
                "positive_ratio": 0.0, "negative_ratio": 0.0,
                "neutral_ratio": 1.0, "total_posts": 0,
            }

        scores = [p.get("sentiment_score", 0.0) for p in coin_posts]
        labels = [p.get("sentiment_label", "neutral") for p in coin_posts]
        n = len(labels)

        return {
            "coin":            coin,
            "avg_sentiment":   round(sum(scores) / n, 4) if n > 0 else 0.0,
            "positive_ratio":  round(labels.count("positive") / n, 3) if n > 0 else 0.0,
            "negative_ratio":  round(labels.count("negative") / n, 3) if n > 0 else 0.0,
            "neutral_ratio":   round(labels.count("neutral")  / n, 3) if n > 0 else 0.0,
            "total_posts":     n,
        }
