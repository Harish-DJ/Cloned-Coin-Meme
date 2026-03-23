import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.sentiment_engine import SentimentEngine

engine = SentimentEngine()
print("Engine pipeline loaded:", engine.pipeline is not None)

test_posts = [
    {"text": "DOGE is going to the moon! BUY NOW! 🚀🚀", "clean_text": "doge is going to the moon buy now"},
    {"text": "PEPE is dead, total scam.", "clean_text": "pepe is dead total scam"}
]

results = engine.analyze_batch(test_posts)
import json
with open("test_sentiment.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)


