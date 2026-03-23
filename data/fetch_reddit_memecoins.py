"""
Reddit Meme Coin Data Collector
================================
Fetches real posts + top comments from meme coin subreddits
using Reddit's public JSON API (no credentials required).

Subreddits targeted:
  - r/dogecoin        → DOGE
  - r/pepecoin        → PEPE
  - r/SHIBArmy        → SHIB
  - r/FlokiInu        → FLOKI
  - r/WIFcoin         → WIF
  - r/bonkcoin        → BONK
  - r/CryptoMoonShots → mixed meme coins

Output: reddit_memecoins_<timestamp>.json
        (same structural philosophy as mock_twitter_200.json)

Requirements:
  pip install requests
"""

import requests
import json
import time
import re
from datetime import datetime, timezone

# ─── CONFIG ──────────────────────────────────────────────────────────────────

SUBREDDITS = [
    {"name": "dogecoin",        "coin": "DOGE"},
    {"name": "pepecoin",        "coin": "PEPE"},
    {"name": "SHIBArmy",        "coin": "SHIB"},
    {"name": "FlokiInu",        "coin": "FLOKI"},
    {"name": "WIFcoin",         "coin": "WIF"},
    {"name": "bonkcoin",        "coin": "BONK"},
    {"name": "CryptoMoonShots", "coin": "MIXED"},
]

POSTS_PER_SUB  = 25   # hot posts per subreddit  (7 subs × 25 = ~175 posts)
COMMENTS_PER_POST = 3  # top comments to fetch per post
SORT = "hot"           # hot | new | top

HEADERS = {
    "User-Agent": "MemeCoinDataCollector/1.0 (research script)"
}

COIN_PATTERN = re.compile(
    r'\b(DOGE|PEPE|SHIB|FLOKI|WIF|BONK|BTC|ETH|SOL)\b',
    re.IGNORECASE
)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def utc_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

def extract_coins(text: str) -> list[str]:
    found = COIN_PATTERN.findall(text or "")
    return list({c.upper() for c in found})

def safe_get(url: str, retries: int = 3, backoff: float = 2.0) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = backoff * (attempt + 1)
                print(f"  ⚠ Rate limited. Waiting {wait}s …")
                time.sleep(wait)
            elif r.status_code == 404:
                print(f"  ✗ 404 Not found: {url}")
                return None
            else:
                print(f"  ✗ HTTP {r.status_code} for {url}")
                return None
        except requests.RequestException as e:
            print(f"  ✗ Request error: {e}")
            time.sleep(backoff)
    return None

# ─── FETCHERS ────────────────────────────────────────────────────────────────

def fetch_posts(subreddit: str, coin: str) -> list[dict]:
    url = f"https://old.reddit.com/r/{subreddit}/{SORT}.json?limit={POSTS_PER_SUB}"
    print(f"  Fetching posts from r/{subreddit} …")
    raw = safe_get(url)
    if not raw:
        return []

    posts = []
    for child in raw.get("data", {}).get("children", []):
        d = child.get("data", {})
        if d.get("stickied"):
            continue  # skip mod/pinned posts

        full_text = f"{d.get('title', '')} {d.get('selftext', '')}"
        posts.append({
            "kind": "t3",
            "source_subreddit": subreddit,
            "primary_coin": coin,
            "data": {
                "id":                    d.get("id"),
                "name":                  d.get("name"),
                "subreddit":             d.get("subreddit"),
                "subreddit_id":          d.get("subreddit_id"),
                "title":                 d.get("title"),
                "selftext":              d.get("selftext", ""),
                "author":                d.get("author"),
                "author_fullname":       d.get("author_fullname"),
                "created_utc":           d.get("created_utc"),
                "created_iso":           utc_iso(d.get("created_utc", 0)),
                "score":                 d.get("score"),
                "upvote_ratio":          d.get("upvote_ratio"),
                "ups":                   d.get("ups"),
                "num_comments":          d.get("num_comments"),
                "url":                   d.get("url"),
                "permalink":             d.get("permalink"),
                "is_self":               d.get("is_self"),
                "link_flair_text":       d.get("link_flair_text"),
                "thumbnail":             d.get("thumbnail"),
                "domain":                d.get("domain"),
                "stickied":              d.get("stickied"),
                "locked":                d.get("locked"),
                "over_18":               d.get("over_18"),
                "spoiler":               d.get("spoiler"),
                "gilded":                d.get("gilded"),
                "total_awards_received": d.get("total_awards_received"),
                "coins_mentioned":       extract_coins(full_text),
            }
        })
    return posts


def fetch_comments(post_id: str, subreddit: str, coin: str) -> list[dict]:
    url = f"https://old.reddit.com/r/{subreddit}/comments/{post_id}.json?limit={COMMENTS_PER_POST}&depth=1&sort=top"
    raw = safe_get(url)
    if not raw or not isinstance(raw, list) or len(raw) < 2:
        return []

    comments = []
    for child in raw[1].get("data", {}).get("children", [])[:COMMENTS_PER_POST]:
        d = child.get("data", {})
        if child.get("kind") != "t1" or not d.get("body") or d.get("body") == "[deleted]":
            continue
        comments.append({
            "kind": "t1",
            "source_subreddit": subreddit,
            "primary_coin": coin,
            "data": {
                "id":                    d.get("id"),
                "name":                  d.get("name"),
                "parent_id":             d.get("parent_id"),
                "link_id":               d.get("link_id"),
                "subreddit":             d.get("subreddit"),
                "subreddit_id":          d.get("subreddit_id"),
                "author":                d.get("author"),
                "author_fullname":       d.get("author_fullname"),
                "body":                  d.get("body"),
                "created_utc":           d.get("created_utc"),
                "created_iso":           utc_iso(d.get("created_utc", 0)),
                "score":                 d.get("score"),
                "ups":                   d.get("ups"),
                "depth":                 d.get("depth"),
                "is_submitter":          d.get("is_submitter"),
                "stickied":              d.get("stickied"),
                "gilded":                d.get("gilded"),
                "total_awards_received": d.get("total_awards_received"),
                "coins_mentioned":       extract_coins(d.get("body", "")),
            }
        })
    return comments

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    all_children   = []
    subreddit_meta = []
    coins_seen     = set()

    for sub in SUBREDDITS:
        name = sub["name"]
        coin = sub["coin"]
        print(f"\n[r/{name}] ({coin})")

        posts = fetch_posts(name, coin)
        print(f"  → {len(posts)} posts fetched")

        sub_post_count    = len(posts)
        sub_comment_count = 0

        for post in posts:
            all_children.append(post)
            post_id = post["data"]["id"]
            coins_seen.update(post["data"]["coins_mentioned"])

            time.sleep(0.6)  # polite delay between comment fetches

            comments = fetch_comments(post_id, name, coin)
            sub_comment_count += len(comments)
            for c in comments:
                coins_seen.update(c["data"]["coins_mentioned"])
                all_children.append(c)

        subreddit_meta.append({
            "subreddit":     name,
            "primary_coin":  coin,
            "posts_fetched": sub_post_count,
            "comments_fetched": sub_comment_count,
        })

        time.sleep(1.0)  # polite delay between subreddits

    # ── assemble output ──────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)

    output = {
        "kind": "Listing",
        "data": {
            "dist":     len(all_children),
            "after":    None,
            "before":   None,
            "children": all_children,
        },
        "meta": {
            "collected_at":     now.isoformat(),
            "sort":             SORT,
            "result_count":     len(all_children),
            "posts_count":      sum(1 for c in all_children if c["kind"] == "t3"),
            "comments_count":   sum(1 for c in all_children if c["kind"] == "t1"),
            "coins_tracked":    sorted(coins_seen),
            "subreddits":       subreddit_meta,
            "time_range": {
                "start": utc_iso(
                    min(
                        (c["data"]["created_utc"] or 0)
                        for c in all_children
                        if c["data"].get("created_utc")
                    )
                ) if all_children else None,
                "end": now.isoformat(),
            }
        }
    }

    filename = f"reddit_memecoins_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'─'*50}")
    print(f"✅ Done! Saved → {filename}")
    print(f"   Total records : {output['data']['dist']}")
    print(f"   Posts         : {output['meta']['posts_count']}")
    print(f"   Comments      : {output['meta']['comments_count']}")
    print(f"   Coins seen    : {', '.join(output['meta']['coins_tracked'])}")


if __name__ == "__main__":
    main()
