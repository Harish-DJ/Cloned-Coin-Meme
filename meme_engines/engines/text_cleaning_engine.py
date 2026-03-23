"""
ENGINE 1: Text Cleaning Engine
--------------------------------
Cleans raw social media posts before feeding into NLP engines.
- Removes URLs, mentions, emojis, punctuation
- Normalizes text (lowercase, whitespace)
- Detects and flags spam/bot posts
Output: clean_text string for Sentiment + Contextual engines
"""

import re
import unicodedata


class TextCleaningEngine:

    SPAM_PATTERNS = [
        r"follow\s+back",
        r"click\s+here",
        r"dm\s+for",
        r"free\s+giveaway",
        r"guaranteed\s+profit",
        r"100x\s+guaranteed",
        r"join\s+now",
    ]

    def __init__(self):
        self._spam_re = re.compile("|".join(self.SPAM_PATTERNS), re.IGNORECASE)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\.\S+", "", text)

    def _remove_mentions(self, text: str) -> str:
        return re.sub(r"@\w+", "", text)

    def _remove_hashtag_symbol(self, text: str) -> str:
        # Keep the word, just strip the # so "DOGE" survives
        return re.sub(r"#(\w+)", r"\1", text)

    def _remove_emojis(self, text: str) -> str:
        return "".join(
            ch for ch in text
            if not unicodedata.category(ch).startswith(("S", "C"))
        )

    def _remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", " ", text)

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def clean(self, text: str) -> dict:
        """
        Clean a single post.

        Returns
        -------
        {
            "clean_text"      : str,
            "is_spam"         : bool,
            "original_length" : int,
            "clean_length"    : int,
        }
        """
        if not text or not isinstance(text, str):
            return {"clean_text": "", "is_spam": False,
                    "original_length": 0, "clean_length": 0}

        is_spam = bool(self._spam_re.search(text))

        t = self._remove_urls(text)
        t = self._remove_mentions(t)
        t = self._remove_hashtag_symbol(t)
        t = self._remove_emojis(t)
        t = self._remove_punctuation(t)
        t = self._normalize_whitespace(t)
        t = t.lower()

        return {
            "clean_text":       t,
            "is_spam":          is_spam,
            "original_length":  len(text),
            "clean_length":     len(t),
        }

    def clean_batch(self, posts: list) -> list:
        """
        Clean a batch of raw post dicts.
        Each post dict must contain at least: { "id": ..., "text": "..." }
        Returns the same list with "clean_text" and "is_spam" injected.
        """
        results = []
        for post in posts:
            cleaned = self.clean(post.get("text", ""))
            results.append({
                **post,
                "clean_text": cleaned["clean_text"],
                "is_spam":    cleaned["is_spam"],
            })
        return results
