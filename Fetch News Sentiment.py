"""
Block 3: News Sentiment — Fetch News Sentiment
Fetch recent news headlines from Google News RSS for each market's topic
and compute sentiment scores. Falls back gracefully when feedparser or
TextBlob are not installed: uses urllib + simple positive/negative word
counting as the sentiment heuristic, or sets sentiment to 0 if fetch fails.
"""

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────
TOP_N_MARKETS         = 150  # markets to fetch news for (by volume activity)
MAX_ARTICLES_PER_Q    = 5    # max articles to read per question
MAX_HTTP_REQUESTS     = 200  # hard cap on total HTTP requests
SLEEP_BETWEEN_REQS    = 0.2  # seconds between requests
GOOGLE_NEWS_BASE      = "https://news.google.com/rss/search"
GOOGLE_NEWS_PARAMS    = {"hl": "en-US", "gl": "US", "ceid": "US:en"}
FETCH_TIMEOUT         = 8   # seconds per HTTP request

# ─── Simple sentiment word lists (fallback when TextBlob not available) ───────
POSITIVE_WORDS = {
    "win", "wins", "winning", "gains", "gain", "rises", "rise", "surges", "surge",
    "growth", "grows", "grow", "up", "positive", "profit", "profits", "success",
    "beat", "beats", "record", "approve", "approves", "agrees", "improve",
    "increase", "increases", "strong", "stronger", "best", "rally", "bullish",
    "boost", "support", "upgrade", "good", "great", "advance", "higher", "top",
}
NEGATIVE_WORDS = {
    "fall", "falls", "falling", "loss", "losses", "drop", "drops", "slump",
    "decline", "declines", "lower", "down", "negative", "fail", "fails",
    "cut", "cuts", "miss", "misses", "weak", "weaker", "worst", "crash",
    "crisis", "threat", "bearish", "concern", "concerns", "warning", "risk",
    "risks", "reject", "rejects", "ban", "decrease", "decreases", "tumble",
}

# ─── Stopwords for keyword extraction ─────────────────────────────────────────
STOPWORDS_NEWS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
    "at", "to", "of", "for", "by", "or", "and", "that", "this", "with",
    "from", "it", "its", "as", "do", "have", "has", "not", "no", "yes",
    "can", "could", "would", "should", "than", "any", "all", "more", "over",
    "about", "after", "before", "during", "between", "through", "into",
    "what", "when", "where", "who", "which", "how", "if", "than", "then",
    "price", "market", "markets", "prediction", "question",
}

# ─── Try to import optional packages ──────────────────────────────────────────
_HAS_FEEDPARSER = False
_HAS_TEXTBLOB = False

try:
    import feedparser
    _HAS_FEEDPARSER = True
    print("[News Sentiment] feedparser available ✓")
except ImportError:
    print("[News Sentiment] feedparser not available — using urllib XML fallback")

try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
    print("[News Sentiment] TextBlob available ✓")
except ImportError:
    print("[News Sentiment] TextBlob not available — using word-count sentiment fallback")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_keywords(title: str, n: int = 3) -> str:
    """Extract top n meaningful keywords from a question title."""
    if not isinstance(title, str):
        return ""
    tokens = re.findall(r"[a-zA-Z]+", title.lower())
    filtered = [t for t in tokens if len(t) > 2 and t not in STOPWORDS_NEWS]
    filtered.sort(key=len, reverse=True)
    return " ".join(filtered[:n])


def simple_sentiment(text: str) -> float:
    """Word-count based sentiment: +1 positive, -1 negative, 0 neutral."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def parse_rss_with_urllib(url: str, max_articles: int = MAX_ARTICLES_PER_Q) -> list:
    """
    Fetch RSS via urllib and parse XML. Returns list of title strings.
    Fallback when feedparser is unavailable.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PredCalResearch/1.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
        titles = []
        for item in root.iter("item"):
            title_el = item.find("title")
            desc_el = item.find("description")
            text = ""
            if title_el is not None and title_el.text:
                text += title_el.text + " "
            if desc_el is not None and desc_el.text:
                text += desc_el.text
            if text.strip():
                titles.append(text.strip())
            if len(titles) >= max_articles:
                break
        return titles
    except Exception:
        return []


def fetch_sentiment_for_keyword(keyword: str, max_articles: int = MAX_ARTICLES_PER_Q) -> dict:
    """
    Fetch Google News RSS for keyword and return aggregated sentiment stats.
    Returns dict: {mean_sentiment, std_sentiment, article_count}
    """
    if not keyword.strip():
        return {"mean_sentiment": 0.0, "std_sentiment": 0.0, "article_count": 0}

    params = {**GOOGLE_NEWS_PARAMS, "q": keyword}
    url = f"{GOOGLE_NEWS_BASE}?{urllib.parse.urlencode(params)}"

    texts = []
    if _HAS_FEEDPARSER:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles]:
                t = entry.get("title", "") + " " + entry.get("summary", "")
                texts.append(t.strip())
        except Exception:
            texts = []
    else:
        texts = parse_rss_with_urllib(url, max_articles)

    sentiments = []
    for text in texts:
        try:
            if _HAS_TEXTBLOB:
                sentiments.append(TextBlob(text).sentiment.polarity)
            else:
                sentiments.append(simple_sentiment(text))
        except Exception:
            pass

    if sentiments:
        return {
            "mean_sentiment": float(np.mean(sentiments)),
            "std_sentiment":  float(np.std(sentiments)),
            "article_count":  len(sentiments),
        }
    return {"mean_sentiment": 0.0, "std_sentiment": 0.0, "article_count": 0}


def enrich_with_news_sentiment(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add news_sentiment_mean, news_sentiment_std, news_article_count to features_df.
    Operates on top N markets by volume (log_volume).
    """
    df = feat_df.copy()

    # Select top N markets to query
    top_idx = (
        df["log_volume"]
        .fillna(0)
        .nlargest(TOP_N_MARKETS)
        .index
    )
    top_markets = df.loc[top_idx].copy()

    print(f"Fetching news sentiment for {len(top_markets)} markets …")
    print(f"(Cap: {MAX_HTTP_REQUESTS} total HTTP requests)")

    request_count  = 0
    enriched_count = 0
    total_articles = 0
    all_sentiments = []

    for idx in top_markets.index:
        if request_count >= MAX_HTTP_REQUESTS:
            print(f"  HTTP cap ({MAX_HTTP_REQUESTS}) reached — stopping early.")
            break

        title   = df.at[idx, "title"]
        keyword = extract_keywords(title, n=3)

        if not keyword:
            continue

        try:
            result = fetch_sentiment_for_keyword(keyword)
        except Exception as e:
            result = {"mean_sentiment": 0.0, "std_sentiment": 0.0, "article_count": 0}

        request_count += 1

        df.at[idx, "news_sentiment_mean"] = result["mean_sentiment"]
        df.at[idx, "news_sentiment_std"]  = result["std_sentiment"]
        df.at[idx, "news_article_count"]  = result["article_count"]

        if result["article_count"] > 0:
            enriched_count += 1
            total_articles  += result["article_count"]
            all_sentiments.append(result["mean_sentiment"])

        time.sleep(SLEEP_BETWEEN_REQS)

    avg_sent = float(np.mean(all_sentiments)) if all_sentiments else 0.0

    print(f"\n[News Sentiment] Summary:")
    print(f"  HTTP requests made  : {request_count}")
    print(f"  Questions enriched  : {enriched_count} / {len(top_markets)}")
    print(f"  Total articles read : {total_articles}")
    print(f"  Mean sentiment      : {avg_sent:.3f} (−1 neg → +1 pos)")
    print(f"  Sentiment engine    : {'TextBlob' if _HAS_TEXTBLOB else 'word-count fallback'}")
    print(f"  RSS parser          : {'feedparser' if _HAS_FEEDPARSER else 'urllib XML fallback'}")

    return df


# ─── Run ──────────────────────────────────────────────────────────────────────
features_df = enrich_with_news_sentiment(features_df)

print(f"\nSentiment column preview:")
sentiment_cols = ["title", "news_sentiment_mean", "news_sentiment_std", "news_article_count"]
mask = features_df["news_article_count"] > 0
if mask.sum() > 0:
    print(features_df.loc[mask, sentiment_cols].head(10).to_string(index=False))
else:
    print("  No articles found for any market (RSS may be blocked in this environment).")
    print("  Sentiment columns set to 0 — downstream blocks will handle this gracefully.")
print(f"\nMarkets with news data: {mask.sum()} / {len(features_df)} (top {TOP_N_MARKETS} queried)")
