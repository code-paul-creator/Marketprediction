"""
Block 2: Feature Engineering — Build Feature Matrix
Extracts features from the combined markets DataFrame for calibration modeling.
"""

import pandas as pd
import numpy as np
import re
from itertools import combinations

# ─── Configuration ────────────────────────────────────────────────────────────
NEAR_CLOSE_DAYS      = 7     # threshold for "near resolution" feature
MID_PROBABILITY_BAND = (0.4, 0.6)  # for overconfidence detection
JACCARD_MATCH_THRESH = 0.15  # min token overlap to consider cross-platform match
MIN_TOKENS_FOR_MATCH = 3     # min meaningful tokens in title for Jaccard match

# ─── Stopwords (minimal, no NLTK needed) ─────────────────────────────────────
STOPWORDS = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
    "at", "to", "of", "for", "by", "or", "and", "by", "that", "this", "with",
    "from", "it", "its", "as", "do", "have", "has", "not", "no", "yes",
    "can", "could", "would", "should", "than", "any", "all", "more", "over",
    "about", "after", "before", "during", "between", "through", "into",
    "what", "when", "where", "who", "which", "how", "if",
}

POLITICS_KEYWORDS = {"election", "elections", "vote", "president", "senate", "congress",
                     "democrat", "republican", "government", "political", "politics",
                     "parliament", "minister", "governor", "mayor", "ballot"}

CRYPTO_KEYWORDS = {"crypto", "bitcoin", "btc", "ethereum", "eth", "blockchain",
                   "defi", "nft", "token", "coin", "binance", "solana", "sol",
                   "altcoin", "stablecoin"}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> set:
    """Lowercase tokenize, remove stopwords and short tokens."""
    if not isinstance(text, str):
        return set()
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def days_to_close(close_time_str) -> float:
    """Parse close time and return days remaining (NaN if parse fails)."""
    if not isinstance(close_time_str, str) or not close_time_str.strip():
        return np.nan
    from datetime import datetime, timezone
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(close_time_str[:26].rstrip("Z"), fmt.rstrip("Z"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (dt - now).total_seconds() / 86400.0
        except ValueError:
            continue
    return np.nan


# ─── Feature Engineering ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix from the raw markets DataFrame.
    Returns a copy of df with additional feature columns.
    """
    feat = df.copy()

    # ── Raw probability ──────────────────────────────────────────────────
    feat["raw_probability"] = feat["probability"].clip(0.001, 0.999)

    # ── Log-odds ─────────────────────────────────────────────────────────
    p = feat["raw_probability"]
    feat["log_odds"] = np.log(p / (1 - p))

    # ── Days to close & near_close ────────────────────────────────────────
    feat["days_to_close"] = feat["close_time"].apply(days_to_close)
    feat["near_close"] = (feat["days_to_close"] < NEAR_CLOSE_DAYS).astype(int)
    # Fill NaN days_to_close with median
    feat["near_close"] = feat["near_close"].fillna(0).astype(int)

    # ── Category / topic flags ────────────────────────────────────────────
    def check_keywords(row, keyword_set):
        combined_text = " ".join([
            str(row.get("category", "")),
            str(row.get("title", ""))
        ]).lower()
        return int(any(kw in combined_text for kw in keyword_set))

    feat["is_political"] = feat.apply(lambda r: check_keywords(r, POLITICS_KEYWORDS), axis=1)
    feat["is_crypto"]    = feat.apply(lambda r: check_keywords(r, CRYPTO_KEYWORDS),   axis=1)

    # ── Extreme probability flag ──────────────────────────────────────────
    feat["is_extreme"] = ((feat["raw_probability"] < 0.05) |
                          (feat["raw_probability"] > 0.95)).astype(int)

    # ── In mid-probability band ───────────────────────────────────────────
    lo, hi = MID_PROBABILITY_BAND
    feat["is_mid_band"] = ((feat["raw_probability"] >= lo) &
                           (feat["raw_probability"] <= hi)).astype(int)

    # ── Platform one-hot ──────────────────────────────────────────────────
    feat["platform_polymarket"] = (feat["source"] == "polymarket").astype(int)
    feat["platform_kalshi"]     = (feat["source"] == "kalshi").astype(int)
    feat["platform_metaculus"]  = (feat["source"] == "metaculus").astype(int)

    # ── Volume normalisation (log scale, fill 0 for NaN) ─────────────────
    vol = feat["volume"].fillna(0).clip(lower=0)
    feat["log_volume"] = np.log1p(vol)

    # ── Cross-platform consensus via Jaccard title matching ───────────────
    print("Computing cross-platform consensus (Jaccard matching) …")
    title_tokens = feat["title"].apply(tokenize)

    n = len(feat)
    # group by source for efficiency
    indices_by_source = {}
    for src in feat["source"].unique():
        indices_by_source[src] = feat.index[feat["source"] == src].tolist()

    # For each market, find counterparts on OTHER platforms
    feat["consensus_deviation"] = 0.0
    feat["cross_platform_mean"] = feat["raw_probability"]   # default = own value
    feat["cross_platform_std"]  = 0.0

    match_count = 0
    for src_a, src_b in combinations(indices_by_source.keys(), 2):
        idxs_a = indices_by_source[src_a]
        idxs_b = indices_by_source[src_b]
        for ia in idxs_a:
            tok_a = title_tokens[ia]
            if len(tok_a) < MIN_TOKENS_FOR_MATCH:
                continue
            best_sim = 0.0
            best_ib  = None
            for ib in idxs_b:
                tok_b = title_tokens[ib]
                if len(tok_b) < MIN_TOKENS_FOR_MATCH:
                    continue
                sim = jaccard(tok_a, tok_b)
                if sim > best_sim:
                    best_sim = sim
                    best_ib  = ib
            if best_sim >= JACCARD_MATCH_THRESH and best_ib is not None:
                match_count += 1
                probs = [feat.at[ia, "raw_probability"], feat.at[best_ib, "raw_probability"]]
                mu    = np.mean(probs)
                std   = np.std(probs)
                feat.at[ia,      "consensus_deviation"] = feat.at[ia,      "raw_probability"] - mu
                feat.at[best_ib, "consensus_deviation"] = feat.at[best_ib, "raw_probability"] - mu
                feat.at[ia,      "cross_platform_mean"] = mu
                feat.at[best_ib, "cross_platform_mean"] = mu
                feat.at[ia,      "cross_platform_std"]  = std
                feat.at[best_ib, "cross_platform_std"]  = std

    print(f"  Cross-platform matches found: {match_count}")

    # ── Placeholder for news sentiment (added in Block 3) ─────────────────
    feat["news_sentiment_mean"] = 0.0
    feat["news_sentiment_std"]  = 0.0
    feat["news_article_count"]  = 0

    return feat


# ─── Run ──────────────────────────────────────────────────────────────────────
features_df = build_features(markets_df)

print(f"\nFeature matrix shape: {features_df.shape}")
print("\nFeature null counts:")
feat_cols = [
    "raw_probability", "log_odds", "near_close", "is_political", "is_crypto",
    "is_extreme", "is_mid_band", "platform_polymarket", "platform_kalshi",
    "platform_metaculus", "log_volume", "consensus_deviation",
    "cross_platform_mean", "cross_platform_std",
    "news_sentiment_mean", "news_sentiment_std", "news_article_count"
]
for col in feat_cols:
    if col in features_df.columns:
        nulls = features_df[col].isnull().sum()
        print(f"  {col:30s}: {nulls} nulls")

print(f"\nSample of feature matrix:")
print(features_df[["source", "title", "raw_probability", "log_odds",
                    "is_political", "is_crypto", "is_extreme"]].head(8).to_string(index=False))
