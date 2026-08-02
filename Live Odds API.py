"""
Block 7: Live Odds Function — Live Odds API
PredCalModel: inference pipeline that takes any question keyword and returns
calibrated odds with confidence intervals, plus batch forecasting.
Expanded to 30 keywords across all major prediction market categories.
"""

import re
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
KEYWORD_MATCH_THRESH  = 0.08    # min Jaccard similarity to consider a market matching
BOOTSTRAP_N_ITER      = 200     # bootstrap samples for confidence interval
BOOTSTRAP_CI          = 0.90    # confidence interval level
MIN_TOKENS_FOR_SEARCH = 2       # min keyword tokens required to search
DEFAULT_PRIOR         = 0.50    # fallback probability if no markets found

# ─── Expanded query list: 30 keywords across all major categories ─────────────
EXAMPLE_QUERIES = [
    # Politics / Elections
    "2026 midterm",
    "trump",
    "harris",
    "republican senate",
    "democratic house",
    "presidential approval",
    "congress",
    # Economics
    "fed rate cut",
    "inflation cpi",
    "recession",
    "gdp growth",
    "unemployment",
    "interest rate",
    # Crypto
    "bitcoin",
    "ethereum",
    "crypto regulation",
    "solana",
    "defi",
    # Geopolitics
    "ukraine war",
    "taiwan",
    "iran nuclear",
    "nato",
    "china",
    # Science / Tech
    "AI regulation",
    "spacex launch",
    "quantum computing",
    "nuclear fusion",
    # Climate
    "hurricane season",
    "carbon tax",
    # Sports
    "world cup",
    "super bowl",
]

# ─── Keyword helpers ──────────────────────────────────────────────────────────
_STOPWORDS_QUERY = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "in", "on",
    "at", "to", "of", "for", "by", "or", "and", "that", "this", "with",
    "from", "it", "its", "as", "do", "have", "has", "not", "no", "yes",
    "can", "could", "would", "should", "than", "any", "all", "more",
    "about", "after", "before", "during", "between", "through", "into",
    "what", "when", "where", "who", "which", "how", "if",
}


def _tokenize_query(text: str) -> set:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 1 and t not in _STOPWORDS_QUERY}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ─── PredCalModel ─────────────────────────────────────────────────────────────

class PredCalModel:
    """
    PredCal: Prediction Market Calibration Model

    Aggregates signals from Polymarket, Kalshi, and Metaculus and produces
    calibrated probability estimates via an ensemble recalibration model
    (Isotonic + Logistic + Beta calibration, weights optimised on held-out data).

    Usage
    -----
    model = PredCalModel(calibration_model, feature_scaler)
    result = model.get_live_odds("bitcoin price")
    results_df = model.batch_forecast(["election 2026", "bitcoin", "AI regulation"])
    """

    def __init__(self, cal_model, f_scaler):
        self.model         = cal_model
        self.scaler        = f_scaler
        self.brier_score   = getattr(cal_model, "brier_score", None)
        self._markets_df   = features_df.copy()
        self._title_tokens = {
            idx: _tokenize_query(str(row["title"]))
            for idx, row in self._markets_df.iterrows()
        }
        self._feature_cols = FEATURE_COLS_CAL

    def _find_matching_markets(self, keyword: str, threshold: float = KEYWORD_MATCH_THRESH):
        """Return slice of markets_df that match keyword via Jaccard similarity."""
        q_tok  = _tokenize_query(keyword)
        if len(q_tok) < MIN_TOKENS_FOR_SEARCH:
            kw_lower = keyword.lower()
            mask = self._markets_df["title"].str.lower().str.contains(kw_lower, na=False)
            return self._markets_df[mask]

        scores = {
            idx: _jaccard(q_tok, tok)
            for idx, tok in self._title_tokens.items()
        }
        matched_idx = [idx for idx, s in scores.items() if s >= threshold]

        if not matched_idx:
            relaxed = threshold / 2
            matched_idx = sorted(scores, key=scores.get, reverse=True)[:5]
            matched_idx = [i for i in matched_idx if scores[i] >= relaxed]

        return self._markets_df.loc[matched_idx] if matched_idx else self._markets_df.iloc[0:0]

    def _build_feature_row(self, market_row: pd.Series) -> np.ndarray:
        """Build a single-row feature array from a market row."""
        row = []
        for col in self._feature_cols:
            val = market_row.get(col, 0)
            try:
                row.append(float(val if val is not None else 0))
            except (TypeError, ValueError):
                row.append(0.0)
        return np.array(row, dtype=float).reshape(1, -1)

    def _calibrate_market(self, market_row: pd.Series) -> float:
        """Return calibrated probability for a single market row."""
        X = self._build_feature_row(market_row)
        X = np.nan_to_num(X, nan=0.0)
        try:
            return float(self.model.predict(X)[0])
        except Exception:
            return float(market_row.get("raw_probability", DEFAULT_PRIOR))

    def _bootstrap_ci(self, calibrated_probs: np.ndarray, n_iter: int = BOOTSTRAP_N_ITER,
                      ci: float = BOOTSTRAP_CI) -> tuple:
        """Bootstrap confidence interval around the mean of calibrated_probs."""
        if len(calibrated_probs) == 0:
            return (DEFAULT_PRIOR, DEFAULT_PRIOR)
        if len(calibrated_probs) == 1:
            p = calibrated_probs[0]
            margin = 0.10
            return (max(0.0, p - margin), min(1.0, p + margin))

        boot_means = []
        rng = np.random.default_rng(0)
        for _ in range(n_iter):
            sample = rng.choice(calibrated_probs, size=len(calibrated_probs), replace=True)
            boot_means.append(sample.mean())

        alpha   = (1 - ci) / 2
        lo, hi  = np.quantile(boot_means, [alpha, 1 - alpha])
        return (float(lo), float(hi))

    def get_live_odds(self, question_keyword: str) -> dict:
        """
        Given a keyword/topic, fetch live market data and return calibrated odds.

        Returns
        -------
        dict with keys:
            question_keyword, markets_found, n_markets_matched,
            raw_probabilities, calibrated_probability, confidence_interval,
            brier_score, last_updated
        """
        matched = self._find_matching_markets(question_keyword)

        markets_found    = []
        raw_probs        = {}
        calibrated_probs = []

        for idx, row in matched.iterrows():
            src  = row.get("source", "unknown")
            prob = float(row.get("raw_probability", DEFAULT_PRIOR))
            cal  = self._calibrate_market(row)

            markets_found.append({
                "platform": src,
                "title":    str(row.get("title", ""))[:80],
                "raw_prob": round(prob, 4),
                "cal_prob": round(cal, 4),
            })

            if src not in raw_probs:
                raw_probs[src] = []
            raw_probs[src].append(prob)
            calibrated_probs.append(cal)

        raw_probs_agg = {src: round(float(np.mean(ps)), 4)
                         for src, ps in raw_probs.items()}

        cal_arr = np.array(calibrated_probs)

        if len(cal_arr) == 0:
            calibrated_prob = DEFAULT_PRIOR
            ci              = (max(0.0, DEFAULT_PRIOR - 0.15), min(1.0, DEFAULT_PRIOR + 0.15))
        else:
            calibrated_prob = float(cal_arr.mean())
            ci              = self._bootstrap_ci(cal_arr)

        return {
            "question_keyword":       question_keyword,
            "markets_found":          markets_found,
            "n_markets_matched":      len(markets_found),
            "raw_probabilities":      raw_probs_agg,
            "calibrated_probability": round(calibrated_prob, 4),
            "confidence_interval":    [round(ci[0], 4), round(ci[1], 4)],
            "brier_score":            self.brier_score,
            "last_updated":           datetime.now(timezone.utc).isoformat(),
        }

    def batch_forecast(self, keywords: list) -> pd.DataFrame:
        """
        Run get_live_odds for multiple keywords.

        Returns
        -------
        pd.DataFrame with columns:
            question_keyword, calibrated_probability, ci_low, ci_high,
            n_markets_matched, match_count, raw_probs_summary, brier_score
        """
        rows = []
        for kw in keywords:
            result = self.get_live_odds(kw)
            rows.append({
                "question_keyword":       result["question_keyword"],
                "calibrated_probability": result["calibrated_probability"],
                "ci_low":                 result["confidence_interval"][0],
                "ci_high":                result["confidence_interval"][1],
                "n_markets_matched":      result["n_markets_matched"],
                "match_count":            result["n_markets_matched"],  # alias for clarity
                "raw_probs_summary":      str(result["raw_probabilities"]),
                "brier_score":            result["brier_score"],
            })
        return pd.DataFrame(rows)


# ─── Instantiate model ────────────────────────────────────────────────────────
pred_cal = PredCalModel(calibration_model, feature_scaler)

# ─── Batch Forecast on all 30 queries ────────────────────────────────────────
print("=" * 80)
print("PredCal Model — Expanded Batch Forecast (30 keywords)")
print("=" * 80)
print(f"Querying {len(EXAMPLE_QUERIES)} topics …\n")

results_df = pred_cal.batch_forecast(EXAMPLE_QUERIES)
results_df_expanded = results_df.sort_values("calibrated_probability", ascending=False).reset_index(drop=True)

# Display full sorted table
pd.set_option("display.max_rows", 50)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)

print(results_df_expanded[[
    "question_keyword", "calibrated_probability", "ci_low", "ci_high",
    "match_count", "brier_score"
]].to_string(index=True))

# ─── Detailed view for top result ────────────────────────────────────────────
top_kw = results_df_expanded.iloc[0]["question_keyword"]
print("\n" + "─" * 80)
print(f"Detailed result for top topic: \"{top_kw}\"")
print("─" * 80)
detail = pred_cal.get_live_odds(top_kw)
print(f"  Calibrated probability : {detail['calibrated_probability']:.1%}")
print(f"  90% CI                 : [{detail['confidence_interval'][0]:.1%}, "
      f"{detail['confidence_interval'][1]:.1%}]")
print(f"  Markets matched        : {detail['n_markets_matched']}")
print(f"  Per-platform raw probs : {detail['raw_probabilities']}")
print(f"  Brier score (model)    : {detail['brier_score']:.4f}")
for i, m in enumerate(detail["markets_found"][:5], 1):
    print(f"    [{i}] {m['platform'].capitalize():10s}  "
          f"raw={m['raw_prob']:.1%}  cal={m['cal_prob']:.1%}  "
          f"\"{m['title'][:55]}\"")

# ─── Category summary ─────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("Summary by match count:")
print(results_df_expanded.groupby(
    pd.cut(results_df_expanded["match_count"], bins=[-1, 0, 2, 5, 20, 9999],
           labels=["0 matches", "1-2", "3-5", "6-20", "20+"])
)["question_keyword"].count().to_string())

# ─── Final summary ────────────────────────────────────────────────────────────
n_markets  = len(features_df)
brier_sc   = calibration_model.brier_score
poly_n     = int((features_df["source"] == "polymarket").sum())
kalshi_n   = int((features_df["source"] == "kalshi").sum())
meta_n     = int((features_df["source"] == "metaculus").sum())

print("\n" + "=" * 80)
print("PredCal Model Summary")
print("=" * 80)
print(f"  Calibration  : {brier_sc:.3f} Brier · {n_markets} live markets tracked")
print(f"  Sources      : Polymarket ({poly_n}), Kalshi ({kalshi_n}), Metaculus ({meta_n})")
print(f"  Architecture : Isotonic + Logistic + Beta Calibration ensemble")
print(f"  Weights      : Iso={calibration_model.weights[0]:.2f}, "
      f"Logit={calibration_model.weights[1]:.2f}, Beta={calibration_model.weights[2]:.2f}")
print(f"  Queries      : {len(EXAMPLE_QUERIES)} keywords | results_df_expanded ({len(results_df_expanded)} rows)")
print(f"  Data source  : {'Synthetic calibration (insufficient live resolved data)' if using_synthetic else 'Live resolved markets'}")
print("=" * 80)
