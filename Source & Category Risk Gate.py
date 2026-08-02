"""
Block 8: Source & Category Risk Gate — PredCal Calibration Trust Layer

This is the core decision surface an autonomous trading agent queries BEFORE
committing capital. It computes per-source and per-category Brier scores,
compares them to the global baseline, and emits a structured sizing decision:
    - size_up   : well-calibrated signal, lean in
    - neutral   : close to baseline, proceed at normal size
    - size_down : historically noisy, reduce position size
    - skip      : calibration so poor the signal is untradeable

Expanded: full per-category risk assessments using category_eval_df from Block 5.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────
GLOBAL_BRIER_BASELINE  = calibration_model.brier_score   # ensemble from Block 4
SIZE_UP_THRESHOLD      = 0.80   # source_brier < GLOBAL * 0.80 → size_up
NEUTRAL_THRESHOLD_HIGH = 1.15   # source_brier < GLOBAL * 1.15 → neutral
SIZE_DOWN_THRESHOLD    = 1.40   # source_brier < GLOBAL * 1.40 → size_down
# above SIZE_DOWN_THRESHOLD → skip

CATEGORY_NOISE_WEIGHT  = 0.40   # how much category Brier compounds the verdict
MIN_MARKETS_FOR_GATE   = 5      # minimum markets per bucket to compute reliable Brier

# ─── Reference gate priors from prototype ─────────────────────────────────────
PROTOTYPE_PRIORS = {
    ("kalshi",     "Finance"):   {"overall_brier": 0.0451, "source_brier": 0.0509,
                                  "category_brier": 0.0908, "decision": "size_down"},
    ("kalshi",     "Politics"):  {"overall_brier": 0.0451, "source_brier": 0.0509,
                                  "category_brier": 0.0520, "decision": "neutral"},
    ("polymarket", "Crypto"):    {"overall_brier": 0.0451, "source_brier": 0.0430,
                                  "category_brier": 0.0460, "decision": "neutral"},
    ("polymarket", "Politics"):  {"overall_brier": 0.0451, "source_brier": 0.0430,
                                  "category_brier": 0.0390, "decision": "size_up"},
    ("metaculus",  "Science"):   {"overall_brier": 0.0451, "source_brier": 0.0290,
                                  "category_brier": 0.0270, "decision": "size_up"},
}


# ─── Brier score proxy for a group of live markets ────────────────────────────

def _group_brier(df_group: pd.DataFrame) -> float:
    """
    Estimate Brier score proxy for a group of markets using calibrated vs raw deviation.
    """
    if len(df_group) < MIN_MARKETS_FOR_GATE:
        return None
    if "calibrated_probability" not in df_group.columns:
        return None
    p_raw = df_group["raw_probability"].fillna(0.5).values
    p_cal = df_group["calibrated_probability"].fillna(0.5).values
    calib_shift = np.mean((p_cal - p_raw) ** 2)
    proxy_brier = GLOBAL_BRIER_BASELINE + calib_shift * 2.0
    return float(proxy_brier)


# ─── Decision engine ──────────────────────────────────────────────────────────

def _make_decision(source_brier: float, category_brier: float,
                   overall: float) -> tuple:
    """Combine source-level and category-level Brier scores into a sizing decision."""
    combined = (1 - CATEGORY_NOISE_WEIGHT) * source_brier + CATEGORY_NOISE_WEIGHT * category_brier
    ratio = combined / overall if overall > 0 else 1.0

    if ratio < SIZE_UP_THRESHOLD:
        decision = "size_up"
        reason   = (f"combined calibration ({combined:.4f}) is {(1-ratio)*100:.0f}% "
                    f"better than global baseline — high-conviction signal")
    elif ratio < NEUTRAL_THRESHOLD_HIGH:
        decision = "neutral"
        reason   = (f"combined calibration ({combined:.4f}) is within ±15% of "
                    f"global baseline — proceed at normal size")
    elif ratio < SIZE_DOWN_THRESHOLD:
        decision = "size_down"
        reason   = (f"historically noisy category with materially worse calibration "
                    f"({combined:.4f}) than global baseline ({overall:.4f})")
    else:
        decision = "skip"
        reason   = (f"calibration ({combined:.4f}) is {(ratio-1)*100:.0f}% worse "
                    f"than baseline — signal is untradeable, skip this bucket")

    return decision, reason


# ─── Main gate computation (source × category) ────────────────────────────────

def compute_risk_gate(markets: pd.DataFrame) -> pd.DataFrame:
    """Compute the risk gate table for every (source, category) bucket."""
    overall_brier = GLOBAL_BRIER_BASELINE
    rows = []

    source_briers = {}
    for src in markets["source"].unique():
        sub = markets[markets["source"] == src]
        sb  = _group_brier(sub)
        if sb is None:
            priors = [v for (s, _), v in PROTOTYPE_PRIORS.items() if s == src]
            sb     = priors[0]["source_brier"] if priors else overall_brier
        source_briers[src] = sb

    markets = markets.copy()
    markets["category_norm"] = (markets["category"]
                                 .fillna("Unknown")
                                 .str.strip()
                                 .str.title()
                                 .replace("", "Unknown"))

    for (src, cat), grp in markets.groupby(["source", "category_norm"]):
        n_markets = len(grp)
        sb        = source_briers.get(src, overall_brier)

        cb = _group_brier(grp)
        if cb is None:
            prior_key = (str(src).lower(), str(cat))
            cb = PROTOTYPE_PRIORS.get(prior_key, {}).get("category_brier", sb)

        decision, reason = _make_decision(sb, cb, overall_brier)

        cw_map = {"size_up": 1.00, "neutral": 0.75, "size_down": 0.50, "skip": 0.00}

        rows.append({
            "source":            src,
            "category":          cat,
            "n_markets":         n_markets,
            "overall_brier":     round(overall_brier, 4),
            "source_brier":      round(sb, 4),
            "category_brier":    round(cb, 4),
            "decision":          decision,
            "reason":            reason,
            "confidence_weight": cw_map[decision],
        })

    gate_df = pd.DataFrame(rows).sort_values(
        ["source", "confidence_weight"], ascending=[True, True]
    ).reset_index(drop=True)

    return gate_df


# ─── Category-level risk gate using category_eval_df from Block 5 ─────────────

def compute_category_risk_gate(cat_eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-category risk gate decisions using category_eval_df from Block 5.
    Uses mock_brier as the category Brier proxy.
    Returns category_risk_df with gate decisions and confidence weights.
    """
    overall = GLOBAL_BRIER_BASELINE
    rows = []

    for _, row in cat_eval_df.iterrows():
        cat = row["category"]
        n   = int(row["market_count"])
        mean_cal = float(row["mean_calibrated_prob"])
        mock_brier = float(row.get("mock_brier", overall))

        # Use mock_brier as both source and category proxy
        decision, reason = _make_decision(mock_brier, mock_brier, overall)
        cw_map = {"size_up": 1.00, "neutral": 0.75, "size_down": 0.50, "skip": 0.00}

        rows.append({
            "category":              cat,
            "market_count":          n,
            "mean_calibrated_prob":  round(mean_cal, 4),
            "brier_estimate":        round(mock_brier, 4),
            "global_brier_baseline": round(overall, 4),
            "brier_ratio":           round(mock_brier / overall if overall > 0 else 1.0, 3),
            "gate_decision":         decision,
            "confidence_weight":     cw_map[decision],
            "reason":                reason,
        })

    cat_risk_df = pd.DataFrame(rows).sort_values(
        "confidence_weight", ascending=False
    ).reset_index(drop=True)
    return cat_risk_df


# ─── Quality-adjusted probability ─────────────────────────────────────────────

def quality_adjusted_forecast(question_keyword: str, gate_df: pd.DataFrame) -> dict:
    """Run PredCal inference and apply risk-gate confidence weights."""
    raw_result = pred_cal.get_live_odds(question_keyword)
    cal_p      = raw_result["calibrated_probability"]
    ci         = raw_result["confidence_interval"]

    matched_markets = raw_result.get("markets_found", [])
    weights = []
    for m in matched_markets:
        src = m["platform"]
        cat_row = gate_df[gate_df["source"] == src]
        if not cat_row.empty:
            weights.append(float(cat_row["confidence_weight"].mean()))

    avg_weight    = float(np.mean(weights)) if weights else 0.75
    adj_cal_p     = cal_p * avg_weight + 0.5 * (1 - avg_weight)
    adj_ci        = [
        max(0.0, ci[0] * avg_weight + 0.5 * (1 - avg_weight)),
        min(1.0, ci[1] * avg_weight + 0.5 * (1 - avg_weight)),
    ]

    return {
        "question_keyword":           question_keyword,
        "calibrated_probability":     round(cal_p,      4),
        "quality_adjusted_prob":      round(adj_cal_p,  4),
        "quality_adjusted_ci":        [round(adj_ci[0], 4), round(adj_ci[1], 4)],
        "avg_confidence_weight":      round(avg_weight,  3),
        "n_markets_matched":          raw_result["n_markets_matched"],
        "raw_probabilities":          raw_result["raw_probabilities"],
        "brier_score":                raw_result["brier_score"],
        "last_updated":               raw_result["last_updated"],
    }


# ─── Run ──────────────────────────────────────────────────────────────────────

# 1. Compute the source × category risk gate table
risk_gate_df = compute_risk_gate(features_df)

print("=" * 90)
print("PREDCAL — SOURCE & CATEGORY RISK GATE")
print("=" * 90)
print(f"\nGlobal Brier baseline : {GLOBAL_BRIER_BASELINE:.4f}")
print(f"Decision thresholds   : size_up <{SIZE_UP_THRESHOLD:.0%}  "
      f"neutral <{NEUTRAL_THRESHOLD_HIGH:.0%}  size_down <{SIZE_DOWN_THRESHOLD:.0%}  else skip\n")

print("── Prototype Reference (sokoclaw/predcal) ──")
print(f"  kalshi / Finance  →  overall={0.0451:.4f}  source={0.0509:.4f}  "
      f"category={0.0908:.4f}  → size_down  (historically noisy)\n")

print("── Live Gate Table (Source × Category) ──")
gate_display = risk_gate_df[[
    "source", "category", "n_markets", "overall_brier",
    "source_brier", "category_brier", "decision", "confidence_weight"
]].copy()

for _, row in gate_display.head(40).iterrows():
    icon = {"size_up": "↑", "neutral": "→", "size_down": "↓", "skip": "✗"}.get(row["decision"], "?")
    print(f"  {icon} {row['source']:12s} | {str(row['category'])[:20]:20s} | n={int(row['n_markets']):4d} | "
          f"Brier: overall={row['overall_brier']:.4f} src={row['source_brier']:.4f} "
          f"cat={row['category_brier']:.4f} | "
          f"decision={row['decision']:9s} | weight={row['confidence_weight']:.2f}")

print(f"\nDecision summary (source × category):")
for dec, grp in risk_gate_df.groupby("decision"):
    icon = {"size_up": "↑", "neutral": "→", "size_down": "↓", "skip": "✗"}.get(dec, "?")
    print(f"  {icon} {dec:10s} : {len(grp)} buckets")


# 2. Per-category risk gate using category_eval_df from Block 5
print("\n" + "=" * 90)
print("PER-CATEGORY RISK GATE (from category_eval_df)")
print("=" * 90)

category_risk_df = compute_category_risk_gate(category_eval_df)

print(f"\n{'Category':<15} {'Count':>7} {'MeanCal':>9} {'BrierEst':>10} {'BrierRatio':>12} {'Gate':>12} {'Weight':>8}")
print("-" * 90)
for _, row in category_risk_df.iterrows():
    icon = {"size_up": "↑", "neutral": "→", "size_down": "↓", "skip": "✗"}.get(row["gate_decision"], "?")
    print(f"  {icon} {row['category']:<13} {int(row['market_count']):>7} "
          f"{row['mean_calibrated_prob']:>9.3f} {row['brier_estimate']:>10.4f} "
          f"{row['brier_ratio']:>12.3f} {row['gate_decision']:>12s} "
          f"{row['confidence_weight']:>8.2f}")

print(f"\nCategory gate summary:")
for dec, grp in category_risk_df.groupby("gate_decision"):
    icon = {"size_up": "↑", "neutral": "→", "size_down": "↓", "skip": "✗"}.get(dec, "?")
    print(f"  {icon} {dec:10s} : {len(grp)} categories — {', '.join(grp['category'].tolist())}")


# 3. Quality-adjusted forecasts for example queries (from Block 7)
print("\n" + "=" * 90)
print("QUALITY-ADJUSTED FORECASTS (sample from 30 keywords)")
print("=" * 90)
qa_results = []
# Use top 10 from results_df_expanded
top_keywords = results_df_expanded.head(10)["question_keyword"].tolist()
for kw in top_keywords:
    qa = quality_adjusted_forecast(kw, risk_gate_df)
    qa_results.append(qa)
    print(f"\n  [{kw}]")
    print(f"    Calibrated prob     : {qa['calibrated_probability']:.1%}")
    print(f"    Quality-adjusted    : {qa['quality_adjusted_prob']:.1%}  "
          f"(weight={qa['avg_confidence_weight']:.2f})")
    print(f"    90% CI (adjusted)   : [{qa['quality_adjusted_ci'][0]:.1%}, "
          f"{qa['quality_adjusted_ci'][1]:.1%}]")
    print(f"    Markets matched     : {qa['n_markets_matched']}")

qa_results_df = pd.DataFrame(qa_results)

print("\n✓ risk_gate_df, category_risk_df, and qa_results_df ready for reporting and API.")
