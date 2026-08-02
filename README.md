# PredCal — Prediction Market Calibration Trust Layer

**GitHub:** [sokoclaw/predcal](https://github.com/sokoclaw/predcal) · **Framing:** Paid calibration risk gate for agentic trading systems

---

PredCal is intentionally narrow. It is **not** a full execution bot. It is the **calibration and trust layer** an autonomous trader queries *before* committing capital.

## Architecture

| Block | Purpose |
|---|---|
| **1 · Fetch Market Data** | Live ingestion from Polymarket CLOB, Kalshi Elections API, Metaculus — ~600 markets |
| **2 · Build Feature Matrix** | Log-odds, near-close, topic flags, cross-platform Jaccard consensus deviation |
| **3 · Fetch News Sentiment** | Google News RSS → TextBlob polarity for top 50 markets |
| **4 · Train Calibration Model** | Isotonic + Logistic + Beta calibration ensemble, weight-optimised on held-out data |
| **5 · Evaluate Calibration** | Brier decomposition (Reliability / Resolution / Uncertainty), ECE, MCE, Sharpness |
| **6 · Calibration Plots** | 4 interactive Plotly charts: reliability diagram, Brier bars, distributions, live dashboard |
| **7 · Live Odds API** | `PredCalModel.get_live_odds(keyword)` — calibrated probability + 90% CI |
| **8 · Source & Category Risk Gate** | Per-source/category Brier → `size_up / neutral / size_down / skip` sizing decision |

## Risk Gate Logic

```json
{
  "source": "kalshi",
  "category": "Finance",
  "overall_brier": 0.0451,
  "source_brier": 0.0509,
  "category_brier": 0.0908,
  "decision": "size_down",
  "reason": "historically noisy category with materially worse calibration than global baseline"
}
```

**Decision thresholds:** `size_up` if combined Brier < 80 % of baseline · `neutral` < 115 % · `size_down` < 140 % · `skip` beyond that.  
Quality-adjusted forecasts shrink toward 50 % in proportion to the confidence weight (1.0 → 0.75 → 0.50 → 0.0).

## Key Findings (prototype)
- Metaculus superforecasters → consistently `size_up` across Science / Tech categories
- Kalshi Finance → `size_down` (category Brier 0.0908 vs baseline 0.0451, ~2× noisier)
- Polymarket Politics → `size_up` (liquid, tight spreads, well-calibrated)
- Ensemble calibration improves raw market Brier by **~15–25 %** (favorite-longshot bias correction)
