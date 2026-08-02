"""
PredCal Pipeline — Final Run Summary
Summarizes what worked, what fell back, and key metrics from this pipeline run.
"""

print("=" * 75)
print("PREDCAL PIPELINE — RUN SUMMARY")
print("=" * 75)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│  Block 1: Fetch Market Data                                    ✅ PASS  │
│  ─────────────────────────────────────────────────────────────────────  │
│  • Polymarket CLOB API  → 218 active markets (68 pages, cursor paging)  │
│  • Kalshi elections API → 200 open markets (probs all 0.50 — API       │
│      returned yes_bid=0 / yes_ask=0; fallback defaulted to 0.5)        │
│  • Metaculus API        → ✗ 403 Forbidden — set to empty DataFrame     │
│  • Total: 418 markets combined                                          │
│  • FALLBACK: Metaculus → empty DF (403); Kalshi probs → 0.50 default  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Block 2: Build Feature Matrix                                 ✅ PASS  │
│  ─────────────────────────────────────────────────────────────────────  │
│  • Feature matrix: (418, 28) — 0 null values                           │
│  • Cross-platform Jaccard matches: 0 (Metaculus absent; all Kalshi     │
│      probs identical at 0.5 → near-zero Jaccard overlap)               │
│  • All feature columns populated with sensible defaults                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Block 3: Fetch News Sentiment                                 ✅ PASS  │
│  ─────────────────────────────────────────────────────────────────────  │
│  • feedparser NOT installed → FALLBACK: urllib XML RSS parser           │
│  • TextBlob NOT installed   → FALLBACK: word-count sentiment heuristic  │
│  • 50 markets queried, 29 enriched (58%), 145 articles total           │
│  • Mean sentiment: +0.221 (slight positive bias, likely sports news)   │
│  • 21 markets returned no articles (RSS blocked or no results)         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Block 4: Train Calibration Model                              ✅ PASS  │
│  ─────────────────────────────────────────────────────────────────────  │
│  • Resolved live markets: 0 (none of the 418 markets are resolved)     │
│  • FALLBACK: Synthetic training data (N=2000, Beta(2,2), FLB=0.85)     │
│  • Isotonic:   Brier=0.2143,  +0.1% vs raw                            │
│  • Logistic:   Brier=0.2102,  +2.0% vs raw                            │
│  • Beta Calib: Brier=0.2092,  +2.4% vs raw                            │
│  • Ensemble:   Brier=0.2092,  +2.4% (weights: iso=0, logit=0, beta=1) │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Block 5: Evaluate Calibration                                 ✅ PASS  │
│  ─────────────────────────────────────────────────────────────────────  │
│  • Raw Market: Brier=0.2158, ECE=0.0917                                │
│  • Ensemble:   Brier=0.2103, ECE=0.0655                                │
│  • Ensemble improvement: −2.5% Brier, −28.6% ECE                      │
│  • MCE: 0.1183 (Ensemble) vs 0.1956 (Raw)                             │
│  ⚠️  All metrics are on SYNTHETIC data — not validated on live markets │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Block 6: Calibration Plots                                    ✅ PASS  │
│  ─────────────────────────────────────────────────────────────────────  │
│  • fig_reliability   — Reliability Diagram (calibration curves)        │
│  • fig_brier         — Brier Score Decomposition (stacked bar)         │
│  • fig_distributions — Probability Distribution by Platform            │
│  • fig_dashboard     — Live Markets Dashboard (418 markets)            │
│  NOTE: Platform distribution chart only shows Polymarket+Kalshi        │
│        (Metaculus unavailable); Kalshi spike at 0.50 due to prob=0.50  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Block 7: Live Odds API                                        ✅ PASS  │
│  ─────────────────────────────────────────────────────────────────────  │
│  • PredCalModel instantiated, tracking 418 live markets                │
│  • Batch forecast demo (5 queries):                                     │
│    - "election 2026" → 2.2% calibrated (51 markets matched)           │
│    - "bitcoin price" → 44.6% (8 markets matched)                       │
│    - "AI regulation" → 50.0% prior (0 markets matched)                 │
│    - "fed rate"      → 50.0% prior (0 markets)                         │
│    - "ukraine"       → 50.0% prior (0 markets)                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Block 8: Source & Category Risk Gate                          ✅ PASS  │
│  ─────────────────────────────────────────────────────────────────────  │
│  • Global Brier baseline: 0.2092                                        │
│  • 193 (source, category) buckets evaluated                            │
│  • All 193 → "neutral" (weight=0.75)                                   │
│  • Kalshi categories are raw event_ticker strings (not clean names)    │
│    → category enrichment would improve bucketing                        │
│  • Quality-adjusted forecasts produced for all 5 example queries       │
└─────────────────────────────────────────────────────────────────────────┘
""")

print("=" * 75)
print("FALLBACKS USED:")
print("  1. Metaculus API 403 → empty DataFrame (graceful skip)")
print("  2. Kalshi yes_bid/yes_ask both 0 → defaulted to prob=0.50")
print("  3. feedparser not installed → urllib XML RSS parser")
print("  4. TextBlob not installed → word-count positive/negative sentiment")
print("  5. 0 resolved live markets → synthetic calibration training data")
print("")
print("KNOWN LIMITATIONS:")
print("  • Calibration metrics are on SYNTHETIC data only (no live ground truth)")
print("  • Kalshi probabilities all 0.50 (API returns zero bid/ask — may require auth)")
print("  • Metaculus API requires auth (403) — only 2 of 3 sources active")
print("  • Kalshi category names are raw event ticker strings, not human-readable")
print("  • Cross-platform consensus deviation is 0 (no cross-platform matches found)")
print("=" * 75)
