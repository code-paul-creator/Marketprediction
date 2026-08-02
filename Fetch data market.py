"""
Block 1: Data Ingestion — Fetch Market Data
Fetches live prediction market data from Polymarket, Kalshi, and Metaculus.
Also fetches resolved Polymarket markets for calibration training.
Data provenance clearly labelled: LIVE vs SIMULATED.

- Polymarket: LIVE via Gamma API (primary, active markets with real prices)
              Falls back to CLOB API if Gamma returns fewer than 50 markets
- Polymarket Resolved: LIVE via Gamma API (closed markets with resolvedPrice)
- Kalshi: Attempts external-api.kalshi.com; falls back to synthetic if $0 prices
- Metaculus: Attempts www.metaculus.com/api2; falls back to synthetic (IP blocked from cloud)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time
import json
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
POLYMARKET_URL          = "https://clob.polymarket.com/markets"
POLYMARKET_GAMMA_URL    = "https://gamma-api.polymarket.com/markets"
KALSHI_URL              = "https://external-api.kalshi.com/trade-api/v2/markets"
METACULUS_URL           = "https://www.metaculus.com/api2/questions/"
REQUEST_TIMEOUT         = 20      # seconds per request
POLYMARKET_LIMIT        = 500     # approximate markets to fetch via pagination
GAMMA_PAGE_SIZE         = 100     # items per page for Gamma API
GAMMA_MAX_PAGES         = 5       # 5 pages × 100 = up to 500 markets
GAMMA_FALLBACK_THRESH   = 50      # fall back to CLOB if Gamma returns fewer than this
RESOLVED_LIMIT          = 500     # resolved markets for calibration training
RESOLVED_OFFSET_STEP    = 100     # pagination step for gamma resolved API
KALSHI_LIMIT            = 200
METACULUS_LIMIT         = 200
KALSHI_SYNTH_N          = 150     # synthetic Kalshi markets to generate if API has no prices
METACULUS_SYNTH_N       = 150     # synthetic Metaculus questions to generate if API blocked
RANDOM_SEED             = 42
REQUEST_HEADERS         = {
    "User-Agent": "Mozilla/5.0 (compatible; PredCalResearch/1.0)",
    "Accept": "application/json",
}

rng = np.random.default_rng(RANDOM_SEED)


# ─── Polymarket (Active markets via Gamma API — primary) ──────────────────────

def fetch_polymarket_gamma(max_markets: int = POLYMARKET_LIMIT) -> pd.DataFrame:
    """
    Fetch active, open markets from Polymarket Gamma API (offset pagination).
    Primary source — returns real prices for actively traded markets.
    URL: https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=100&offset=N
    Key fields: question, outcomePrices, volume, endDate, active, closed, category/tags
    """
    records = []
    page_size = GAMMA_PAGE_SIZE
    max_pages = GAMMA_MAX_PAGES
    pages_fetched = 0
    now = datetime.now(timezone.utc)

    for page_num in range(max_pages):
        offset = page_num * page_size
        url = (
            f"{POLYMARKET_GAMMA_URL}"
            f"?active=true&closed=false&limit={page_size}&offset={offset}"
        )
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [Polymarket Gamma] fetch error (offset={offset}): {e}")
            break

        # Gamma API returns a list directly
        if isinstance(data, list):
            markets = data
        elif isinstance(data, dict):
            markets = data.get("data", data.get("markets", []))
        else:
            markets = []

        if not markets:
            print(f"  [Polymarket Gamma] Empty response at offset={offset}, stopping pagination.")
            break

        pages_fetched += 1

        for m in markets:
            # Only keep active=true, closed=false
            if not m.get("active", False) or m.get("closed", True):
                continue

            # Parse outcomePrices: JSON string like '["0.63", "0.37"]' or list
            outcome_prices_raw = m.get("outcomePrices")
            prob = None
            try:
                if isinstance(outcome_prices_raw, str):
                    prices_list = json.loads(outcome_prices_raw)
                elif isinstance(outcome_prices_raw, list):
                    prices_list = outcome_prices_raw
                else:
                    prices_list = []

                if prices_list:
                    prob = float(str(prices_list[0]).strip())
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                prob = None

            # Skip if outcomePrices is missing/invalid, or is exactly [0.5, 0.5] with zero volume
            volume = float(m.get("volume", 0) or 0)
            if prob is None:
                continue
            if abs(prob - 0.5) < 1e-9 and volume == 0:
                continue  # untraded market with flat/default price

            # Filter to real probability range
            if not (0.01 <= prob <= 0.99):
                continue

            # Parse endDate → days_to_close
            end_date_str = m.get("endDate", "")
            days_to_close = 0
            if end_date_str:
                try:
                    # Handle both 'Z' and '+00:00' suffixes
                    end_dt_str = end_date_str.replace("Z", "+00:00")
                    end_dt = datetime.fromisoformat(end_dt_str)
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=timezone.utc)
                    days_to_close = max(0, (end_dt - now).days)
                except (ValueError, AttributeError):
                    days_to_close = 0

            # Extract category/tags
            category = m.get("category", "") or ""
            if not category:
                tags = m.get("tags", [])
                if isinstance(tags, list) and tags:
                    # tags may be dicts with 'label' or just strings
                    first_tag = tags[0]
                    if isinstance(first_tag, dict):
                        category = first_tag.get("label", first_tag.get("name", ""))
                    else:
                        category = str(first_tag)

            records.append({
                "source":           "polymarket",
                "question_id":      m.get("id", m.get("slug", "")),
                "title":            m.get("question", ""),
                "probability":      prob,
                "volume":           volume,
                "close_time":       end_date_str,
                "days_to_close":    days_to_close,
                "category":         category,
                "resolved":         False,
                "outcome":          np.nan,
                "data_provenance":  "live",
            })

        if len(markets) < page_size:
            # Last page — no more to fetch
            break

    df = pd.DataFrame(records) if records else pd.DataFrame()
    print(f"✅ [Polymarket Gamma] fetched {len(df)} active markets (LIVE, {pages_fetched} pages)")
    return df


def fetch_polymarket_clob(max_markets: int = POLYMARKET_LIMIT) -> pd.DataFrame:
    """
    Fallback: Fetch active markets from Polymarket CLOB API (cursor-based pagination).
    Used only when Gamma API returns fewer than GAMMA_FALLBACK_THRESH markets.
    """
    records = []
    cursor = "MQ=="   # starting cursor
    pages_fetched = 0
    now = datetime.now(timezone.utc)

    while len(records) < max_markets:
        try:
            resp = requests.get(
                POLYMARKET_URL,
                params={"next_cursor": cursor},
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [Polymarket CLOB] fetch error (cursor={cursor}): {e}")
            break

        markets = data.get("data", [])
        if not markets:
            break

        for m in markets:
            if not m.get("active") or m.get("closed"):
                continue
            tokens = m.get("tokens", [])
            prob = None
            for tok in tokens:
                if str(tok.get("outcome", "")).upper() == "YES":
                    try:
                        prob = float(tok["price"])
                    except (KeyError, TypeError, ValueError):
                        pass
                    break
            if prob is None and len(tokens) > 0:
                try:
                    prob = float(tokens[0].get("price", 0.5))
                except (TypeError, ValueError):
                    prob = 0.5
            if prob is None:
                prob = 0.5

            # Fix days_to_close: use end_date_iso from CLOB
            end_date_str = m.get("end_date_iso", "")
            days_to_close = 0
            if end_date_str:
                try:
                    end_dt_str = end_date_str.replace("Z", "+00:00")
                    end_dt = datetime.fromisoformat(end_dt_str)
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=timezone.utc)
                    days_to_close = max(0, (end_dt - now).days)
                except (ValueError, AttributeError):
                    days_to_close = 0

            prob = max(0.001, min(0.999, prob))

            records.append({
                "source":           "polymarket",
                "question_id":      m.get("condition_id", m.get("market_slug", "")),
                "title":            m.get("question", ""),
                "probability":      prob,
                "volume":           float(m.get("volume", 0) or 0),
                "close_time":       end_date_str,
                "days_to_close":    days_to_close,
                "category":         m.get("category", ""),
                "resolved":         False,
                "outcome":          np.nan,
                "data_provenance":  "live",
            })

        pages_fetched += 1
        next_cursor = data.get("next_cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.3)

    df = pd.DataFrame(records) if records else pd.DataFrame()
    print(f"✅ [Polymarket CLOB] fetched {len(df)} active markets (LIVE fallback, {pages_fetched} pages)")
    return df


def fetch_polymarket(max_markets: int = POLYMARKET_LIMIT) -> pd.DataFrame:
    """
    Fetch active markets from Polymarket.
    Primary: Gamma API (active=true, closed=false, offset pagination).
    Fallback: CLOB API if Gamma returns fewer than GAMMA_FALLBACK_THRESH markets.
    """
    df = fetch_polymarket_gamma(max_markets=max_markets)
    if len(df) < GAMMA_FALLBACK_THRESH:
        print(f"  [Polymarket] Gamma returned only {len(df)} markets (< {GAMMA_FALLBACK_THRESH}) — falling back to CLOB API")
        df = fetch_polymarket_clob(max_markets=max_markets)
    return df


# ─── Polymarket Resolved Markets (Gamma API) ──────────────────────────────────

def fetch_polymarket_resolved(max_markets: int = RESOLVED_LIMIT) -> pd.DataFrame:
    """
    Fetch resolved/settled markets from Polymarket Gamma API.
    Returns DataFrame with columns: title, resolved_outcome, raw_probability, volume, category, source.
    """
    records = []
    offset = 0
    batch_size = RESOLVED_OFFSET_STEP
    attempts_without_progress = 0

    print(f"[Polymarket Resolved] Fetching up to {max_markets} resolved markets via Gamma API ...")

    while len(records) < max_markets:
        # Try primary endpoint first
        urls_to_try = [
            f"{POLYMARKET_GAMMA_URL}?closed=true&limit={batch_size}&offset={offset}",
            f"{POLYMARKET_GAMMA_URL}?closed=true&active=false&archived=true&limit={batch_size}&offset={offset}",
        ]

        data = None
        for url in urls_to_try:
            try:
                resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                if data:
                    break
            except Exception as e:
                print(f"  [Gamma API] fetch error ({url}): {e}")
                data = None

        if data is None:
            print("  [Polymarket Resolved] Both Gamma API endpoints failed.")
            break

        # Gamma API may return list or dict with 'data' key
        if isinstance(data, list):
            markets = data
        elif isinstance(data, dict):
            markets = data.get("data", data.get("markets", []))
        else:
            markets = []

        if not markets:
            print(f"  [Polymarket Resolved] No more markets at offset={offset}")
            break

        new_this_batch = 0
        for m in markets:
            # Extract resolvedPrice — must be "0.0" or "1.0" (or 0/1 as numeric)
            resolved_price_raw = m.get("resolvedPrice", m.get("resolution", None))
            if resolved_price_raw is None:
                continue
            try:
                resolved_price = float(resolved_price_raw)
            except (TypeError, ValueError):
                continue

            # Filter to binary outcomes only (0.0 or 1.0)
            if resolved_price not in (0.0, 1.0):
                continue

            # Extract raw probability (last trade price before close)
            outcome_prices = m.get("outcomePrices", [])
            raw_prob = 0.5
            try:
                if isinstance(outcome_prices, str):
                    outcome_prices = json.loads(outcome_prices)
                if outcome_prices and len(outcome_prices) >= 2:
                    raw_prob = float(str(outcome_prices[0]).strip())
                else:
                    raw_prob = float(m.get("lastTradePrice", 0.5) or 0.5)
            except (TypeError, ValueError, json.JSONDecodeError):
                raw_prob = 0.5
            raw_prob = max(0.001, min(0.999, float(raw_prob)))

            title = m.get("question", m.get("title", ""))
            volume = float(m.get("volume", 0) or 0)
            category = m.get("category", "")

            records.append({
                "title":            title,
                "resolved_outcome": float(resolved_price),
                "raw_probability":  raw_prob,
                "volume":           volume,
                "category":         category,
                "source":           "polymarket_resolved",
            })
            new_this_batch += 1

        if new_this_batch == 0:
            attempts_without_progress += 1
            if attempts_without_progress >= 3:
                print(f"  [Polymarket Resolved] No new valid records for 3 batches — stopping.")
                break
        else:
            attempts_without_progress = 0

        offset += batch_size
        time.sleep(0.3)

    resolved_df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["title", "resolved_outcome", "raw_probability", "volume", "category", "source"]
    )

    n_valid = len(resolved_df)
    print(f"✅ [Polymarket Resolved] fetched {n_valid} resolved markets with valid resolvedPrice (YES/NO outcome)")
    if n_valid < 50:
        print(f"⚠️  Only {n_valid} resolved markets with valid outcomes — using synthetic training if <50")

    return resolved_df


# ─── Kalshi ───────────────────────────────────────────────────────────────────

KALSHI_CATEGORY_TEMPLATES = [
    ("Economics", ["Will US GDP growth exceed 2% in Q3 2026?",
                   "Will the Fed cut rates at September 2026 FOMC meeting?",
                   "Will US unemployment stay below 5% by year-end 2026?",
                   "Will inflation (CPI) drop below 3% by December 2026?"]),
    ("Politics",  ["Will a US infrastructure bill pass by end of 2026?",
                   "Will Trump's approval rating exceed 50% in August 2026?",
                   "Will the Senate flip in 2026 midterms?",
                   "Will any new cabinet secretary be confirmed this month?"]),
    ("Tech",      ["Will AI GDP contribution exceed $500B in 2026?",
                   "Will Apple release an AI chip by end of 2026?",
                   "Will a major tech company lay off > 10,000 employees in Q3?",
                   "Will cryptocurrency market cap exceed $4T by year-end?"]),
    ("Climate",   ["Will 2026 be the hottest year on record?",
                   "Will a new climate treaty be signed in 2026?",
                   "Will global EV sales exceed 25M units in 2026?"]),
    ("Sports",    ["Will the Yankees make the 2026 World Series?",
                   "Will the NBA Finals go to 7 games in 2026?",
                   "Will a new 100m world record be set at 2026 World Athletics?"]),
]

def generate_synthetic_kalshi(n: int = KALSHI_SYNTH_N) -> pd.DataFrame:
    """
    Generate synthetic Kalshi markets when live API has $0 prices.
    Uses N(0.5, 0.18) distribution clipped to [0.03, 0.97], reflecting
    real-money market characteristics (slight favorite-longshot bias, wider spread).
    """
    all_cats = []
    all_titles = []
    for cat, titles in KALSHI_CATEGORY_TEMPLATES:
        for t in titles:
            all_cats.append(cat)
            all_titles.append(t)

    records = []
    now = datetime.now(timezone.utc)

    for i in range(n):
        prob = float(np.clip(rng.normal(0.5, 0.18), 0.03, 0.97))
        days_out = int(rng.integers(7, 180))
        cat_idx = i % len(all_cats)
        volume = float(np.clip(rng.lognormal(8, 1.5), 100, 500_000))
        records.append({
            "source":          "kalshi",
            "question_id":     f"KALSHI-SYN-{i:04d}",
            "title":           f"{all_titles[cat_idx]} (variant {i // len(all_cats) + 1})",
            "probability":     prob,
            "volume":          volume,
            "close_time":      (now + timedelta(days=days_out)).isoformat(),
            "days_to_close":   days_out,
            "category":        all_cats[cat_idx],
            "resolved":        False,
            "outcome":         np.nan,
            "data_provenance": "simulated",
        })

    df = pd.DataFrame(records)
    print(f"⚠️  [Kalshi] API returned $0 prices for all markets — using {len(df)} SIMULATED markets")
    print(f"   Distribution: N(0.5, 0.18) clipped [0.03, 0.97] | prob range [{df['probability'].min():.2f}, {df['probability'].max():.2f}]")
    return df


def fetch_kalshi(limit: int = KALSHI_LIMIT) -> pd.DataFrame:
    """
    Fetch open markets from Kalshi external API.
    Price fields: yes_bid_dollars / yes_ask_dollars (string floats in $).
    Falls back to synthetic if all markets have $0 prices.
    """
    records = []
    now = datetime.now(timezone.utc)
    try:
        resp = requests.get(
            KALSHI_URL,
            params={"limit": limit, "status": "open"},
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_markets = data.get("markets", [])

        zero_price_count = 0
        for m in raw_markets:
            bid = float(m.get("yes_bid_dollars", 0) or 0)
            ask = float(m.get("yes_ask_dollars", 0) or 0)
            if bid == 0 and ask == 0:
                zero_price_count += 1
                continue

            prob = (bid + ask) / 2.0
            prob = max(0.01, min(0.99, prob))

            # days_to_close
            close_time_str = m.get("close_time", "")
            days_to_close = 0
            if close_time_str:
                try:
                    ct_str = close_time_str.replace("Z", "+00:00")
                    close_dt = datetime.fromisoformat(ct_str)
                    if close_dt.tzinfo is None:
                        close_dt = close_dt.replace(tzinfo=timezone.utc)
                    days_to_close = max(0, (close_dt - now).days)
                except (ValueError, AttributeError):
                    days_to_close = 0

            records.append({
                "source":          "kalshi",
                "question_id":     m.get("ticker", ""),
                "title":           m.get("title", ""),
                "probability":     prob,
                "volume":          float(m.get("volume", 0) or 0),
                "close_time":      close_time_str,
                "days_to_close":   days_to_close,
                "category":        m.get("category", ""),
                "resolved":        False,
                "outcome":         np.nan,
                "data_provenance": "live",
            })

        print(f"✅ [Kalshi] fetched {len(records)} open markets with prices (skipped {zero_price_count} zero-liquidity)")

    except Exception as e:
        print(f"⚠️  [Kalshi] API error: {e}")
        records = []

    if len(records) == 0:
        return generate_synthetic_kalshi()

    return pd.DataFrame(records)


# ─── Metaculus ────────────────────────────────────────────────────────────────

METACULUS_TOPIC_TEMPLATES = [
    ("Politics",    ["Will the US pass a major climate bill by 2027?",
                     "Will there be a US government shutdown in 2026?",
                     "Will any G7 country hold a snap election in 2026?"]),
    ("Economics",   ["Will global GDP growth exceed 3% in 2026?",
                     "Will the US enter recession in 2026?",
                     "Will Bitcoin exceed $200,000 by end of 2026?"]),
    ("Science",     ["Will a new SARS-like pandemic emerge by 2027?",
                     "Will a commercial fusion reactor achieve Q>1 by 2030?",
                     "Will AI pass the ARC-AGI benchmark by 2027?"]),
    ("Technology",  ["Will GPT-5 be released before end of 2026?",
                     "Will a self-driving car achieve Level 5 by 2027?",
                     "Will quantum supremacy be achieved for a practical problem by 2026?"]),
    ("Geopolitics", ["Will Russia and Ukraine sign a ceasefire in 2026?",
                     "Will North Korea conduct a nuclear test in 2026?",
                     "Will Taiwan experience a military blockade in 2026?"]),
    ("Health",      ["Will mRNA cancer vaccines enter Phase 3 trials by 2027?",
                     "Will a new flu pandemic begin in 2026?",
                     "Will COVID-19 be declared fully endemic by WHO in 2026?"]),
]

def generate_synthetic_metaculus(n: int = METACULUS_SYNTH_N) -> pd.DataFrame:
    """
    Generate synthetic Metaculus-style questions when the API is blocked.
    Uses Beta(3,3) distribution (community consensus tends toward 30-70%).
    """
    all_cats, all_titles = [], []
    for cat, titles in METACULUS_TOPIC_TEMPLATES:
        for t in titles:
            all_cats.append(cat)
            all_titles.append(t)

    records = []
    now = datetime.now(timezone.utc)

    for i in range(n):
        prob = float(np.clip(rng.beta(3, 3), 0.05, 0.95))
        days_out = int(rng.integers(30, 730))
        cat_idx = i % len(all_cats)
        volume = float(np.clip(rng.lognormal(3, 1.2), 10, 5_000))
        records.append({
            "source":          "metaculus",
            "question_id":     f"META-SYN-{i:04d}",
            "title":           f"{all_titles[cat_idx]} (variant {i // len(all_cats) + 1})",
            "probability":     prob,
            "volume":          volume,
            "close_time":      (now + timedelta(days=days_out)).isoformat(),
            "days_to_close":   days_out,
            "category":        all_cats[cat_idx],
            "resolved":        False,
            "outcome":         np.nan,
            "data_provenance": "simulated",
        })

    df = pd.DataFrame(records)
    print(f"⚠️  [Metaculus] IP-blocked or unavailable — using {len(df)} SIMULATED questions")
    print(f"   Distribution: Beta(3,3) clipped [0.05, 0.95] | prob range [{df['probability'].min():.2f}, {df['probability'].max():.2f}]")
    return df


def fetch_metaculus(limit: int = METACULUS_LIMIT) -> pd.DataFrame:
    """
    Fetch open forecast questions from Metaculus REST API.
    Falls back to synthetic data if IP-blocked.
    """
    records = []
    now = datetime.now(timezone.utc)
    try:
        resp = requests.get(
            METACULUS_URL,
            params={"limit": limit, "status": "open", "type": "forecast"},
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        questions = data.get("results", [])

        for q in questions:
            comm = q.get("community_prediction", {})
            prob = None
            if isinstance(comm, dict):
                prob = comm.get("full", {}).get("q2") if isinstance(comm.get("full"), dict) else None
            if prob is None:
                continue

            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue

            prob = max(0.01, min(0.99, prob))

            close_time_str = q.get("close_time", "")
            days_to_close = 0
            if close_time_str:
                try:
                    ct_str = close_time_str.replace("Z", "+00:00")
                    close_dt = datetime.fromisoformat(ct_str)
                    if close_dt.tzinfo is None:
                        close_dt = close_dt.replace(tzinfo=timezone.utc)
                    days_to_close = max(0, (close_dt - now).days)
                except (ValueError, AttributeError):
                    days_to_close = 0

            records.append({
                "source":          "metaculus",
                "question_id":     str(q.get("id", "")),
                "title":           q.get("title", ""),
                "probability":     prob,
                "volume":          float(q.get("number_of_predictions", 0) or 0),
                "close_time":      close_time_str,
                "days_to_close":   days_to_close,
                "category":        q.get("categories", [{}])[0].get("name", "") if q.get("categories") else "",
                "resolved":        False,
                "outcome":         np.nan,
                "data_provenance": "live",
            })

        if records:
            print(f"✅ [Metaculus] fetched {len(records)} open questions (LIVE)")
            return pd.DataFrame(records)

    except Exception as e:
        print(f"[Metaculus] fetch error: {e}")

    return generate_synthetic_metaculus()


# ─── Combine All Sources ──────────────────────────────────────────────────────

def fetch_all_markets() -> tuple:
    """Fetch and combine markets from all sources."""
    print("=" * 60)
    print("Fetching prediction market data …")
    print("=" * 60)

    poly_df  = fetch_polymarket()
    kal_df   = fetch_kalshi()
    meta_df  = fetch_metaculus()

    # Ensure days_to_close column exists in all DataFrames
    now = datetime.now(timezone.utc)
    for df_part in [poly_df, kal_df, meta_df]:
        if "days_to_close" not in df_part.columns:
            if "close_time" in df_part.columns:
                def _parse_dtc(s):
                    try:
                        ct = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
                        if ct.tzinfo is None:
                            ct = ct.replace(tzinfo=timezone.utc)
                        return max(0, (ct - now).days)
                    except Exception:
                        return 0
                df_part["days_to_close"] = df_part["close_time"].apply(_parse_dtc)
            else:
                df_part["days_to_close"] = 0

    all_parts = [df for df in [poly_df, kal_df, meta_df] if not df.empty]
    if not all_parts:
        markets_df = pd.DataFrame()
    else:
        markets_df = pd.concat(all_parts, ignore_index=True)

    # Ensure consistent columns
    for col in ["source", "question_id", "title", "probability", "volume",
                "close_time", "days_to_close", "category", "resolved", "outcome", "data_provenance"]:
        if col not in markets_df.columns:
            markets_df[col] = np.nan if col in ["probability", "volume", "outcome"] else ""

    print()
    print("=" * 60)
    print(f"Total active markets: {len(markets_df)}")
    for src, grp in markets_df.groupby("source"):
        print(f"  {src.capitalize():<12}: {len(grp)}")
    print()

    return markets_df


# ─── Run ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Fetching prediction market data …")
print("=" * 60)

poly_df  = fetch_polymarket()
kal_df   = fetch_kalshi()
meta_df  = fetch_metaculus()

# Ensure days_to_close in all parts
now = datetime.now(timezone.utc)
for _df in [poly_df, kal_df, meta_df]:
    if not _df.empty and "days_to_close" not in _df.columns:
        def _parse_dtc(s):
            try:
                ct = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
                if ct.tzinfo is None:
                    ct = ct.replace(tzinfo=timezone.utc)
                return max(0, (ct - now).days)
            except Exception:
                return 0
        _df["days_to_close"] = _df["close_time"].apply(_parse_dtc)

markets_df = pd.concat(
    [df for df in [poly_df, kal_df, meta_df] if not df.empty],
    ignore_index=True
)

# Ensure consistent columns
for _col in ["source", "question_id", "title", "probability", "volume",
             "close_time", "days_to_close", "category", "resolved", "outcome", "data_provenance"]:
    if _col not in markets_df.columns:
        markets_df[_col] = np.nan if _col in ["probability", "volume", "outcome"] else ""

print()
print("=" * 60)
print(f"Total active markets: {len(markets_df)}")
for _src, _grp in markets_df.groupby("source"):
    print(f"  {_src.capitalize():<12}: {len(_grp)}")

print()
print("=" * 60)
print("Fetching resolved markets for calibration training …")
print("=" * 60)

resolved_df = fetch_polymarket_resolved()
n_valid_resolved = len(resolved_df)
print(f"\nResolved markets summary:")
print(f"  Total with valid resolvedPrice : {n_valid_resolved}")

print(f"\n✅ Data ingestion complete: {len(markets_df)} active + {n_valid_resolved} resolved markets")


# ─── Summary of Polymarket Gamma fetch (Block 1 verification) ─────────────────

poly_only = markets_df[markets_df["source"] == "polymarket"].copy()
print()
print("=" * 60)
print("POLYMARKET FETCH SUMMARY (Gamma API)")
print("=" * 60)
print(f"Markets fetched : {len(poly_only)}")

if len(poly_only) > 0:
    probs = poly_only["probability"]
    print(f"Probability stats:")
    print(f"  min  = {probs.min():.4f}")
    print(f"  max  = {probs.max():.4f}")
    print(f"  mean = {probs.mean():.4f}")
    print(f"  std  = {probs.std():.4f}")
    n_not_half = (probs != 0.5).sum()
    print(f"  markets with prob != 0.5 : {n_not_half}")
    n_with_volume = (poly_only["volume"] > 0).sum()
    print(f"  markets with volume > 0  : {n_with_volume}")
    n_nonneg_days = (poly_only["days_to_close"] >= 0).sum()
    print(f"  markets with days_to_close >= 0 : {n_nonneg_days}")

    print()
    print("Sample of 5 markets:")
    sample_cols = ["title", "probability", "volume", "days_to_close"]
    sample = poly_only[sample_cols].head(5)
    for _, row in sample.iterrows():
        print(f"  • {row['title'][:70]}")
        print(f"    prob={row['probability']:.3f}  vol={row['volume']:,.0f}  days_to_close={row['days_to_close']}")
else:
    print("⚠️  No Polymarket markets fetched!")
