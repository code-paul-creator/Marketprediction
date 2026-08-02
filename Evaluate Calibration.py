"""
Block 5: Brier Score & Calibration Evaluation — Evaluate Calibration
Full evaluation suite: Brier score decomposition, ECE, MCE, sharpness.
NEW: Per-category calibration breakdown and volume quintile analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
CALIBRATION_BINS = 10    # bins for reliability / ECE / MCE
RANDOM_SEED      = 42
N_VOLUME_QUINTILES = 5   # quintiles for volume analysis

# ─── Re-generate test set predictions ────────────────────────────────────────
from sklearn.model_selection import train_test_split
from scipy.special import expit as _sig

if using_synthetic:
    rng2     = np.random.default_rng(RANDOM_SEED)
    p_true2  = rng2.beta(BETA_ALPHA, BETA_BETA_PARAM, SYNTHETIC_N)
    p_bias2  = p_true2 ** FLB_EXPONENT
    plat2    = rng2.integers(0, 3, SYNTHETIC_N)
    noise2   = np.where(plat2 == 2,
                        rng2.normal(0, METACULUS_NOISE_STD, SYNTHETIC_N),
                        rng2.normal(0, POLYMARKET_NOISE_STD, SYNTHETIC_N))
    p_rep2   = np.clip(p_bias2 + noise2, 0.001, 0.999)
    news2    = rng2.normal(0, 0.3, SYNTHETIC_N)
    p_adj2   = np.clip(p_true2 + NEWS_EFFECT_SIZE * news2, 0.001, 0.999)
    y_all2   = rng2.binomial(1, p_adj2).astype(float)

    log_odds2     = np.log(p_rep2 / (1 - p_rep2))
    near_close2   = rng2.binomial(1, 0.15, SYNTHETIC_N).astype(int)
    is_pol2       = rng2.binomial(1, 0.30, SYNTHETIC_N).astype(int)
    is_crypto2    = rng2.binomial(1, 0.15, SYNTHETIC_N).astype(int)
    is_ext2       = ((p_rep2 < 0.05) | (p_rep2 > 0.95)).astype(int)
    is_mid2       = ((p_rep2 >= 0.4) & (p_rep2 <= 0.6)).astype(int)
    vol2          = np.log1p(rng2.exponential(500, SYNTHETIC_N))
    cons_dev2     = rng2.normal(0, 0.03, SYNTHETIC_N)
    sent_std2     = np.abs(rng2.normal(0, 0.1, SYNTHETIC_N))
    plat_poly2    = (plat2 == 0).astype(int)
    plat_kal2     = (plat2 == 1).astype(int)
    plat_meta2    = (plat2 == 2).astype(int)

    X_all2 = np.column_stack([
        p_rep2, log_odds2, near_close2, is_pol2, is_crypto2,
        is_ext2, is_mid2, plat_poly2, plat_kal2, plat_meta2,
        vol2, cons_dev2, news2, sent_std2,
    ])
    _, X_test2, _, y_test2 = train_test_split(
        X_all2, y_all2, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
    )
else:
    if len(resolved_df) >= REAL_DATA_THRESHOLD:
        real_data2 = resolved_df.copy()
        p_real2 = real_data2["raw_probability"].clip(0.001, 0.999).values
        y_all2  = real_data2["resolved_outcome"].values
        log_odds2_r    = np.log(p_real2 / (1 - p_real2))
        near_close2_r  = np.zeros(len(real_data2), dtype=int)
        is_pol2_r      = real_data2["category"].str.lower().str.contains("politic|election|government", na=False).astype(int).values
        is_crypto2_r   = real_data2["category"].str.lower().str.contains("crypto|bitcoin|eth|blockchain", na=False).astype(int).values
        is_ext2_r      = ((p_real2 < 0.05) | (p_real2 > 0.95)).astype(int)
        is_mid2_r      = ((p_real2 >= 0.4) & (p_real2 <= 0.6)).astype(int)
        log_vol2_r     = np.log1p(real_data2["volume"].fillna(0).clip(lower=0).values)
        X_all2 = np.column_stack([
            p_real2, log_odds2_r, near_close2_r, is_pol2_r, is_crypto2_r,
            is_ext2_r, is_mid2_r,
            np.ones(len(real_data2)), np.zeros(len(real_data2)), np.zeros(len(real_data2)),
            log_vol2_r, np.zeros(len(real_data2)), np.zeros(len(real_data2)), np.zeros(len(real_data2)),
        ])
        _, X_test2, _, y_test2 = train_test_split(
            X_all2, y_all2, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
        )
    else:
        resolved_mask2 = features_df["resolved"] & features_df["outcome"].notna()
        train_data2    = features_df[resolved_mask2][FEATURE_COLS_CAL + ["outcome"]].dropna()
        X_all2         = train_data2[FEATURE_COLS_CAL].values
        y_all2         = train_data2["outcome"].values
        _, X_test2, _, y_test2 = train_test_split(
            X_all2, y_all2, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
        )

p_test2      = X_test2[:, 0]    # raw probabilities on test set
p_ensemble2  = calibration_model.predict(X_test2)

# Individual models
p_iso2   = calibration_model.iso.predict(p_test2)
X_sc2    = calibration_model.scaler.transform(X_test2)
p_log2   = calibration_model.logit.predict_proba(X_sc2)[:, 1]
_a, _b   = calibration_model.beta_params
_logit_p = np.log(np.clip(p_test2, 1e-6, 1-1e-6) / (1 - np.clip(p_test2, 1e-6, 1-1e-6)))
p_beta2  = np.clip(_sig(_a * _logit_p + _b), 1e-7, 1-1e-7)

# ─── Brier decomposition helper ────────────────────────────────────────────────
def brier_decomposition(probs, outcomes, n_bins=CALIBRATION_BINS):
    probs    = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    bins     = np.linspace(0, 1, n_bins + 1)
    bin_ids  = np.digitize(probs, bins[1:-1])
    base     = outcomes.mean()
    rel = res = 0.0
    n   = len(probs)
    for k in range(n_bins):
        mask = bin_ids == k
        n_k  = mask.sum()
        if n_k == 0:
            continue
        p_k = probs[mask].mean()
        o_k = outcomes[mask].mean()
        rel += (n_k / n) * (p_k - o_k) ** 2
        res += (n_k / n) * (o_k - base) ** 2
    unc = base * (1 - base)
    return {"brier_score": rel - res + unc, "reliability": rel,
            "resolution": res, "uncertainty": unc}

# ─── ECE / MCE ────────────────────────────────────────────────────────────────

def compute_ece_mce(probs, outcomes, n_bins=CALIBRATION_BINS):
    probs    = np.asarray(probs)
    outcomes = np.asarray(outcomes)
    bins     = np.linspace(0, 1, n_bins + 1)
    bin_ids  = np.digitize(probs, bins[1:-1])
    ece = mce = 0.0
    n   = len(probs)
    for k in range(n_bins):
        mask = bin_ids == k
        n_k  = mask.sum()
        if n_k == 0:
            continue
        calib_err = abs(probs[mask].mean() - outcomes[mask].mean())
        ece  += (n_k / n) * calib_err
        mce   = max(mce, calib_err)
    return ece, mce


def sharpness(probs):
    probs = np.asarray(probs)
    return float(np.mean(probs * (1 - probs)))


# ─── Evaluate all methods ─────────────────────────────────────────────────────
methods_eval = {
    "Raw Market":       np.clip(p_test2, 1e-7, 1-1e-7),
    "Isotonic":         np.clip(p_iso2,  1e-7, 1-1e-7),
    "Logistic":         np.clip(p_log2,  1e-7, 1-1e-7),
    "Beta Calibration": p_beta2,
    "Ensemble":         p_ensemble2,
}

rows = []
for name, probs in methods_eval.items():
    decomp   = brier_decomposition(probs, y_test2)
    ece, mce = compute_ece_mce(probs, y_test2)
    sharp    = sharpness(probs)
    ll       = log_loss(y_test2, probs)
    rows.append({
        "Method":      name,
        "Brier":       decomp["brier_score"],
        "Reliability": decomp["reliability"],
        "Resolution":  decomp["resolution"],
        "Uncertainty": decomp["uncertainty"],
        "ECE":         ece,
        "MCE":         mce,
        "Sharpness":   sharp,
        "Log Loss":    ll,
    })

eval_results = pd.DataFrame(rows).set_index("Method")

# ─── Calibration curves for plotting ─────────────────────────────────────────
calib_curves = {}
for name, probs in methods_eval.items():
    try:
        frac_pos, mean_pred = calibration_curve(y_test2, probs, n_bins=CALIBRATION_BINS)
        calib_curves[name] = {"mean_pred": mean_pred, "frac_pos": frac_pos}
    except Exception as e:
        calib_curves[name] = {"mean_pred": np.array([]), "frac_pos": np.array([])}

# ─── Print formatted table ────────────────────────────────────────────────────
print("=" * 100)
print("CALIBRATION EVALUATION RESULTS")
print("=" * 100)
print(f"\n{'Method':<20} {'Brier':>7} {'Reliab.':>8} {'Resolut.':>9} {'Uncert.':>8} "
      f"{'ECE':>7} {'MCE':>7} {'Sharp.':>8} {'LogLoss':>8}")
print("-" * 100)
for method, row in eval_results.iterrows():
    print(f"  {method:<18} {row['Brier']:7.4f} {row['Reliability']:8.4f} "
          f"{row['Resolution']:9.4f} {row['Uncertainty']:8.4f} "
          f"{row['ECE']:7.4f} {row['MCE']:7.4f} {row['Sharpness']:8.4f} "
          f"{row['Log Loss']:8.4f}")
print("=" * 100)

raw_bs  = eval_results.loc["Raw Market",  "Brier"]
ens_bs  = eval_results.loc["Ensemble",    "Brier"]
imp_pct = (raw_bs - ens_bs) / raw_bs * 100
print(f"\n✓ Ensemble improves Brier score by {imp_pct:.1f}% vs raw market baseline")
print(f"  Raw Brier: {raw_bs:.4f}  →  Ensemble Brier: {ens_bs:.4f}")
print(f"  ECE improvement: {eval_results.loc['Raw Market','ECE']:.4f} → {eval_results.loc['Ensemble','ECE']:.4f}")


# ─── Compute calibrated_probability for features_df (needed for per-category analysis) ──────
feat_matrix = features_df[FEATURE_COLS_CAL].copy().fillna(0)
p_calibrated_all = calibration_model.predict(feat_matrix.values)
features_df["calibrated_probability"] = p_calibrated_all
features_df["calibration_shift"]      = (features_df["calibrated_probability"]
                                          - features_df["raw_probability"]).abs()


# ─── NEW: Per-category calibration breakdown ──────────────────────────────────

CATEGORY_MAP = {
    "politics":   ["politic", "election", "vote", "senate", "congress", "democrat", "republican", "government"],
    "crypto":     ["crypto", "bitcoin", "btc", "ethereum", "eth", "blockchain", "defi", "solana", "nft"],
    "economics":  ["econom", "gdp", "inflation", "cpi", "recession", "fed", "interest rate", "unemployment", "finance"],
    "science":    ["science", "climate", "health", "ai", "tech", "fusion", "crispr", "quantum", "spacex"],
    "sports":     ["sport", "world cup", "super bowl", "nba", "nfl", "olympic", "baseball", "soccer", "football"],
    "geopolitics":["ukraine", "taiwan", "nato", "iran", "china", "russia", "war", "nuclear", "geopolit"],
}

def classify_category(row) -> str:
    """Classify market into standard category based on title and category field."""
    text = (str(row.get("category", "")) + " " + str(row.get("title", ""))).lower()
    for cat, keywords in CATEGORY_MAP.items():
        if any(kw in text for kw in keywords):
            return cat
    return "other"

features_df["category_std"] = features_df.apply(classify_category, axis=1)

# Compute mock Brier for each category using calibrated probabilities
rng_cat = np.random.default_rng(RANDOM_SEED + 1)

cat_rows = []
for cat, grp in features_df.groupby("category_std"):
    n = len(grp)
    mean_raw  = float(grp["raw_probability"].mean())
    mean_cal  = float(grp["calibrated_probability"].mean())

    p_cal_arr = grp["calibrated_probability"].values
    y_mock = rng_cat.binomial(1, np.clip(p_cal_arr, 0.01, 0.99)).astype(float)
    mock_brier = float(brier_score_loss(y_mock, np.clip(p_cal_arr, 1e-7, 1-1e-7)))

    cat_rows.append({
        "category":              cat,
        "market_count":          n,
        "mean_raw_probability":  round(mean_raw, 4),
        "mean_calibrated_prob":  round(mean_cal, 4),
        "calibration_shift":     round(float(grp["calibration_shift"].mean()), 4),
        "mock_brier":            round(mock_brier, 4),
    })

category_eval_df = pd.DataFrame(cat_rows).sort_values("market_count", ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("PER-CATEGORY CALIBRATION BREAKDOWN")
print("=" * 80)
print(f"\n{'Category':<15} {'Count':>7} {'Mean Raw':>10} {'Mean Cal':>10} {'Cal Shift':>11} {'Mock Brier':>12}")
print("-" * 80)
for _, row in category_eval_df.iterrows():
    print(f"  {row['category']:<13} {int(row['market_count']):>7} "
          f"{row['mean_raw_probability']:>10.3f} {row['mean_calibrated_prob']:>10.3f} "
          f"{row['calibration_shift']:>11.4f} {row['mock_brier']:>12.4f}")
print("=" * 80)


# ─── NEW: Volume quintile analysis ───────────────────────────────────────────

# Use pd.qcut with duplicates='drop' and get actual number of bins
log_vol_series = features_df["log_volume"].fillna(0)

# Try quintiles, fall back gracefully if duplicates reduce bin count
actual_q = N_VOLUME_QUINTILES
quint_labels = None
while actual_q >= 2:
    try:
        quint_cut, quint_bins = pd.qcut(
            log_vol_series,
            q=actual_q,
            retbins=True,
            duplicates="drop"
        )
        n_actual_bins = len(quint_bins) - 1
        quint_labels = [f"Q{i+1}" for i in range(n_actual_bins)]
        quint_cut = pd.qcut(
            log_vol_series,
            q=actual_q,
            labels=quint_labels,
            duplicates="drop"
        )
        break
    except ValueError:
        actual_q -= 1

if actual_q < 2 or quint_labels is None:
    # Fallback: manual tertiles
    quint_cut = pd.cut(log_vol_series,
                       bins=3,
                       labels=["Low", "Mid", "High"])
    quint_labels = ["Low", "Mid", "High"]

features_df["volume_quintile"] = quint_cut

quint_rows = []
for q_label, grp in features_df.groupby("volume_quintile", observed=True):
    n = len(grp)
    mean_raw = float(grp["raw_probability"].mean())
    mean_cal = float(grp["calibrated_probability"].mean())
    sharp_val = float(grp["calibrated_probability"].std())
    log_vol_range = f"[{grp['log_volume'].min():.1f}, {grp['log_volume'].max():.1f}]"

    quint_rows.append({
        "quintile":             str(q_label),
        "market_count":         n,
        "log_vol_range":        log_vol_range,
        "mean_raw_probability": round(mean_raw, 4),
        "mean_calibrated_prob": round(mean_cal, 4),
        "sharpness":            round(sharp_val, 4),
    })

volume_quintile_df = pd.DataFrame(quint_rows)

print("\n" + "=" * 90)
print("VOLUME QUINTILE ANALYSIS")
print("=" * 90)
print(f"\n{'Quintile':>9} {'Count':>7} {'Log Vol Range':>18} {'Mean Raw':>10} {'Mean Cal':>10} {'Sharpness':>11}")
print("-" * 90)
for _, row in volume_quintile_df.iterrows():
    print(f"  {row['quintile']:>7} {int(row['market_count']):>7} {row['log_vol_range']:>18} "
          f"{row['mean_raw_probability']:>10.3f} {row['mean_calibrated_prob']:>10.3f} "
          f"{row['sharpness']:>11.4f}")
print("=" * 90)
print("  (Higher sharpness = more decisive market — probability spread further from 50%)")

print(f"\n✅ category_eval_df ({len(category_eval_df)} categories) and "
      f"volume_quintile_df ({len(volume_quintile_df)} quintiles) ready for Block 6 & 8.")
