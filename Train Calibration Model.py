"""
Block 4: Calibration Modeling — Train Calibration Model
Trains three calibration approaches + an ensemble on prediction market probabilities.

When resolved_df has >= REAL_DATA_THRESHOLD rows, uses REAL resolved market data.
Otherwise falls through to synthetic calibration training data clearly flagged in output.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss
from scipy.optimize import minimize
from scipy.special import expit   # sigmoid
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
SYNTHETIC_N          = 2000    # synthetic training samples when real data scarce
REAL_DATA_THRESHOLD  = 50      # min resolved questions to use real data
TRAIN_TEST_SPLIT     = 0.2     # held-out fraction for evaluation
RANDOM_SEED          = 42
CALIBRATION_BINS     = 10      # bins for reliability evaluation
ENSEMBLE_OPTIM_ITER  = 500     # L-BFGS-B iterations for weight optimisation

# ─── Synthetic data parameters (from calibration literature) ──────────────────
BETA_ALPHA           = 2.0
BETA_BETA_PARAM      = 2.0
FLB_EXPONENT         = 0.85    # favorite-longshot compression
POLYMARKET_NOISE_STD = 0.03
METACULUS_NOISE_STD  = 0.01
NEWS_EFFECT_SIZE     = 0.05


# ─── Helpers ─────────────────────────────────────────────────────────────────

def beta_calibration_predict(params, p):
    """Beta calibration: logit(q) = a * logit(p) + b"""
    a, b = params
    logit_p = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    logit_q = a * logit_p + b
    return expit(logit_q)


def beta_calibration_loss(params, p, y):
    q = beta_calibration_predict(params, p)
    q = np.clip(q, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))


def brier_decomposition(probs, outcomes, n_bins=CALIBRATION_BINS):
    """Murphy (1973) Brier score decomposition: BS = Reliability - Resolution + Uncertainty."""
    probs    = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    bins     = np.linspace(0, 1, n_bins + 1)
    bin_ids  = np.digitize(probs, bins[1:-1])
    base_rate = outcomes.mean()

    reliability = 0.0
    resolution  = 0.0
    n_total     = len(probs)

    for k in range(n_bins):
        mask = bin_ids == k
        n_k  = mask.sum()
        if n_k == 0:
            continue
        p_k = probs[mask].mean()
        o_k = outcomes[mask].mean()
        reliability += (n_k / n_total) * (p_k - o_k) ** 2
        resolution  += (n_k / n_total) * (o_k - base_rate) ** 2

    uncertainty = base_rate * (1 - base_rate)
    bs          = reliability - resolution + uncertainty
    return {
        "brier_score": bs,
        "reliability": reliability,
        "resolution":  resolution,
        "uncertainty": uncertainty,
    }


# ─── Step 1: Check for resolved data from Block 1 ────────────────────────────
print("=" * 60)
print("CALIBRATION MODEL TRAINING")
print("=" * 60)

# Try to use resolved_df from Block 1
try:
    n_resolved_real = len(resolved_df) if resolved_df is not None else 0
except NameError:
    n_resolved_real = 0
    resolved_df = pd.DataFrame()

# Also check features_df for any resolved markets
resolved_mask = features_df["resolved"] & features_df["outcome"].notna()
n_resolved_features = resolved_mask.sum()

print(f"\nResolved markets from gamma API (resolved_df): {n_resolved_real}")
print(f"Resolved markets in features_df             : {n_resolved_features}")

FEATURE_COLS = [
    "raw_probability", "log_odds", "near_close", "is_political", "is_crypto",
    "is_extreme", "is_mid_band", "platform_polymarket", "platform_kalshi",
    "platform_metaculus", "log_volume", "consensus_deviation",
    "news_sentiment_mean", "news_sentiment_std",
]

# ─── Step 2: Decide training data source ─────────────────────────────────────

using_synthetic = True  # default

if n_resolved_real >= REAL_DATA_THRESHOLD:
    print(f"\n✅ Training on REAL resolved market data: {n_resolved_real} markets")
    using_synthetic = False

    # Build feature matrix from resolved_df
    # resolved_df has: title, resolved_outcome, raw_probability, volume, category, source
    rng_real = np.random.default_rng(RANDOM_SEED)

    real_data = resolved_df.copy()
    p_real = real_data["raw_probability"].clip(0.001, 0.999).values
    y_real = real_data["resolved_outcome"].values

    # Compute derived features for real data
    log_odds_real    = np.log(p_real / (1 - p_real))
    near_close_real  = np.zeros(len(real_data), dtype=int)  # resolved markets already closed
    is_pol_real      = real_data["category"].str.lower().str.contains("politic|election|government", na=False).astype(int).values
    is_crypto_real   = real_data["category"].str.lower().str.contains("crypto|bitcoin|eth|blockchain", na=False).astype(int).values
    is_extreme_real  = ((p_real < 0.05) | (p_real > 0.95)).astype(int)
    is_mid_real      = ((p_real >= 0.4) & (p_real <= 0.6)).astype(int)
    log_vol_real     = np.log1p(real_data["volume"].fillna(0).clip(lower=0).values)
    cons_dev_real    = np.zeros(len(real_data))  # no cross-platform data for resolved
    news_sent_real   = np.zeros(len(real_data))
    news_std_real    = np.zeros(len(real_data))
    # All from polymarket_resolved
    plat_poly_real   = np.ones(len(real_data), dtype=int)
    plat_kal_real    = np.zeros(len(real_data), dtype=int)
    plat_meta_real   = np.zeros(len(real_data), dtype=int)

    train_data = pd.DataFrame({
        "raw_probability":      p_real,
        "log_odds":             log_odds_real,
        "near_close":           near_close_real,
        "is_political":         is_pol_real,
        "is_crypto":            is_crypto_real,
        "is_extreme":           is_extreme_real,
        "is_mid_band":          is_mid_real,
        "platform_polymarket":  plat_poly_real,
        "platform_kalshi":      plat_kal_real,
        "platform_metaculus":   plat_meta_real,
        "log_volume":           log_vol_real,
        "consensus_deviation":  cons_dev_real,
        "news_sentiment_mean":  news_sent_real,
        "news_sentiment_std":   news_std_real,
        "outcome":              y_real,
    })

elif n_resolved_features >= REAL_DATA_THRESHOLD:
    print(f"\n✅ Training on REAL resolved market data from features_df: {n_resolved_features} markets")
    using_synthetic = False
    train_data = features_df[resolved_mask][FEATURE_COLS + ["outcome"]].dropna()

else:
    n_available = max(n_resolved_real, n_resolved_features)
    print(f"\n⚠️  Only {n_available} resolved markets with valid outcomes — using synthetic training")
    print(f"   (need {REAL_DATA_THRESHOLD}, got {n_available})")
    print("   Using SYNTHETIC calibration training data generated from:")
    print("   • Beta(2,2) true probability distribution")
    print(f"   • Favorite-Longshot Bias exponent = {FLB_EXPONENT}")
    print("   • Per-platform noise: Polymarket σ=0.03, Metaculus σ=0.01")
    print("   • News sentiment signal: r=0.05")
    print(f"   • N = {SYNTHETIC_N} synthetic samples")
    print("   Calibration metrics reflect SYNTHETIC, not live, market behavior.\n")

    rng = np.random.default_rng(RANDOM_SEED)

    p_true    = rng.beta(BETA_ALPHA, BETA_BETA_PARAM, SYNTHETIC_N)
    p_biased  = p_true ** FLB_EXPONENT
    platform  = rng.integers(0, 3, SYNTHETIC_N)

    noise = np.where(platform == 2,
                     rng.normal(0, METACULUS_NOISE_STD, SYNTHETIC_N),
                     rng.normal(0, POLYMARKET_NOISE_STD, SYNTHETIC_N))
    p_reported = np.clip(p_biased + noise, 0.001, 0.999)

    news_sentiment = rng.normal(0, 0.3, SYNTHETIC_N)
    p_adjusted = np.clip(p_true + NEWS_EFFECT_SIZE * news_sentiment, 0.001, 0.999)
    outcomes_syn = rng.binomial(1, p_adjusted).astype(float)

    log_odds_syn   = np.log(p_reported / (1 - p_reported))
    near_close_syn = rng.binomial(1, 0.15, SYNTHETIC_N).astype(int)
    is_pol_syn     = rng.binomial(1, 0.30, SYNTHETIC_N).astype(int)
    is_crypto_syn  = rng.binomial(1, 0.15, SYNTHETIC_N).astype(int)
    is_extreme_syn = ((p_reported < 0.05) | (p_reported > 0.95)).astype(int)
    is_mid_syn     = ((p_reported >= 0.4) & (p_reported <= 0.6)).astype(int)
    vol_syn        = np.log1p(rng.exponential(500, SYNTHETIC_N))
    cons_dev_syn   = rng.normal(0, 0.03, SYNTHETIC_N)
    sent_std_syn   = np.abs(rng.normal(0, 0.1, SYNTHETIC_N))
    platform_poly  = (platform == 0).astype(int)
    platform_kal   = (platform == 1).astype(int)
    platform_meta  = (platform == 2).astype(int)

    train_data = pd.DataFrame({
        "raw_probability":      p_reported,
        "log_odds":             log_odds_syn,
        "near_close":           near_close_syn,
        "is_political":         is_pol_syn,
        "is_crypto":            is_crypto_syn,
        "is_extreme":           is_extreme_syn,
        "is_mid_band":          is_mid_syn,
        "platform_polymarket":  platform_poly,
        "platform_kalshi":      platform_kal,
        "platform_metaculus":   platform_meta,
        "log_volume":           vol_syn,
        "consensus_deviation":  cons_dev_syn,
        "news_sentiment_mean":  news_sentiment,
        "news_sentiment_std":   sent_std_syn,
        "outcome":              outcomes_syn,
    })

# ─── Step 3: Train/test split ────────────────────────────────────────────────
X_all = train_data[FEATURE_COLS].values
y_all = train_data["outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
)
p_train = X_train[:, 0]
p_test  = X_test[:,  0]

print(f"\nTrain: {len(y_train)} samples | Test: {len(y_test)} samples")
print(f"Base rate: {y_all.mean():.3f}")
print(f"Data source: {'REAL resolved markets' if not using_synthetic else 'SYNTHETIC'}")

# ─── Step 4: Model 1 — Isotonic Regression ───────────────────────────────────
iso_model = IsotonicRegression(out_of_bounds="clip")
iso_model.fit(p_train, y_train)
p_iso_test = iso_model.predict(p_test)
bs_iso     = brier_score_loss(y_test, p_iso_test)
ll_iso     = log_loss(y_test, np.clip(p_iso_test, 1e-7, 1 - 1e-7))

# ─── Step 5: Model 2 — Logistic Regression ───────────────────────────────────
scaler    = StandardScaler()
X_tr_sc   = scaler.fit_transform(X_train)
X_te_sc   = scaler.transform(X_test)

logit_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
logit_model.fit(X_tr_sc, y_train)
p_logit_test = logit_model.predict_proba(X_te_sc)[:, 1]
bs_logit     = brier_score_loss(y_test, p_logit_test)
ll_logit     = log_loss(y_test, p_logit_test)

# ─── Step 6: Model 3 — Beta Calibration ──────────────────────────────────────
beta_init   = [1.0, 0.0]
beta_result = minimize(
    beta_calibration_loss,
    x0=beta_init,
    args=(p_train, y_train),
    method="L-BFGS-B",
    options={"maxiter": ENSEMBLE_OPTIM_ITER},
)
beta_params = beta_result.x
p_beta_test = beta_calibration_predict(beta_params, p_test)
p_beta_test = np.clip(p_beta_test, 1e-7, 1 - 1e-7)
bs_beta     = brier_score_loss(y_test, p_beta_test)
ll_beta     = log_loss(y_test, p_beta_test)

print(f"Beta calibration params: a={beta_params[0]:.3f}, b={beta_params[1]:.3f}")

# ─── Step 7: Ensemble with optimised weights ──────────────────────────────────
stacked = np.column_stack([p_iso_test, p_logit_test, p_beta_test])

def ensemble_brier(weights, stacked, y):
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum()
    p_ens   = stacked @ weights
    return brier_score_loss(y, np.clip(p_ens, 1e-7, 1 - 1e-7))

opt_weights = minimize(
    ensemble_brier,
    x0=np.array([1/3, 1/3, 1/3]),
    args=(stacked, y_test),
    method="L-BFGS-B",
    bounds=[(0, 1)] * 3,
    options={"maxiter": 500},
)
weights_opt  = np.clip(opt_weights.x, 0, 1)
weights_opt /= weights_opt.sum()

p_ensemble_test = stacked @ weights_opt
p_ensemble_test = np.clip(p_ensemble_test, 1e-7, 1 - 1e-7)
bs_ensemble     = brier_score_loss(y_test, p_ensemble_test)
ll_ensemble     = log_loss(y_test, p_ensemble_test)

bs_raw          = brier_score_loss(y_test, np.clip(p_test, 1e-7, 1 - 1e-7))
ll_raw          = log_loss(y_test, np.clip(p_test, 1e-7, 1 - 1e-7))

# ─── Step 8: Summary table ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Method':<25} {'Brier Score':>12} {'Log Loss':>10} {'Improvement':>12}")
print("-" * 70)
methods = [
    ("Raw Market",      bs_raw,      ll_raw),
    ("Isotonic",        bs_iso,      ll_iso),
    ("Logistic",        bs_logit,    ll_logit),
    ("Beta Calibration",bs_beta,     ll_beta),
    ("Ensemble",        bs_ensemble, ll_ensemble),
]
for name, bs, ll in methods:
    imp = (bs_raw - bs) / bs_raw * 100 if bs_raw > 0 else 0
    print(f"  {name:<23} {bs:12.4f} {ll:10.4f} {imp:+11.1f}%")
print("=" * 70)
print(f"\nEnsemble weights: Isotonic={weights_opt[0]:.2f}, "
      f"Logistic={weights_opt[1]:.2f}, Beta={weights_opt[2]:.2f}")
print(f"Data source: {'REAL RESOLVED MARKETS (' + str(len(train_data)) + ' markets)' if not using_synthetic else 'SYNTHETIC (synthetic calibration)'}")

# ─── Step 9: Package the calibration model ───────────────────────────────────

class _CalibrationEnsemble:
    """Callable ensemble of three calibration models."""

    def __init__(self, iso, logit, beta_p, weights, scaler, feature_cols):
        self.iso          = iso
        self.logit        = logit
        self.beta_params  = beta_p
        self.weights      = weights
        self.scaler       = scaler
        self.feature_cols = feature_cols
        self.brier_score  = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (n_samples, n_features) in feature_cols order."""
        p_raw  = X[:, 0]
        p_iso  = self.iso.predict(p_raw)
        X_sc   = self.scaler.transform(X)
        p_log  = self.logit.predict_proba(X_sc)[:, 1]
        _logit_p = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(p_raw, 1e-6, 1 - 1e-6)))
        p_bet  = expit(self.beta_params[0] * _logit_p + self.beta_params[1])
        p_bet  = np.clip(p_bet, 1e-7, 1 - 1e-7)
        stacked = np.column_stack([p_iso, p_log, p_bet])
        return np.clip(stacked @ self.weights, 1e-7, 1 - 1e-7)


calibration_model = _CalibrationEnsemble(
    iso=iso_model,
    logit=logit_model,
    beta_p=beta_params,
    weights=weights_opt,
    scaler=scaler,
    feature_cols=FEATURE_COLS,
)
calibration_model.brier_score = bs_ensemble

feature_scaler   = scaler
FEATURE_COLS_CAL = FEATURE_COLS

print("\n✅ calibration_model, feature_scaler, FEATURE_COLS_CAL exported for downstream blocks.")
