"""
Block 6: Visualizations — Calibration Plots
Six publication-quality interactive charts using Plotly.
Original 4 + 2 new charts: category heatmap and volume vs probability scatter.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "perfect":    "#5A5A5A",
    "raw":        "#9E4A4A",
    "isotonic":   "#3D5A80",
    "logistic":   "#6B8F71",
    "beta":       "#C9A227",
    "ensemble":   "#8C6A9E",
    "polymarket": "#3D5A80",
    "kalshi":     "#B07D62",
    "metaculus":  "#6B8F71",
}
# Category colors — must be valid hex (no alpha suffix for Plotly bar)
CATEGORY_COLORS = {
    "politics":    "#3D5A80",
    "crypto":      "#C9A227",
    "economics":   "#6B8F71",
    "science":     "#8C6A9E",
    "sports":      "#B07D62",
    "geopolitics": "#9E4A4A",
    "other":       "#5A5A5A",
}
# Lighter versions as separate valid rgba strings
CATEGORY_COLORS_LIGHT = {k: f"rgba({int(v[1:3],16)},{int(v[3:5],16)},{int(v[5:7],16)},0.45)"
                          for k, v in CATEGORY_COLORS.items()}
TEMPLATE = "plotly_white"

# ─── Chart 1: Reliability Diagram ────────────────────────────────────────────
print("Building Chart 1: Reliability Diagram …")

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    name="Perfect calibration",
    line=dict(color=COLORS["perfect"], width=1.5, dash="dot"),
))

curve_styles = {
    "Raw Market":       (COLORS["raw"],      "dash",       4),
    "Isotonic":         (COLORS["isotonic"], "solid",      2),
    "Beta Calibration": (COLORS["beta"],     "longdash",   2),
    "Ensemble":         (COLORS["ensemble"], "solid",      3),
}

for curve_name, (color, dash, width) in curve_styles.items():
    if curve_name not in calib_curves:
        continue
    mp  = calib_curves[curve_name]["mean_pred"]
    fpo = calib_curves[curve_name]["frac_pos"]
    if len(mp) == 0:
        continue
    fig1.add_trace(go.Scatter(
        x=mp, y=fpo,
        mode="lines+markers",
        name=curve_name,
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=6),
    ))

fig1.update_layout(
    template=TEMPLATE,
    title=dict(text="<b>Reliability Diagram</b> — Calibration Curves", font=dict(size=18)),
    xaxis=dict(title="Mean Predicted Probability", range=[0, 1], tickformat=".0%"),
    yaxis=dict(title="Fraction of Positives (Outcome Rate)", range=[0, 1], tickformat=".0%"),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1),
    width=700, height=550,
)
fig_reliability = fig1

# ─── Chart 2: Brier Score Decomposition Bar Chart ────────────────────────────
print("Building Chart 2: Brier Score Decomposition …")

methods_order = ["Raw Market", "Isotonic", "Logistic", "Beta Calibration", "Ensemble"]
bar_colors    = [COLORS["raw"], COLORS["isotonic"], COLORS["logistic"], COLORS["beta"], COLORS["ensemble"]]

fig2 = go.Figure()

for comp, opacity in [("Reliability", 0.9), ("Resolution", 0.65), ("Uncertainty", 0.45)]:
    vals  = [eval_results.loc[m, comp] for m in methods_order if m in eval_results.index]
    names = [m for m in methods_order if m in eval_results.index]
    fig2.add_trace(go.Bar(
        name=comp, x=names, y=vals,
        marker_color=[c for c, m in zip(bar_colors, methods_order) if m in eval_results.index],
        marker_opacity=opacity,
        text=[f"{v:.4f}" for v in vals],
        textposition="inside",
    ))

brier_vals = [eval_results.loc[m, "Brier"] for m in methods_order if m in eval_results.index]
names_used = [m for m in methods_order if m in eval_results.index]
fig2.add_trace(go.Scatter(
    x=names_used, y=brier_vals,
    mode="markers+text",
    name="Total Brier",
    marker=dict(color="#222222", size=12, symbol="diamond"),
    text=[f"<b>{v:.4f}</b>" for v in brier_vals],
    textposition="top center",
))

fig2.update_layout(
    template=TEMPLATE, barmode="stack",
    title=dict(text="<b>Brier Score Decomposition</b> by Calibration Method", font=dict(size=18)),
    xaxis=dict(title=""),
    yaxis=dict(title="Score Component"),
    legend=dict(x=1.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    width=800, height=520,
)
fig_brier = fig2

# ─── Chart 3: Probability Distribution by Platform ───────────────────────────
print("Building Chart 3: Probability Distribution by Platform …")

platform_map = {
    "polymarket": (COLORS["polymarket"], "Polymarket"),
    "kalshi":     (COLORS["kalshi"],     "Kalshi"),
    "metaculus":  (COLORS["metaculus"],  "Metaculus"),
}

fig3 = go.Figure()
for src, (color, label) in platform_map.items():
    sub = features_df[features_df["source"] == src]["raw_probability"]
    if sub.empty:
        continue
    fig3.add_trace(go.Histogram(
        x=sub, name=label, opacity=0.72, nbinsx=40,
        marker_color=color, histnorm="probability density",
    ))

fig3.add_vline(x=0.5, line_dash="dot", line_color=COLORS["perfect"],
               annotation_text="50%", annotation_position="top right")
fig3.update_layout(
    template=TEMPLATE, barmode="overlay",
    title=dict(text="<b>Probability Distribution Across Platforms</b>", font=dict(size=18)),
    xaxis=dict(title="Implied Probability", tickformat=".0%"),
    yaxis=dict(title="Density"),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    width=750, height=500,
)
fig_distributions = fig3

# ─── Chart 4: Live Markets Dashboard ─────────────────────────────────────────
print("Building Chart 4: Live Markets Dashboard …")

if "calibrated_probability" not in features_df.columns:
    feat_matrix = features_df[FEATURE_COLS_CAL].copy().fillna(0)
    p_calibrated_all = calibration_model.predict(feat_matrix.values)
    features_df["calibrated_probability"] = p_calibrated_all
    features_df["calibration_shift"]      = (features_df["calibrated_probability"]
                                              - features_df["raw_probability"]).abs()

top10_idx = features_df["calibration_shift"].nlargest(10).index

fig4 = go.Figure()
for src, (color, label) in platform_map.items():
    sub = features_df[features_df["source"] == src]
    if sub.empty:
        continue
    vol_norm = np.log1p(sub["volume"].fillna(0))
    size_vals = (vol_norm / vol_norm.max() * 20 + 4).clip(4, 24).tolist() if vol_norm.max() > 0 else [8] * len(sub)
    fig4.add_trace(go.Scatter(
        x=sub["raw_probability"], y=sub["calibrated_probability"],
        mode="markers", name=label,
        marker=dict(color=color, size=size_vals, opacity=0.65, line=dict(width=0.5, color="white")),
        text=sub["title"].str[:60] + "…",
        hovertemplate="<b>%{text}</b><br>Raw: %{x:.1%}<br>Calibrated: %{y:.1%}<br><extra>" + label + "</extra>",
    ))

for idx in top10_idx[:5]:
    row_ann = features_df.loc[idx]
    short_title = str(row_ann["title"])[:30] + "…"
    fig4.add_annotation(
        x=row_ann["raw_probability"], y=row_ann["calibrated_probability"],
        text=short_title, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
        arrowcolor="#888", font=dict(size=9, color="#333"),
        bgcolor="rgba(255,255,255,0.75)", bordercolor="#ccc", borderwidth=1,
        ax=40, ay=-30,
    )

fig4.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines", name="No adjustment",
    line=dict(color=COLORS["perfect"], width=1.2, dash="dot"), showlegend=True,
))
fig4.update_layout(
    template=TEMPLATE,
    title=dict(text="<b>Live Markets Dashboard</b> — Raw vs Calibrated Probability", font=dict(size=18)),
    xaxis=dict(title="Raw Market Probability", range=[0, 1], tickformat=".0%"),
    yaxis=dict(title="Calibrated Probability", range=[0, 1], tickformat=".0%"),
    legend=dict(x=1.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    width=850, height=600, hovermode="closest",
)
fig_dashboard = fig4

# ─── Chart 5: Category Calibration Heatmap / Bar Chart ───────────────────────
print("Building Chart 5: Category Calibration Heatmap …")

if "category_eval_df" not in dir() or category_eval_df is None:
    _cat_tmp = []
    for cat, grp in features_df.groupby("category_std"):
        _cat_tmp.append({
            "category": cat,
            "market_count": len(grp),
            "mean_raw_probability": float(grp["raw_probability"].mean()),
            "mean_calibrated_prob": float(grp["calibrated_probability"].mean()),
        })
    category_eval_df = pd.DataFrame(_cat_tmp)

cat_df = category_eval_df.sort_values("market_count", ascending=False)
cat_labels = cat_df["category"].tolist()
cat_counts = cat_df["market_count"].tolist()
cat_raw    = [float(x) for x in cat_df["mean_raw_probability"].tolist()]
cat_cal    = [float(x) for x in cat_df["mean_calibrated_prob"].tolist()]
cat_bar_colors       = [CATEGORY_COLORS.get(c, "#5A5A5A")        for c in cat_labels]
cat_bar_colors_light = [CATEGORY_COLORS_LIGHT.get(c, "rgba(90,90,90,0.45)") for c in cat_labels]

fig5 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Market Count by Category", "Mean Calibrated Probability by Category"),
    horizontal_spacing=0.12,
)

fig5.add_trace(go.Bar(
    x=cat_labels, y=cat_counts,
    marker_color=cat_bar_colors,
    text=[str(v) for v in cat_counts],
    textposition="outside",
    name="Market Count",
    showlegend=False,
), row=1, col=1)

fig5.add_trace(go.Bar(
    x=cat_labels, y=cat_raw,
    name="Mean Raw Prob",
    marker_color=cat_bar_colors_light,
    text=[f"{v:.1%}" for v in cat_raw],
    textposition="outside",
), row=1, col=2)

fig5.add_trace(go.Bar(
    x=cat_labels, y=cat_cal,
    name="Mean Calibrated Prob",
    marker_color=cat_bar_colors,
    text=[f"{v:.1%}" for v in cat_cal],
    textposition="outside",
), row=1, col=2)

fig5.add_hline(y=0.5, line_dash="dot", line_color="#5A5A5A",
               annotation_text="50%", annotation_position="right",
               row=1, col=2)

max_prob_y = max(cat_cal + cat_raw) * 1.25
fig5.update_layout(
    template=TEMPLATE, barmode="group",
    title=dict(text="<b>Market Category Distribution & Mean Calibrated Probability</b>", font=dict(size=17)),
    yaxis=dict(title="Market Count"),
    yaxis2=dict(title="Probability", tickformat=".0%", range=[0, max_prob_y]),
    legend=dict(x=1.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    width=1050, height=520,
)
fig_category = fig5

# ─── Chart 6: Volume vs Probability Sharpness Scatter ────────────────────────
print("Building Chart 6: Volume vs Probability Sharpness …")

fig6 = go.Figure()

for src, (color, label) in platform_map.items():
    sub = features_df[features_df["source"] == src]
    if sub.empty:
        continue
    fig6.add_trace(go.Scatter(
        x=sub["log_volume"],
        y=sub["calibrated_probability"],
        mode="markers",
        name=label,
        marker=dict(color=color, size=4, opacity=0.55, line=dict(width=0)),
        text=sub["title"].str[:50] + "…",
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Log Volume: %{x:.2f}<br>"
            "Calibrated Prob: %{y:.1%}<br>"
            "<extra>" + label + "</extra>"
        ),
    ))

fig6.add_hline(y=0.5, line_dash="dash", line_color="#5A5A5A", line_width=1.5,
               annotation_text="50% (no signal)", annotation_position="right",
               annotation_font_color="#5A5A5A")

fig6.update_layout(
    template=TEMPLATE,
    title=dict(text="<b>Market Probability vs Volume (all active markets)</b>", font=dict(size=17)),
    xaxis=dict(title="Log Volume (log₁₊ dollars traded)"),
    yaxis=dict(title="Calibrated Probability", tickformat=".0%", range=[0, 1]),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1),
    width=850, height=560,
    hovermode="closest",
)
fig_volume = fig6

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n✓ 6 charts created total:")
print("  fig_reliability   — Reliability Diagram")
print("  fig_brier         — Brier Score Decomposition")
print("  fig_distributions — Probability Distribution by Platform")
print(f"  fig_dashboard     — Live Markets Dashboard ({len(features_df)} markets)")
print("  fig_category      — Market Category Distribution & Mean Calibrated Probability")
print("  fig_volume        — Market Probability vs Volume (all active markets)")
