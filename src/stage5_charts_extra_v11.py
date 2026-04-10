"""
Stage 5 - Charts
Run AFTER stage5_evaluate.py.
Reads evaluation_set.csv, results CSVs, and graph.graphml.
Produces 11 PNG charts for the thesis. Note the 5a and 5b
enumeration end with Chartt 10, but we have 11 PNGs

Charts produced:
  chart_degree_ccdf.png              - 3.3.3  Node degree CCDF
  chart_roc_curve.png                - 3.7    ROC Curve (Method A)
  chart_tpr_by_level.png             - 3.7    TPR by OSINT Level per Method
  chart_f1_by_method.png             - 3.7    F1 by Detection Method (aggregated)
  chart_rss_distribution_lowmed.png  - 3.7    RSS Distribution (Low + Medium)
  chart_rss_distribution_highall.png - 3.7    RSS Distribution (High + All combined)
  chart_confusion_matrices.png       - 3.7    Confusion Matrices
  chart_f1_by_level.png              - 3.7    F1 by OSINT Level per Method
  chart_tau_sweep.png                - 3.7    Threshold Sweep (Validation Split)
  chart_detection_outcome.png        - 3.7    Detection Outcome (1×3 horizontal)
  chart_detection_outcome_3x1.png    - 3.7    Detection Outcome (3×1 vertical)

Related Thesis Sections:
    3.3.3 (Descriptive Statistics)
    3.7 (Results)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, EVALUATION_SET

# Load data -----------------------------------------------------------------
df         = pd.read_csv(EVALUATION_SET)
df_summary = pd.read_csv(PROCESSED_DIR / "results_summary.csv")
df_thresh  = pd.read_csv(PROCESSED_DIR / "results_threshold.csv")

tau_star = float(df_thresh["tau_star"].iloc[0])

# Colour palette ------------------------------------------------------------
METHOD_COLORS = {"A_RSS": "#095256", "B_Auth": "#087f8c", "C_Hybrid": "#5aaa95"}
METHOD_LABELS = {
    "A_RSS":    "Method A: RSS",
    "B_Auth":   "Method B: Header Auth",
    "C_Hybrid": "Method C: Hybrid",
}

print(f"Loaded {len(df)} rows, tau* = {tau_star:.4f}")


# region Chart 1: Degree Distribution CCDF  (S3.3.3)
GRAPH_PATH = PROCESSED_DIR / "graph.graphml"


def ccdf(values: np.ndarray):
    """Return (k, P(X >= k)) arrays, excluding zeros."""
    v = np.asarray(values, dtype=int)
    v = v[v > 0]
    if len(v) == 0:
        return np.array([]), np.array([])
    v_sorted = np.sort(v)
    uniq, first_idx = np.unique(v_sorted, return_index=True)
    surv = 1.0 - (first_idx / len(v_sorted))
    return uniq, surv


G = nx.read_graphml(GRAPH_PATH)

rows_deg = []
for node in sorted(G.nodes()):
    in_d  = int(G.in_degree(node))
    out_d = int(G.out_degree(node))
    rows_deg.append({"node": node, "in_degree": in_d,
                     "out_degree": out_d, "total_degree": in_d + out_d})

df_deg = pd.DataFrame(rows_deg).sort_values(
    "total_degree", ascending=False
).reset_index(drop=True)
df_deg.to_csv(PROCESSED_DIR / "node_degree_summary.csv", index=False)

in_arr  = df_deg["in_degree"].to_numpy(dtype=int)
out_arr = df_deg["out_degree"].to_numpy(dtype=int)

nonzero_in  = in_arr[in_arr > 0]
nonzero_out = out_arr[out_arr > 0]

in_x, in_y   = ccdf(in_arr)
out_x, out_y = ccdf(out_arr)

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
ax.plot(in_x, in_y, marker="o", markersize=2.5, linewidth=1.0, color="#095256",
        label=(f"In-degree "
               f"(n={len(nonzero_in):,}, "
               f"median={int(np.median(nonzero_in))}, "
               f"max={int(nonzero_in.max()):,})"))
ax.plot(out_x, out_y, marker="s", markersize=2.5, linewidth=1.0, color="#087f8c",
        label=(f"Out-degree "
               f"(n={len(nonzero_out):,}, "
               f"median={int(np.median(nonzero_out))}, "
               f"max={int(nonzero_out.max()):,})"))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Degree k (log scale)")
ax.set_ylabel(r"P(X $\geq$ k) (log scale)")
ax.set_title("CCDF of In- and Out-Degree Distribution")
ax.grid(True, which="both", alpha=0.25)
ax.legend(frameon=True, fontsize=8.5)
note = (f"Directed graph | Nodes = {G.number_of_nodes():,} | "
        f"Edges = {G.number_of_edges():,}")
ax.text(0.99, 0.01, note, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                          "alpha": 0.8, "edgecolor": "gray"})
p_ccdf = PROCESSED_DIR / "chart_degree_ccdf.png"
fig.savefig(p_ccdf, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Written {p_ccdf}")

# endregion

# region Validation / test split  (this is re-used by ROC and tau sweep)
rng = np.random.RandomState(42)
val_mask  = np.zeros(len(df), dtype=bool)
strat_key = df["osint_level"].to_numpy()
for level in np.unique(strat_key):
    level_idx = np.where(strat_key == level)[0]
    rng.shuffle(level_idx)
    n_val = len(level_idx) // 2
    val_mask[level_idx[:n_val]] = True

df_val  = df[val_mask]
df_test = df[~val_mask]
# endregion

# region Chart 2: ROC Curve - Method A: Relationship Strength Scoring (RSS)
test_rss_arr = df_test["rss"].to_numpy()
test_y_arr   = df_test["y"].to_numpy()

# AUC-ROC via trapezoidal rule.  Lower score = more suspicious (sklearn does not work - inverted)
def manual_auc_roc_fixed(y_true, scores):
    order    = np.argsort(scores)
    y_sorted = y_true[order]
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr_arr = np.concatenate([[0.0], np.cumsum(y_sorted) / n_pos])
    fpr_arr = np.concatenate([[0.0], np.cumsum(1 - y_sorted) / n_neg])
    return float(np.trapz(tpr_arr, fpr_arr))


# Return (fpr, tpr) arrays.  Lower score = more suspicious
def roc_curve_fixed(y_true, scores):
    thresholds = np.sort(np.unique(scores))
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    fpr_list, tpr_list = [0.0], [0.0]
    for thr in thresholds:
        preds = (scores < thr).astype(int)
        tpr_list.append(
            int(np.sum((preds == 1) & (y_true == 1))) / n_pos if n_pos else 0.0)
        fpr_list.append(
            int(np.sum((preds == 1) & (y_true == 0))) / n_neg if n_neg else 0.0)
    fpr_list.append(1.0)
    tpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list)


auc_fixed = manual_auc_roc_fixed(test_y_arr, test_rss_arr)
fpr_pts, tpr_pts = roc_curve_fixed(test_y_arr, test_rss_arr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_pts, tpr_pts, color="#095256", linewidth=2,
        label=f"Method A (AUC = {auc_fixed:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Method A: Relationship Strength Scoring (Test Split)")
ax.legend(loc="lower right")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.grid(True, alpha=0.3)
fig.tight_layout()
p_roc = PROCESSED_DIR / "chart_roc_curve.png"
fig.savefig(p_roc, dpi=200)
plt.close(fig)
print(f"Written {p_roc}")

# endregion

# region Chart 3: TPR by OSINT Level per Method  (grouped bars)
fig, ax = plt.subplots(figsize=(8, 5))
bar_levels = ["Low", "Medium", "High"]
bar_width  = 0.22
x_pos      = np.arange(len(bar_levels))

for i, m_name in enumerate(["A_RSS", "B_Auth", "C_Hybrid"]):
    level_tpr = []
    for lvl in bar_levels:
        row = df_summary[(df_summary["method"] == m_name) &
                         (df_summary["osint_level"] == lvl)]
        level_tpr.append(float(row["tpr"].iloc[0]) if len(row) > 0 else 0.0)
    drawn = ax.bar(x_pos + i * bar_width, level_tpr, bar_width,
                   label=METHOD_LABELS[m_name], color=METHOD_COLORS[m_name])
    for rect, val in zip(drawn, level_tpr):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("OSINT Level")
ax.set_ylabel("True Positive Rate")
ax.set_title("TPR by OSINT Level and Detection Method (Test Split)")
ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels(bar_levels)
ax.set_ylim(0, 1.15)
ax.axhline(y=0.80, color="#5aaa95", linestyle=":", linewidth=1.2, alpha=0.8,
           label="Pass threshold (0.80)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
p_tpr = PROCESSED_DIR / "chart_tpr_by_level.png"
fig.savefig(p_tpr, dpi=200)
plt.close(fig)
print(f"Written {p_tpr}")

# endregion

# region Chart 4: F1 by Detection Method  (aggregated bar)
fig, ax = plt.subplots(figsize=(7, 5))
methods_ordered = ["A_RSS", "B_Auth", "C_Hybrid"]
method_bar_labels = [
    "Method A:\nRSS",
    "Method B:\nHeader Auth",
    "Method C:\nHybrid",
]
agg_f1 = []
for m_name in methods_ordered:
    row = df_summary[(df_summary["method"] == m_name) &
                     (df_summary["osint_level"] == "All")]
    agg_f1.append(float(row["f1"].iloc[0]) if len(row) > 0 else 0.0)

bars = ax.bar(method_bar_labels, agg_f1, width=0.45,
              color=[METHOD_COLORS[m] for m in methods_ordered])
for rect, val in zip(bars, agg_f1):
    ax.text(rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.02,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10)

ax.set_xlabel("Detection Method")
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score by Detection Method (Test Split)")
ax.set_ylim(0, 1.15)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
p_f1m = PROCESSED_DIR / "chart_f1_by_method.png"
fig.savefig(p_f1m, dpi=200)
plt.close(fig)
print(f"Written {p_f1m}")

# endregion

# region Chart 5a: RSS Distribution - Low + Medium (1x2)
RSS_DIST_COLORS = {
    "Low":    {"legit": "#5aaa95", "spear": "#ff8552", "tau": "#095256"},
    "Medium": {"legit": "#5aaa95", "spear": "#e9d758", "tau": "#095256"},
    "High":   {"legit": "#5aaa95", "spear": "#087f8c", "tau": "#095256"},
    "All":    {"legit": "#5aaa95", "spear": "#087f8c", "tau": "#095256"},
}

DIST_LABEL_SIZE  = 13
DIST_TICK_SIZE   = 13
DIST_TITLE_SIZE  = 15
DIST_LEGEND_SIZE = 13

legit_rss = df.loc[df["osint_level"] == "Legit", "rss"].to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for i, lvl in enumerate(["Low", "Medium"]):
    ax = axes[i]
    pal = RSS_DIST_COLORS[lvl]
    spear_rss = df.loc[df["osint_level"] == lvl, "rss"].to_numpy()

    ax.hist(legit_rss, bins=80, alpha=0.6, color=pal["legit"],
            label="Legitimate", density=True, zorder=1)
    ax.axvline(x=tau_star, color=pal["tau"], linestyle="--", linewidth=1.8,
               label=f"\u03c4* = {tau_star:.4f}", zorder=2)
    ax.hist(spear_rss, bins=max(5, len(np.unique(spear_rss))), alpha=0.85,
            color=pal["spear"], label=f"Spear ({lvl})", density=True, zorder=3)

    ax.set_xlabel("RSS Score", fontsize=DIST_LABEL_SIZE)
    ax.set_ylabel("Probability Density" if i == 0 else "", fontsize=DIST_LABEL_SIZE)
    ax.set_title(f"{lvl} OSINT", fontsize=DIST_TITLE_SIZE)
    ax.legend(fontsize=DIST_LEGEND_SIZE)
    ax.tick_params(labelsize=DIST_TICK_SIZE)
    ax.grid(True, alpha=0.2)

fig.suptitle("RSS Score Distribution: Legitimate vs Low and Medium\nLevel Spear-Phishing (Full Evaluation Set)",
             fontsize=17, y=1.01)
fig.tight_layout()
p_rss_a = PROCESSED_DIR / "chart_rss_distribution_low_med.png"
fig.savefig(p_rss_a, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Written {p_rss_a}")
# endregion

# region Chart 5b: RSS Distribution - High + All (1x2)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for i, (lvl, label, spear_data) in enumerate([
    ("High", "High", df.loc[df["osint_level"] == "High", "rss"].to_numpy()),
    ("All",  "All Levels Combined", df.loc[df["y"] == 1, "rss"].to_numpy()),
]):
    ax = axes[i]
    pal = RSS_DIST_COLORS[lvl]

    ax.hist(legit_rss, bins=80, alpha=0.6, color=pal["legit"],
            label="Legitimate", density=True, zorder=1)
    ax.axvline(x=tau_star, color=pal["tau"], linestyle="--", linewidth=1.8,
               label=f"\u03c4* = {tau_star:.4f}", zorder=2)
    spear_label = f"Spear ({lvl})" if lvl != "All" else "All Spear"
    n_bins = max(5, len(np.unique(spear_data))) if lvl != "All" else max(10, len(np.unique(spear_data)))
    ax.hist(spear_data, bins=n_bins, alpha=0.85,
            color=pal["spear"], label=spear_label, density=True, zorder=3)

    ax.set_xlabel("RSS Score", fontsize=DIST_LABEL_SIZE)
    ax.set_ylabel("Probability Density" if i == 0 else "", fontsize=DIST_LABEL_SIZE)
    ax.set_title(label if lvl == "All" else f"{lvl} OSINT", fontsize=DIST_TITLE_SIZE)
    ax.legend(fontsize=DIST_LEGEND_SIZE)
    ax.tick_params(labelsize=DIST_TICK_SIZE)
    ax.grid(True, alpha=0.2)

fig.suptitle("RSS Score Distribution: Legitimate vs High and All\nLevel Spear-Phishing (Full Evaluation Set)",
             fontsize=17, y=1.01)
fig.tight_layout()
p_rss_b = PROCESSED_DIR / "chart_rss_distribution_high_all.png"
fig.savefig(p_rss_b, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Written {p_rss_b}")

# endregion

# region Chart 6: Confusion Matrix Heatmaps - one per method
#       Layout: [[TP, FN], [FP, TN]]  (clockwise from top-left)
#       Colours: TP #095256, TN #087f8c, FP+FN #5aaa95
rss_all  = df_test["rss"].to_numpy()
y_all    = df_test["y"].to_numpy()
auth_all = df_test["header_auth_pass"].to_numpy().astype(bool)

pred_methods = {
    "A - RSS":         (rss_all < tau_star).astype(int),
    "B - Header Auth": (~auth_all).astype(int),
    "C - Hybrid":      ((rss_all < tau_star) | (~auth_all)).astype(int),
}

CLR_TP = "#095256"
CLR_TN = "#087f8c"
CLR_OTHER = "#5aaa95"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for idx, (m_label, m_pred) in enumerate(pred_methods.items()):
    ax   = axes[idx]
    tp_v = int(np.sum((m_pred == 1) & (y_all == 1)))
    fp_v = int(np.sum((m_pred == 1) & (y_all == 0)))
    fn_v = int(np.sum((m_pred == 0) & (y_all == 1)))
    tn_v = int(np.sum((m_pred == 0) & (y_all == 0)))

    # Layout: row 0 = Actual Pos, row 1 = Actual Neg
    #         col 0 = Pred Pos,   col 1 = Pred Neg
    # Clockwise from top-left: TP, FN, TN, FP
    cm = np.array([[tp_v, fn_v],
                   [fp_v, tn_v]])

    cell_colors = np.array([[CLR_TP,    CLR_OTHER],
                            [CLR_OTHER, CLR_TN   ]])

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    for r in range(2):
        for c in range(2):
            rect_patch = Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=cell_colors[r, c], edgecolor="white",
                                       linewidth=2)
            ax.add_patch(rect_patch)
            val = cm[r, c]
            ax.text(c, r, f"{val:,}", ha="center", va="center",
                    fontsize=15, fontweight="bold", color="white")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Pos", "Pred Neg"])
    ax.set_yticklabels(["Actual Pos", "Actual Neg"])
    ax.set_title(m_label, fontsize=14, fontweight="bold")

fig.suptitle("Confusion Matrices (Test Split)", fontsize=16, y=1.02)
fig.tight_layout()
p_cm = PROCESSED_DIR / "chart_confusion_matrices.png"
fig.savefig(p_cm, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Written {p_cm}")

# endregion

# region Chart 7: F1 by OSINT Level per Method (grouped bars)
fig, ax = plt.subplots(figsize=(8, 5))
bar_levels = ["Low", "Medium", "High"]
bar_width  = 0.22
x_pos      = np.arange(len(bar_levels))

for i, m_name in enumerate(["A_RSS", "B_Auth", "C_Hybrid"]):
    level_f1 = []
    for lvl in bar_levels:
        row = df_summary[(df_summary["method"] == m_name) &
                         (df_summary["osint_level"] == lvl)]
        level_f1.append(float(row["f1"].iloc[0]) if len(row) > 0 else 0.0)
    drawn = ax.bar(x_pos + i * bar_width, level_f1, bar_width,
                   label=METHOD_LABELS[m_name], color=METHOD_COLORS[m_name])
    for rect, val in zip(drawn, level_f1):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

ax.set_xlabel("OSINT Level", fontsize=11)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("F1 Score by OSINT Level and Detection Method (Test Split)", fontsize=12)
ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels(bar_levels, fontsize=10)
ax.tick_params(labelsize=10)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
p_f1l = PROCESSED_DIR / "chart_f1_by_level.png"
fig.savefig(p_f1l, dpi=200)
plt.close(fig)
print(f"Written {p_f1l}")

# endregion

# region Chart 8: Tau Sweep - dual-panel (log + linear zoom) on validation set
#        Palette: Pass 80:#ffbe0b  F1:#fb5607  FPR:#ff006e  tau*:#8338ec  TPR:#114b5f
val_rss = df_val["rss"].to_numpy()
val_y   = df_val["y"].to_numpy()

candidates = np.sort(np.unique(val_rss))
if len(candidates) > 2000:
    step = len(candidates) // 2000
    candidates_plot = candidates[::step]
else:
    candidates_plot = candidates

# Force-include tau* so the exact F1 peak is never skipped by downsampling
candidates_plot = np.sort(np.unique(np.append(candidates_plot, tau_star)))

sweep_taus, sweep_f1s, sweep_tprs, sweep_fprs = [], [], [], []

for cand in candidates_plot:
    cand_pred = (val_rss < cand).astype(int)
    c_tp = int(np.sum((cand_pred == 1) & (val_y == 1)))
    c_fp = int(np.sum((cand_pred == 1) & (val_y == 0)))
    c_fn = int(np.sum((cand_pred == 0) & (val_y == 1)))
    c_tn = int(np.sum((cand_pred == 0) & (val_y == 0)))
    c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
    c_rec  = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
    c_f1   = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
    c_fpr  = c_fp / (c_fp + c_tn) if (c_fp + c_tn) > 0 else 0.0
    sweep_taus.append(cand)
    sweep_f1s.append(c_f1)
    sweep_tprs.append(c_rec)
    sweep_fprs.append(c_fpr)

sweep_taus = np.array(sweep_taus)
sweep_f1s  = np.array(sweep_f1s)
sweep_tprs = np.array(sweep_tprs)
sweep_fprs = np.array(sweep_fprs)

# Colours
_C_PASS = "#ffbe0b"
_C_F1   = "#fb5607"
_C_FPR  = "#ff006e"
_C_TAU  = "#8338ec"
_C_TPR  = "#114b5f"

fig, (ax_log, ax_zoom) = plt.subplots(
    1, 2, figsize=(14, 5.5),
    gridspec_kw={"width_ratios": [1.6, 1]},
)

# shared helper
_mk = dict(markersize=3)

def _draw_panel(panel_ax, taus, f1s, tprs, fprs, show_legend=False):
    panel_ax.plot(taus, tprs, color=_C_TPR, linewidth=2, marker="o",
            label="TPR", **_mk)
    panel_ax.plot(taus, f1s, color=_C_F1, linewidth=2, marker="D",
            label="F1", **_mk)
    panel_ax.plot(taus, fprs, color=_C_FPR, linewidth=2, linestyle=":",
            marker="s", label="FPR", **_mk)
    panel_ax.axhline(y=0.80, color=_C_PASS, linestyle="-.", linewidth=1.8,
               alpha=0.8, label="TPR ≥ 0.80")
    panel_ax.axhline(y=0.05, color=_C_PASS, linestyle="--", linewidth=1.4,
               alpha=0.7, label="FPR < 0.05")
    panel_ax.axvline(x=tau_star, color=_C_TAU, linestyle="--", linewidth=2,
               alpha=0.85, label=f"\u03c4* = {tau_star:.4f}")
    panel_ax.set_ylim(-0.03, 1.08)
    panel_ax.grid(True, alpha=0.2)
    if show_legend:
        panel_ax.legend(fontsize=11, loc="center right",
                  frameon=True, fancybox=True, framealpha=0.9)

# Panel A: log x-axis (full range)
_draw_panel(ax_log, sweep_taus, sweep_f1s, sweep_tprs, sweep_fprs,
            show_legend=True)
ax_log.set_xscale("log")
ax_log.set_xlabel("Threshold \u03c4 (log scale)", fontsize=13)
ax_log.set_ylabel("Score", fontsize=13)
ax_log.set_title("A   Full range (log scale)", fontsize=14,
                 fontweight="bold", loc="left")

# Panel B: linear zoom tau:\u03c4 include: \u2208 [0, 1.2]
zoom_mask = sweep_taus <= 1.2
_draw_panel(ax_zoom,
            sweep_taus[zoom_mask], sweep_f1s[zoom_mask],
            sweep_tprs[zoom_mask], sweep_fprs[zoom_mask],
            show_legend=False)
ax_zoom.set_xlabel("Threshold \u03c4", fontsize=13)
ax_zoom.set_ylabel("")
ax_zoom.set_title("B   Zoom: \u03c4 \u2208 [0, 1.2]", fontsize=14,
                  fontweight="bold", loc="left")

# F1 peak annotation on zoom panel
best_idx = int(np.argmax(sweep_f1s))
best_f1  = sweep_f1s[best_idx]
best_tau = sweep_taus[best_idx]
if best_tau <= 1.2:
    ax_zoom.annotate(
        f"F1 = {best_f1:.4f}",
        xy=(best_tau, best_f1),
        xytext=(best_tau + 0.30, best_f1 + 0.15),
        fontsize=12, fontweight="bold", color=_C_F1,
        arrowprops=dict(arrowstyle="->", color=_C_F1, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_C_F1, lw=1),
    )

fig.suptitle(
    "Threshold Sweep: F1, TPR and FPR vs \u03c4 (Validation Split)",
    fontsize=16, y=1.01,
)
fig.tight_layout()
p_tau = PROCESSED_DIR / "chart_tau_sweep.png"
fig.savefig(p_tau, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Written {p_tau}")
# endregion

# region Chart 9: Detection Outcome by OSINT Level  (stacked bar)  [font +4 total]
DET_FONT     = 14
DET_LEGEND   = 12
DET_TITLE    = 14
DET_LABEL    = 12
DET_TICK     = 11

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
method_panel_titles = [
    "A - Relationship Strength Scoring (RSS)",
    "B - Header Authentication",
    "C - Hybrid (RSS + Header Auth)",
]
method_keys  = ["A_RSS", "B_Auth", "C_Hybrid"]
osint_levels = ["Low", "Medium", "High"]

for idx, (m_key, m_title) in enumerate(zip(method_keys, method_panel_titles)):
    ax = axes[idx]
    detected, missed = [], []
    for lvl in osint_levels:
        row = df_summary[(df_summary["method"] == m_key) &
                         (df_summary["osint_level"] == lvl)]
        if len(row) > 0:
            detected.append(int(row["tp"].iloc[0]))
            missed.append(int(row["fn"].iloc[0]))
        else:
            detected.append(0)
            missed.append(0)

    x_pos    = np.arange(len(osint_levels))
    bars_det = ax.bar(x_pos, detected, 0.5, label="Detected (TP)",
                      color="#087f8c", alpha=0.85)
    bars_mis = ax.bar(x_pos, missed, 0.5, bottom=detected,
                      label="Missed (FN)", color="#5aaa95", alpha=0.85)

    for bar_d, bar_m, d_val, m_val in zip(bars_det, bars_mis, detected, missed):
        if d_val > 0:
            ax.text(bar_d.get_x() + bar_d.get_width() / 2, d_val / 2,
                    str(d_val), ha="center", va="center", fontsize=DET_FONT,
                    fontweight="bold", color="white")
        if m_val > 0:
            ax.text(bar_m.get_x() + bar_m.get_width() / 2, d_val + m_val / 2,
                    str(m_val), ha="center", va="center", fontsize=DET_FONT,
                    fontweight="bold", color="white")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(osint_levels, fontsize=DET_TICK)
    ax.tick_params(labelsize=DET_TICK)
    ax.set_xlabel("OSINT Level", fontsize=DET_LABEL)
    if idx == 0:
        ax.set_ylabel("Spear-Phishing Emails", fontsize=DET_LABEL)
    ax.set_title(m_title, fontweight="bold", fontsize=DET_TITLE)
    ax.legend(fontsize=DET_LEGEND)
    ax.grid(True, axis="y", alpha=0.2)

fig.suptitle("Detection Outcome: Detected vs Missed\nby Method and OSINT Level (Test Split)",
             fontsize=15, y=1.02)
fig.tight_layout()
p_det = PROCESSED_DIR / "chart_detection_outcome.png"
fig.savefig(p_det, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Written {p_det}")
# endregion

# region Chart 10: Detection Outcome (3×1 vertical)
DET_FONT     = 14
DET_LEGEND   = 12
DET_TITLE    = 14
DET_LABEL    = 12
DET_TICK     = 11

fig, axes = plt.subplots(3, 1, figsize=(7, 14), sharey=True)
method_panel_titles = [
    "A - Relationship Strength Scoring (RSS)",
    "B - Header Authentication",
    "C - Hybrid (RSS + Header Auth)",
]
method_keys  = ["A_RSS", "B_Auth", "C_Hybrid"]
osint_levels = ["Low", "Medium", "High"]

for idx, (m_key, m_title) in enumerate(zip(method_keys, method_panel_titles)):
    ax = axes[idx]
    detected, missed = [], []
    for lvl in osint_levels:
        row = df_summary[(df_summary["method"] == m_key) &
                         (df_summary["osint_level"] == lvl)]
        if len(row) > 0:
            detected.append(int(row["tp"].iloc[0]))
            missed.append(int(row["fn"].iloc[0]))
        else:
            detected.append(0)
            missed.append(0)

    x_pos    = np.arange(len(osint_levels))
    bars_det = ax.bar(x_pos, detected, 0.5, label="Detected (TP)",
                      color="#087f8c", alpha=0.85)
    bars_mis = ax.bar(x_pos, missed, 0.5, bottom=detected,
                      label="Missed (FN)", color="#5aaa95", alpha=0.85)

    for bar_d, bar_m, d_val, m_val in zip(bars_det, bars_mis, detected, missed):
        if d_val > 0:
            ax.text(bar_d.get_x() + bar_d.get_width() / 2, d_val / 2,
                    str(d_val), ha="center", va="center", fontsize=DET_FONT,
                    fontweight="bold", color="white")
        if m_val > 0:
            ax.text(bar_m.get_x() + bar_m.get_width() / 2, d_val + m_val / 2,
                    str(m_val), ha="center", va="center", fontsize=DET_FONT,
                    fontweight="bold", color="white")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(osint_levels, fontsize=DET_TICK)
    ax.tick_params(labelsize=DET_TICK)
    ax.set_xlabel("OSINT Level", fontsize=DET_LABEL)
    if idx == 0:
        ax.set_ylabel("Spear-Phishing Emails", fontsize=DET_LABEL)
    ax.set_title(m_title, fontweight="bold", fontsize=DET_TITLE)
    ax.legend(fontsize=DET_LEGEND)
    ax.grid(True, axis="y", alpha=0.2)

fig.suptitle("Detection Outcome: Detected vs Missed\nby Method and OSINT Level (Test Split)",
             fontsize=15, y=1.02)
fig.tight_layout()
p_det_3x1 = PROCESSED_DIR / "chart_detection_outcome_3x1.png"
fig.savefig(p_det_3x1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Written {p_det_3x1}")
# endregion


# Print Summary
print(f"\nCorrected AUC-ROC: {auc_fixed:.6f}")
print("Done - 11 charts generated.")
