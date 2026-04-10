"""
Stage 5 - Evaluate
Input  : data/02_Processed/evaluation_set.csv
Output : data/02_Processed/results_threshold.csv
         data/02_Processed/results_summary.csv
         data/02_Processed/results_stats.csv
         
Methods:
  A  RSS alone    - flag if rss < tau
  B  Header auth  - flag if header_auth_pass == False
  C  Hybrid       - flag if rss < tau OR header_auth_pass == False

Related Thesis Sections:
    3.5.3 (Stage 5 Evaluate: Baseline Comparison)
    3.7 (Results)
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, EVALUATION_SET

# Logging set-up
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
_log_path = PROCESSED_DIR / "stage5_evaluate.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(_log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SEED = 42

# region HELPER FUNCTIONS
# -----------------------------------------------------------------------------
# Return (tp, fp, fn, tn) as plain integers
def confusion_counts(y_true, y_pred):
    tp_c = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp_c = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn_c = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn_c = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp_c, fp_c, fn_c, tn_c

# Return dict with tpr, fpr, precision, f1, accuracy
def derived_metrics(tp_c, fp_c, fn_c, tn_c):
    tpr_v = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
    fpr_v = fp_c / (fp_c + tn_c) if (fp_c + tn_c) > 0 else 0.0
    prec_v = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
    f1_v = 2 * prec_v * tpr_v / (prec_v + tpr_v) if (prec_v + tpr_v) > 0 else 0.0
    acc_v = (tp_c + tn_c) / (tp_c + fp_c + fn_c + tn_c) if (tp_c + fp_c + fn_c + tn_c) > 0 else 0.0
    return {
        "tpr": round(tpr_v, 6), "fpr": round(fpr_v, 6),
        "precision": round(prec_v, 6), "f1": round(f1_v, 6),
        "accuracy": round(acc_v, 6),
    }


# AUC-ROC via trapezoidal rule <- Lower score = more suspicious
def manual_auc_roc(y_true, scores):
    neg_scores = -scores
    order = np.argsort(neg_scores)[::-1]
    y_sorted = y_true[order]
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr_arr = np.concatenate([[0.0], np.cumsum(y_sorted) / n_pos])
    fpr_arr = np.concatenate([[0.0], np.cumsum(1 - y_sorted) / n_neg])
    return float(np.trapz(tpr_arr, fpr_arr))


# Return (fpr_array, tpr_array) for plotting. Lower score = more suspicious
def roc_curve_points(y_true, scores):
    neg_scores = -scores
    thresholds = np.sort(np.unique(neg_scores))[::-1]
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    fpr_list, tpr_list = [0.0], [0.0]
    for thr in thresholds:
        preds = (neg_scores >= thr).astype(int)
        tpr_list.append(int(np.sum((preds == 1) & (y_true == 1))) / n_pos if n_pos else 0.0)
        fpr_list.append(int(np.sum((preds == 1) & (y_true == 0))) / n_neg if n_neg else 0.0)
    return np.array(fpr_list), np.array(tpr_list)


# McNemar test. Falls back to exact binomial when discordant pair count ios less than < 25
def mcnemar_test(pred_x, pred_y, y_true):
    cx = (pred_x == y_true).astype(int)
    cy = (pred_y == y_true).astype(int)
    n01 = int(np.sum((cx == 1) & (cy == 0)))
    n10 = int(np.sum((cx == 0) & (cy == 1)))
    n_disc = n01 + n10
    if n_disc == 0:
        return 0.0, 1.0, "none", n_disc
    if n_disc < 25:
        p_val = float(stats.binomtest(n01, n_disc, 0.5).pvalue)
        return 0.0, p_val, "exact_binomial", n_disc
    chi2_v = float((n01 - n10) ** 2 / n_disc)
    p_val = float(stats.chi2.sf(chi2_v, df=1))
    return chi2_v, p_val, "chi2", n_disc

# endregion

# region LOAD & SPLIT
# -----------------------------------------------------------------------------
log.info("Reading evaluation set: %s", EVALUATION_SET)
df = pd.read_csv(EVALUATION_SET)
log.info("Loaded %d rows", len(df))

rng = np.random.RandomState(SEED)
val_mask = np.zeros(len(df), dtype=bool)
strat_key = df["osint_level"].values

for level in np.unique(strat_key):
    level_idx = np.where(strat_key == level)[0]
    rng.shuffle(level_idx)
    n_val = len(level_idx) // 2
    val_mask[level_idx[:n_val]] = True

df_val = df[val_mask].copy().reset_index(drop=True)
df_test = df[~val_mask].copy().reset_index(drop=True)

log.info("Validation rows : %d  (spear=%d)", len(df_val), df_val["y"].sum())
log.info("Test rows       : %d  (spear=%d)", len(df_test), df_test["y"].sum())
for lvl in ["Legit", "Low", "Medium", "High"]:
    log.info("  %-8s  val=%d  test=%d",
             lvl, (df_val["osint_level"] == lvl).sum(), (df_test["osint_level"] == lvl).sum())
# endregion

# region TAU SWEEP - validation set, F1-max, lowest-tau tie-break
# -----------------------------------------------------------------------------
val_rss = df_val["rss"].values
val_y = df_val["y"].values
candidates = np.sort(np.unique(val_rss))
log.info("Tau sweep: %d unique RSS candidates", len(candidates))

best_f1 = -1.0
best_tau = candidates[0]

for cand in candidates:
    cand_pred = (val_rss < cand).astype(int)
    c_tp, c_fp, c_fn, c_tn = confusion_counts(val_y, cand_pred)
    c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
    c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
    c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
    if c_f1 > best_f1:
        best_f1 = c_f1
        best_tau = cand
    elif c_f1 == best_f1 and cand < best_tau:
        best_tau = cand

tau_star = best_tau

val_pred_star = (val_rss < tau_star).astype(int)
vtp, vfp, vfn, vtn = confusion_counts(val_y, val_pred_star)

log.info("=" * 60)
log.info("THRESHOLD SELECTED")
log.info("  tau*       = %.4f", tau_star)
log.info("  val F1     = %.4f", best_f1)
log.info("  val TP=%d  FP=%d  FN=%d  TN=%d", vtp, vfp, vfn, vtn)
log.info("=" * 60)

thresh_path = PROCESSED_DIR / "results_threshold.csv"
pd.DataFrame([{
    "tau_star": tau_star, "val_f1": round(best_f1, 6),
    "val_tp": vtp, "val_fp": vfp, "val_fn": vfn, "val_tn": vtn,
}]).to_csv(thresh_path, index=False)
log.info("Written %s", thresh_path)

# endregion

# region TEST SET - Methods A, B, C × per-level + aggregated
# -----------------------------------------------------------------------------
test_rss = df_test["rss"].values
test_y = df_test["y"].values
test_auth = df_test["header_auth_pass"].values.astype(bool)
test_level = df_test["osint_level"].values

pred_a = (test_rss < tau_star).astype(int)
pred_b = (~test_auth).astype(int)
pred_c = ((test_rss < tau_star) | (~test_auth)).astype(int)

summary_rows = []
methods_map = {"A_RSS": pred_a, "B_Auth": pred_b, "C_Hybrid": pred_c}

for m_name, m_pred in methods_map.items():
    for lvl in ["Low", "Medium", "High", "All"]:
        if lvl == "All":
            mask = np.ones(len(test_y), dtype=bool)
        else:
            mask = (test_level == lvl) | (test_level == "Legit")

        s_tp, s_fp, s_fn, s_tn = confusion_counts(test_y[mask], m_pred[mask])
        met = derived_metrics(s_tp, s_fp, s_fn, s_tn)
        row_data = {"method": m_name, "osint_level": lvl,
                    "tp": s_tp, "fp": s_fp, "fn": s_fn, "tn": s_tn,
                    **met, "auc_roc": None}
        summary_rows.append(row_data)
        log.info("  %s | %-7s | TPR=%.4f FPR=%.4f F1=%.4f | TP=%d FP=%d FN=%d TN=%d",
                 m_name, lvl, met["tpr"], met["fpr"], met["f1"], s_tp, s_fp, s_fn, s_tn)

# endregion

# region AUC-ROC for Method A
# -----------------------------------------------------------------------------
auc_all = manual_auc_roc(test_y, test_rss)
log.info("AUC-ROC Method A (All): %.6f", auc_all)

for row_data in summary_rows:
    if row_data["method"] == "A_RSS" and row_data["osint_level"] == "All":
        row_data["auc_roc"] = round(auc_all, 6)

for lvl in ["Low", "Medium", "High"]:
    lvl_mask = (test_level == lvl) | (test_level == "Legit")
    auc_lvl = manual_auc_roc(test_y[lvl_mask], test_rss[lvl_mask])
    log.info("AUC-ROC Method A (%s): %.6f", lvl, auc_lvl)
    for row_data in summary_rows:
        if row_data["method"] == "A_RSS" and row_data["osint_level"] == lvl:
            row_data["auc_roc"] = round(auc_lvl, 6)

summary_path = PROCESSED_DIR / "results_summary.csv"
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
log.info("Written %s", summary_path)

# endregion

# region STATISTICAL TESTS - paired t-test + McNemar Test
# -----------------------------------------------------------------------------
correct_a = (test_y == pred_a).astype(int)
correct_b = (test_y == pred_b).astype(int)
correct_c = (test_y == pred_c).astype(int)

stats_rows = []

# C vs A
t_ca, p_ca = stats.ttest_rel(correct_c, correct_a)
mc_stat_ca, mc_p_ca, mc_meth_ca, mc_n_ca = mcnemar_test(pred_c, pred_a, test_y)
log.info("C vs A - ttest: t=%.4f p=%.6f | McNemar(%s): stat=%.4f p=%.6f n_disc=%d",
         float(t_ca), float(p_ca), mc_meth_ca, mc_stat_ca, mc_p_ca, mc_n_ca)
stats_rows.append({
    "comparison": "C_Hybrid_vs_A_RSS",
    "ttest_stat": round(float(t_ca), 6), "ttest_p": round(float(p_ca), 6),
    "mcnemar_stat": round(mc_stat_ca, 6), "mcnemar_p": round(mc_p_ca, 6),
    "mcnemar_method": mc_meth_ca, "n_discordant": mc_n_ca,
})

# C vs B
t_cb, p_cb = stats.ttest_rel(correct_c, correct_b)
mc_stat_cb, mc_p_cb, mc_meth_cb, mc_n_cb = mcnemar_test(pred_c, pred_b, test_y)
log.info("C vs B - ttest: t=%.4f p=%.6f | McNemar(%s): stat=%.4f p=%.6f n_disc=%d",
         float(t_cb), float(p_cb), mc_meth_cb, mc_stat_cb, mc_p_cb, mc_n_cb)
stats_rows.append({
    "comparison": "C_Hybrid_vs_B_Auth",
    "ttest_stat": round(float(t_cb), 6), "ttest_p": round(float(p_cb), 6),
    "mcnemar_stat": round(mc_stat_cb, 6), "mcnemar_p": round(mc_p_cb, 6),
    "mcnemar_method": mc_meth_cb, "n_discordant": mc_n_cb,
})

stats_path = PROCESSED_DIR / "results_stats.csv"
pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
log.info("Written %s", stats_path)

# endregion

# Pipeline completed!
log.info("=" * 60)
log.info("STAGE 5 COMPLETE")
log.info("  tau*                : %.4f", tau_star)
log.info("  AUC-ROC (Method A) : %.6f", auc_all)
log.info("  Outputs:")
log.info("    %s", thresh_path)
log.info("    %s", summary_path)
log.info("    %s", stats_path)
log.info("=" * 60)
