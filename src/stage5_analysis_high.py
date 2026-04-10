"""
Stage 5 — High-OSINT Deep Dive

This script investigates why High OSINT evades detection and what it would
cost (in false positives) to catch more of them

Reads:
  evaluation_set.csv  — full evaluation set (legitimate + injected rows)
  score_table.csv     — all scored edges with s1–s6 sub-scores

Produces:
  high_osint_subscore_profile.csv  — s1–s6 + rss for the High OSINT test-set
                                     rows, showing which trust signals the
                                     attacker exploits
  tau_tradeoff_table.csv           — TPR per OSINT level + global FPR at
                                     multiple tau thresholds, illustrating the
                                     cost of lowering the detection boundary

Related Thesis Sections:
    3.7.1 (Table 7 — tau tradeoff)
    3.8 (Discussion)
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, EVALUATION_SET

SEED = 42
SCORE_TABLE = PROCESSED_DIR / "score_table.csv"

df_eval = pd.read_csv(EVALUATION_SET)
df_scores = pd.read_csv(SCORE_TABLE)

print(f"evaluation_set : {len(df_eval)} rows")
print(f"score_table    : {len(df_scores)} rows")

rng = np.random.RandomState(SEED)
val_mask = np.zeros(len(df_eval), dtype=bool)
strat_key = df_eval["osint_level"].to_numpy()
for level in np.unique(strat_key):
    level_idx = np.where(strat_key == level)[0]
    rng.shuffle(level_idx)
    n_val = len(level_idx) // 2
    val_mask[level_idx[:n_val]] = True

df_test = df_eval[~val_mask].copy().reset_index(drop=True)
print(f"test set       : {len(df_test)} rows")


def extract_email(addr: str) -> str:
    match = re.search(r"<([^>]+)>", str(addr))
    return match.group(1).strip().lower() if match else str(addr).strip().lower()


df_high = df_test[df_test["osint_level"] == "High"].copy()
print(f"High OSINT test rows: {len(df_high)}")

score_lookup: dict[tuple[str, str], dict[str, float]] = {}
for _, row in df_scores.iterrows():
    key = (str(row["sender"]).strip().lower(), str(row["recipient"]).strip().lower())
    score_lookup[key] = {
        "s1": float(row["s1"]),
        "s2": float(row["s2"]),
        "s3": float(row["s3"]),
        "s4": float(row["s4"]),
        "s5": float(row["s5"]),
        "s6": float(row["s6"]),
    }

print(f"score_lookup entries: {len(score_lookup)}")

print("\nDiagnostic — first 3 High OSINT lookup attempts:")
s1_vals, s2_vals, s3_vals, s4_vals, s5_vals, s6_vals = [], [], [], [], [], []

for _, row in df_high.iterrows():
    sender_bare = extract_email(row["sender"])
    recipient_bare = extract_email(row["recipient"])
    key = (sender_bare, recipient_bare)
    found = score_lookup.get(key)

    if len(s1_vals) < 3:
        print(f"  sender_raw   = {row['sender']}")
        print(f"  sender_bare  = {sender_bare}")
        print(f"  recipient    = {recipient_bare}")
        print(f"  key in lookup: {key in score_lookup}")
        if not found:
            partial = [k for k in score_lookup if k[0] == sender_bare][:2]
            print(f"  partial matches for sender: {partial}")
        print()

    if found:
        s1_vals.append(found["s1"])
        s2_vals.append(found["s2"])
        s3_vals.append(found["s3"])
        s4_vals.append(found["s4"])
        s5_vals.append(found["s5"])
        s6_vals.append(found["s6"])
    else:
        s1_vals.append(float("nan"))
        s2_vals.append(float("nan"))
        s3_vals.append(float("nan"))
        s4_vals.append(float("nan"))
        s5_vals.append(float("nan"))
        s6_vals.append(float("nan"))

df_high["s1"] = s1_vals
df_high["s2"] = s2_vals
df_high["s3"] = s3_vals
df_high["s4"] = s4_vals
df_high["s5"] = s5_vals
df_high["s6"] = s6_vals

matched = df_high["s1"].notna().sum()
print(f"Merge matched: {matched} / {len(df_high)} rows")

profile_cols = [
    "sender", "recipient", "osint_level",
    "s1", "s2", "s3", "s4", "s5", "s6", "rss",
]
df_profile = df_high[profile_cols].sort_values("rss", ascending=True).reset_index(drop=True)

profile_path = PROCESSED_DIR / "high_osint_subscore_profile.csv"
df_profile.to_csv(profile_path, index=False)
print(f"Written {profile_path}  ({len(df_profile)} rows)")

print("\nSub-score summary (High OSINT):")
for col in ["s1", "s2", "s3", "s4", "s5", "s6", "rss"]:
    vals = df_profile[col].to_numpy()
    n_valid = int(np.sum(~np.isnan(vals)))
    if n_valid > 0:
        print(
            f"  {col:>3s}  min={np.nanmin(vals):.4f}  median={np.nanmedian(vals):.4f}"
            f"  max={np.nanmax(vals):.4f}  mean={np.nanmean(vals):.4f}  (n={n_valid})"
        )
    else:
        print(f"  {col:>3s}  ALL NaN — check diagnostic output above")

tau_candidates = [0.02, 1.0, 5.0, 10.0, 15.0, 20.0, 28.0, 40.0, 65.0]

test_rss = df_test["rss"].to_numpy()
test_y = df_test["y"].to_numpy()
test_level = df_test["osint_level"].to_numpy()

legit_mask = test_level == "Legit"
n_legit = int(np.sum(legit_mask))

tradeoff_rows = []
for tau in tau_candidates:
    preds = (test_rss < tau).astype(int)

    global_fp = int(np.sum((preds == 1) & (test_y == 0)))
    global_fpr = global_fp / n_legit if n_legit > 0 else 0.0

    row: dict[str, float | int] = {
        "tau": tau, "global_fp": global_fp, "global_fpr": round(global_fpr, 6),
    }

    for lvl in ["Low", "Medium", "High"]:
        lvl_mask = test_level == lvl
        n_pos = int(np.sum(lvl_mask))
        if n_pos == 0:
            row[f"tpr_{lvl.lower()}"] = float("nan")
            row[f"tp_{lvl.lower()}"] = 0
            continue
        lvl_tp = int(np.sum((preds[lvl_mask] == 1) & (test_y[lvl_mask] == 1)))
        row[f"tp_{lvl.lower()}"] = lvl_tp
        row[f"tpr_{lvl.lower()}"] = round(lvl_tp / n_pos, 6)

    tradeoff_rows.append(row)

col_order = [
    "tau",
    "tp_low", "tpr_low",
    "tp_medium", "tpr_medium",
    "tp_high", "tpr_high",
    "global_fp", "global_fpr",
]
df_tradeoff = pd.DataFrame(tradeoff_rows)[col_order]

tradeoff_path = PROCESSED_DIR / "tau_tradeoff_table.csv"
df_tradeoff.to_csv(tradeoff_path, index=False)
print(f"\nWritten {tradeoff_path}  ({len(df_tradeoff)} rows)")
print(df_tradeoff.to_string(index=False))