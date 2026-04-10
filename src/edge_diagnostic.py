"""
Edge diagnostic tool for Appendix B.2.
Prints four blocks for any sender-recipient pair:
  Block 1 - Test set context and edge identification
  Block 2 - Sub-score decomposition (from score_table.csv)
  Block 3 - Edge context / raw counts (from enron_parsed.csv)
  Block 4 - Detection verdict against tau

Usage:
  python edge_diagnostic.py <sender> <recipient>
  python edge_diagnostic.py   (defaults to the known single FP)

Related Thesis Section: 3.8.1	(Interpretation of Results)
"""

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EVALUATION_SET, SCORE_TABLE, PARSED_CSV

SEED = 42
TAU = 0.0215
SEP = "=" * 64

DEFAULT_SENDER = "melanie.hunter@neg.pge.com"
DEFAULT_RECIPIENT = "llightfoot@coral-energy.com"


def build_test_split(df_eval: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    val_mask = np.zeros(len(df_eval), dtype=bool)
    strat_key = df_eval["osint_level"].to_numpy()
    for level in np.unique(strat_key):
        level_idx = np.where(strat_key == level)[0]
        rng.shuffle(level_idx)
        n_val = len(level_idx) // 2
        val_mask[level_idx[:n_val]] = True
    return df_eval[~val_mask].copy().reset_index(drop=True)


def print_block1(df_test: pd.DataFrame, sender: str, recipient: str):
    print(f"\n{SEP}")
    print("BLOCK 1 - Test Set Context and Edge Identification")
    print(SEP)

    legit_count = (df_test["y"] == 0).sum()
    legit_below = df_test.loc[
        (df_test["y"] == 0) & (df_test["rss"] < TAU)
    ]
    print(f"  test set rows        : {len(df_test):,}")
    print(f"  legit rows           : {legit_count:,}")
    print(f"  legit with rss < tau : {len(legit_below)}")
    print(f"  tau                  : {TAU}")
    print()

    match = df_test.loc[
        (df_test["sender"] == sender) & (df_test["recipient"] == recipient)
    ]
    if match.empty:
        print(f"  ** Pair not found in test split: {sender} -> {recipient}")
        print(f"     Checking full evaluation set is recommended.")
        return False

    row = match.iloc[0]
    cols = ["sender", "recipient", "timestamp", "rss",
            "header_auth_pass", "y", "osint_level"]
    for c in cols:
        print(f"  {c:>20s} = {row[c]}")
    return True


def print_block2(sender: str, recipient: str):
    print(f"\n{SEP}")
    print("BLOCK 2 - Sub-Score Decomposition (from score_table.csv)")
    print(SEP)

    df_scores = pd.read_csv(SCORE_TABLE)
    match = df_scores.loc[
        (df_scores["sender"] == sender) & (df_scores["recipient"] == recipient)
    ]
    if match.empty:
        print(f"  ** Pair not found in score_table.csv")
        print(f"     Sender is absent from the communication graph (RSS = 0).")
        return

    row = match.iloc[0]
    labels = {
        "s1": "Trust Degree (TD)",
        "s2": "Communication Reciprocity (CR)",
        "s3": "Communication Interaction Avg (CIA)",
        "s4": "Average Response Time (ART)",
        "s5": "Clustering Coefficient (CC)",
        "s6": "Betweenness Centrality (BC)",
    }
    for col, label in labels.items():
        if col in row.index:
            print(f"  {col}  {label:<40s} = {row[col]:.10f}")

    rss_val = row["rss"] if "rss" in row.index else float("nan")
    print(f"\n  {'RSS (weighted composite)':<46s} = {rss_val:.10f}")
    print(f"  {'tau*':<46s} = {TAU}")
    print(f"  {'Deficit (tau* - RSS)':<46s} = {TAU - rss_val:.10f}")


def print_block3(sender: str, recipient: str):
    print(f"\n{SEP}")
    print("BLOCK 3 - Edge Context (from enron_parsed.csv)")
    print(SEP)

    df_parsed = pd.read_csv(PARSED_CSV, usecols=["sender", "recipient"])

    w_ij = len(df_parsed[
        (df_parsed["sender"] == sender) & (df_parsed["recipient"] == recipient)
    ])
    w_ji = len(df_parsed[
        (df_parsed["sender"] == recipient) & (df_parsed["recipient"] == sender)
    ])
    total_sent_i = len(df_parsed[df_parsed["sender"] == sender])
    total_sent_j = len(df_parsed[df_parsed["sender"] == recipient])

    print(f"  W(i,j)  count({sender} -> {recipient})  = {w_ij}")
    print(f"  W(j,i)  count({recipient} -> {sender})  = {w_ji}")
    print(f"  total_sent(sender)    = {total_sent_i}")
    print(f"  total_sent(recipient) = {total_sent_j}")


def print_block4(sender: str, recipient: str, df_test: pd.DataFrame):
    print(f"\n{SEP}")
    print("BLOCK 4 - Detection Verdict")
    print(SEP)

    match = df_test.loc[
        (df_test["sender"] == sender) & (df_test["recipient"] == recipient)
    ]
    if match.empty:
        print("  ** Cannot produce verdict; pair not in test split.")
        return

    row = match.iloc[0]
    rss_val = row["rss"]
    y_true = int(row["y"])
    auth_pass = row["header_auth_pass"]

    rss_flag = rss_val < TAU
    auth_flag = not auth_pass
    hybrid_flag = rss_flag or auth_flag

    print(f"  RSS flag   (rss < tau)         : {'FLAGGED' if rss_flag else 'PASS'}")
    print(f"  Auth flag  (auth failed)       : {'FLAGGED' if auth_flag else 'PASS'}")
    print(f"  Hybrid     (RSS OR Auth)       : {'FLAGGED' if hybrid_flag else 'PASS'}")
    print(f"  Ground truth label (y)         : {y_true}  ({'spear-phish' if y_true else 'legitimate'})")

    if y_true == 0 and rss_flag:
        print(f"  Conclusion                     : FALSE POSITIVE")
    elif y_true == 1 and not rss_flag:
        print(f"  Conclusion                     : FALSE NEGATIVE")
    elif y_true == 1 and rss_flag:
        print(f"  Conclusion                     : TRUE POSITIVE")
    else:
        print(f"  Conclusion                     : TRUE NEGATIVE")

    print(f"\n  This case is interpreted in Section 3.8.1 and visualised in Figure 10.")


def main():
    parser = argparse.ArgumentParser(
        description="Edge diagnostic for Appendix B.2"
    )
    parser.add_argument(
        "sender", nargs="?", default=DEFAULT_SENDER,
        help=f"Sender email address (default: {DEFAULT_SENDER})"
    )
    parser.add_argument(
        "recipient", nargs="?", default=DEFAULT_RECIPIENT,
        help=f"Recipient email address (default: {DEFAULT_RECIPIENT})"
    )
    args = parser.parse_args()

    sender = args.sender.strip().lower()
    recipient = args.recipient.strip().lower()

    print(f"Edge Diagnostic: {sender} -> {recipient}")

    df_eval = pd.read_csv(EVALUATION_SET)
    df_test = build_test_split(df_eval)

    found = print_block1(df_test, sender, recipient)
    print_block2(sender, recipient)
    print_block3(sender, recipient)
    if found:
        print_block4(sender, recipient, df_test)

    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
