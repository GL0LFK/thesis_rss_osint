"""
Stage 4 (Inject) (v2 - HIGH OSINT now carries real RSS)
Input  : data/02_Processed/score_table.csv
Configs: data/03_Labels_Attack_Injection/osint_levels.yaml
         data/03_Labels_Attack_Injection/inject_config.yaml
Output : data/02_Processed/evaluation_set.csv

Columns: sender, recipient, timestamp, rss, header_auth_pass, y, osint_level

Related Thesis Sections:
    3.5.1 (Stage 4 Inject: Injection Methodology)
    3.5.2 (OSINT Preparedness Levels)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import random

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, LABELS_DIR, SCORE_TABLE, EVALUATION_SET

# Setting up Logging
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
_log_path = PROCESSED_DIR / "stage4_inject.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(_log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Load configs
osint_path = LABELS_DIR / "osint_levels.yaml"
config_path = LABELS_DIR / "inject_config.yaml"

log.info("Loading OSINT config: %s", osint_path)
with open(osint_path, "r", encoding="utf-8") as f:
    osint_cfg = yaml.safe_load(f)

log.info("Loading inject config: %s", config_path)
with open(config_path, "r", encoding="utf-8") as f:
    inject_cfg = yaml.safe_load(f)

base_seed = int(inject_cfg.get("random_seed", 42))
ratio = float(inject_cfg["injection"].get("spear_phish_ratio", 0.001))

log.info("Random seed           : %d", base_seed)
log.info("Spear-phish ratio     : %.6f", ratio)

# Dedicated Random Number Generators to ensure deterministic behaviour
rng_global = random.Random(base_seed)
rng_medium = random.Random(base_seed + 1)   # Bernoulli trial at Medium OSINT
rng_high   = random.Random(base_seed + 2)   # HIGH OSINT edge sampling

# Load scores
log.info("Reading scores: %s", SCORE_TABLE)
df_scores = pd.read_csv(SCORE_TABLE)
log.info("Loaded %d scored edges", len(df_scores))

# Compute injection counts (0.1% of final Evaluation Set)
N_legit = len(df_scores)
N_spear = int(round(ratio * N_legit))
per_level = N_spear // 3
N_spear = per_level * 3

log.info("Legitimate rows       : %d", N_legit)
log.info("Injected total spear  : %d", N_spear)
log.info("Per OSINT level       : %d", per_level)

if N_spear == 0:
    log.warning("Computed 0 spear-phish to inject; check ratio or dataset size.")

# Build legitimate evaluation base
now = datetime(2001, 1, 1)

# Legitimate email data frame
legit_df = df_scores[["sender", "recipient", "rss"]].copy()
legit_df["timestamp"] = [now + timedelta(minutes=i) for i in range(N_legit)]
legit_df["header_auth_pass"] = True
legit_df["y"] = 0
legit_df["osint_level"] = "Legit"

# Select targets for LOW and MEDIUM injection
# Sample distinct legitimate row indices as attack targets
low_med_indices = rng_global.sample(range(N_legit), per_level * 2)
low_idx = low_med_indices[:per_level] # first half
med_idx = low_med_indices[per_level:] # second half

# HIGH OSINT: sample real high-RSS edges (sender-recipient pair)
# Sort by RSS descending; sample from top quartile (Q3+) to ensure the attacker
# picks genuinely trusted pairs
df_sorted = df_scores.sort_values("rss", ascending=False).reset_index(drop=True)
top_quartile_cutoff = int(len(df_sorted) * 0.25)
top_quartile = df_sorted.iloc[:top_quartile_cutoff]

high_sample_indices = rng_high.sample(range(len(top_quartile)), per_level)
high_edges = top_quartile.iloc[high_sample_indices]

log.info("HIGH OSINT: sampled %d edges from top quartile (RSS >= %.2f)",
         per_level, top_quartile["rss"].min())

# This is a helper: fake sender construction
FREE_MAIL = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]

# Convert the sender into display name: gipsz.elek -> Gipsz Elek
def _fake_name_from_email(email: str) -> str:
    tokens = [t for t in email.split("@", 1)[0].replace(".", " ").replace("_", " ").split() if t]
    if not tokens:
        return "Unknown Sender"
    if len(tokens) == 1:
        return f"{tokens[0].capitalize()} User"
    return f"{tokens[0].capitalize()} {tokens[-1].capitalize()}"

# Random string for the email address 783e4rbvdshj@gmail.com
def _random_string(length: int, rng: random.Random) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


# Build injected rows
injected_rows = []

# LOW OSINT
# External unknown sender, free-mail domain, header_auth_pass = False, rss = 0
for idx in low_idx:
    row = legit_df.iloc[idx]
    victim = row["recipient"]
    display = _fake_name_from_email(victim)
    local = _random_string(10, rng_global)
    domain = rng_global.choice(FREE_MAIL)

    injected_rows.append({
        "sender": f"{display} <{local}@{domain}>",
        "recipient": victim,
        "timestamp": row["timestamp"],
        "rss": 0.0,
        "header_auth_pass": False,
        "y": 1,
        "osint_level": "Low",
    })

# MEDIUM OSINT
# Impersonated employee, non-Enron domain, Bernoulli(0.62), rss = 0
P_MEDIUM_PASS = 0.62

for idx in med_idx:
    row = legit_df.iloc[idx]
    victim = row["recipient"]
    real_sender = row["sender"]
    display = _fake_name_from_email(real_sender)
    base_local = real_sender.split("@", 1)[0]
    domain = rng_global.choice(FREE_MAIL)

    injected_rows.append({
        "sender": f"{display} <{base_local}@{domain}>",
        "recipient": victim,
        "timestamp": row["timestamp"],
        "rss": 0.0,
        "header_auth_pass": bool(rng_medium.random() < P_MEDIUM_PASS),
        "y": 1,
        "osint_level": "Medium",
    })

# HIGH OSINT 
# Impersonated trusted contact: real sender address, real RSS, header_auth_pass = True
for _, edge in high_edges.iterrows():
    real_sender = edge["sender"]
    victim = edge["recipient"]
    real_rss = float(edge["rss"])
    display = _fake_name_from_email(real_sender)

    injected_rows.append({
        "sender": f"{display} <{real_sender}>",
        "recipient": victim,
        "timestamp": now + timedelta(minutes=rng_high.randint(0, N_legit)),
        "rss": real_rss,
        "header_auth_pass": True,
        "y": 1,
        "osint_level": "High",
    })

# Assemble the Evaluation Set
eval_cols = ["sender", "recipient", "timestamp", "rss", "header_auth_pass", "y", "osint_level"]

injected_df = pd.DataFrame(injected_rows, columns=eval_cols)
evaluation_df = pd.concat([legit_df[eval_cols], injected_df], ignore_index=True)

# Diagnostics (number of ..)
n_legit_final = (evaluation_df["y"] == 0).sum()
n_spear_final = (evaluation_df["y"] == 1).sum()
n_low  = (evaluation_df["osint_level"] == "Low").sum()
n_med  = (evaluation_df["osint_level"] == "Medium").sum()
n_high = (evaluation_df["osint_level"] == "High").sum()

# Medium OSINT Level Bernoulli diagnostics
med_mask = evaluation_df["osint_level"] == "Medium"
med_pass = evaluation_df.loc[med_mask, "header_auth_pass"].sum()
med_fail = med_mask.sum() - med_pass

# High RSS diagnostics
high_mask = evaluation_df["osint_level"] == "High"
high_rss_vals = evaluation_df.loc[high_mask, "rss"]

log.info("Final evaluation rows : %d", len(evaluation_df))
log.info("  Legitimate          : %d", n_legit_final)
log.info("  Spear-phish total   : %d", n_spear_final)
log.info("    Low               : %d  (rss=0, auth=False)", n_low)
log.info("    Medium            : %d  (rss=0, auth pass=%d / fail=%d, p=%.2f)",
         n_med, med_pass, med_fail, P_MEDIUM_PASS)
log.info("    High              : %d  (real RSS, auth=True)", n_high)
log.info("    High RSS range    : min=%.2f  mean=%.2f  max=%.2f",
         high_rss_vals.min(), high_rss_vals.mean(), high_rss_vals.max())

# Assert legitimate header_auth_pass = True
assert evaluation_df.loc[evaluation_df["y"] == 0, "header_auth_pass"].all(), \
    "FATAL: some legitimate rows have header_auth_pass = False"
log.info("ASSERT PASSED: all legitimate rows have header_auth_pass = True")

# Write CSV
log.info("Writing %s", EVALUATION_SET)
evaluation_df.to_csv(EVALUATION_SET, index=False)

log.info("=" * 60)
log.info("STAGE 4 COMPLETE")
log.info("  evaluation_set rows : %d", len(evaluation_df))
log.info("  output → %s", EVALUATION_SET)
log.info("=" * 60)