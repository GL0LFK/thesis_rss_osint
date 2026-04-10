"""
Stage 3 — Score
Input  : data/02_Processed/graph.graphml
Output : data/02_Processed/score_table.csv

Columns: sender, recipient, s1, s2, s3, s4, s5, s6, rss

Related Thesis Section: 3.4.3 (Stage 3 Score: Relationship Score Baseline)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, GRAPH_GRAPHML, SCORE_TABLE
import random
random.seed(42)

# Logging setup
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
_log_path = PROCESSED_DIR / "stage3_score.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(_log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Load graph 
log.info("Reading %s", GRAPH_GRAPHML)
G = nx.read_graphml(GRAPH_GRAPHML)
log.info("Loaded graph: %d nodes, %d directed edges", G.number_of_nodes(), G.number_of_edges())

# Precompute node-level metrics (s5, s6), because these are node level scores requiring undirected graph
log.info("Computing clustering coefficients (s5)...")
G_undirected = G.to_undirected()
# How tightly connected each node's neighbours are; one float per node
clustering = nx.clustering(G_undirected)

log.info("Computing betweenness centrality (s6)... this takes time")
random.seed(42)
# Sample is set to 1000 nodes, otherwise it would take forever for 75k nodes. The accuracy is still OK!
bc = nx.betweenness_centrality(G, normalized=True, k=1000, weight=None, seed=42)

# Extract all betweenness centrality values for min-max range
bc_values = list(bc.values())
bc_min = min(bc_values)
bc_max = max(bc_values)

# Min-max normalise; if all values are identical, set all to 0
if bc_max == bc_min:
    s6_from_bc = {n: 0.0 for n in G.nodes}
else:
    span = bc_max - bc_min
    s6_from_bc = {n: (val - bc_min) / span for n, val in bc.items()}

# Helper: parse timestamps back into sorted list of floats (hours) 
from datetime import datetime

# Parse the ISO string created by Stage 2
def _parse_timestamp_list(ts_str: str):
    if not ts_str:
        return []
    parts = str(ts_str).split("|")
    out = [] # I pack the datetime objects here
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(datetime.fromisoformat(p))
        except ValueError:
            # Should not happen; Stage 2 I convert to ISO; log and skip
            log.warning("BAD_TS  %s", p)
    out.sort()
    return out


# Compute pairwise scores
log.info("Computing edge scores (s1–s6, rss)...")

rows = []

# Precompute total outgoing weights for s1 (eg. Count of all emails sent by the sender u)
out_weight_sum = {}
for u, v, data in G.edges(data=True):
    w = float(data.get("weight", 1.0))
    out_weight_sum[u] = out_weight_sum.get(u, 0.0) + w

for u, v, data in G.edges(data=True):
    # rename NetworkX default u and v variables
    sender = str(u)
    recipient = str(v)

    w_ij = float(data.get("weight", 1.0))

    # s1 — Trust Degree
    denom = out_weight_sum.get(u, 0.0)
    s1 = w_ij / denom if denom > 0 else 0.0 # do not divide by zero

    # Reciprocal weight and timestamps, calculated for some of the sub-scores
    rev_data = G.get_edge_data(v, u, default=None)
    w_ji = float(rev_data.get("weight", 0.0)) if rev_data else 0.0

    # s2 — Communication Reciprocity
    s2 = 1.0 if w_ji > 0 else 0.0

    # s3 — Communication Interaction Average
    if w_ij > 0 and w_ji > 0:
        s3 = min(w_ij, w_ji) / max(w_ij, w_ji)
    else:
        s3 = 0.0

    # s4 — Average Response Time (inverted)
    if s2 == 0.0:
        s4 = 0.0 # the recipient never responded back to the sender so there is nothing to measure
    else:
        ts_ij = _parse_timestamp_list(data.get("timestamps", ""))
        ts_ji = _parse_timestamp_list(rev_data.get("timestamps", "")) if rev_data else []

        j_idx = 0
        latencies = []
        for t_sent in ts_ij:
            while j_idx < len(ts_ji) and ts_ji[j_idx] <= t_sent:
                j_idx += 1
            if j_idx >= len(ts_ji):
                break
            t_reply = ts_ji[j_idx]
            delta_h = (t_reply - t_sent).total_seconds() / 3600.0
            if delta_h >= 0:
                latencies.append(delta_h)

        if latencies:
            mean_latency_hours = sum(latencies) / len(latencies)
            s4 = 1.0 / (1.0 + mean_latency_hours)
        else:
            s4 = 0.0

    # s5 — Clustering coefficient (node-level)
    # Lookup sender's clustering coefficient from precomputed dictionary; default 0.0 if missing
    s5 = float(clustering.get(u, 0.0))

    # s6 — Betweenness centrality (node-level, min-max normalised)
    # Lookup sender's normalised betweenness centrality from precomputed dictionary; default 0.0 if missing
    s6 = float(s6_from_bc.get(u, 0.0))

    rss = 100.0 * (
        0.25 * s1
        + 0.20 * s2
        + 0.20 * s3
        + 0.15 * s4
        + 0.10 * s5
        + 0.10 * s6
    )

    rows.append(
        {
            "sender": sender,
            "recipient": recipient,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "s5": s5,
            "s6": s6,
            "rss": rss,
        }
    )

# Write CSV
log.info("Writing %s", SCORE_TABLE)
df_scores = pd.DataFrame(rows, columns=["sender", "recipient", "s1", "s2", "s3", "s4", "s5", "s6", "rss"])
df_scores.to_csv(SCORE_TABLE, index=False)

log.info("=" * 60)
log.info("STAGE 3 COMPLETE")
log.info("  edge_count (scored rows) : %d", len(rows))
log.info("  output → %s", SCORE_TABLE)
log.info("=" * 60)
