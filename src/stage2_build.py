"""
Stage 2 (Build)
Input  : data/01_Interim/enron_parsed.csv
Output : data/02_Processed/graph.graphml

Related Thesis Sections
    3.3.2 (Stage 2 Build: Graph Building Process)
    §3.3.3 (Descriptive Statistics)
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INTERIM_DIR, PROCESSED_DIR, PARSED_CSV, GRAPH_GRAPHML

# Setup Logging
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
_log_path = INTERIM_DIR / "stage2_build.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(_log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Load CSV fiel
log.info("Reading %s", PARSED_CSV)
df = pd.read_csv(
    PARSED_CSV,
    parse_dates=["timestamp"], # convert back the datetimes tx stripped in Stage 1, to ISO strings so it can be consumed by Stage 3
    dtype={"sender": str, "recipient": str},
)
log.info("Loaded %d rows", len(df))

# Build edge accumulator
# edge_data[(sender, recipient)] = {"weight": int, "timestamps": [datetime, ...]}
log.info("Aggregating edges...")
edge_data = defaultdict(lambda: {"weight": 0, "timestamps": []})

for sender, recipient, ts in zip(df["sender"], df["recipient"], df["timestamp"]):
    key = (sender, recipient)
    edge_data[key]["weight"] += 1
    edge_data[key]["timestamps"].append(ts)

log.info("Unique directed edges before graph construction: %d", len(edge_data))

# Build the directed graph
log.info("Building Directed Graph...")
G = nx.DiGraph()

# Sort timestamps for Stage 3 (s4) it relies on chrono. order
# pipe-delimited serialisation --> Stage 3 will deserialise it for the date times
for (sender, recipient), attrs in edge_data.items():
    attrs["timestamps"].sort()
    ts_str = "|".join(t.isoformat() for t in attrs["timestamps"])
    G.add_edge(
        sender,
        recipient,
        weight=attrs["weight"],
        timestamps=ts_str,
    )

# Assert zero self-loops
self_loop_edge_count = nx.number_of_selfloops(G)
assert self_loop_edge_count == 0, (
    f"FATAL: {self_loop_edge_count} self-loop(s) found in graph. "
    "Stage 1 dedup failed."
)

# Statistics computation
log.info("Computing graph statistics...")

node_count          = G.number_of_nodes()
directed_edge_count = G.number_of_edges()

external_address_count = sum(
    1 for n in G.nodes() if "@enron.com" not in str(n)
)

weakly_connected_components = nx.number_weakly_connected_components(G)
graph_density               = nx.density(G)

# Clustering Coefficient (undirected projection)
G_undirected             = G.to_undirected()
mean_clustering_coeff    = nx.average_clustering(G_undirected)

# Write graph to a file
log.info("Writing %s", GRAPH_GRAPHML)
nx.write_graphml(G, GRAPH_GRAPHML)
log.info("graph.graphml written and locked read-only.")

# Report final statistics
log.info("=" * 60)
log.info("STAGE 2 COMPLETE")
log.info("  node_count                    : %d", node_count)
log.info("  directed_edge_count           : %d", directed_edge_count)
log.info("  self_loop_edge_count          : %d", self_loop_edge_count)
log.info("  external_address_count        : %d", external_address_count)
log.info("  weakly_connected_components   : %d", weakly_connected_components)
log.info("  graph_density                 : %.8f", graph_density)
log.info("  mean_clustering_coefficient   : %.6f", mean_clustering_coeff)
log.info("  output → %s", GRAPH_GRAPHML)
log.info("=" * 60)
