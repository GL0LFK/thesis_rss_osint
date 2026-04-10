# Data Dictionary — score_table.csv

**Stage:** Stage 3 (Score)  
**Thesis:** 3.4.3 (Stage 3 Score: Relationship Score Baseline)  
**Location:** `data/02_Processed/score_table.csv`  
**Row semantics:** One row per directed edge (sender → recipient) in the communication graph.

| Column    | Type  | Range   | Description                                                                                          |
|-----------|-------|---------|------------------------------------------------------------------------------------------------------|
| sender    | str   | —       | Source node (email address) of the directed edge.                                                     |
| recipient | str   | —       | Target node (email address) of the directed edge.                                                     |
| s1        | float | [0, 1]  | **Trust Degree.** Fraction of sender's total outgoing emails directed to this recipient.              |
| s2        | float | {0, 1}  | **Communication Reciprocity.** 1 if the recipient ever emailed the sender back; 0 otherwise.          |
| s3        | float | [0, 1]  | **Communication Interaction Average.** min(W_ij, W_ji) / max(W_ij, W_ji); 0 if no reciprocal edge.   |
| s4        | float | [0, 1]  | **Average Response Time (inverted).** 1 / (1 + mean_latency_hours); 0 if no replies exist.            |
| s5        | float | [0, 1]  | **Clustering Coefficient.** How tightly connected the sender's neighbours are (node-level, undirected).|
| s6        | float | [0, 1]  | **Betweenness Centrality.** Min-max normalised; how often the sender bridges shortest paths (node-level).|
| rss       | float | [0, 100]| **Relationship Strength Score.** Weighted composite: 100 × (0.25·s1 + 0.20·s2 + 0.20·s3 + 0.15·s4 + 0.10·s5 + 0.10·s6). |

**Notes:**
- s5 and s6 are sender-level metrics (describe node i, not the pair i→j).
- s6 uses approximate betweenness centrality (k=1000 sampled nodes, seed=42).
- A pair absent from the graph receives RSS = 0 (maximum anomaly signal).
