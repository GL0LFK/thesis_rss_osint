"""
Related Thesis Section: 3.2 (System Design and Data Preparation)
"""
from pathlib import Path

# Project root — one level up from src/
BASE_PATH = Path(__file__).resolve().parent.parent

# Data sub-folders
RAW_DIR       = BASE_PATH / "data" / "00_Raw"
INTERIM_DIR   = BASE_PATH / "data" / "01_Interim"
PROCESSED_DIR = BASE_PATH / "data" / "02_Processed"
LABELS_DIR    = BASE_PATH / "data" / "03_Labels_Attack_Injection"
DICT_DIR      = BASE_PATH / "data" / "99_Data_Dictionary"

# Stage 1 — raw email tree - !!unpack the zip file before first run!!
MAILDIR       = RAW_DIR / "enron_mail_20150507" / "maildir"

# Named artefacts
PARSED_CSV    = INTERIM_DIR   / "enron_parsed.csv"
GRAPH_GRAPHML = PROCESSED_DIR / "graph.graphml"
SCORE_TABLE   = PROCESSED_DIR / "score_table.csv"
EVALUATION_SET = PROCESSED_DIR / "evaluation_set.csv"