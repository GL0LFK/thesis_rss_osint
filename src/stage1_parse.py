"""
Stage 1 — Parse
Input  : data/00_Raw/enron_mail_20150507/maildir  (raw .eml tree, no extensions)
Output : data/01_Interim/enron_parsed.csv         columns: sender, recipient, timestamp
Log    : data/01_Interim/malformed_drops.log

Related Thesis Section: 3.3.1 (Stage 1 Parse: Graph Model Design)
"""

import os
import sys
import logging
from pathlib import Path

import mailparser
from dateutil.parser import parse as dateutil_parse
import pandas as pd

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MAILDIR, INTERIM_DIR, PARSED_CSV
from utils import normalise_address, strip_tzinfo

# Logging setup
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
_log_path = INTERIM_DIR / "malformed_drops.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(_log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Counters 
raw_file_count             = 0
malformed_drops            = 0
self_loop_drops            = 0
multi_recipient_expansions = 0

seen_msg_ids  = set()   # unique Message-IDs  →  feeds unique_message_ids counter
seen_row_keys = set()   # (msg_id::recipient) or (sender|recipient|ts)  →  row dedup
rows          = []

# Dry run control
# TEST PUPROSES ONLY 
# Limit to the first N employee dirs
# Set to None for the full corpus run
DRY_RUN_EMPLOYEES = None

# Folder Walk - this takes a long time
if DRY_RUN_EMPLOYEES is not None:
    employee_dirs = sorted(d for d in MAILDIR.iterdir() if d.is_dir())[:DRY_RUN_EMPLOYEES]
    log.info("DRY RUN — %d employee dirs: %s", DRY_RUN_EMPLOYEES, [d.name for d in employee_dirs])
else:
    employee_dirs = [MAILDIR]
    log.info("FULL RUN — walking %s", MAILDIR)

for root_dir in employee_dirs:
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            raw_file_count += 1

            _open_path = ("\\\\?\\" + str(fpath.resolve())) if os.name == "nt" else str(fpath)
            mail = None
            try:
                with open(_open_path, "rb") as _fh:
                    _raw = _fh.read()
                mail = mailparser.parse_from_bytes(_raw)
            except Exception as exc:
                malformed_drops += 1
                log.warning("MALFORMED  %s  |  %s", fpath, exc)
                continue

            # Sender 
            sender = None
            try:
                sender = normalise_address(mail.from_[0][1])
                if not sender:
                    raise ValueError("empty sender address")
            except (IndexError, TypeError, ValueError):
                malformed_drops += 1
                log.warning("NO_SENDER  %s", fpath)
                continue

            # Recipients 
            recipients = None
            try:
                recipients = [normalise_address(p[1]) for p in mail.to if p[1]]
                if not recipients:
                    raise ValueError("empty recipient list")
            except (IndexError, TypeError, ValueError):
                malformed_drops += 1
                log.warning("NO_RECIPIENT  %s", fpath)
                continue

            # Timestamp
            # result and strip tzinfo to keep the pipeline timezone-naive
            # in Stage 2 we convert this to ISO strings so it can be consumed by Stage 3
            ts = None
            try:
                if mail.date is None:
                    raise ValueError("no date parsed")
                ts = strip_tzinfo(mail.date)
            except (OverflowError, ValueError) as exc:
                malformed_drops += 1
                log.warning("NO_DATE  %s  |  %s", fpath, exc)
                continue

            # Message-ID 
            msg_id = (mail.message_id or "").strip()
            if msg_id:
                seen_msg_ids.add(msg_id)

            # Multi-recipient expansion counter (before filtering)
            # This will add 2+ million rows
            if len(recipients) > 1:
                multi_recipient_expansions += len(recipients) - 1

            # Per-recipient rows 
            for recipient in recipients:

                # self-loop
                if sender == recipient:
                    self_loop_drops += 1
                    continue

                # dedup: same (msg_id, recipient) or same (sender, recipient, ts)
                row_key = (
                    f"{msg_id}::{recipient}"
                    if msg_id
                    else f"{sender}|{recipient}|{ts.isoformat()}"
                )
                if row_key in seen_row_keys:
                    continue
                seen_row_keys.add(row_key)

                rows.append({
                    "sender":    sender,
                    "recipient": recipient,
                    "timestamp": ts,
                })

# Write CSV
df = pd.DataFrame(rows, columns=["sender", "recipient", "timestamp"])
df.to_csv(PARSED_CSV, index=False)

# Final counters 
unique_message_ids = len(seen_msg_ids)
output_row_count   = len(df)

log.info("=" * 60)
log.info("STAGE 1 COMPLETE")
log.info("  raw_file_count             : %d", raw_file_count)
log.info("  unique_message_ids         : %d", unique_message_ids)
log.info("  malformed_drops            : %d", malformed_drops)
log.info("  self_loop_drops            : %d", self_loop_drops)
log.info("  multi_recipient_expansions : %d", multi_recipient_expansions)
log.info("  output_row_count           : %d", output_row_count)
log.info("  output   → %s", PARSED_CSV)
log.info("  drop log → %s", _log_path)
log.info("=" * 60)