# Data Dictionary — enron_parsed.csv

**Stage:** Stage 1 (Parse)  
**Thesis:** 3.3.1 (Stage 1 Parse: Graph Model Design)  
**Location:** `data/01_Interim/enron_parsed.csv`  
**Row semantics:** One row per unique sender → recipient message (multi-recipient emails expanded to N rows).

| Column      | Type     | Example                          | Description                                                                 |
|-------------|----------|----------------------------------|-----------------------------------------------------------------------------|
| sender      | str      | `john.doe@enron.com`             | Normalised (lowercase, stripped) email address from the `From:` header.     |
| recipient   | str      | `jane.smith@enron.com`           | Normalised email address from the `To:` header. One recipient per row.      |
| timestamp   | datetime | `2001-03-15 09:42:00`            | Parsed from the `Date:` header via `dateutil.parser`; timezone-info stripped.|

**Filters applied (in order):**
1. Malformed files that cannot be parsed are dropped and logged.
2. Records with missing or unparseable sender, recipient, or timestamp are dropped.
3. All addresses normalised to lowercase with whitespace stripped.
4. Deduplicated by Message-ID (or composite key sender+recipient+timestamp if absent).
5. Multi-recipient `To:` expanded to one row per recipient.
6. Self-loops (sender == recipient) dropped.
