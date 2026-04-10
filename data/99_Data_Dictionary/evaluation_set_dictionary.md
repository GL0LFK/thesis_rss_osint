# Data Dictionary — evaluation_set.csv

**Stage:** Stage 4 (Inject)  
**Thesis:** 3.5.1 (Stage 4 Inject: Injection Methodology)  
**Location:** `data/02_Processed/evaluation_set.csv`  
**Row semantics:** One row per email (legitimate or synthetically injected spear-phishing).

| Column           | Type  | Values / Range                        | Description                                                                                     |
|------------------|-------|---------------------------------------|-------------------------------------------------------------------------------------------------|
| sender           | str   | email or `"Display Name <addr>"`      | Sender address. Legitimate rows use bare addresses; injected rows use display-name format.       |
| recipient        | str   | email address                         | Target recipient address.                                                                        |
| timestamp        | datetime | `2001-01-01 00:00:00` + offset      | Synthetic placeholder timestamps (not from original headers).                                    |
| rss              | float | [0, 100]                              | Relationship Strength Score. 0 for Low/Medium injections; real RSS for High injections.          |
| header_auth_pass | bool  | True / False                          | Simulated SPF/DKIM/DMARC result. True for all legitimate rows.                                   |
| y                | int   | {0, 1}                                | Ground-truth label. 0 = legitimate, 1 = injected spear-phishing.                                 |
| osint_level      | str   | `Legit`, `Low`, `Medium`, `High`      | OSINT preparedness level of the injection; `Legit` for legitimate rows.                           |

**OSINT level properties:**

| Level  | RSS   | header_auth_pass       | Sender identity                          |
|--------|-------|------------------------|------------------------------------------|
| Legit  | real  | True                   | Original sender from the graph.           |
| Low    | 0     | False                  | Random external address, freemail domain. |
| Medium | 0     | Bernoulli(p=0.62)      | Impersonated employee, freemail domain.   |
| High   | real  | True                   | Real trusted sender address, real RSS.    |
