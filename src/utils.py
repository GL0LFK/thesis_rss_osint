"""
Shared deterministic helpers used across pipeline stages
No side effects. No I/O. No state

Related Thesis Section: 3.2 (System Design and Data Preparation)
"""

# Lowercase and strip whitespace from an email address
def normalise_address(addr: str) -> str:
    return addr.strip().lower()


def strip_tzinfo(dt):
    """
    Return a timezone-naive copy of dt
    Enron headers are mixed-zone, because all timestamps are kept naive throughout
    the pipeline to satisfy the determinism constraint
    Returns None if dt is None
    """
    if dt is None:
        return None
    return dt.replace(tzinfo=None)