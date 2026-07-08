"""Local feedback store for confirmed decisions — the Excel-free learning loop.

When a human approves/corrects a prediction, the mapping must be remembered so the next
pipeline run retrains on it. The data now comes from Databricks (read-only), so instead
of writing back to a workbook we append the confirmed rows to a small append-only CSV
here. data_source.fetch_learning_and_empty() merges this file onto the Databricks
learning rows, closing the feedback loop without any .xlsx.

Columns are the canonical matcher names (see matcher.COLS) plus the audit stamp, so an
appended record is a drop-in learning row.

Env var:
    LEARNING_STORE_PATH   CSV location (default: ./learning_store.csv in the CWD)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from matcher import COLS

# Audit columns recorded alongside each confirmed decision (used by review._learning_record).
REVIEWED_BY = "Reviewed_By"
REVIEWED_AT = "Reviewed_At"
REVIEW_SOURCE = "Review_Source"

# The full column set of a stored record, in a stable order.
STORE_COLS = [
    COLS["sub_account_name"], COLS["sub_account_id"], COLS["resource_group"],
    COLS["tag_dcs"], COLS["tag_app"], COLS["recharging_item_id"],
    REVIEWED_BY, REVIEWED_AT, REVIEW_SOURCE,
]


def _path() -> Path:
    return Path(os.getenv("LEARNING_STORE_PATH", "learning_store.csv"))


def load() -> pd.DataFrame:
    """All committed decisions as a DataFrame (empty with STORE_COLS if none yet)."""
    p = _path()
    if not p.exists():
        return pd.DataFrame(columns=STORE_COLS)
    df = pd.read_csv(p, dtype=str).fillna("")
    for col in STORE_COLS:  # tolerate an older/narrower file
        if col not in df.columns:
            df[col] = ""
    return df[STORE_COLS]


def append(records: list[dict]) -> list[dict]:
    """Append confirmed learning records to the store. Returns the records appended."""
    if not records:
        return []
    df = load()
    df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)[STORE_COLS]
    df.to_csv(_path(), index=False)
    return records


def remove(record: dict) -> bool:
    """Remove a previously appended record (identified by its audit stamp + id +
    account). Removes at most one matching row; returns whether one was found."""
    p = _path()
    if not p.exists():
        return False
    df = load()
    mask = pd.Series(True, index=df.index)
    for col in (REVIEWED_AT, REVIEWED_BY, COLS["recharging_item_id"],
                COLS["sub_account_name"]):
        mask &= df[col].astype(str) == str(record.get(col, ""))
    if not mask.any():
        return False
    df = df.drop(index=df.index[mask][0]).reset_index(drop=True)
    df.to_csv(p, index=False)
    return True
