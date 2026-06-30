"""Review-queue layer: bridges the matcher's output to the enrichment agent, and
commits confirmed decisions back into the learning data (the feedback loop).

The deterministic matcher (matcher.py) ranks IDs and flags low-confidence rows for
review. This module:
  1. parse_candidates  — turns a row's `Top_Matches` string into the candidate ID
     list the agent is allowed to choose from.
  2. run_review        — runs EnrichmentAgent over the Needs_Review rows only,
     attaching the agent's proposal/evidence as new columns.
  3. commit_decision   — appends a human-confirmed (row -> Recharging_Item_ID)
     mapping to GO_MAPPING_LEARNING with an audit stamp, so the next pipeline run
     retrains on it. This is what makes the system improve each cycle.

No function here ever auto-writes a Recharging_Item_ID into the prediction output —
a human Accept/Override is always the trigger for commit_decision.
"""

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agent import EnrichmentAgent
from matcher import EMPTY_COLS, LEARNING_COLS

LEARNING_SHEET = "GO_MAPPING_LEARNING"
# A candidate token looks like "PSO_ITM_1214 (0.41)" or "PSO_ITM_1214 (exact name+RG)".
# We want the ID, i.e. everything before the first " (".
_CANDIDATE = re.compile(r"^\s*(.+?)\s*\(")


def parse_candidates(top_matches: str) -> list[str]:
    """Extract the candidate Recharging_Item_IDs from a matcher `Top_Matches` cell.

    Handles both the classifier format ("ID (0.41) | ID (0.22) | ID (0.10)") and the
    exact-match format ("ID (exact name+RG)"). Order and dedup are preserved.
    """
    ids: list[str] = []
    for token in str(top_matches or "").split("|"):
        m = _CANDIDATE.match(token)
        cand = (m.group(1) if m else token).strip()
        if cand and cand not in ids:
            ids.append(cand)
    return ids


def _row_to_agent_input(row: pd.Series) -> dict:
    """Map a GO_MAPPING_EMPTY row to the dict shape EnrichmentAgent.investigate wants."""
    def field(col_key: str) -> str:
        v = str(row.get(EMPTY_COLS[col_key], "") or "").strip()
        return "" if v.lower() == "nan" else v

    return {
        "name": field("sub_account_name"),
        "resource_group": field("resource_group"),
        "tag_dcs": field("tag_dcs"),
        "tag_app": field("tag_app"),
        "sub_account_id": field("sub_account_id"),
    }


# Columns the agent contributes to the results frame, with their blank defaults.
# Kept as object dtype so a row can hold a str ID, an int confidence or a bool.
AGENT_COL_DEFAULTS = {"Agent_Proposed_ID": "", "Agent_Confidence": 0,
                      "Agent_Needs_Human": False, "Agent_Reasoning": "",
                      "Agent_Evidence": ""}
AGENT_COLS = list(AGENT_COL_DEFAULTS)


def run_review(results: pd.DataFrame, agent: EnrichmentAgent | None = None,
               max_rows: int | None = None, progress=None) -> pd.DataFrame:
    """Run the enrichment agent over the Needs_Review rows, constrained to each row's
    own Top_Matches candidates. Returns a copy of `results` with AGENT_COLS attached
    (blank on rows that were not reviewed).

    `progress(done, total)` is an optional callback for UI progress bars.
    """
    out = results.copy()
    for col, default in AGENT_COL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = pd.Series([default] * len(out), index=out.index, dtype=object)

    review_idx = out.index[out["Needs_Review"]].tolist()
    if max_rows is not None:
        review_idx = review_idx[:max_rows]
    if not review_idx:
        return out

    agent = agent or EnrichmentAgent()
    for n, idx in enumerate(review_idx, 1):
        row = out.loc[idx]
        candidates = parse_candidates(row.get("Top_Matches", ""))
        proposal = agent.investigate(_row_to_agent_input(row), candidates)
        out.at[idx, "Agent_Proposed_ID"] = proposal.get("recommended_id") or ""
        out.at[idx, "Agent_Confidence"] = proposal.get("confidence", 0)
        out.at[idx, "Agent_Needs_Human"] = bool(proposal.get("needs_human", True))
        out.at[idx, "Agent_Reasoning"] = proposal.get("reasoning", "")
        out.at[idx, "Agent_Evidence"] = " ; ".join(proposal.get("evidence", []) or [])
        if progress:
            progress(n, len(review_idx))
    return out


# ── Feedback loop: commit a confirmed decision back to the learning data ──────
AUDIT_COLS = {"reviewed_by": "Reviewed_By", "reviewed_at": "Reviewed_At",
              "source": "Review_Source"}


def commit_decision(row: pd.Series, recharging_item_id: str, reviewed_by: str,
                    workbook: str, source: str = "human_review") -> None:
    """Append a confirmed (row -> Recharging_Item_ID) mapping to GO_MAPPING_LEARNING.

    Writes into the same workbook so the next `build_index()` trains on it. A one-time
    .BACKUP copy is made the first time we touch the file, and audit columns record
    who/when/how. All other sheets are preserved verbatim.
    """
    wb = Path(workbook)
    backup = wb.with_suffix(".BACKUP" + wb.suffix)
    if not backup.exists():
        shutil.copy2(wb, backup)

    sheets = pd.read_excel(wb, sheet_name=None)
    learning = sheets[LEARNING_SHEET]

    new = {
        LEARNING_COLS["sub_account_name"]: str(row.get(EMPTY_COLS["sub_account_name"], "") or ""),
        LEARNING_COLS["sub_account_id"]: str(row.get(EMPTY_COLS["sub_account_id"], "") or ""),
        LEARNING_COLS["resource_group"]: str(row.get(EMPTY_COLS["resource_group"], "") or ""),
        LEARNING_COLS["tag_dcs"]: str(row.get(EMPTY_COLS["tag_dcs"], "") or ""),
        LEARNING_COLS["tag_app"]: str(row.get(EMPTY_COLS["tag_app"], "") or ""),
        LEARNING_COLS["recharging_item_id"]: recharging_item_id,
        AUDIT_COLS["reviewed_by"]: reviewed_by,
        AUDIT_COLS["reviewed_at"]: datetime.now(timezone.utc).isoformat(timespec="seconds"),
        AUDIT_COLS["source"]: source,
    }
    sheets[LEARNING_SHEET] = pd.concat([learning, pd.DataFrame([new])], ignore_index=True)

    with pd.ExcelWriter(wb, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


if __name__ == "__main__":
    # Offline smoke test for the parser — no Bedrock, no Excel writes.
    for s in ["PSO_ITM_1214 (0.41) | PSO_ITM_429 (0.22) | XX_GOSHARED (0.10)",
              "PSO_ITM_530 (exact name+RG)", ""]:
        print(repr(s), "->", parse_candidates(s))
