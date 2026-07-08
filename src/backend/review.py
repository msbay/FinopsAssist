"""Review-queue layer: bridges the matcher's output to the FinOps assistant, and
commits confirmed decisions back into the learning data (the feedback loop).

The deterministic matcher (matcher.py) ranks IDs and flags low-confidence rows for
review. This module:
  1. run_review        — runs FinopsAssistant over the Needs_Review rows only,
     attaching the agent's proposal/evidence as new columns.
  2. commit_decisions  — appends human-confirmed (row -> Recharging_Item_ID) mappings
     to the local learning store (learning_store.py) with an audit stamp, so the next
     pipeline run retrains on them. This is what makes the system improve each cycle.
  3. recall_decision   — undoes a committed decision.

No function here ever auto-writes a Recharging_Item_ID into the prediction output —
a human Accept/Override is always the trigger for a commit.
"""

import re
from datetime import datetime, timezone

import learning_store
import pandas as pd
from agent import FinopsAssistant
from matcher import (
    ACTION_AGENT,
    EMPTY_COLS,
    LEARNING_COLS,
    PROVIDER_AWS,
    RechargingMatcher,
)

# Minimum € spend for a review row to be worth an LLM call. Rows at/below this are left
# for a human (no agent suggestion) so tokens aren't spent investigating trivial spend.
MIN_LLM_COST_EUR = 50.0


def _row_to_agent_input(row: pd.Series) -> dict:
    """Map a GO_MAPPING_EMPTY row to the dict shape FinopsAssistant.investigate wants.

    Uses the matcher's provider-aware extraction so AWS rows carry their V5 enrichment
    (owner / dcs / description / name) — the same fields the classifier sees — and Azure
    rows do not. The shared axa tags are also withheld from AWS rows (blanked), so the
    LLM sees exactly what the AWS classifier does; Azure still passes them.
    """
    fields, bucket = RechargingMatcher._extract(row, EMPTY_COLS)
    is_aws = bucket == PROVIDER_AWS
    sub_id = str(row.get(EMPTY_COLS["sub_account_id"], "") or "").strip()
    return {
        "provider": str(row.get(EMPTY_COLS["provider"], "") or "").strip(),
        "name": fields["name"],
        "resource_group": fields["resource_group"],
        "tag_dcs": "" if is_aws else fields["tag_dcs"],
        "tag_app": "" if is_aws else fields["tag_app"],
        "sub_account_id": "" if sub_id.lower() == "nan" else sub_id,
        "aws_owner": fields["aws_owner"],
        "aws_dcs": fields["aws_dcs"],
        "aws_desc": fields["aws_desc"],
        "aws_name": fields["aws_name"],
    }


# Columns the agent contributes to the results frame, with their blank defaults.
# Kept as object dtype so a row can hold a str ID, an int confidence or a bool.
AGENT_COL_DEFAULTS = {"Agent_Proposed_ID": "", "Agent_Confidence": 0,
                      "Agent_Needs_Human": False, "Agent_Reasoning": "",
                      "Agent_Evidence": "", "Agent_Tokens": 0}


_SCORED = re.compile(r"^\s*(.+?)\s*\(([0-9.]+)\)\s*$")


def _parse_scored(candidates) -> list[tuple[str, float | None]]:
    """Parse a 'id (0.12) | id (0.10)' string (or a plain id list) into (id, prob)
    pairs, preserving order (the classifier's confidence ranking)."""
    if not isinstance(candidates, str):
        return [(c, None) for c in candidates]
    out = []
    for token in candidates.split("|"):
        m = _SCORED.match(token)
        if m:
            out.append((m.group(1).strip(), float(m.group(2))))
        elif token.strip():
            out.append((token.strip(), None))
    return out


def _named_candidates(candidates, matcher) -> list[dict]:
    """Attach rank, probability and the semantic tree (Family > Product > Item name)
    to each candidate, using the matcher's learned hierarchy. `candidates` is the
    ranked 'id (prob) | ...' string (or a plain id list); order = classifier ranking."""
    cands = []
    for rank, (cid, prob) in enumerate(_parse_scored(candidates), 1):
        fam, prod, name = matcher._ancestry(cid) if matcher is not None else ("", "", cid)
        cands.append({"id": cid, "name": name, "family": fam, "product": prod,
                      "prob": prob, "rank": rank})
    return cands


def run_review(results: pd.DataFrame, matcher=None, agent: FinopsAssistant | None = None,
               max_rows: int | None = None, progress=None, only_idx=None) -> pd.DataFrame:
    """Run the FinOps assistant over the Needs_Review rows, constrained to each row's
    own Top_Matches candidates. Returns a copy of `results` with AGENT_COLS attached
    (blank on rows that were not reviewed).

    `matcher` supplies the learned Family/Product/name tree so the agent reasons over
    semantic names rather than opaque ids. `progress(done, total)` is an optional
    callback for UI progress bars. `only_idx`, if given, restricts the run to that
    explicit set of row indices (the rows a user selected in the Review Queue) — still
    intersected with the eligible "Send to agent" set so no-signal rows are never wasted.
    """
    out = results.copy()
    for col, default in AGENT_COL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = pd.Series([default] * len(out), index=out.index, dtype=object)

    # Route by the validated ladder: only "Send to agent" rows go to the LLM (low
    # confidence WITH signal). High-confidence rows are for human approval; no-signal
    # rows have nothing to reason over and need owner/enrichment — the agent skips both.
    if "Suggested_Action" in out.columns:
        eligible = out["Suggested_Action"] == ACTION_AGENT
    else:  # older frames without the column: fall back to the equivalent predicate
        eligible = out["Needs_Review"]
    if only_idx is not None:  # honour an explicit user selection, but stay within eligible
        eligible = eligible & out.index.isin(list(only_idx))
    review_idx = out.index[eligible].tolist()
    # Skip low-spend rows entirely: the LLM tokens aren't justified below MIN_LLM_COST_EUR.
    # Such rows stay in the review queue for a human (no agent suggestion). Then prioritise
    # the rest by € spend so a capped run (max_rows) spends its budget where the money is.
    cost_col = EMPTY_COLS["cost"]
    if cost_col in out.columns:
        cost = pd.to_numeric(out[cost_col], errors="coerce").fillna(0)
        review_idx = [i for i in review_idx if cost.at[i] > MIN_LLM_COST_EUR]
        review_idx.sort(key=lambda i: cost.at[i], reverse=True)
    if max_rows is not None:
        review_idx = review_idx[:max_rows]
    if not review_idx:
        return out

    agent = agent or FinopsAssistant()
    for n, idx in enumerate(review_idx, 1):
        row = out.loc[idx]
        # No-signal rows are in the queue too, but there's nothing for the agent to reason
        # over — mark them for a human WITHOUT spending an LLM call (uniform card, 0 tokens).
        if str(row.get("Match_Method", "")) == "no_signal":
            out.at[idx, "Agent_Proposed_ID"] = ""
            out.at[idx, "Agent_Confidence"] = 0
            out.at[idx, "Agent_Needs_Human"] = True
            out.at[idx, "Agent_Reasoning"] = ("No signal (opaque/empty name, no resource "
                                              "group or tags) — needs a human / owner.")
            out.at[idx, "Agent_Evidence"] = ""
            out.at[idx, "Agent_Tokens"] = 0
            if progress:
                progress(n, len(review_idx))
            continue
        # Prefer the wider, high-recall nucleus set (Candidate_IDs); fall back to Top_Matches.
        cand_src = row.get("Candidate_IDs") or row.get("Top_Matches", "")
        candidates = _named_candidates(cand_src, matcher)
        proposal = agent.investigate(_row_to_agent_input(row), candidates)
        # Always surface a best guess: if the agent abstained, fall back to the top-ranked
        # candidate (the classifier's #1) so the prediction cell is never empty.
        pred = proposal.get("recommended_id") or ""
        if not pred and candidates:
            pred = candidates[0].get("id", "")
        out.at[idx, "Agent_Proposed_ID"] = pred
        out.at[idx, "Agent_Confidence"] = proposal.get("confidence", 0)
        out.at[idx, "Agent_Needs_Human"] = bool(proposal.get("needs_human", True))
        out.at[idx, "Agent_Reasoning"] = proposal.get("reasoning", "")
        out.at[idx, "Agent_Evidence"] = " ; ".join(proposal.get("evidence", []) or [])
        out.at[idx, "Agent_Tokens"] = int(proposal.get("total_tokens", 0))
        # Rows stay in the Review queue (the admin approves the agent's prediction there);
        # run_review only fills the Agent_* columns, it never re-routes the row.
        if progress:
            progress(n, len(review_idx))
    return out


# ── Feedback loop: commit a confirmed decision to the local learning store ────
def _learning_record(row: pd.Series, recharging_item_id: str, reviewed_by: str,
                     source: str, at: str) -> dict:
    """One learning-store row (canonical columns + audit stamp) for a confirmed decision."""
    return {
        LEARNING_COLS["sub_account_name"]: str(row.get(EMPTY_COLS["sub_account_name"], "") or ""),
        LEARNING_COLS["sub_account_id"]: str(row.get(EMPTY_COLS["sub_account_id"], "") or ""),
        LEARNING_COLS["resource_group"]: str(row.get(EMPTY_COLS["resource_group"], "") or ""),
        LEARNING_COLS["tag_dcs"]: str(row.get(EMPTY_COLS["tag_dcs"], "") or ""),
        LEARNING_COLS["tag_app"]: str(row.get(EMPTY_COLS["tag_app"], "") or ""),
        LEARNING_COLS["recharging_item_id"]: recharging_item_id,
        learning_store.REVIEWED_BY: reviewed_by,
        learning_store.REVIEWED_AT: at,
        learning_store.REVIEW_SOURCE: source,
    }


def commit_decisions(decisions, reviewed_by: str,
                     source: str = "human_review") -> list[dict]:
    """Append several confirmed (row -> Recharging_Item_ID) mappings to the local
    learning store (learning_store.py) in a single write (for dashboard batch-approve).

    `decisions` is an iterable of (row: pd.Series, recharging_item_id: str). Rows with a
    blank id are skipped (nothing to learn). Audit columns record who/when/how. Returns
    the list of appended learning records (each carries the Reviewed_At stamp used to
    recall it). The next fetch_learning_and_empty() merges these onto the Databricks
    learning rows, so the model retrains on them.
    """
    records, at = [], datetime.now(timezone.utc).isoformat(timespec="seconds")
    for row, rid in decisions:
        rid = str(rid or "").strip()
        if rid:
            records.append(_learning_record(row, rid, reviewed_by, source, at))
    return learning_store.append(records)


def recall_decision(record: dict) -> bool:
    """Undo a previously committed decision: remove the matching row from the learning
    store (identified by its audit stamp + id + account). Removes at most one row and
    returns whether a match was found."""
    return learning_store.remove(record)
