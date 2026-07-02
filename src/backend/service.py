"""Backend service layer — the first step of the front/back split.

Wraps the deterministic matcher + the LLM review behind a small, UI-agnostic API that
the FastAPI routes (api.py) call. No Streamlit here. Everything the Streamlit app used to
do in-session now lives server-side:

    run_batch    → train the matcher on history + predict this month's accounts
    rows         → the predictions, split into ready-to-approve / to-review buckets
    start_review → analyse the review rows with the LLM (background thread, polled)
    commit       → append confirmed decisions to the learning store (feedback loop)

State is an in-memory dict of batches for now. Phase 1 replaces `_BATCHES` and the Excel
`commit_decisions` write with a database — the route/signature contracts stay the same.
"""

import io
import threading
import uuid

import pandas as pd
from matcher import ACTION_AGENT, ACTION_APPROVE, EMPTY_COLS, RechargingMatcher
from review import (
    AGENT_COL_DEFAULTS,
    AUDIT_COLS,
    commit_decisions,
    recall_decision,
    run_review,
)

WORKBOOK = "GO Report Extract GROUPED_V3.xlsx"
APPROVE_THRESHOLD = 50.0
ACTION_DONE = "Approved ✓"


class Batch:
    """One classified batch held in memory: the trained matcher, the prediction frame,
    a lock guarding commits, and the current LLM review job (or None)."""

    def __init__(self, matcher: RechargingMatcher, results: pd.DataFrame):
        self.matcher = matcher
        self.results = results
        self.lock = threading.Lock()
        self.job: dict | None = None
        self.approvals: list[dict] = []  # committed decisions this batch (for history/recall)


_BATCHES: dict[str, Batch] = {}


def _cost(df: pd.DataFrame) -> pd.Series:
    col = EMPTY_COLS["cost"]
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def run_batch(learning_bytes: bytes | None = None, empty_bytes: bytes | None = None) -> str:
    """Train the matcher on GO_MAPPING_LEARNING and predict GO_MAPPING_EMPTY. Returns a
    new batch_id. With no uploaded bytes, uses the bundled workbook."""
    lsrc = io.BytesIO(learning_bytes) if learning_bytes else WORKBOOK
    esrc = io.BytesIO(empty_bytes) if empty_bytes else WORKBOOK
    matcher = RechargingMatcher()
    matcher.build_index(pd.read_excel(lsrc, sheet_name="GO_MAPPING_LEARNING"))
    results = matcher.predict(pd.read_excel(esrc, sheet_name="GO_MAPPING_EMPTY"),
                              confidence_threshold=APPROVE_THRESHOLD)
    # Seed empty agent columns so review rows are describable before the LLM runs.
    for col, default in AGENT_COL_DEFAULTS.items():
        results[col] = pd.Series([default] * len(results), index=results.index, dtype=object)
    batch_id = uuid.uuid4().hex[:12]
    _BATCHES[batch_id] = Batch(matcher, results)
    return batch_id


def get_batch(batch_id: str) -> Batch:
    batch = _BATCHES.get(batch_id)
    if batch is None:
        raise KeyError(batch_id)
    return batch


def review_status(batch_id: str) -> dict:
    """Progress of the LLM review job for a batch."""
    job = get_batch(batch_id).job
    if not job:
        return {"running": False, "done": 0, "total": 0, "error": None}
    return {"running": job["running"], "done": job["done"],
            "total": job["total"], "error": job["error"]}


def summary(batch_id: str) -> dict:
    """Batch-level KPIs (row counts + € coverage) plus the review job status."""
    r = get_batch(batch_id).results
    active = r[r["Suggested_Action"] != ACTION_DONE]
    cost = _cost(active)
    approve = active["Suggested_Action"] == ACTION_APPROVE
    review = active["Suggested_Action"] == ACTION_AGENT
    return {
        "batch_id": batch_id,
        "total_rows": int(len(active)),
        "ready_to_approve": int(approve.sum()),
        "to_review": int(review.sum()),
        "approved": int((r["Suggested_Action"] == ACTION_DONE).sum()),
        "total_spend_eur": round(float(cost.sum()), 2),
        "approve_ready_spend_eur": round(float(cost[approve].sum()), 2),
        "review_spend_eur": round(float(cost[review].sum()), 2),
        "review": review_status(batch_id),
    }


def _row_dict(idx, row: pd.Series) -> dict:
    def field(key: str) -> str:
        v = row.get(EMPTY_COLS[key], "")
        return "" if pd.isna(v) else str(v)

    def num(v) -> float:
        return float(pd.to_numeric(v, errors="coerce") or 0)

    return {
        "row_id": int(idx),
        "cost_eur": num(row.get(EMPTY_COLS["cost"], 0)),
        "sub_account_name": field("sub_account_name"),
        "resource_group": field("resource_group"),
        "tag_dcs": field("tag_dcs"),
        "tag_app": field("tag_app"),
        "predicted_recharging_item_id": str(row.get("Predicted_Recharging_Item_ID", "") or ""),
        "confidence": num(row.get("Confidence", 0)),
        "suggested_action": str(row.get("Suggested_Action", "") or ""),
        "agent_prediction": str(row.get("Agent_Proposed_ID", "") or ""),
        "agent_confidence": num(row.get("Agent_Confidence", 0)),
        "agent_explanation": str(row.get("Agent_Reasoning", "") or ""),
        "agent_tokens": int(num(row.get("Agent_Tokens", 0))),
    }


def rows(batch_id: str, bucket: str = "all") -> list[dict]:
    """Predictions for a batch, € descending. bucket = all | approve | review | done."""
    r = get_batch(batch_id).results
    mask = {
        "approve": r["Suggested_Action"] == ACTION_APPROVE,
        "review": r["Suggested_Action"] == ACTION_AGENT,
        "done": r["Suggested_Action"] == ACTION_DONE,
        "all": r["Suggested_Action"] != ACTION_DONE,
    }.get(bucket, r["Suggested_Action"] != ACTION_DONE)
    sub = r[mask].assign(_c=_cost(r[mask])).sort_values("_c", ascending=False)
    return [_row_dict(idx, row) for idx, row in sub.iterrows()]


def start_review(batch_id: str) -> dict:
    """Analyse the review rows with the LLM in a background thread. Idempotent while a job
    is already running. Returns the current job status."""
    batch = get_batch(batch_id)
    r = batch.results
    review_idx = r.index[r["Suggested_Action"] == ACTION_AGENT].tolist()
    if not review_idx:
        return review_status(batch_id)
    if batch.job and batch.job["running"]:
        return review_status(batch_id)

    job = {"done": 0, "total": len(review_idx), "running": True, "error": None}
    batch.job = job

    def worker():
        try:
            out = run_review(r, batch.matcher,
                             progress=lambda d, t: job.update(done=d, total=t))
            with batch.lock:  # merge only the agent columns, preserving any commits made meanwhile
                agent_cols = [c for c in AGENT_COL_DEFAULTS if c in out.columns]
                common = batch.results.index.intersection(out.index)
                batch.results.loc[common, agent_cols] = out.loc[common, agent_cols]
        except Exception as e:  # noqa: BLE001 — surfaced via job["error"]
            job["error"] = str(e)
        finally:
            job["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return review_status(batch_id)


def commit(batch_id: str, decisions: list[dict], reviewed_by: str = "api") -> dict:
    """Commit confirmed (row_id -> recharging_item_id) decisions: append to the learning
    store, mark the rows done, and record them for history/recall. `decisions` items:
    {"row_id": int, "recharging_item_id": str}."""
    batch = get_batch(batch_id)
    with batch.lock:
        r = batch.results
        pending = []  # (idx, row, rid, prev_action, prev_review)
        for d in decisions:
            idx, rid = d.get("row_id"), str(d.get("recharging_item_id", "") or "").strip()
            if idx in r.index and rid:
                pending.append((idx, r.loc[idx], rid,
                                str(r.at[idx, "Suggested_Action"]),
                                bool(r.at[idx, "Needs_Review"])))
        records = commit_decisions([(row, rid) for _, row, rid, _, _ in pending],
                                   reviewed_by, WORKBOOK)
        for (idx, row, rid, prev_action, prev_review), rec in zip(pending, records):
            r.at[idx, "Predicted_Recharging_Item_ID"] = rid
            r.at[idx, "Needs_Review"] = False
            r.at[idx, "Suggested_Action"] = ACTION_DONE
            batch.approvals.append({
                "approval_id": uuid.uuid4().hex[:12],
                "row_id": int(idx),
                "name": str(row.get(EMPTY_COLS["sub_account_name"], f"row {idx}") or f"row {idx}"),
                "recharging_item_id": rid,
                "reviewed_at": rec.get(AUDIT_COLS["reviewed_at"], ""),
                "source": rec.get(AUDIT_COLS["source"], ""),
                "record": rec, "prev_action": prev_action, "prev_review": prev_review,
            })
    return {"committed": len(records), "skipped": len(decisions) - len(records)}


def reroute(batch_id: str, row_ids: list[int]) -> dict:
    """Move rows (e.g. rejected from Ready-to-approve) back into the Review queue."""
    batch = get_batch(batch_id)
    with batch.lock:
        r = batch.results
        moved = 0
        for idx in row_ids:
            if idx in r.index:
                r.at[idx, "Needs_Review"] = True
                r.at[idx, "Suggested_Action"] = ACTION_AGENT
                moved += 1
    return {"rerouted": moved}


def history(batch_id: str) -> list[dict]:
    """Decisions committed for this batch (most recent first), for the History view."""
    approvals = get_batch(batch_id).approvals
    keys = ("approval_id", "row_id", "name", "recharging_item_id", "reviewed_at", "source")
    return [{k: a[k] for k in keys} for a in reversed(approvals)]


def recall(batch_id: str, approval_id: str) -> dict:
    """Undo a committed decision: remove it from the learning store and restore the row's
    pre-approval routing so it re-opens for review. Raises KeyError if not found."""
    batch = get_batch(batch_id)
    with batch.lock:
        pos = next((i for i, a in enumerate(batch.approvals)
                    if a["approval_id"] == approval_id), None)
        if pos is None:
            raise KeyError(approval_id)
        appr = batch.approvals.pop(pos)
        removed = recall_decision(WORKBOOK, appr["record"])
        idx = appr["row_id"]
        if idx in batch.results.index:
            batch.results.at[idx, "Needs_Review"] = appr["prev_review"]
            batch.results.at[idx, "Suggested_Action"] = appr["prev_action"]
    return {"recalled": True, "learning_row_removed": removed}
