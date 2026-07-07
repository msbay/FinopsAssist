"""FinOps Assistant — HTTP API (FastAPI).

The backend of the front/back split. It exposes the cost-allocation pipeline as a REST
API so any client (the Streamlit console today, a React app tomorrow, or a scheduled job)
can drive it. Business logic lives in service.py; this module is only routing + schemas.

Run (from the project root, so the workbook path resolves):
    uvicorn api:app --app-dir src/backend --reload
Docs:
    http://127.0.0.1:8000/docs
"""

import logging
import os
import sys
from secrets import compare_digest

# Put this module's own dir on the path so the sibling engine modules (service, matcher,
# review, agent, ...) import by bare name regardless of how uvicorn is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import service
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

# Load .env so it configures the backend in local dev (prod injects real env vars, which
# take precedence — load_dotenv does not override an already-set variable).
load_dotenv()

logger = logging.getLogger("finops.api")

# ── Authentication ────────────────────────────────────────────────────────────
# A shared SERVICE token for the frontend (the Next.js BFF injects it on every call).
# User/SSO auth is handled upstream by the frontend; this only proves the caller is the
# trusted frontend, so the internal API can't be driven by anything else.
#   • FINOPS_API_TOKEN unset  → auth DISABLED (local dev convenience).
#   • FINOPS_API_TOKEN set     → every request except the open paths must send
#                                `Authorization: Bearer <token>`.
API_TOKEN = os.getenv("FINOPS_API_TOKEN")
_OPEN_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

if not API_TOKEN:
    logger.warning("FINOPS_API_TOKEN is not set — API authentication is DISABLED. "
                   "Fine for local dev; this MUST be set in production.")


def require_token(request: Request, authorization: str | None = Header(default=None)) -> None:
    """Validate the shared service token (constant-time) on every protected route.

    No-op when FINOPS_API_TOKEN is unset (dev) or for the open paths (health + API docs).
    """
    if not API_TOKEN or request.url.path in _OPEN_PATHS:
        return
    if not authorization or not compare_digest(authorization, f"Bearer {API_TOKEN}"):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


app = FastAPI(
    title="FinOps Assistant API",
    version="0.1.0",
    summary="Map new cloud resources to a Recharging Item — classifier + LLM review.",
    dependencies=[Depends(require_token)],  # applies to every route; open paths exempt above
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ReviewStatus(BaseModel):
    running: bool
    done: int
    total: int
    error: str | None = None


class BatchSummary(BaseModel):
    batch_id: str
    total_rows: int
    ready_to_approve: int
    to_review: int
    approved: int
    total_spend_eur: float
    approve_ready_spend_eur: float
    review_spend_eur: float
    review: ReviewStatus


class Row(BaseModel):
    row_id: int
    cost_eur: float
    provider: str = ""
    sub_account_name: str
    resource_group: str
    tag_dcs: str
    tag_app: str
    # AWS enrichment (blank for Azure) — lets the frontend show AWS vs Azure tables with
    # the columns that apply to each.
    aws_owner: str = ""
    aws_dcs: str = ""
    aws_desc: str = ""
    aws_name: str = ""
    predicted_recharging_item_id: str
    confidence: float
    suggested_action: str
    agent_prediction: str
    agent_confidence: float
    agent_explanation: str
    agent_tokens: int = 0


class Decision(BaseModel):
    row_id: int
    recharging_item_id: str = Field(min_length=1)


class CommitRequest(BaseModel):
    reviewed_by: str = "api"
    decisions: list[Decision]


class CommitResult(BaseModel):
    committed: int
    skipped: int


class RerouteRequest(BaseModel):
    row_ids: list[int]


class HistoryEntry(BaseModel):
    approval_id: str
    row_id: int
    name: str
    recharging_item_id: str
    reviewed_at: str
    source: str


class RecallRequest(BaseModel):
    approval_id: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/batches", response_model=BatchSummary, status_code=201)
async def create_batch(file: UploadFile | None = File(default=None)) -> dict:
    """Run the pipeline: train on history + predict this month's accounts. Optionally
    upload an .xlsx (with GO_MAPPING_LEARNING + GO_MAPPING_EMPTY); otherwise the bundled
    workbook is used. Returns the new batch summary."""
    data = await file.read() if file else None
    batch_id = service.run_batch(learning_bytes=data, empty_bytes=data)
    return service.summary(batch_id)


def _summary_or_404(batch_id: str) -> dict:
    try:
        return service.summary(batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found")


@app.get("/batches/{batch_id}", response_model=BatchSummary)
def get_batch(batch_id: str) -> dict:
    return _summary_or_404(batch_id)


@app.get("/batches/{batch_id}/rows", response_model=list[Row])
def get_rows(batch_id: str, bucket: str = "all") -> list[dict]:
    """Predictions (€ descending). `bucket` = all | approve | review | done."""
    try:
        return service.rows(batch_id, bucket)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found")


@app.post("/batches/{batch_id}/review", response_model=ReviewStatus)
def start_review(batch_id: str) -> dict:
    """Kick off (or report) the LLM analysis of the review rows. Runs in the background —
    poll GET /batches/{id}/review for progress."""
    try:
        return service.start_review(batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found")


@app.get("/batches/{batch_id}/review", response_model=ReviewStatus)
def get_review(batch_id: str) -> dict:
    try:
        return service.review_status(batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found")


@app.post("/batches/{batch_id}/decisions", response_model=CommitResult)
def commit_decisions(batch_id: str, req: CommitRequest) -> dict:
    """Commit confirmed (row -> Recharging_Item_ID) decisions to the learning store."""
    try:
        return service.commit(batch_id, [d.model_dump() for d in req.decisions],
                              req.reviewed_by)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found")


@app.post("/batches/{batch_id}/reroute", response_model=dict)
def reroute(batch_id: str, req: RerouteRequest) -> dict:
    """Move rows (rejected from Ready-to-approve) back into the Review queue."""
    try:
        return service.reroute(batch_id, req.row_ids)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found")


@app.get("/batches/{batch_id}/history", response_model=list[HistoryEntry])
def get_history(batch_id: str) -> list[dict]:
    try:
        return service.history(batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found")


@app.post("/batches/{batch_id}/recall", response_model=dict)
def recall(batch_id: str, req: RecallRequest) -> dict:
    """Undo a committed decision (remove from learning store, re-open the row)."""
    try:
        return service.recall(batch_id, req.approval_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch or approval not found")
