"""Thin HTTP client for the FinOps Assistant API.

This is the seam of the front/back split: the Streamlit app (app.py) talks to the backend
ONLY through these functions — it no longer imports matcher/review/agent directly. Point
it at another host with the FINOPS_API_URL env var.
"""

import os

import httpx
from dotenv import load_dotenv

# Load .env so the frontend reads the SAME FINOPS_API_URL / FINOPS_API_TOKEN as the
# backend (api.py also calls load_dotenv). Without this the Streamlit process would send
# no token while the backend requires one -> 401. Real env vars still take precedence.
load_dotenv()

API_URL = os.environ.get("FINOPS_API_URL", "http://127.0.0.1:8000")
# Service token for the backend (only enforced when the backend has FINOPS_API_TOKEN set).
API_TOKEN = os.environ.get("FINOPS_API_TOKEN")
_auth = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
_client = httpx.Client(base_url=API_URL, timeout=300, headers=_auth)


def _json(resp: httpx.Response):
    resp.raise_for_status()
    return resp.json()


def run_batch() -> dict:
    """Train + predict, pulling data directly from Databricks + the AWS accounts API."""
    return _json(_client.post("/batches"))


def summary(batch_id: str) -> dict:
    return _json(_client.get(f"/batches/{batch_id}"))


def rows(batch_id: str, bucket: str = "all") -> list[dict]:
    return _json(_client.get(f"/batches/{batch_id}/rows", params={"bucket": bucket}))


def start_review(batch_id: str) -> dict:
    return _json(_client.post(f"/batches/{batch_id}/review"))


def review_status(batch_id: str) -> dict:
    return _json(_client.get(f"/batches/{batch_id}/review"))


def commit(batch_id: str, decisions: list[dict], reviewed_by: str = "ui") -> dict:
    return _json(_client.post(f"/batches/{batch_id}/decisions",
                              json={"reviewed_by": reviewed_by, "decisions": decisions}))


def reroute(batch_id: str, row_ids: list[int]) -> dict:
    return _json(_client.post(f"/batches/{batch_id}/reroute", json={"row_ids": row_ids}))


def history(batch_id: str) -> list[dict]:
    return _json(_client.get(f"/batches/{batch_id}/history"))


def recall(batch_id: str, approval_id: str) -> dict:
    return _json(_client.post(f"/batches/{batch_id}/recall",
                              json={"approval_id": approval_id}))
