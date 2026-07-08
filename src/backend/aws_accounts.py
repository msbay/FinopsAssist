"""AWS account tags — read from Postgres by default, or the finops-backend HTTP API.

Source of the enrichment (owner / global.dcs / description / name) for AWS accounts,
keyed by account_id and mapped to the `AwsAccountTags.*` column names the matcher expects
(see matcher.AWS_TAG_SOURCES). data_source.py joins the result onto the Databricks rows on
SubAccountId → account_id.

Two interchangeable sources — the toggle is env-driven so it can be set on the uvicorn
command line (e.g. `AWS_ACCOUNTS_SOURCE=api uvicorn api:app …`) or in the deployment env:

  • Default — Postgres (aws_raw_data.aws_accounts), read directly. No arg needed.
  • API     — set AWS_ACCOUNTS_API_URL=<url>  (or AWS_ACCOUNTS_SOURCE=api to use the
              default corporate URL). Handy from the corporate network where the API is
              already exposed. When either is set, the API is used and the DB is ignored.

Both sources yield the same account record shape, so `_account_to_enrichment` maps either.

DB connection (same env vars as the source service, so its secret works unchanged):
    DATABASE_URL              full DSN, e.g. postgresql://user:pass@host:5432/dbname
  — or the individual parts —
    DB_HOST, DB_PORT (5432), DB_NAME, DB_USER, DB_PASSWORD
Optional:
    DB_SSLMODE                e.g. require / verify-full / disable
    AWS_ACCOUNTS_TABLE        schema-qualified table (default aws_raw_data.aws_accounts)

API config:
    AWS_ACCOUNTS_SOURCE       set to "api" to use the API (default corporate URL)
    AWS_ACCOUNTS_API_URL      the accounts endpoint (setting it also selects API mode)
    AWS_ACCOUNTS_API_TOKEN    optional bearer token if the endpoint requires auth
"""

from __future__ import annotations

import json
import logging
import os

import httpx
import psycopg2
from matcher import AWS_DCS_COL, AWS_DESC_COL, AWS_NAME_COL, AWS_OWNER_COL
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("finops.aws_accounts")

# Schema-qualified source table (trusted config, not request input).
AWS_ACCOUNTS_TABLE = os.getenv("AWS_ACCOUNTS_TABLE", "aws_raw_data.aws_accounts")

# Default API endpoint used when AWS_ACCOUNTS_SOURCE=api is set without an explicit URL.
DEFAULT_API_URL = (
    "https://finops-backend.ago-fr-dev-int.merlot.eu-central-1.aws.openpaas."
    "axa-cloud.com/data/aws-accounts"
)

# Canonical enrichment column names the matcher reads (matcher.AWS_TAG_SOURCES) — imported
# from matcher so there's one source of truth. Values absent for an account are left blank,
# so a partially-tagged account still enriches.
OWNER_COL = AWS_OWNER_COL
DCS_COL = AWS_DCS_COL
DESC_COL = AWS_DESC_COL
NAME_COL = AWS_NAME_COL
ENRICHMENT_COLS = (OWNER_COL, DCS_COL, DESC_COL, NAME_COL)


def _use_api() -> tuple[bool, str]:
    """(use_api, url): API mode when AWS_ACCOUNTS_API_URL is set or SOURCE=api."""
    url = os.getenv("AWS_ACCOUNTS_API_URL", "").strip()
    if url:
        return True, url
    if os.getenv("AWS_ACCOUNTS_SOURCE", "").strip().lower() == "api":
        return True, DEFAULT_API_URL
    return False, ""


def _connect():
    """Open a Postgres connection from DATABASE_URL or the individual DB_* env vars."""
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn, connect_timeout=10)
    params = {
        "host": os.getenv("DB_HOST", ""),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", ""),
        "user": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
        "connect_timeout": 10,
    }
    sslmode = os.getenv("DB_SSLMODE")
    if sslmode:
        params["sslmode"] = sslmode
    return psycopg2.connect(**params)


def _accounts_from_db() -> list[dict]:
    """All rows of the aws_accounts table as a list of dicts."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {AWS_ACCOUNTS_TABLE}")
            return [dict(r) for r in cur.fetchall()]


def _accounts_from_api(url: str) -> list[dict]:
    """All AWS accounts from the finops-backend HTTP API."""
    token = os.getenv("AWS_ACCOUNTS_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(timeout=60, verify=False) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _account_to_enrichment(acc: dict) -> dict:
    """Map one aws_accounts record to the AwsAccountTags.* enrichment columns."""
    tags = acc.get("tags") or {}
    if isinstance(tags, str):  # defensive — jsonb is usually decoded to a dict already
        try:
            tags = json.loads(tags)
        except (ValueError, TypeError):
            tags = {}
    return {
        OWNER_COL: str(acc.get("owner_email") or tags.get("owner") or ""),
        DCS_COL: str(tags.get("global.dcs") or ""),
        DESC_COL: str(tags.get("description") or ""),
        NAME_COL: str(tags.get("name") or acc.get("account_name") or ""),
    }


def fetch_account_tags() -> dict[str, dict]:
    """Read all AWS accounts (from the DB by default, or the API when selected) and return
    {account_id: {enrichment columns}}.

    Best-effort: on any error (source unreachable, auth failure, unexpected shape) it logs
    a warning and returns an empty dict, so the pipeline still runs — AWS rows simply keep
    empty enrichment, as specified.
    """
    use_api, url = _use_api()
    src = "API" if use_api else "DB"
    try:
        accounts = _accounts_from_api(url) if use_api else _accounts_from_db()
    except Exception as e:  # noqa: BLE001 — enrichment is best-effort
        logger.warning("AWS accounts %s unavailable (%s); AWS rows keep empty enrichment",
                       src, e)
        return {}

    lookup: dict[str, dict] = {}
    for acc in accounts:
        acct_id = str(acc.get("account_id") or "").strip()
        if acct_id:
            lookup[acct_id] = _account_to_enrichment(acc)
    logger.info("AWS accounts %s: %d accounts loaded", src, len(lookup))
    return lookup


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tags = fetch_account_tags()
    print(f"Loaded {len(tags)} AWS accounts")
    for acct_id, enrichment in list(tags.items())[:3]:
        print(f"  {acct_id}: {enrichment}")
