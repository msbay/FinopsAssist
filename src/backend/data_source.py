"""Databricks data source — replaces the Excel workbook.

Reads from:
    adb_finops_dv_en1.gold.gld_rectifier_resources_mappedto_data_enrichment

Rows with a non-empty Recharging_Item_ID → GO_MAPPING_LEARNING (training data)
Rows with empty/null Recharging_Item_ID  → GO_MAPPING_EMPTY   (to predict)

Authentication: Azure AD Service Principal (client credentials flow).

Required env vars:
    DATABRICKS_HOST            e.g. adb-30728...azuredatabricks.net
    DATABRICKS_HTTP_PATH       e.g. /sql/1.0/warehouses/12a91db8d835fc5c
    DATABRICKS_CATALOG_TABLE   e.g. adb_finops_dv_en1.gold.gld_rectifier_...
    AZURE_SPN_CLIENT_ID        Service Principal app/client ID
    AZURE_SPN_CLIENT_SECRET    Service Principal secret
    AZURE_SPN_TENANT_ID        Azure AD tenant ID
"""

from __future__ import annotations

import logging
import os

import aws_accounts
import learning_store
import pandas as pd
import urllib3
from databricks import sql as dbsql
from dotenv import load_dotenv
from matcher import COLS, has_real_id

logger = logging.getLogger("finops.data_source")

# Suppress InsecureRequestWarning from corporate proxy SSL bypass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")
DATABRICKS_TABLE = os.getenv(
    "DATABRICKS_CATALOG_TABLE",
    "adb_finops_dv_en1.gold.gld_rectifier_resources_mappedto_data_enrichment",
)

# Azure AD SPN for auth
_SPN_CLIENT_ID = os.getenv("AZURE_SPN_CLIENT_ID", "")
_SPN_CLIENT_SECRET = os.getenv("AZURE_SPN_CLIENT_SECRET", "")
_SPN_TENANT_ID = os.getenv("AZURE_SPN_TENANT_ID", "")


def _check_config() -> None:
    missing = []
    if not DATABRICKS_HOST:
        missing.append("DATABRICKS_HOST")
    if not DATABRICKS_HTTP_PATH:
        missing.append("DATABRICKS_HTTP_PATH")
    if not _SPN_CLIENT_ID:
        missing.append("AZURE_SPN_CLIENT_ID")
    if not _SPN_CLIENT_SECRET:
        missing.append("AZURE_SPN_CLIENT_SECRET")
    if not _SPN_TENANT_ID:
        missing.append("AZURE_SPN_TENANT_ID")
    if missing:
        raise RuntimeError(
            f"Databricks config incomplete. Missing env vars: {', '.join(missing)}"
        )


def _get_access_token() -> str:
    """Get an Azure AD access token for Databricks using client credentials."""
    import httpx

    url = f"https://login.microsoftonline.com/{_SPN_TENANT_ID}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": _SPN_CLIENT_ID,
        "client_secret": _SPN_CLIENT_SECRET,
        "scope": "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default",  # Azure Databricks resource ID
    }
    with httpx.Client(timeout=30, verify=False) as client:
        resp = client.post(url, data=data)
        if resp.status_code != 200:
            print(f"  Token error {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
        return resp.json()["access_token"]


def _connect():
    """Open a Databricks SQL connection using SPN OAuth (U2M/M2M)."""
    _check_config()
    token = _get_access_token()

    # The databricks-sql-connector needs extra kwargs for Azure AD SPN auth.
    # We pass the token directly and set auth_type to skip its own OAuth flow.
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    return dbsql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=token,
        _tls_no_verify=True,
        _transport_kwargs={"_ssl_context": ssl_context},
    )


# ---------------------------------------------------------------------------
# Query: the exact columns the app needs. The table is ALREADY aggregated (one row per
# resource); its cost column is literally named `sum(axa_EffectiveCost_EUR)`, so we just
# SELECT — no GROUP BY. The concatenated tag columns and the cost column are aliased to
# the canonical matcher names right in SQL, so the frame comes back ready to use with no
# post-fetch renaming.
# ---------------------------------------------------------------------------
_SELECT_COLS = [
    "SubAccountId", "SubAccountName", "axa_Azure_ResourceGroupName", "BillingAccountId",
    "GlobalCustomerID", "GlobalCustomerName", "ProviderName", "Recharging_Item_ID",
    "Recharging_Item_Name", "Product_Name", "Product_Family_information",
    "Product_Manager_information", "Product_Manager_Email",
    "Transversal_Service_Owner_information", "Transversal_Service_Owner_Email",
]
# Source column → canonical matcher name (matcher.COLS). The concatenated tag columns and
# the pre-aggregated cost column are renamed in SQL.
_COL_ALIASES = {
    "axa_tags_global_app_concatenated": "axa_tags_global_app",
    "axa_tags_global_dcs_concatenated": "axa_tags_global_dcs",
    "sum(axa_EffectiveCost_EUR)": "Sumaxa_EffectiveCost_EUR",  # matcher.COLS["cost"]
}


def _build_query() -> str:
    cols = ", ".join(f"`{c}`" for c in _SELECT_COLS)
    aliased = ", ".join(f"`{src}` AS `{dst}`" for src, dst in _COL_ALIASES.items())
    return f"SELECT {cols}, {aliased} FROM {DATABRICKS_TABLE}"


# ---------------------------------------------------------------------------
# AWS enrichment: join the AWS accounts API onto the Databricks rows
# ---------------------------------------------------------------------------
def _enrich_aws(df: pd.DataFrame, account_tags: dict[str, dict]) -> pd.DataFrame:
    """Add the AwsAccountTags.* columns by joining SubAccountId → the accounts API's
    account_id. Rows without a match (non-AWS, or an AWS account absent from the API)
    keep empty enrichment, as specified."""
    for col in aws_accounts.ENRICHMENT_COLS:  # ensure the columns always exist
        if col not in df.columns:
            df[col] = ""
    if not account_tags or df.empty:
        return df
    ids = df[COLS["sub_account_id"]].astype(str).str.strip()
    for col in aws_accounts.ENRICHMENT_COLS:
        df[col] = ids.map(lambda i: account_tags.get(i, {}).get(col, "")).fillna("")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_all(account_tags: dict[str, dict] | None = None) -> pd.DataFrame:
    """Fetch the aggregated resource table from Databricks, enriched with AWS account
    tags from the AWS accounts Postgres DB (joined on SubAccountId). Pass `account_tags` to
    reuse an already-fetched lookup; otherwise it is fetched once here."""
    if account_tags is None:
        account_tags = aws_accounts.fetch_account_tags()
    query = _build_query()
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    df = _enrich_aws(df, account_tags)
    return df


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a full DataFrame into (learning, empty) by Recharging_Item_ID."""
    id_col = COLS["recharging_item_id"]
    if id_col not in df.columns:
        raise KeyError(
            f"Column '{id_col}' not found in Databricks table. "
            f"Available: {list(df.columns)}"
        )
    learning = has_real_id(df[id_col])
    return df[learning].reset_index(drop=True), df[~learning].reset_index(drop=True)


def fetch_learning_and_empty() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch once, split into (learning, empty), then append the locally-committed
    decisions (the Excel-free feedback loop) onto the learning frame. Use this to avoid
    two round-trips."""
    # Fetch the AWS accounts API once per run and reuse it for both the Databricks rows
    # and the learning-store rows below.
    account_tags = aws_accounts.fetch_account_tags()
    learning, empty = _split(fetch_all(account_tags))

    store = learning_store.load()
    if not store.empty:
        # Committed decisions carry SubAccountId, so enrich them the same way (blank when
        # the account isn't in the AWS API) before they join the training frame.
        store = _enrich_aws(store, account_tags)
        learning = pd.concat([learning, store], ignore_index=True)
        logger.info("Learning store: merged %d committed decision(s)", len(store))
    return learning, empty


def fetch_learning() -> pd.DataFrame:
    """Rows with a real Recharging_Item_ID -> training data."""
    learning, _ = fetch_learning_and_empty()
    return learning


# ---------------------------------------------------------------------------
# Connectivity test
# ---------------------------------------------------------------------------
def test_connection() -> None:
    """Quick connectivity + schema check. Run: python data_source.py"""
    print("Checking Databricks configuration...")
    _check_config()
    print(f"  Host : {DATABRICKS_HOST}")
    print(f"  Path : {DATABRICKS_HTTP_PATH}")
    print(f"  Table: {DATABRICKS_TABLE}")
    print(f"  SPN  : {_SPN_CLIENT_ID[:8]}...  (tenant: {_SPN_TENANT_ID[:8]}...)")

    print("\nAcquiring Azure AD token...")
    token = _get_access_token()
    print(f"  Token acquired ({len(token)} chars)")

    print("\nConnecting to Databricks SQL...")
    with _connect() as conn:
        with conn.cursor() as cursor:
            # 1. Schema check
            cursor.execute(f"DESCRIBE TABLE {DATABRICKS_TABLE}")
            schema = cursor.fetchall()
            print(f"\nTable schema ({len(schema)} columns):")
            for row in schema:
                print(f"  {row[0]:45s} {row[1]}")

            # 2. Row counts
            cursor.execute(f"SELECT COUNT(*) FROM {DATABRICKS_TABLE}")
            total = cursor.fetchone()[0]

            id_col = COLS["recharging_item_id"]
            # Try the canonical name first; if it doesn't exist, scan for a match
            try:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {DATABRICKS_TABLE} "
                    f"WHERE `{id_col}` IS NOT NULL AND TRIM(`{id_col}`) != '' "
                    f"AND UPPER(TRIM(`{id_col}`)) NOT IN ('XX_TOIDENTIFY', 'NAN')"
                )
                learning_count = cursor.fetchone()[0]
            except Exception:
                learning_count = "?"

            print("\nRow counts:")
            print(f"  Total rows          : {total}")
            print(f"  With Recharging_Item_ID (LEARNING): {learning_count}")
            if isinstance(learning_count, int):
                print(f"  Without (EMPTY)     : {total - learning_count}")

            # 3. Sample data — run the actual aggregated query the app uses.
            cursor.execute(f"{_build_query()} LIMIT 3")
            sample = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            print(f"\nSample row columns: {cols}")
            if sample:
                print("First row (truncated):")
                for col, val in zip(cols, sample[0]):
                    print(f"  {col:45s} = {str(val)[:80]}")

    print("\nChecking AWS accounts API enrichment...")
    tags = aws_accounts.fetch_account_tags()
    print(f"  AWS accounts loaded: {len(tags)}")

    print("\nDatabricks connectivity test PASSED")


if __name__ == "__main__":
    test_connection()
