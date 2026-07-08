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

import os

import pandas as pd
import urllib3
from databricks import sql as dbsql
from dotenv import load_dotenv

from matcher import COLS, PLACEHOLDER_IDS

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
# Column mapping: Databricks column names → canonical names used by the app
# ---------------------------------------------------------------------------
# The Databricks table may use different column names than the Excel sheets.
# This mapping is applied after fetching; keys = Databricks names (lowercase),
# values = canonical names the matcher expects. Unmapped columns pass through.
# Adjust this dict once you see the actual Databricks schema.
_COLUMN_RENAME: dict[str, str] = {
    "axa_tags_global_app_concatenated": "axa_tags_global_app",
    "axa_tags_global_dcs_concatenated": "axa_tags_global_dcs",
    "sum(axa_EffectiveCost_EUR)": "Sumaxa_EffectiveCost_EUR",
}


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Databricks columns to the canonical names the matcher expects."""
    if _COLUMN_RENAME:
        df = df.rename(columns=_COLUMN_RENAME)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_all() -> pd.DataFrame:
    """Fetch the full table from Databricks as a DataFrame."""
    with _connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DATABRICKS_TABLE}")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    df = _rename_columns(df)
    return df


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a full DataFrame into (learning, empty) by Recharging_Item_ID."""
    id_col = COLS["recharging_item_id"]
    if id_col not in df.columns:
        raise KeyError(
            f"Column '{id_col}' not found in Databricks table. "
            f"Available: {list(df.columns)}"
        )
    ids = df[id_col].astype(str).str.strip().str.upper()
    has_id = df[id_col].notna() & ~ids.isin(PLACEHOLDER_IDS | {"", "NAN", "NONE"})
    return df[has_id].reset_index(drop=True), df[~has_id].reset_index(drop=True)


def fetch_learning_and_empty() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch once, split into (learning, empty). Use this to avoid two round-trips."""
    return _split(fetch_all())


def fetch_learning() -> pd.DataFrame:
    """Rows with a real Recharging_Item_ID -> training data."""
    learning, _ = fetch_learning_and_empty()
    return learning


def fetch_empty() -> pd.DataFrame:
    """Rows with empty/null Recharging_Item_ID -> to predict."""
    _, empty = fetch_learning_and_empty()
    return empty


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

            print(f"\nRow counts:")
            print(f"  Total rows          : {total}")
            print(f"  With Recharging_Item_ID (LEARNING): {learning_count}")
            if isinstance(learning_count, int):
                print(f"  Without (EMPTY)     : {total - learning_count}")

            # 3. Sample data
            cursor.execute(f"SELECT * FROM {DATABRICKS_TABLE} LIMIT 3")
            sample = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            print(f"\nSample row columns: {cols}")
            if sample:
                print(f"First row (truncated):")
                for col, val in zip(cols, sample[0]):
                    print(f"  {col:45s} = {str(val)[:80]}")

    print("\nDatabricks connectivity test PASSED")


if __name__ == "__main__":
    test_connection()
