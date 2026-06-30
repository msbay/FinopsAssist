"""Tools for the enrichment agent.

Design note: the agent's whole purpose is to recover the *missing* signal that the
deterministic matcher lacks — who owns a subscription, what product/team a resource
group belongs to. That information lives in cloud / CMDB systems, not in the GO
Report. Those tools are STUBBED for now (no access yet) and return a clear
"not configured" marker so the agent degrades gracefully instead of hallucinating.

The one tool that works today is `find_similar_mappings`, which retrieves real
historical rows from GO_MAPPING_LEARNING — concrete evidence for the agent to reason
over. All tools are read-only. No tool ever writes a Recharging_Item_ID.
"""

from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer

from matcher import LEARNING_COLS

INPUT_FILE = "GO Report Extract LIGHT_V2.xlsx"
NOT_CONFIGURED = (
    "ACCESS_NOT_CONFIGURED: this data source is not connected yet. "
    "Do not guess its contents; rely on other evidence."
)


@lru_cache(maxsize=1)
def _reference_index():
    """Lazily load GO_MAPPING_LEARNING and build a char n-gram index over row text."""
    df = pd.read_excel(INPUT_FILE, sheet_name="GO_MAPPING_LEARNING")
    id_col = LEARNING_COLS["recharging_item_id"]
    df = df.dropna(subset=[id_col]).copy()

    def field(row, key):
        v = str(row.get(LEARNING_COLS[key], "") or "").lower().strip()
        return "" if v == "nan" else v

    texts, rows = [], []
    for _, r in df.iterrows():
        name, rg = field(r, "sub_account_name"), field(r, "resource_group")
        dcs, app = field(r, "tag_dcs"), field(r, "tag_app")
        texts.append(" | ".join(x for x in [name, rg, dcs, app] if x))
        rows.append({"name": name, "resource_group": rg, "tag_dcs": dcs,
                     "tag_app": app, "recharging_item_id": str(r[id_col])})
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vec.fit_transform(texts)
    return vec, matrix, rows


@tool
def find_similar_mappings(account_name: str, resource_group: str = "",
                          tags: str = "", k: int = 5) -> str:
    """Find historical accounts most similar to this one and the Recharging_Item_ID
    each was mapped to. Use this to see how comparable resources were classified
    before. Returns the top-k matches with a similarity score (0-1)."""
    vec, matrix, rows = _reference_index()
    query = " | ".join(x for x in [account_name.lower().strip(),
                                   resource_group.lower().strip(),
                                   tags.lower().strip()] if x)
    if not query:
        return "No query text provided."
    sims = (vec.transform([query]) @ matrix.T).toarray()[0]
    top = sims.argsort()[::-1][:k]
    lines = []
    for i in top:
        r = rows[i]
        lines.append(f"- sim={sims[i]:.2f} | {r['name']} | rg={r['resource_group']} | "
                     f"dcs={r['tag_dcs']} -> {r['recharging_item_id']}")
    return "Most similar historical mappings:\n" + "\n".join(lines)


# ── Cloud / CMDB tools: stubbed until access is granted ──────────────────────
@tool
def lookup_subscription_owner(subscription_id: str) -> str:
    """Look up the owning team / cost centre / product for a cloud subscription or
    account id. Ideal for opaque GUID account names, where the id is a real key into
    the cloud provider or CMDB. (Not connected yet.)"""
    return NOT_CONFIGURED


@tool
def lookup_resource_group_metadata(resource_group: str) -> str:
    """Look up live metadata (owner tags, application, environment) for an Azure
    resource group from the cloud provider. (Not connected yet.)"""
    return NOT_CONFIGURED


@tool
def search_internal_knowledge(query: str) -> str:
    """Search internal documentation / wiki for what a naming token or code means
    (e.g. 'what is bkphost', 'who owns GDAI'). (Not connected yet.)"""
    return NOT_CONFIGURED


# Registry the agent binds. Add real cloud tools here once access is configured.
TOOLS = [
    find_similar_mappings,
    lookup_subscription_owner,
    lookup_resource_group_metadata,
    search_internal_knowledge,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


if __name__ == "__main__":
    # Offline smoke test — exercises the real retrieval tool, no Bedrock needed.
    print(find_similar_mappings.invoke(
        {"account_name": "ago-gl-bkphost-dv-01", "resource_group": "z-ago-support-dv15",
         "tags": "cloud product"}))
