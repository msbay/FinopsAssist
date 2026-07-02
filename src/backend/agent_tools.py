"""Retrieval evidence for the enrichment agent.

`find_similar_mappings` retrieves real historical rows from GO_MAPPING_LEARNING —
concrete, read-only evidence the agent reasons over. It never writes a
Recharging_Item_ID.

(Cloud / CMDB / wiki enrichment — looking up a subscription's owner or what a naming
token means — is the planned next step; see the README roadmap. It isn't wired yet,
so the agent reasons from history alone for now.)
"""

from functools import lru_cache

import pandas as pd
from matcher import LEARNING_COLS, normalize_schema, trainable_rows
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_FILE = "GO Report Extract GROUPED_V3.xlsx"


@lru_cache(maxsize=1)
def _reference_index():
    """Lazily load GO_MAPPING_LEARNING and build a char n-gram index over row text."""
    df = normalize_schema(pd.read_excel(INPUT_FILE, sheet_name="GO_MAPPING_LEARNING"))
    id_col = LEARNING_COLS["recharging_item_id"]
    df = trainable_rows(df, id_col)  # drop blank/XX_TOIDENTIFY placeholders

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
    return vec, vec.fit_transform(texts), rows


def find_similar_mappings(account_name: str, resource_group: str = "",
                          tags: str = "", k: int = 5) -> str:
    """The k historical accounts most similar to this one and the Recharging_Item_ID
    each was mapped to — evidence for how comparable resources were classified before.
    Returns the top-k matches with a similarity score (0-1)."""
    vec, matrix, rows = _reference_index()
    query = " | ".join(x for x in [account_name.lower().strip(),
                                   resource_group.lower().strip(),
                                   tags.lower().strip()] if x)
    if not query:
        return "No query text provided."
    sims = (vec.transform([query]) @ matrix.T).toarray()[0]
    # Walk neighbours by similarity, collapsing repeats of the same account->ID pair so the
    # k lines carry k *distinct* precedents (more signal, fewer tokens). Drop the sim score
    # and empty tag labels; keep only the fields that exist.
    lines, seen = [], set()
    for i in sims.argsort()[::-1]:
        r = rows[i]
        key = (r["name"], r["recharging_item_id"])
        if key in seen:
            continue
        seen.add(key)
        tagbits = " ".join(x for x in [r["tag_dcs"], r["tag_app"]] if x)
        extra = "".join([f" | rg={r['resource_group']}" if r["resource_group"] else "",
                         f" | {tagbits}" if tagbits else ""])
        lines.append(f"- {r['name']}{extra} -> {r['recharging_item_id']}")
        if len(lines) >= k:
            break
    return "Similar history:\n" + "\n".join(lines)


if __name__ == "__main__":
    # Offline smoke test — no Bedrock needed.
    print(find_similar_mappings(account_name="ago-gl-bkphost-dv-01",
                                resource_group="z-ago-support-dv15", tags="cloud product"))
