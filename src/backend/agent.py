"""FinOps assistant for the review queue.

This LLM layer sits *beside* the deterministic pipeline, not inside it (the
load -> validate -> match -> save spine stays deterministic). It runs only on
low-confidence rows the matcher flagged and returns a *proposal* — it never writes a
Recharging_Item_ID. A human (or a deterministic rule) commits.

Design: a single retrieve-then-reason LLM call, not a tool-calling loop. The only
read-only evidence today is `find_similar_mappings` (historical neighbours), so we
fetch it deterministically and hand it to the model in one prompt. That avoids the
multi-step loop's repeated re-sending of context + tool schemas, which was ~10x more
tokens per row. When live cloud/CMDB tools are connected, reintroduce a bounded loop.

Run:
    python src/agent.py            # demo on one synthetic review row (needs Bedrock)
"""

import json
import sys

sys.path.insert(0, "src")

from agent_tools import find_similar_mappings
from langchain_core.messages import HumanMessage, SystemMessage
from main import get_llm

SYSTEM_PROMPT = """You are a FinOps assistant for AXA Group Operations mapping a new cloud \
resource to a Recharging_Item (cost category; tree: Product Family > Product > Item). The \
classifier was unsure — pick the best candidate.

Signals, by priority:
- Tags: `app` ≈ the Product, `dcs` ≈ the Product Family.
- Azure: the ResourceGroup name is the most specific clue to the Item (e.g. "dynatrace" \
-> Dynatrace). AWS has no ResourceGroup — use the account name + tags.
- SubAccountName (AWS account / Azure subscription) is broad context only; one \
subscription can hold several Items, so never map on it alone.
- Similar historical mappings show how comparable resources were classified.
- Candidates are ranked by classifier probability p (a prior): prefer #1 unless \
tags/RG/evidence point elsewhere; break near-ties with the signals above.

ALWAYS recommend your single best candidate from the list as recommended_id — never null \
and never invent one. When the evidence is weak or conflicting, still pick the most likely \
candidate and express the doubt with a LOW confidence (and needs_human=true).
Be terse. Reply with ONLY this JSON (no prose): {"recommended_id": <str>, \
"confidence": <0-100>, "needs_human": <bool>, "reasoning": <=15 words, \
"evidence": [<=2 short strings]}"""


def _as_candidate(c) -> dict:
    """Accept either a plain id string or a {id,name,family,product,prob,rank} dict."""
    if isinstance(c, dict):
        return {"id": c.get("id", ""), "name": c.get("name", "") or c.get("id", ""),
                "family": c.get("family", ""), "product": c.get("product", ""),
                "prob": c.get("prob"), "rank": c.get("rank")}
    return {"id": c, "name": c, "family": "", "product": "", "prob": None, "rank": None}


def _format_candidates(cands: list[dict]) -> str:
    """Ranked list (classifier confidence order) with probability + Family>Product
    context inline, so the model sees the statistical prior AND the semantics."""
    lines = []
    for c in cands:
        rank = f"#{c['rank']}" if c.get("rank") else "-"
        prob = f" p={c['prob']:.2f}" if c.get("prob") is not None else ""
        ctx = " > ".join(x for x in [c.get("family"), c.get("product")] if x)
        ctx = f" [{ctx}]" if ctx else ""
        lines.append(f'{rank} {c["id"]}{prob} {c["name"]}{ctx}')
    return "\n".join(lines)


def _retrieve_evidence(row: dict, k: int = 3) -> str:
    """Deterministically fetch the k most similar historical mappings (no LLM call)."""
    tags = " ".join(x for x in [row.get("tag_dcs", ""), row.get("tag_app", "")] if x)
    try:
        return find_similar_mappings(account_name=row.get("name", ""),
                                     resource_group=row.get("resource_group", ""), tags=tags, k=k)
    except Exception:  # noqa: BLE001 — evidence is best-effort
        return "No similar historical mappings available."


def _row_prompt(row: dict, cands: list[dict], evidence: str) -> str:
    # No inline hints here — the signal priorities live in SYSTEM_PROMPT, so repeating
    # them per row is wasted tokens. Just the facts + the candidates + the evidence.
    return (
        "Resource to map:\n"
        f"  Provider: {row.get('provider', '')}\n"
        f"  SubAccountName: {row.get('name', '')}\n"
        f"  ResourceGroup: {row.get('resource_group', '')}\n"
        f"  tag_dcs: {row.get('tag_dcs', '')}\n"
        f"  tag_app: {row.get('tag_app', '')}\n\n"
        "Candidates (ranked by classifier p):\n"
        f"{_format_candidates(cands)}\n\n"
        f"{evidence}"
    )


def _token_usage(ai) -> tuple[int, int]:
    """(input_tokens, output_tokens) from an AIMessage; 0s if not reported."""
    u = getattr(ai, "usage_metadata", None) or {}
    return int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)


def _parse_proposal(text: str) -> dict:
    """Extract the JSON proposal from the model's final message; degrade gracefully."""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"recommended_id": None, "confidence": 0, "needs_human": True,
                "reasoning": "Could not parse agent output.", "evidence": [text[:200]]}


class FinopsAssistant:
    """Proposes (never commits) a mapping for one review row, via a single
    retrieve-then-reason LLM call."""

    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def investigate(self, row: dict, candidates: list) -> dict:
        """`candidates` may be plain ids or {id,name,family,product} dicts. The model
        reasons over the names but must return one of the candidate ids. The returned
        proposal also carries the Bedrock token usage."""
        cands = [_as_candidate(c) for c in candidates]
        candidate_ids = {c["id"] for c in cands}
        evidence = _retrieve_evidence(row)  # deterministic retrieval, no LLM round-trip
        ai = self.llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                              HumanMessage(content=_row_prompt(row, cands, evidence))])
        in_tok, out_tok = _token_usage(ai)

        proposal = _parse_proposal(ai.content if isinstance(ai.content, str) else str(ai.content))
        # Guardrail: never let a non-candidate ID through.
        if proposal.get("recommended_id") and proposal["recommended_id"] not in candidate_ids:
            proposal["needs_human"] = True
            proposal["reasoning"] = "Proposed ID was not in candidate list; escalating."
            proposal["recommended_id"] = None
        proposal["input_tokens"] = in_tok
        proposal["output_tokens"] = out_tok
        proposal["total_tokens"] = in_tok + out_tok
        return proposal


def _demo():
    row = {"name": "ago-gl-bkphost-dv-01", "resource_group": "z-ago-support-dv15",
           "tag_dcs": "cloud product", "tag_app": "", "sub_account_id": ""}
    # Candidates carry the semantic tree (Family > Product > Item name) so the agent
    # reasons about names, not opaque ids.
    candidates = [
        {"id": "PSO_ITM_1214", "name": "Backup & Recovery", "family": "Cloud Products",
         "product": "Managed Public IaaS (MPI)"},
        {"id": "PSO_ITM_429", "name": "MPI – shared landing zone", "family": "Cloud Products",
         "product": "Managed Public IaaS (MPI)"},
        {"id": "XX_GOSHARED", "name": "GO Shared Departments", "family": "(GO Shared Departments)",
         "product": "(IT own Cloud consumption)"},
    ]
    print("Investigating demo review row...\n")
    try:
        agent = FinopsAssistant()
        proposal = agent.investigate(row, candidates)
        print("Proposal:\n" + json.dumps(proposal, indent=2))
    except Exception as e:  # noqa: BLE001 — surface connection/credential errors clearly
        print(f"Agent run failed (likely Bedrock credentials/model access): {e}")


if __name__ == "__main__":
    _demo()
