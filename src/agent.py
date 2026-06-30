"""Enrichment agent for the review queue.

This is the agentic layer that sits *beside* the deterministic pipeline, not inside
it (the load -> validate -> match -> save spine stays deterministic). It runs only
on low-confidence rows the matcher flagged, investigates them with read-only tools,
and returns a *proposal* — it never writes a Recharging_Item_ID. A human (or a
deterministic rule) commits.

Why an agent and not one LLM call: each row needs a different investigation path
(retrieve similar rows, look up the subscription owner, check what a token means),
so the model decides which tools to call and when, over several steps.

Run:
    python src/agent.py            # demo on one synthetic review row (needs Bedrock)
"""

import json
import sys

sys.path.insert(0, "src")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_tools import TOOLS, TOOLS_BY_NAME
from main import get_llm

MAX_STEPS = 5

SYSTEM_PROMPT = """You are a FinOps enrichment agent. A cloud account/resource could
not be confidently mapped to a Recharging_Item_ID by the automated matcher. Your job
is to INVESTIGATE using the available tools and recommend the best mapping.

Rules:
- Use tools to gather evidence. Prefer find_similar_mappings to see how comparable
  resources were classified historically.
- If a tool returns ACCESS_NOT_CONFIGURED, do not invent its contents — reason from
  what evidence you do have.
- You may ONLY recommend an ID from the provided candidate list. If the evidence is
  insufficient, set "needs_human": true and leave recommended_id null. Never invent
  an ID.
- When done, reply with ONLY a JSON object (no prose) of the form:
  {"recommended_id": <str or null>, "confidence": <0-100>, "needs_human": <bool>,
   "reasoning": <one sentence>, "evidence": [<short strings>]}"""


def _row_prompt(row: dict, candidates: list[str]) -> str:
    return (
        "Row to classify:\n"
        f"  SubAccountName: {row.get('name', '')}\n"
        f"  ResourceGroup:  {row.get('resource_group', '')}\n"
        f"  tag_dcs:        {row.get('tag_dcs', '')}\n"
        f"  tag_app:        {row.get('tag_app', '')}\n"
        f"  SubAccountId:   {row.get('sub_account_id', '')}\n\n"
        f"Candidate IDs (choose one, or needs_human): {candidates}"
    )


def _parse_proposal(text: str) -> dict:
    """Extract the JSON proposal from the model's final message; degrade gracefully."""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"recommended_id": None, "confidence": 0, "needs_human": True,
                "reasoning": "Could not parse agent output.", "evidence": [text[:200]]}


class EnrichmentAgent:
    """Tool-using agent that proposes (never commits) a mapping for one review row."""

    def __init__(self, llm=None, max_steps: int = MAX_STEPS):
        self.max_steps = max_steps
        self.llm = (llm or get_llm()).bind_tools(TOOLS)

    def investigate(self, row: dict, candidates: list[str]) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=_row_prompt(row, candidates))]
        ai = AIMessage(content="")
        for _ in range(self.max_steps):
            ai = self.llm.invoke(messages)
            messages.append(ai)
            if not getattr(ai, "tool_calls", None):
                break
            for call in ai.tool_calls:
                tool = TOOLS_BY_NAME.get(call["name"])
                output = tool.invoke(call["args"]) if tool else f"Unknown tool {call['name']}"
                messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

        proposal = _parse_proposal(ai.content if isinstance(ai.content, str) else str(ai.content))
        # Guardrail: never let a non-candidate ID through.
        if proposal.get("recommended_id") and proposal["recommended_id"] not in candidates:
            proposal["needs_human"] = True
            proposal["reasoning"] = "Proposed ID was not in candidate list; escalating."
            proposal["recommended_id"] = None
        return proposal


def _demo():
    row = {"name": "ago-gl-bkphost-dv-01", "resource_group": "z-ago-support-dv15",
           "tag_dcs": "cloud product", "tag_app": "", "sub_account_id": ""}
    candidates = ["PSO_ITM_1214", "PSO_ITM_429", "XX_GOSHARED"]
    print("Investigating demo review row...\n")
    try:
        agent = EnrichmentAgent()
        proposal = agent.investigate(row, candidates)
        print("Proposal:\n" + json.dumps(proposal, indent=2))
    except Exception as e:  # noqa: BLE001 — surface connection/credential errors clearly
        print(f"Agent run failed (likely Bedrock credentials/model access): {e}")


if __name__ == "__main__":
    _demo()
