# FinOps Assistant

Automated `Recharging_Item_ID` prediction for the monthly GO Report mapping process. A fast **deterministic matcher** classifies every row and produces a calibrated confidence; an **enrichment agent** (a single retrieve-then-reason LLM call) investigates only the low-confidence rows and proposes a mapping with evidence; a human confirms, and the confirmed decision **feeds back into the learning data** so the system improves each cycle. Everything is **prioritised by € spend** — the biggest cloud costs get mapped first.

The system runs as a **FastAPI backend** (engine + LLM) and a **Streamlit thin-client** dashboard that talk over HTTP — see [Architecture](#architecture).

---

## Problem

Each month, ~800 new cloud accounts / subscriptions / resource groups appear in the GO Report that must be mapped to a `Recharging_Item_ID` (one of ~76 cost categories). Today an analyst does this by hand, comparing each new entry against ~3,800 historically mapped ones — looking at account names, resource groups, and tags. It is slow, repetitive, and error-prone, and the hardest rows (opaque names, near-ties between categories) are exactly the ones a human spends the most time on.

---

## How it works (end to end)

```
   GO Report .xlsx
   (GO_MAPPING_EMPTY = rows to classify,
    GO_MAPPING_LEARNING = ~3,800 historical mappings)
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  ENGINE  — deterministic   (matcher.py)                       │
│  exact lookup  →  char n-gram logistic-regression classifier  │
│  emits per row:  Predicted_ID · Confidence · Top_Matches      │
│                  · Needs_Review · Match_Method · Review_Reason │
└─────────────────────────────────────────────────────────────┘
          │  high confidence (≈60%)        │  low confidence / flagged (≈40%)
          ▼  auto-accepted                  ▼
                                  ┌─────────────────────────────────────────┐
                                  │  AGENT  — investigates  (agent.py +       │
                                  │           agent_tools.py + review.py)     │
                                  │  • candidates = that row's Top_Matches    │
                                  │  • retrieves similar history (evidence),  │
                                  │    reasons over the Family>Product>Item   │
                                  │    tree in one LLM call                   │
                                  │  • proposes ONE candidate ID + evidence   │
                                  │  • never invents an ID, never commits     │
                                  └─────────────────────────────────────────┘
                                                   │  proposal
                                                   ▼
                                  ┌─────────────────────────────────────────┐
                                  │  HUMAN  — Accept / Override  (app.py)     │
                                  │  confirmed mapping is appended to         │
                                  │  GO_MAPPING_LEARNING (audited)            │
                                  └─────────────────────────────────────────┘
                                                   │
                                                   └──► next run trains on it  ⟲
```

The two AI/LLM touchpoints are **scoped and optional**: the matcher is pure scikit-learn and always runs; the agent runs **only on the rows the matcher flags**. If Bedrock is unavailable the deterministic results still stand — you just don't get agent proposals.

---

## Architecture

The app is split into a **FastAPI backend** and a **Streamlit thin-client frontend** that
communicate **only over HTTP**. The frontend renders UI and calls the API; it imports no
model/ML code. Either half can be replaced (e.g. a React frontend) without touching the other.

```
┌─────────────────────────────┐        HTTP (JSON)        ┌────────────────────────────────┐
│  FRONTEND  (src/frontend/)  │  ──────────────────────▶  │  BACKEND  (src/backend/)       │
│  app.py       Streamlit UI  │                           │  api.py      FastAPI routes    │
│  api_client.py  HTTP client │  ◀──────────────────────  │  service.py  logic + state     │
│  port 8501 (browser)        │      (summary, rows)      │  ─────────── engine ───────────│
└─────────────────────────────┘                           │  matcher · review · agent ·    │
                                                           │  agent_tools · main            │
                                                           │  port 8000 (/docs)             │
                                                           └────────────────────────────────┘
```

Inside the backend, three layers separate deterministic processing from agentic reasoning
from HTTP:

```
┌──────────────────────────────────────────────────────────────┐
│        API      (api.py)  — routes + schemas                  │
│        SERVICE  (service.py)  — orchestration + batch state    │
├──────────────────────────────────────────────────────────────┤
│        AGENT  (agent.py · agent_tools.py · review.py)          │
│  Enrichment agent, run on the review rows only. Constrained to │
│  the matcher's candidates; proposes, never commits. Confirmed  │
│  decisions feed back to the learning data.                     │
├──────────────────────────────────────────────────────────────┤
│        ENGINE  (matcher.py)                                    │
│  Deterministic: exact lookup + char n-gram classifier,         │
│  calibrated confidence.                                        │
└──────────────────────────────────────────────────────────────┘
```

> **Why FastAPI + a thin client?** It moves all state and logic server-side (one source of
> truth), lets the LLM run as a background job, and makes the system ready for auth, scaling,
> and a different UI — none of which are possible when logic is embedded in a Streamlit script.
> The current backend keeps batch state **in memory** and still writes the Excel learning
> store; swapping both for a database is the next hardening step and won't change the API.

### Why this design?

| Alternative considered | Why we rejected it |
|---|---|
| Let one LLM agent run the *whole* pipeline (decide tool order, do the matching) | Non-deterministic execution is a compliance risk for a financial pipeline. The match/validate spine stays deterministic; the agent is **scoped to the review layer**, with read-only tools, constrained to the classifier's candidate list, and proposing rather than committing. |
| LLM matching all 811 rows | Slow (~3 min vs ~3 sec), expensive, non-deterministic, and no real numeric confidence. A trained classifier is faster, cheaper, and gives a calibrated probability. The agent earns its cost only on the ~40% the classifier is unsure about. |
| Neural sentence embeddings (e.g. `bge-small`) | Benchmarked: they don't beat character n-grams here. Cloud account names are structured identifiers (`prod-axa-dcs-01`), not prose — char n-grams capture the naming conventions directly. |
| Hand-weighted fuzzy scoring | Benchmarked: a learned classifier beats hand-tuned field weights (+3–5pp) and yields a *calibrated* confidence for free. |
| Let the agent pick any `Recharging_Item_ID` | It could hallucinate an ID that doesn't fit the row — unacceptable for cost allocation. A guardrail forces `needs_human` if the proposal isn't in the candidate list. |
| Chat-only interface for overrides | Typing "override row 47 to PSO_ITM_361" for 50 rows is unusable. Tables, dropdowns, and one-click Accept are faster for data correction. |

---

## The Engine — matching logic (`src/backend/matcher.py`)

### Text representation

Each row becomes one string for comparison:

```
{SubAccountName} | {ResourceGroupName} | {tag_dcs} | {tag_app}
```

- **`SubAccountName`** — primary identifier; carries naming conventions (`prod-axa-dcs-01`).
- **`ResourceGroupName`** — highly informative for Azure (`z-ago-finops-cfp-ew1-rg01`), almost always present; the strongest secondary signal.
- **`tag_dcs` / `tag_app`** — highly predictive and ~100% populated in the current data: `dcs` is an ~89%-pure hint for the Product **Family** and `app` an ~97%-pure hint for the **Product**. The char n-grams already absorb this signal, so isolating the tags as separate categorical features does **not** improve accuracy (benchmarked).
- **`SubAccountId`** is excluded — a GUID/numeric id has no lexical similarity to other ids.

### Two-layer prediction

**Layer 1 — exact lookup (cheap, certain).** A row whose `(SubAccountName, ResourceGroup)` was already seen inherits that `Recharging_Item_ID` at confidence 100. An *unambiguous* name (always one ID historically) gets 95. ~44% of monthly rows are accounts seen before — handled deterministically, no model.

**Layer 2 — learned classifier (generalizes to new accounts).** Everything else goes to a **multinomial logistic regression** over **character n-gram TF-IDF** (`char_wb`, n-grams 2–4). Char n-grams capture `prod`, `dcs`, `-rg01` and naming-convention variants directly; benchmarked against `BAAI/bge-small-en-v1.5` embeddings, the neural signal added nothing.

### Review routing

Beyond the confidence threshold, the matcher isolates rows the *clues themselves* can't determine (`_evidence_reason`):

- **no clue** — opaque/short name, no resource group or tags → always review.
- **weak** — name only → always review.
- **check with LLM** — has a resource group and/or tags but the pattern is ambiguous → review when confidence is low; this is where the agent adds the most value.

### Confidence score (0–100)

The classifier's own `predict_proba` for its top pick, scaled to 0–100 (exact lookups get 100/95):

- **≥ 70** — high (green), auto-accept
- **50–69** — medium (yellow), worth checking
- **< 50** — low (red), flagged for review

Per-row outputs: `Predicted_Recharging_Item_ID`, `Confidence`, `Top_Matches` (top-5, the agent's candidate list — `TOP_K_CANDIDATES`), `Needs_Review`, `Match_Method`, `Review_Reason`, plus the hierarchy columns below.

### Hierarchy: Family → Product → Recharging Item

The labels form a strict tree — each Recharging Item belongs to exactly one **Product**, which belongs to one **Product Family** (the business unit). The matcher predicts the *item* with the flat classifier (best accuracy) and reads its Product/Family off the learned tree, so the three levels are always consistent.

Per-level confidence comes from **marginalizing the item probability vector up the tree**: `Family_conf = Σ P(item)` over items in the predicted family, likewise for Product (nested inside the family). Because coarser levels aggregate more probability mass, confidence rises up the tree and is guaranteed `Family ≥ Product ≥ Item`.

Why this and not a hierarchical *classifier*? Benchmarked — top-down constrained models do **not** beat the flat classifier on item accuracy (a wrong Family would force a wrong Item, i.e. error propagation). The value of the hierarchy is **calibrated per-level confidence and review routing**, not higher leaf accuracy: a wrong item is often still the right Product/Family.

Extra outputs: `Predicted_Product_Family`, `Family_Confidence`, `Predicted_Product_Name`, `Product_Confidence`, `Predicted_Recharging_Item_Name`.

---

## The Agent — enrichment of review rows (`src/backend/agent.py`, `agent_tools.py`, `review.py`)

For the rows the matcher flags, the agent recovers signal the classifier lacks.

**What it receives.** The flagged row plus its **candidate set** (`Candidate_IDs` — the adaptive nucleus set, wider than the displayed `Top_Matches`), presented as the Family → Product → Item tree with human-readable names. It may only recommend an ID from that set.

**How it works** — a single **retrieve-then-reason** call (`FinopsAssistant.investigate`), not a multi-step tool loop:
1. `agent_tools.find_similar_mappings` (TF-IDF char-ngram retrieval over `GO_MAPPING_LEARNING`) is called **deterministically** to fetch the most similar historical mappings — "how were comparable accounts classified before?".
2. That evidence + the candidate tree go into **one** LLM prompt; the model reasons semantically over the names and returns a proposal.

This replaced an earlier tool-calling loop that re-sent the growing context + tool schemas each step (~10× the tokens). A bounded loop can be reintroduced once live cloud/CMDB enrichment tools exist — see the roadmap.

**What it returns** — a structured proposal, never a write:

```json
{"recommended_id": "PSO_ITM_530", "confidence": 64, "needs_human": false,
 "reasoning": "one sentence", "evidence": ["similar rows used …"]}
```

**Guardrails.** The agent must return one of the candidate IDs (never null); if it abstains, `run_review` falls back to the classifier's top-ranked candidate, so the prediction cell is never empty. `review.run_review` runs the agent across the flagged rows and attaches `Agent_Proposed_ID / Agent_Confidence / Agent_Needs_Human / Agent_Reasoning / Agent_Evidence / Agent_Tokens`.

### Feedback loop (`review.commit_decisions`)

When a human **Accepts** or **Overrides**, the confirmed `(name, resource group, tags) → Recharging_Item_ID` is appended to `GO_MAPPING_LEARNING` with an audit stamp (`Reviewed_By`, `Reviewed_At`, `Review_Source`). A one-time `.BACKUP` copy is made on first write and all other sheets are preserved. The next run retrains on it, so:

- the classifier sees more examples → **fewer review rows next cycle**, and
- `find_similar_mappings` retrieves richer history → **better agent proposals**.

This is the compounding loop: the tool gets smarter every month from the analysts' own decisions.

---

## Evaluation

Accuracy is measured honestly on **genuinely new accounts**: `GO_MAPPING_LEARNING` is split **by `SubAccountName`** (group split) so no account appears in both train and test. A plain random split lets near-duplicate rows of the same account leak across and inflates accuracy by ~16 points. ~56% of each month's rows are accounts never seen before, so this is the case that matters. The headline is averaged over several splits (a single split swings ~±8pp by luck).

| Metric (held-out new accounts) | Value |
|---|---|
| **Item** accuracy | **~82%** (averaged over splits) |
| **Product** accuracy | **~86%** |
| **Family** accuracy | **~87%** |
| Auto-accept band (conf ≥70) | ~75% of rows at **~95% accuracy** |
| Expected Calibration Error | ~4–5 (lower = better) |

Training excludes the `XX_TOIDENTIFY` placeholder and blank targets (`matcher.trainable_rows`), so the model never learns "to identify" as a class — which is what keeps the auto-accept zone calibrated.

Accuracy rises up the tree (Family ≥ Product ≥ Item): a wrong item is often still the right product/family, which is what makes the per-level confidence useful for review routing. The remaining ~44% of production rows are *seen* accounts that hit exact lookup at 100%, so blended accuracy is higher than the item headline. The hardest cases — `XX_*` catch-all buckets and bare GUIDs with no resource group or tags — are exactly what gets routed to the agent.

---

## Dashboard (`src/frontend/app.py`)

A Streamlit **thin client** — it holds only the current `batch_id` and drives everything
through the API (`api_client.py`). The sidebar switches between features (via `st.navigation`):

- **🏷️ Cost Allocation** — *live*, described below.
- **🛠️ Tag Remediation** — *coming soon*: propose the correct tag value for non-compliant resources (same agent pattern as Cost Allocation).

### Cost Allocation feature

**Run pipeline** (sidebar) calls the backend to train + predict, then kicks off the LLM
analysis of the **whole review queue** as a background job. The page stays usable while it
runs. Overview cards (row counts + € coverage) sit above three tabs:

- **✅ Ready to approve** — the classifier-confident rows (≥ threshold). Select rows → **Approve** (commits the predicted ID) or **Reject → send to review**.
- **🔍 Review queue** — every non-confident row, each already carrying the **agent's prediction, agent confidence (blue), the classifier's own top-1 + confidence, an explanation, and token cost**. A live progress bar shows the LLM job. Select rows → **Approve** the agent's prediction, or **Reject & correct** with a value you type.
- **🗂 History** — decisions committed this session, each with **Recall** (removes it from the learning store and re-opens the row).

Nothing is ever auto-committed — a human always confirms. Every commit appends to the
learning store and improves the next run.

**Data source:** upload a GO Report `.xlsx` in the sidebar, or leave it empty to use the bundled file. Uploaded files are *batches to classify*; confirmed decisions always append back to the canonical learning workbook so knowledge accrues in one place.

### Cost prioritisation (`Sumaxa_EffectiveCost_EUR`)

FinOps cares about mapping **the biggest spend first**, not clearing rows in arbitrary order — and cloud cost is heavily concentrated (in the current batch the **top ~10% of rows hold ~74% of the €**). So cost is a first-class dimension:

- **The agent spends its budget where the money is.** `review.run_review` sorts the flagged rows by € descending before the `max_rows` cap, so "run the agent on N rows" always processes the N **highest-cost** rows.
- **The dashboard is framed around € coverage**, not row counts — the headline becomes "auto-mapped covers X% of spend", and the review queue shows how much € each batch of decisions clears. Mapping the top ~15–20 review rows typically covers ~80% of the review spend.

`Sumaxa_EffectiveCost_EUR` is carried through as a canonical column (`matcher.COLS["cost"]`); it is **not** a model input — it only drives prioritisation and reporting.

---

## Project structure

```
FinopsAssist/
├── src/
│   ├── backend/                    # FastAPI service + engine  (pip install -e ".[backend]")
│   │   ├── api.py                  # FastAPI routes + Pydantic schemas
│   │   ├── service.py              # Orchestration + in-memory batch store
│   │   ├── matcher.py              # Engine: exact lookup + char n-gram classifier
│   │   ├── agent.py                # FinopsAssistant — single retrieve-then-reason call
│   │   ├── agent_tools.py          # find_similar_mappings — historical-neighbour retrieval
│   │   ├── review.py               # Matcher→agent bridge + feedback-loop write-back
│   │   └── main.py                 # Bedrock client + get_llm()
│   └── frontend/                   # Streamlit thin client  (pip install -e ".[frontend]")
│       ├── app.py                  # Streamlit UI (Home + Cost Allocation)
│       └── api_client.py           # HTTP client → backend (FINOPS_API_URL)
├── GO Report Extract GROUPED_V3.xlsx  # Learning + empty sheets (the backend's data store)
├── pyproject.toml                     # deps split into [backend] / [frontend] extras
├── .env                               # AWS credentials + config (gitignored)
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate                 # macOS/Linux  (.venv\Scripts\activate on Windows)
pip install -e ".[backend,frontend,dev]"  # local dev (both halves)
# on separate hosts you'd install only one side:
#   pip install -e ".[backend]"           # API host
#   pip install -e ".[frontend]"          # UI host
```

Create a `.env` in the project root (gitignored — never commit real keys):

```env
AWS_REGION=eu-central-1
BEDROCK_MODEL_ID=eu.amazon.nova-2-lite-v1:0
AWS_ACCESS_KEY_ID=<your-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
# AWS_SESSION_TOKEN=                  # only for temporary credentials
```

`get_llm()` passes these explicitly to boto3 and **fails fast if any are missing** — it never falls back to ambient `~/.aws` credentials (avoids hitting the wrong account).

---

## Run

Two processes, both started **from the project root** (so the learning-store path resolves).
Start the backend first:

```bash
uvicorn api:app --app-dir src/backend            # backend  → http://127.0.0.1:8000  (docs at /docs)
streamlit run src/frontend/app.py                # frontend → http://localhost:8501
```

- **AWS account enrichment source.** By default the backend reads AWS account tags directly from Postgres (`DB_*` / `DATABASE_URL`). To use the exposed HTTP API instead — e.g. from the corporate network — set the source on the same command line (uvicorn doesn't forward custom flags, so it's an env var; identical to setting it in an OpenShift Deployment `env:`):
  ```bash
  # Use the API with the built-in default corporate URL:
  AWS_ACCOUNTS_SOURCE=api uvicorn api:app --app-dir src/backend

  # …or point at an explicit URL (also selects API mode):
  AWS_ACCOUNTS_API_URL=https://finops-backend.ago-fr-dev-int.merlot.eu-central-1.aws.openpaas.axa-cloud.com/data/aws-accounts \
    uvicorn api:app --app-dir src/backend
  ```
  No arg → DB. Startup logs show which ran (`AWS accounts DB: N accounts loaded` vs `AWS accounts API: N accounts loaded`). Either way it's best-effort: if the source is unreachable, AWS rows keep empty enrichment and the run continues.
- While developing, use `uvicorn api:app --app-dir src/backend --reload --reload-dir src/backend`. The `--reload-dir src/backend` is important — **without it the reloader watches the whole tree including `.venv`**, which loops endlessly on large packages and never starts. For just *running* the app, omit `--reload`.
- Point the UI at a remote backend with `FINOPS_API_URL=http://host:8000 streamlit run src/frontend/app.py`.
- Databricks connectivity check: `python src/backend/data_source.py`. Bedrock check: `python src/backend/main.py`.

The backend needs valid Bedrock credentials in `.env`; without them the classifier still
runs and the review queue can be filled in manually (the LLM step surfaces a warning).

---

## Troubleshooting

**Backend "keeps loading" / never starts.** You launched with `--reload` and no `--reload-dir`, so the file-watcher is scanning `.venv` (e.g. a large package like `transformers`/`torch`) in a loop. Run without `--reload`, or scope it: `--reload --reload-dir src/backend`. Confirm the API is up at `http://127.0.0.1:8000/health`.

**First run is slow.** Expected. Train + predict is a few seconds, but the LLM then analyses the *whole* review queue in the background (~1 Bedrock call/row, a few minutes on a full batch) — watch the Review-queue progress bar. The first run also pays one-time warm-ups (heavy imports, building the evidence index, first Bedrock connection). Ready-to-approve is usable immediately while the review job runs.

**`transformers`/`torch` in your `.venv` (heavy, slow startup).** These are **not** project dependencies — they came from a stray install in that venv. Recreate a clean env:
```bash
# from the project root, venv deactivated
rm -rf .venv                                  # Windows: rmdir /s /q .venv
python -m venv .venv && source .venv/bin/activate
pip install -e ".[backend,frontend,dev]"      # transformers/torch will NOT be pulled
```

**`Run pipeline` errors / "Backend unavailable".** The UI can't reach the API. Make sure the backend is running first, and that `FINOPS_API_URL` (default `http://127.0.0.1:8000`) points at it.

---

## Data requirements

Input file: `GO Report Extract GROUPED_V3.xlsx` with these sheets:

| Sheet | Purpose | Key columns |
|---|---|---|
| `GO_MAPPING_LEARNING` | Historical mappings (reference) | `SubAccountName`, `axa_Azure_ResourceGroupName`, `axa_tags_global_dcs`, `axa_tags_global_app`, `Recharging_Item_ID`, `Product_Name`, `Product_Family_information` |
| `GO_MAPPING_EMPTY` | New items to predict | `SubAccountName`, `ProviderName`, `axa_Azure_ResourceGroupName`, `axa_tags_global_dcs`, `axa_tags_global_app`, `Sumaxa_EffectiveCost_EUR` |

`ProviderName` distinguishes **AWS** accounts (no resource group) from **Azure** subscriptions/resource groups; `Sumaxa_EffectiveCost_EUR` is the € spend used to prioritise the review queue (not a model input).

The V3 schema uses the same plain column names in both sheets. Older LIGHT_V2 files (with `Custom.focus_costs[...]` / `focus_costs[...]` prefixes) still load: `matcher.normalize_schema()` renames legacy columns to the canonical names on read.

---

## Tuning

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `APPROVE_THRESHOLD` | `backend/service.py` | `50.0` | Confidence cut for Ready-to-approve vs Review; higher = more conservative |
| `NGRAM_RANGE` | `backend/matcher.py` constant | `(2, 4)` | Character n-gram size for the classifier |
| `CLF_C` | `backend/matcher.py` constant | `50.0` | Logistic-regression regularization (higher = less regularization) |
| evidence `k` | `backend/agent.py` `_retrieve_evidence` | `3` | Historical similar mappings fetched as evidence per review row |

---

## Potential improvement — a value-of-information arbiter

The current design treats the cloud/agent as a **rescuer**: it only fires on rows the classifier already doubts (low confidence). Rethinking the problem from first principles suggests a better topology, given two real constraints — cloud tags are **partial / inconsistent** (evidence, not ground truth) and Azure/AWS access is **per-account lookups only** (enrichment is scarce, can't be run on every row).

**Core insight.** A `Recharging_Item_ID` is a *cost-ownership* label; name/tags/history are only proxies for it. The classifier's ~81% item-accuracy ceiling on new accounts is an **information** limit, not a model limit — the input underdetermines the answer. The cloud can supply the missing signal, but it's noisy and rationed, so the goal is to spend a lookup **only where it will change the decision**.

This means escalating on **novelty and cost**, not just low confidence — because the classifier's worst failure is being *confidently wrong* on accounts unlike anything in history, which a confidence threshold never catches.

Proposed routing brain (a small change in front of the existing agent):

1. **Novelty signal** — add *max similarity of the account to history* as a first-class routing feature. Low similarity ⇒ the classifier is extrapolating ⇒ enrich, even if its probability looks peaked. *(Highest-leverage first build.)*
2. **Cost weighting** — *partially live*: the review queue and the agent already process flagged rows by `Sumaxa_EffectiveCost_EUR` descending (see [Cost prioritisation](#cost-prioritisation-sumaxa_effectivecost_eur)). The remaining refinement is to weight by `expected_accuracy_gain × cost` — i.e. also factor in *how uncertain* a row is, not just its spend, so a scarce cloud lookup goes to the high-spend **and** high-uncertainty rows first.
3. **Cloud as weighted evidence + agreement check** — when a per-account lookup returns an owner/cost-centre tag, cross-check it against what history mapped similar accounts to: **agree → auto-accept at high confidence; disagree → surface as the highest-value human review** with both sides. This turns inconsistent tags from a liability into a calibration signal.
4. **Aggressive caching** — cache lookups by account ID. Accounts recur month to month, so the rationed API is hit ~once per account ever; incremental cost approaches zero after the first pass, which is what makes selective enrichment affordable.

Net: the cheap classifier still decides most rows; a *selective* cloud-enrichment + reconciliation step is spent only where novelty and spend say it pays off. Everything worth keeping today (exact lookup, calibrated classifier, candidate-list guardrail, feedback loop) stays — only the routing logic in `matcher.py` changes.

---

## Roadmap — ownership-confirmation workflow

> **Status: design agreed, not yet built.** Everything above describes what runs today (an
> Excel-backed batch tool). This section records the target architecture the tool is moving
> toward, for the org container deployment.

### The reframe

Cost allocation — attributing every cloud account to the team that should pay for it — is slow
for **two** hands-on reasons: working out **who likely owns** an account, and **getting that
owner to confirm** before the cost is written to the master. Today's tool automates the first
(classifier + agent) but leaves the second — the email chasing and the manual master edit —
entirely manual. The target closes that gap.

### Databricks is the master; the app never writes it

The `.xlsx` files are **extracts from a governed Databricks table** — Databricks is the system of
record, not the spreadsheet. Target:

- **Read-only** from Databricks (accounts, cost, historical mappings, and the
  `recharging_item ↔ owner` reference) via a SQL Warehouse + service principal.
- The app **never writes the master table.** Confirmed decisions flow through a **governed
  promotion job the data team controls** — append to an app-owned staging table → `MERGE` into the
  golden table; or a signed file/API batch if no Databricks write is permitted.

This deletes the runtime Excel writes and makes the container **stateless**. (Note the cross-cloud
split: data on **Azure**/Databricks, the LLM agent on **AWS Bedrock** — two identities to manage,
or consolidate onto one cloud's LLM.)

### Confirmation workflow

Because `recharging_item ↔ owner` is **1-to-1**, predicting the recharging ID also names the owner
to contact:

1. Classifier + agent predict the recharging ID (⇒ the likely owner).
2. The owner is asked **"Is this account yours?"** via a **Microsoft Teams Adaptive Card** (email
   with a one-click magic link as fallback) — they confirm *ownership*, never a code.
3. **Yes** ⇒ the recharging ID is confirmed and the account is staged for promotion.
4. **Anything else** ("no" / "not mine" / "not sure" / no response after reminders) ⇒ **no
   automated second guess.** The account escalates to a **FinOps admin** who decides (reassign,
   investigate, park in a provisional bucket, or leave unassigned).

Per-account state machine: `Predicted → Request sent → Awaiting → Confirmed → Ready to promote`,
with branches for auto-reminders and admin escalation. Every transition is timestamped — the audit
trail finance needs.

### Where state lives

- **Databricks** — source of truth (read-only), and via the governed job the master mapping.
- **Postgres (app-owned)** — the *workflow* state only: requests, statuses, confirmations, audit
  log. **Not** the cost data. This is what lets a multi-day, multi-party approval survive restarts
  and run on more than one replica.

### Prerequisites (container-readiness)

Before the org container deployment the code needs: config via **environment variables** (workbook
/ DB path, credentials) instead of a bundled `.env`; **removing** the static-AWS-keys requirement
and `verify=False` in `main.get_llm()` so it can use a platform role identity over TLS; and
centralising the hard-coded workbook path (`service.WORKBOOK`, `agent_tools.INPUT_FILE`).

### Phasing

1. **MVP speedboat** — predict the owner + generate a ready-to-send Teams/email draft the analyst
   sends manually. Zero integration; proves the predictions.
2. **Workflow + tracking** — one-click send, live status board, magic-link responses, Postgres.
3. **Teams + automation** — interactive Adaptive Cards, auto-reminders.
4. **Governed hand-off** — confirmed batch flows into the data team's promotion job.

### Open governance questions

- **Hand-off** — app-owned Databricks staging table (recommended), or a file/API batch into the
  existing ingestion?
- **Provisional bucket** — may a non-confirmed account's cost be parked in a temporary
  "shared / unallocated" item, or must it stay unassigned until resolved?
- **SLA** — how many reminders / how long before an account auto-escalates to the admin?

---

## Known limits / next steps

- **Add cloud/CMDB/wiki enrichment** (owner lookup, resource-group metadata, naming glossary) — the biggest unlock; turns the agent from "smart pattern-matcher over history" into one that resolves accounts history can't explain. Would reintroduce a bounded tool-calling loop alongside the current retrieval.
- **Agent throughput/cost** — ~40% of rows are flagged; each agent investigation is a live LLM call, so the queue is capped per run. Batch/cache for full-batch enrichment.
- **Persistence & concurrency** — the backend keeps batch state **in memory** and `commit_decisions` rewrites the Excel workbook; safe for one process / small team one-at-a-time. The target design (see **Roadmap** above) reads the master from **Databricks** (read-only) and moves *workflow* state to **Postgres**, making the container stateless; neither changes the API contracts, and it unblocks running the backend on multiple replicas.
