# FinOps Assistant

Automated `Recharging_Item_ID` prediction for the monthly GO Report mapping process. A fast **deterministic matcher** classifies every row and produces a calibrated confidence; a **tool-using enrichment agent** investigates only the low-confidence rows and proposes a mapping with evidence; a human confirms, and the confirmed decision **feeds back into the learning data** so the system improves each cycle. All of it is driven from an interactive dashboard.

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
                                  │  • tools: find_similar_mappings (live),   │
                                  │    cloud/CMDB/wiki lookups (stubbed)      │
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

Three layers, separating deterministic processing from agentic reasoning from user interaction:

```
┌──────────────────────────────────────────────────────────────┐
│                    INTERFACE  (app.py)                         │
│  Streamlit: Results tab · Review Queue tab · chat sidebar      │
├──────────────────────────────────────────────────────────────┤
│        AGENT  (agent.py · agent_tools.py · review.py)          │
│  Tool-using enrichment agent, run on Needs_Review rows only.   │
│  Constrained to the matcher's candidates; proposes, never      │
│  commits. Confirmed decisions feed back to the learning data.  │
├──────────────────────────────────────────────────────────────┤
│        ENGINE  (matcher.py · run_pipeline.py)                  │
│  Deterministic: load → validate → match → output.              │
│  Exact lookup + char n-gram classifier, calibrated confidence. │
└──────────────────────────────────────────────────────────────┘
```

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

## The Engine — matching logic (`src/matcher.py`)

### Text representation

Each row becomes one string for comparison:

```
{SubAccountName} | {ResourceGroupName} | {tag_dcs} | {tag_app}
```

- **`SubAccountName`** — primary identifier; carries naming conventions (`prod-axa-dcs-01`).
- **`ResourceGroupName`** — highly informative for Azure (`z-ago-finops-cfp-ew1-rg01`), almost always present; the strongest secondary signal.
- **`tag_dcs` / `tag_app`** — useful when present but ~35% null, so they act as tiebreakers.
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

Per-row outputs: `Predicted_Recharging_Item_ID`, `Confidence`, `Top_Matches` (top-3, the agent's candidate list), `Needs_Review`, `Match_Method`, `Review_Reason`.

---

## The Agent — enrichment of review rows (`src/agent.py`, `src/agent_tools.py`, `src/review.py`)

For the rows the matcher flags, the agent recovers signal the classifier lacks.

**What it receives.** The flagged row plus its **candidate list parsed from `Top_Matches`** (`review.parse_candidates`). It may only recommend an ID from that list.

**How it investigates** (`EnrichmentAgent.investigate`, up to 5 steps). It decides which tools to call:

| Tool (`agent_tools.py`) | Status | What it does |
|---|---|---|
| `find_similar_mappings` | **live** | TF-IDF char-ngram retrieval over `GO_MAPPING_LEARNING` — "how were comparable accounts classified before?", with similarity scores. The engine of the agent's value today. |
| `lookup_subscription_owner` | stubbed | Owning team / cost centre for a subscription or opaque GUID account. |
| `lookup_resource_group_metadata` | stubbed | Live owner/app/environment tags for a resource group. |
| `search_internal_knowledge` | stubbed | What a naming token or code means (e.g. `bkphost`, `GDAI`). |

Stubbed tools return a clear `ACCESS_NOT_CONFIGURED` marker; the prompt instructs the agent **not to guess** their contents, so it degrades gracefully instead of hallucinating. Connecting them (registry in `agent_tools.py`) is the biggest future unlock — it's what makes opaque GUID accounts resolvable.

**What it returns** — a structured proposal, never a write:

```json
{"recommended_id": "PSO_ITM_530", "confidence": 64, "needs_human": false,
 "reasoning": "one sentence", "evidence": ["similar rows used …"]}
```

**Guardrails.** Any ID outside the candidate list is rejected → `needs_human`. The agent proposes; a human (or rule) commits. `review.run_review` runs the agent across the flagged rows and attaches `Agent_Proposed_ID / Agent_Confidence / Agent_Needs_Human / Agent_Reasoning / Agent_Evidence`.

### Feedback loop (`review.commit_decision`)

When a human **Accepts** or **Overrides**, the confirmed `(name, resource group, tags) → Recharging_Item_ID` is appended to `GO_MAPPING_LEARNING` with an audit stamp (`Reviewed_By`, `Reviewed_At`, `Review_Source`). A one-time `.BACKUP` copy is made on first write and all other sheets are preserved. The next run retrains on it, so:

- the classifier sees more examples → **fewer review rows next cycle**, and
- `find_similar_mappings` retrieves richer history → **better agent proposals**.

This is the compounding loop: the tool gets smarter every month from the analysts' own decisions.

---

## Evaluation (`src/benchmark.py`)

Accuracy is measured honestly on **genuinely new accounts**: `GO_MAPPING_LEARNING` is split **by `SubAccountName`** (group split) so no account appears in both train and test. A plain random split lets near-duplicate rows of the same account leak across and inflates accuracy by ~16 points. ~56% of each month's rows are accounts never seen before, so this is the case that matters. The headline is averaged over several splits (a single split swings ~±8pp by luck).

| Metric (held-out new accounts) | Value |
|---|---|
| Accuracy | **~79%** |
| Auto-accept band (conf ≥70) | ~51% of rows at **~96% accuracy** |
| Flagged for review (conf <50) | ~32% of rows |
| Expected Calibration Error | ~13 (lower = better) |

The remaining ~44% of production rows are *seen* accounts that hit exact lookup at 100%, so the blended accuracy is higher than the 79% headline. The hardest cases — `XX_*` catch-all buckets and bare GUIDs with no resource group or tags — are exactly what gets routed to the agent.

Run: `python src/benchmark.py`.

---

## Headless pipeline (`src/run_pipeline.py`)

A rigid, deterministic 4-step batch run for when you just want an enriched Excel file (no UI):

1. **Load & validate** — read both sheets, verify required columns (fail fast), warn on missing names.
2. **Build index & predict** — train the matcher, predict every `GO_MAPPING_EMPTY` row. Pure scikit-learn.
3. **LLM checkpoint** — for `Needs_Review` rows only, a single Bedrock call writes a one-sentence `LLM_Justification`. (This is the lightweight batch counterpart to the interactive agent; if Bedrock is down the pipeline still completes with the column left blank.)
4. **Save** — write `GO_predictions.xlsx` (all original columns + prediction columns), sorted worst-confidence-first.

> The **interactive app** uses the full tool-using `EnrichmentAgent` and the feedback loop; the **headless pipeline** uses the lighter single-shot justification. Same engine, two consumption modes.

---

## Dashboard (`src/app.py`)

A multi-feature Streamlit app. The sidebar switches between features (via `st.navigation`):

- **🏷️ Cost Allocation** — *live*, described below.
- **🧹 Tag Hygiene** — *coming soon*: propose the correct tag value for non-compliant resources (same agent pattern as Cost Allocation).

New features are added as a page function + one entry in the navigation catalogue.

### Cost Allocation feature

Interactive UI — the analyst sees data, not a chatbox. Three tabs:

- **📊 Results** — metrics bar (total / auto / needs-review / avg confidence), color-coded table (🟢≥70 🟡50–69 🔴<50), filters by status and predicted ID (native pandas, no LLM), and **Download to Excel**.
- **🔍 Review Queue** — choose how many flagged rows to investigate, run the agent, then one expandable card per row: the proposed ID, confidence, **reasoning**, **evidence**, a candidate radio, and **Accept & commit** (writes back to the learning data).
- **ℹ️ About** — legend for the confidence bands and match methods.
- **Chat sidebar** — high-level questions about the current batch only (the LLM gets a compact summary, never raw rows). e.g. "How many rows need review?", "Top cost categories?".

**Data source:** upload a GO Report `.xlsx` in the sidebar, or leave it empty to use the bundled file. Uploaded files are *batches to classify*; confirmed decisions always append back to the canonical learning workbook so knowledge accrues in one place.

---

## Project structure

```
FinopsAssist/
├── src/
│   ├── main.py            # Bedrock client + get_llm() + connectivity test
│   ├── matcher.py         # Engine: exact lookup + char n-gram classifier
│   ├── agent.py           # EnrichmentAgent — tool-using investigator
│   ├── agent_tools.py     # Agent tools (find_similar_mappings live; rest stubbed)
│   ├── review.py          # Matcher→agent bridge + feedback-loop write-back
│   ├── run_pipeline.py    # Headless deterministic 4-step batch pipeline
│   ├── benchmark.py       # Group-split accuracy & calibration benchmark
│   └── app.py             # Multi-feature Streamlit app (Home + Cost Allocation)
├── GO Report Extract LIGHT_V2.xlsx   # Input data (learning + empty sheets)
├── GO_predictions.xlsx               # Output of run_pipeline.py (generated)
├── pyproject.toml
├── .env                              # AWS credentials + config (gitignored)
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # macOS/Linux  (.venv\Scripts\activate on Windows)
pip install -e ".[dev]"
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

```bash
python src/main.py            # test Bedrock connectivity
streamlit run src/app.py      # interactive dashboard (recommended)
python src/run_pipeline.py    # headless batch → GO_predictions.xlsx
python src/benchmark.py       # accuracy & calibration (group split by account)
python src/agent.py           # single-row agent demo
```

---

## Data requirements

Input file: `GO Report Extract LIGHT_V2.xlsx` with these sheets:

| Sheet | Purpose | Key columns |
|---|---|---|
| `GO_MAPPING_LEARNING` | Historical mappings (reference) | `Custom.focus_costs[SubAccountName]`, `…[axa_Azure_ResourceGroupName]`, `…[axa_tags_global_dcs]`, `…[axa_tags_global_app]`, `Custom.gld_referential_mdm_axagorechargingitem[Recharging_Item_ID]` |
| `GO_MAPPING_EMPTY` | New items to predict | `focus_costs[SubAccountName]`, `…[axa_Azure_ResourceGroupName]`, `…[axa_tags_global_dcs]`, `…[axa_tags_global_app]` |

Column names differ between the two sheets (`Custom.focus_costs[...]` vs `focus_costs[...]`); handled via separate mappings in `matcher.py`.

---

## Tuning

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `confidence_threshold` | `matcher.py` `predict()` | `50.0` | Lower = fewer rows flagged for review; higher = more conservative |
| `NGRAM_RANGE` | `matcher.py` constant | `(2, 4)` | Character n-gram size for the classifier (swept in `benchmark.py`) |
| `CLF_C` | `matcher.py` constant | `50.0` | Logistic-regression regularization (higher = less regularization) |
| `MAX_STEPS` | `agent.py` constant | `5` | Max tool-use iterations per review row |
| Rows to investigate | app Review Queue | `10` | Cap on agent calls per run (each is a live Bedrock call) |

---

## Potential improvement — a value-of-information arbiter

The current design treats the cloud/agent as a **rescuer**: it only fires on rows the classifier already doubts (low confidence). Rethinking the problem from first principles suggests a better topology, given two real constraints — cloud tags are **partial / inconsistent** (evidence, not ground truth) and Azure/AWS access is **per-account lookups only** (enrichment is scarce, can't be run on every row).

**Core insight.** A `Recharging_Item_ID` is a *cost-ownership* label; name/tags/history are only proxies for it. The classifier's ~79% ceiling on new accounts is an **information** limit, not a model limit — the input underdetermines the answer. The cloud can supply the missing signal, but it's noisy and rationed, so the goal is to spend a lookup **only where it will change the decision**.

This means escalating on **novelty and cost**, not just low confidence — because the classifier's worst failure is being *confidently wrong* on accounts unlike anything in history, which a confidence threshold never catches.

Proposed routing brain (a small change in front of the existing agent):

1. **Novelty signal** — add *max similarity of the account to history* as a first-class routing feature. Low similarity ⇒ the classifier is extrapolating ⇒ enrich, even if its probability looks peaked. *(Highest-leverage first build.)*
2. **Cost weighting** — rank flagged rows by `expected_accuracy_gain × Sumaxa_EffectiveCost_EUR`. A wrong recharge on a €50k account matters far more than on a €5 one; spend scarce lookups and human attention on the high-spend, high-uncertainty rows first. (The EUR column already exists and is currently unused.)
3. **Cloud as weighted evidence + agreement check** — when a per-account lookup returns an owner/cost-centre tag, cross-check it against what history mapped similar accounts to: **agree → auto-accept at high confidence; disagree → surface as the highest-value human review** with both sides. This turns inconsistent tags from a liability into a calibration signal.
4. **Aggressive caching** — cache lookups by account ID. Accounts recur month to month, so the rationed API is hit ~once per account ever; incremental cost approaches zero after the first pass, which is what makes selective enrichment affordable.

Net: the cheap classifier still decides most rows; a *selective* cloud-enrichment + reconciliation step is spent only where novelty and spend say it pays off. Everything worth keeping today (exact lookup, calibrated classifier, candidate-list guardrail, feedback loop) stays — only the routing logic in `matcher.py` changes.

---

## Known limits / next steps

- **Connect the stubbed agent tools** (cloud/CMDB/wiki) — the biggest unlock; turns the agent from "smart pattern-matcher over history" into one that resolves accounts history can't explain.
- **Agent throughput/cost** — ~40% of rows are flagged; each agent investigation is a live LLM call, so the queue is capped per run. Batch/cache for full-batch enrichment.
- **Write-back concurrency** — `commit_decision` rewrites the workbook; safe for a small team one-at-a-time. Move the learning store to SQLite when concurrent edits matter.
