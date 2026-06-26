# FinOps Co-Pilot

Automated Recharging Item ID prediction for the monthly GO Report mapping process. Combines fast deterministic matching with targeted LLM reasoning, exposed through an interactive dashboard.

---

## Problem

Each month, ~800 new cloud accounts/subscriptions/resource groups appear in the GO Report that need to be mapped to a `Recharging_Item_ID` (one of ~76 cost categories). Today this is done manually by analysts who compare new entries against ~3,800 historically mapped ones, looking at account names, resource groups, and tags. This is slow, repetitive, and error-prone.

## Solution Architecture

The system follows a three-layer design that separates deterministic processing from LLM reasoning from user interaction:

```
┌──────────────────────────────────────────────────────────┐
│                    INTERFACE (app.py)                     │
│  Streamlit dashboard: table, filters, overrides, chat    │
├──────────────────────────────────────────────────────────┤
│                 BRAIN (LLM checkpoint)                    │
│  Called only for low-confidence rows (~5%)                │
│  Adds Reasoning_Justification column                     │
├──────────────────────────────────────────────────────────┤
│              ENGINE (matcher.py + run_pipeline.py)        │
│  Deterministic: load → validate → match → output         │
│  Embeddings + fuzzy matching, confidence scoring          │
└──────────────────────────────────────────────────────────┘
```

### Why this design?

| Alternative considered | Why we rejected it |
|---|---|
| Full LLM agent deciding tool order | Non-deterministic execution is a compliance risk for a financial pipeline. The LLM might skip validation or change behavior across model versions. |
| LLM matching all 811 rows | Slow (~3 min vs. ~3 sec), expensive, non-deterministic, and no real numeric confidence score. Embeddings + fuzzy matching are faster, cheaper, and give explainable scores. |
| Chat-only interface for overrides | Typing "override row 47 to PSO_ITM_361" for 50 rows is unusable. Dropdowns and clicks are faster for data correction. |
| Feeding raw rows to LLM for filtering | Context window bloat and unnecessary cost. Filtering is a pandas operation, not an LLM task. |

---

## Matching Logic (`src/matcher.py`)

### Text Representation

Each row is converted to a single string for comparison:

```
{SubAccountName} | {ResourceGroupName} | {tag_dcs} | {tag_app}
```

**Why these fields?**
- `SubAccountName` — primary identifier, carries naming conventions (e.g. `prod-axa-dcs-01`)
- `ResourceGroupName` — highly informative for Azure resources (e.g. `z-ago-finops-cfp-ew1-rg01`), always present. The original proposal ignored this field — we added it because it's the most reliable secondary signal.
- `tag_dcs` / `tag_app` — useful when present, but **35% null** in the data, so they act as tiebreakers, not primary signals.

`SubAccountId` is excluded from the text representation because it's a numeric/GUID identifier with no semantic or lexical similarity to other IDs.

### Hybrid Scoring (40% Semantic / 60% Fuzzy)

Each new item is compared against all ~3,700 reference items using two complementary signals:

**1. Semantic similarity (40% weight)** — `sentence-transformers` with `BAAI/bge-small-en-v1.5`
- Embeds the text representation into a 384-dim vector
- Cosine similarity against all reference vectors
- Catches "means the same thing but worded differently"
- At ~3,700 reference items, this is an in-memory NumPy dot product — no vector database needed

**2. Fuzzy string matching (60% weight)** — `rapidfuzz` token-sort ratio
- Character-level similarity normalized to 0-1
- Catches naming convention variants: `prod-axa-dcs-01` vs `prod-axa-dcs-02`
- Critical because cloud account names are structured identifiers, not natural language

**Why 60% fuzzy / 40% semantic (not the reverse)?**

Cloud infrastructure naming follows strict conventions. A semantic model doesn't know that `prod` means production or that `dcs` is a division code, but fuzzy matching catches these trivially. We weight fuzzy higher because the data is structured identifiers, not prose. This can be tuned after evaluating results on real data.

### Embedding Model Choice: `BAAI/bge-small-en-v1.5`

| Model | Size | Why / why not |
|---|---|---|
| **`BAAI/bge-small-en-v1.5`** (chosen) | 33 MB | Fast, small, runs locally. Short structured strings don't need a large model. |
| `BAAI/bge-m3` | 2 GB | Overkill. Designed for long multilingual documents, not 5-word account names. |
| `text-embedding-3-small` (OpenAI) | API | Data residency concern — internal cost/ownership data should not leave the infrastructure. |
| `EmbeddingGemma-300M` | 300 MB | Good alternative if multilingual support is needed later. |

The model runs **fully on-premises** with no external API calls for the matching step. It can be swapped by changing the `model_name` parameter.

### Confidence Score (0–100)

Confidence combines three signals into a single number:

```
confidence = (0.5 × best_score + 0.3 × margin + 0.2 × agreement) × 100
```

| Signal | Weight | What it measures |
|---|---|---|
| `best_score` | 50% | How similar is the top match (hybrid score) |
| `margin` | 30% | Gap between the best match and the best match with a *different* Recharging_Item_ID. High margin = clear winner. |
| `agreement` | 20% | Fraction of top-5 neighbors sharing the same ID. 5/5 = unambiguous. |

- **≥ 70**: High confidence (green) — auto-accept
- **50–69**: Medium confidence (yellow) — worth checking
- **< 50**: Low confidence (red) — flagged for review, sent to LLM

The threshold (`confidence_threshold=50.0`) controls where "needs review" starts. Adjustable based on the team's risk tolerance.

### Data Cleaning

- Rows in `GO_MAPPING_LEARNING` with a null `Recharging_Item_ID` are excluded from the index (4% of rows)
- `NaN` values in tags are filtered out of the text representation (not passed as the string "nan")
- Column names differ between sheets (`Custom.focus_costs[...]` vs `focus_costs[...]`) — handled via separate column mappings

---

## Pipeline (`src/run_pipeline.py`)

A rigid, deterministic 4-step sequence. The LLM never decides the order — it's called at a fixed checkpoint.

### Step 1: Load & Validate
- Reads both sheets from the Excel file
- Verifies all required columns exist (fails fast with a clear error)
- Warns if any rows have missing `SubAccountName`

### Step 2: Build Index & Predict
- Creates embeddings for all reference items
- Runs hybrid matching for every row in `GO_MAPPING_EMPTY`
- Pure Python/NumPy — deterministic, no external calls

### Step 3: LLM Checkpoint (Bedrock)
- **Only fires for rows where `Needs_Review = True`** (typically ~5%)
- Sends each ambiguous row + its top-3 candidates to the LLM
- The LLM picks the best match and writes a one-sentence justification
- If Bedrock is unavailable, the pipeline still completes — justification column is left empty
- This is the only non-deterministic step, and it's isolated and optional

### Step 4: Save Results
- Writes `GO_predictions.xlsx` with all original columns plus:
  - `Predicted_Recharging_Item_ID`
  - `Confidence`
  - `Top_Matches` (top-3 for audit trail)
  - `Needs_Review` (boolean)
  - `LLM_Justification` (for reviewed rows)
- Results are sorted by confidence ascending (worst first for quick review)

---

## Dashboard (`src/app.py`)

Interactive Streamlit UI. The analyst sees data, not a chatbox.

### Main Panel
- **Metrics bar**: total rows, high confidence count, needs review count, average confidence
- **Color-coded data table**: green (≥70), yellow (50–69), red (<50)
- **Filters**: by review status and by predicted ID (native pandas filtering, no LLM)
- **Manual override**: select row index → pick correct ID from dropdown → apply. No typing.
- **Excel export**: download the final results including any overrides

### Chat Sidebar
- For **high-level analysis only**, not for data manipulation
- The LLM receives a compact summary (counts, averages, top IDs) — never raw rows
- If the user asks to filter data, the LLM returns a pandas query to execute server-side
- Examples: "Why did unmatched rows spike this month?", "Summarize the main cost categories"

---

## Project Structure

```
FinopsAgent/
├── src/
│   ├── main.py            # LLM setup (Bedrock client, get_llm helper)
│   ├── matcher.py         # Hybrid matching engine
│   ├── run_pipeline.py    # Deterministic 4-step pipeline
│   └── app.py             # Streamlit dashboard
├── GO Report Extract LIGHT_V2.xlsx   # Input data
├── GO_predictions.xlsx               # Output (generated)
├── pyproject.toml
├── .env                   # AWS credentials + config
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and configure:

```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
```

## Run

```bash
# CLI pipeline (headless, writes GO_predictions.xlsx)
python src/run_pipeline.py

# Interactive dashboard
streamlit run src/app.py

# Test Bedrock connectivity
python src/main.py test
```

---

## Data Requirements

Input file: `GO Report Extract LIGHT_V2.xlsx` with these sheets:

| Sheet | Purpose | Key columns |
|---|---|---|
| `GO_MAPPING_LEARNING` | Historical mappings (reference) | `Custom.focus_costs[SubAccountName]`, `Custom.focus_costs[axa_Azure_ResourceGroupName]`, `Custom.focus_costs[axa_tags_global_dcs]`, `Custom.focus_costs[axa_tags_global_app]`, `Custom.gld_referential_mdm_axagorechargingitem[Recharging_Item_ID]` |
| `GO_MAPPING_EMPTY` | New items to predict | `focus_costs[SubAccountName]`, `focus_costs[axa_Azure_ResourceGroupName]`, `focus_costs[axa_tags_global_dcs]`, `focus_costs[axa_tags_global_app]` |

---

## Tuning

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `semantic_weight` | `matcher.py` `_hybrid_scores()` | `0.4` | Higher = more weight on meaning; lower = more weight on character similarity |
| `confidence_threshold` | `matcher.py` `predict()` | `50.0` | Lower = fewer rows flagged for review; higher = more conservative |
| `top_k` | `matcher.py` `predict()` | `5` | Number of neighbors for agreement scoring |
| `model_name` | `RechargingMatcher.__init__()` | `BAAI/bge-small-en-v1.5` | Swap embedding model without code changes |
