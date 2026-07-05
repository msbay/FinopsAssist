# Frontend Integration Spec — FinOps Assistant API

For the Next.js team. This maps every screen to the backend endpoints it calls, the exact
payloads, and the one non-obvious flow (the background LLM job you poll). The backend is a
FastAPI service and is **reused unchanged** — this is the whole integration surface.

- **Live contract:** the backend serves an OpenAPI schema at **`/docs`** (Swagger) and
  **`/openapi.json`**. You can codegen a typed TypeScript client from it.
- **Reference client:** [`src/frontend/api_client.py`](../src/frontend/api_client.py) is a
  ~60-line list of every call and its payload (the current Streamlit client).
- **Schemas source of truth:** [`src/backend/api.py`](../src/backend/api.py).

---

## 1. Integration model

Use a **backend-for-frontend (BFF)** pattern:

```
User (SSO) → Next.js (your auth shell) ──server-side fetch──► FastAPI backend
                                         (inject service token)  (internal ClusterIP, no public Route)
```

- The **browser never calls FastAPI directly** — Next.js server routes proxy to it. So there is
  **no CORS** to configure, and the backend stays network-internal.
- Inject a **service token** on the server-side call: header `Authorization: Bearer <token>`,
  where the token is the backend's `FINOPS_API_TOKEN` (delivered to both pods via a Secret). The
  backend validates it on every route; **`/health`, `/docs`, `/redoc`, `/openapi.json` are open**
  (so probes and Swagger keep working). A missing/wrong token returns **401**. When
  `FINOPS_API_TOKEN` is unset on the backend (local dev only) auth is disabled.
- Pass the **authenticated SSO username** as `reviewed_by` on commits (see §6) so the audit trail
  records the real person, not `"ui"`.
- **Base URL:** one env var, e.g. `FINOPS_API_URL` (today defaults to `http://127.0.0.1:8000`).

---

## 2. State model — read this first

- A **batch** is one pipeline run. Everything is keyed by a `batch_id` (string) returned from
  `POST /batches`. The frontend holds only that id.
- **The backend keeps batch state in memory (single replica).** A backend restart drops all
  batches → any `/batches/{id}` call then returns **404**. Handle 404 by clearing the stored
  `batch_id` and prompting the user to re-run (see §7).
- Rows are returned **sorted by € cost descending** by the server. You don't need to re-sort.

---

## 3. Endpoint reference

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/health` | — | `{ "status": "ok" }` |
| `POST` | `/batches` | `multipart/form-data`, optional field `file` (`.xlsx`) | `BatchSummary` (201) |
| `GET` | `/batches/{id}` | — | `BatchSummary` |
| `GET` | `/batches/{id}/rows?bucket=` | — (query: `all`\|`approve`\|`review`\|`done`) | `Row[]` |
| `POST` | `/batches/{id}/review` | — | `ReviewStatus` (starts/echoes the LLM job) |
| `GET` | `/batches/{id}/review` | — | `ReviewStatus` (poll this) |
| `POST` | `/batches/{id}/decisions` | `CommitRequest` | `CommitResult` |
| `POST` | `/batches/{id}/reroute` | `{ "row_ids": number[] }` | `{ "rerouted": number }` |
| `GET` | `/batches/{id}/history` | — | `HistoryEntry[]` |
| `POST` | `/batches/{id}/recall` | `{ "approval_id": string }` | `{ "recalled": bool, "learning_row_removed": bool }` |

If `file` is omitted from `POST /batches`, the server uses its bundled workbook (useful for demos).

---

## 4. Data shapes (TypeScript)

```ts
interface ReviewStatus {
  running: boolean;
  done: number;
  total: number;
  error: string | null;   // non-null ⇒ the LLM job failed; rows still exist, minus agent fields
}

interface BatchSummary {
  batch_id: string;
  total_rows: number;
  ready_to_approve: number;
  to_review: number;
  approved: number;
  total_spend_eur: number;
  approve_ready_spend_eur: number;
  review_spend_eur: number;
  review: ReviewStatus;   // embedded job status at fetch time
}

interface Row {
  row_id: number;                       // stable id used in commit/reroute
  cost_eur: number;
  sub_account_name: string;
  resource_group: string;
  tag_dcs: string;
  tag_app: string;
  predicted_recharging_item_id: string; // the CLASSIFIER's pick
  confidence: number;                   // 0–100, classifier (green ≥70 / amber 50–69 / red <50)
  suggested_action: string;
  agent_prediction: string;             // the LLM AGENT's pick (review rows only; "" until job done)
  agent_confidence: number;             // 0–100, agent (render on a distinct blue scale)
  agent_explanation: string;
  agent_tokens: number;                 // LLM tokens spent on this row (0 = handled w/o an LLM call)
}

interface HistoryEntry {
  approval_id: string;                  // used by recall
  row_id: number;
  name: string;                         // account name
  recharging_item_id: string;
  reviewed_at: string;                  // ISO-8601 UTC
  source: string;
}

interface Decision { row_id: number; recharging_item_id: string; } // recharging_item_id must be non-empty
interface CommitRequest { reviewed_by: string; decisions: Decision[]; }
interface CommitResult { committed: number; skipped: number; }
```

---

## 5. Screen-by-screen mapping

There is one live feature — **Cost Allocation** — with a run action and three tabs. (The
Streamlit "Home" hub is just navigation; your portal replaces it.)

### 5.0 Run pipeline (the entry action)

1. Optional: user uploads a GO report `.xlsx`.
2. `POST /batches` with the file (or none) → store `batch_id` from the returned `BatchSummary`.
3. **Immediately** `POST /batches/{id}/review` to kick off the LLM analysis of the review rows
   (a background job on the server).
4. Begin polling (see §8) and render the overview + tabs from the `BatchSummary`.

> `POST /batches` runs train + predict synchronously (seconds; the **first** call is slower — cold
> imports + index build). Show a spinner. The LLM work does **not** block this request.

### 5.1 Overview cards

Source: `GET /batches/{id}` → `BatchSummary`.

- **Total rows** = `total_rows` · **Ready to approve** = `ready_to_approve` · **To review** = `to_review`
- **Total spend** = `total_spend_eur`
- **Approve-ready spend** = `approve_ready_spend_eur` (also show as % of total)
- **Spend to review** = `review_spend_eur` (also as %)

### 5.2 Tab — ✅ Ready to approve

- **Load:** `GET /batches/{id}/rows?bucket=approve`.
- **Columns:** `cost_eur`, `predicted_recharging_item_id`, `confidence` (colour-band it),
  `sub_account_name`, `resource_group`, `tag_dcs`, `tag_app`.
- **Actions** (multi-select rows):
  - **Approve selected** → `POST /batches/{id}/decisions` with
    `decisions: [{ row_id, recharging_item_id: predicted_recharging_item_id }]`.
  - **Reject → send to review** → `POST /batches/{id}/reroute` with `{ row_ids: [...] }`.
- **After the call:** refetch this bucket + the summary.

### 5.3 Tab — 🔍 Review queue

- **Load:** `GET /batches/{id}/rows?bucket=review`.
- **While the LLM job runs:** show a progress bar from the polled `ReviewStatus` (§8); the
  `agent_prediction`/`agent_confidence`/`agent_explanation` fields are `""`/`0` until it finishes.
- **Columns:** `cost_eur`, `sub_account_name`, `resource_group`,
  **Agent prediction** (`agent_prediction`), **Agent confidence** (`agent_confidence`, blue scale),
  **Classifier prediction** (`predicted_recharging_item_id`), **Classifier confidence**
  (`confidence`), **Explanation** (`agent_explanation`), **Tokens** (`agent_tokens`).
- **Actions** (multi-select rows):
  - **Approve & commit** → `POST .../decisions` with `recharging_item_id: agent_prediction`.
  - **Reject & correct** → a text input for a `recharging_item_id`, then `POST .../decisions`
    with that typed value. Disable until the input is non-empty.
- **After the call:** refetch this bucket + the summary.

> Note the difference: **Ready-to-approve commits the _classifier's_ pick; the Review queue
> commits the _agent's_ pick (or a human correction).** Same endpoint, different `recharging_item_id`.

### 5.4 Tab — 🗂 History

- **Load:** `GET /batches/{id}/history` → `HistoryEntry[]` (most recent first).
- **Columns:** `name`, `recharging_item_id`, `reviewed_at`, `source`.
- **Action per row:** **Recall** → `POST /batches/{id}/recall` with `{ approval_id }`. This removes
  the row from the learning store and re-opens the account for review.
- **After the call:** refetch history + summary (+ the review bucket, since a row re-appears there).

---

## 6. Business rules the UI must enforce

- **Nothing is ever auto-committed** — every approval is an explicit user action.
- `recharging_item_id` in a decision **must be non-empty** (the API rejects blanks with 422).
  Validate client-side, especially for "Reject & correct".
- Send `reviewed_by` = the **authenticated SSO username** on every commit.
- Confidence rendering: classifier `confidence` uses green ≥70 / amber 50–69 / red <50; render
  `agent_confidence` on a **distinct** (e.g. blue) scale so the two aren't confused.
- After any mutating call (`decisions`, `reroute`, `recall`), refetch the **summary** and the
  **affected bucket(s)** — counts and rows move between buckets server-side.

---

## 7. Error handling

| Status | Meaning | UI behaviour |
|---|---|---|
| `404` on any `/batches/{id}` | Batch not found — backend restarted (in-memory state) | Clear stored `batch_id`; prompt "Run pipeline" again |
| `422` | Validation (e.g. empty `recharging_item_id`) | Prevent client-side; surface field error |
| `ReviewStatus.error != null` | The LLM job failed (e.g. Bedrock creds) | Show a warning; rows still work — user can enter values manually |
| network/5xx | Backend down | Surface a retriable error; don't crash the page |

---

## 8. The polling flow (the one tricky part)

The LLM analysis of the review queue is a **server-side background job**. You start it once and
poll for progress.

```
POST /batches                      → { batch_id, review: { running:false, ... } }
POST /batches/{id}/review          → { running:true, done:0, total:N }   // kicks off the job
                                       (idempotent: calling again while running just echoes status)

loop every ~2s:
  GET /batches/{id}/review         → { running, done, total, error }
    • running == true   → show progress: done / total
    • total === 0       → no review rows; treat as done, stop
    • running == false  → job finished:
                            - refetch GET /batches/{id}/rows?bucket=review  (agent_* now filled)
                            - refetch GET /batches/{id}                     (summary)
                            - stop polling
    • error != null     → stop polling, show warning (manual entry still works)
```

Notes:
- Poll only while the Review tab is relevant; a 2s interval matches the current UX.
- The job processes the **highest-€ rows first**, so partial results are the most valuable rows.
- Users can work in the **Ready-to-approve** tab while the review job runs — it's independent.

---

## 9. What you do NOT need to build

- No ML, no model calls, no Excel handling — all server-side.
- No direct auth to Bedrock/data — the backend owns that.
- No client-side sorting of rows by cost — the server returns them €-descending.
