"""FinOps Assistant — Streamlit UI (thin client).

The FRONTEND of the front/back split. It renders the UI and talks to the backend ONLY
through api_client (HTTP) — it imports no matcher / review / agent / model code. Point it
at a different backend with the FINOPS_API_URL env var.

Run the backend first, then this UI (from the project root):
    uvicorn api:app --app-dir src/backend   # backend  → http://127.0.0.1:8000
    streamlit run src/frontend/app.py       # frontend → http://localhost:8501
"""

import os
import sys

# Put this module's own dir on the path so `import api_client` resolves.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import api_client
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FinOps Assistant", layout="wide", page_icon="💸")
st.markdown("<style>.block-container{padding-top:2.5rem;}</style>", unsafe_allow_html=True)


def init_state():
    if "batch_id" not in st.session_state:
        st.session_state.batch_id = None


init_state()


# ── Styling helpers (presentation only) ───────────────────────────────────────
def color_confidence(val):
    if val >= 70:
        return "background-color: #c6efce; color: #006100"
    if val >= 50:
        return "background-color: #ffeb9c; color: #9c5700"
    return "background-color: #ffc7ce; color: #9c0006"


def color_agent(val):
    # Blue scale — distinct from the classifier's green/yellow/red; darker = more certain.
    if val >= 70:
        return "background-color: #9ec5ff; color: #0a2e6b"
    if val >= 50:
        return "background-color: #d6e4ff; color: #0a3d91"
    return "background-color: #eef3fc; color: #5b6b8c"


def stat_card(label, value_html, help_text=""):
    tip = f' title="{help_text}"' if help_text else ""
    return (
        "<div style='border:1px solid rgba(49,51,63,0.2);border-radius:0.6rem;"
        "padding:0.75rem 1rem;height:6rem;box-sizing:border-box;display:flex;"
        "flex-direction:column;justify-content:space-between'>"
        f"<div style='font-size:0.8rem;color:#6b7280'{tip}>{label}</div>{value_html}</div>")


def big(txt):
    return f"<span style='font-size:1.9rem;font-weight:600;color:#1A1A2E'>{txt}</span>"


def pct_eur(pct_str, eur, bg, fg):
    return (
        "<div style='display:flex;align-items:baseline;justify-content:space-between;"
        f"gap:0.5rem'>{big(pct_str)}"
        f"<span style='background:{bg};color:{fg};padding:2px 10px;border-radius:10px;"
        f"font-weight:600;font-size:0.9rem'>€{eur:,.0f}</span></div>")


def card(col, *args, **kwargs):
    col.markdown(stat_card(*args, **kwargs), unsafe_allow_html=True)


def _do(fn, msg):
    """Run a backend call, then toast + rerun; surface errors instead of crashing."""
    try:
        fn()
        st.success(msg)
        st.rerun()
    except Exception as e:  # noqa: BLE001
        st.error(f"Backend error: {e}")


# ── Catalogue of features (drives the Home cards + nav) ────────────────────────
FEATURES = [
    {"key": "cost", "icon": "🏷️", "title": "Cost Allocation", "status": "live",
     "desc": "Map new cloud accounts to a Recharging_Item_ID using history + an "
             "investigation agent, with human review and a learning feedback loop."},
    {"key": "tags", "icon": "🛠️", "title": "Tag Remediation", "status": "soon",
     "desc": "Audit mandatory tags (owner, cost-centre) and propose the correct tag value "
             "for non-compliant resources — the judgment tag-coverage tools leave to you."},
]


# ── Page: Home ────────────────────────────────────────────────────────────────
def home_page():
    st.markdown("<h1 style='text-align:center; margin-bottom:0'>💸 FinOps Assistant</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#6b7280; font-size:1.05rem'>"
                "A hub for FinOps tasks. Pick a feature below or from the sidebar.</p>",
                unsafe_allow_html=True)
    st.divider()
    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        with cols[i % 2]:
            with st.container(border=True, height=280):
                live = feat["status"] == "live"
                st.markdown(f"### {feat['icon']} {feat['title']}")
                st.caption("🟢 Available" if live else "⚪ Coming soon")
                st.write(feat["desc"])
                if live:
                    st.page_link(PAGES_BY_KEY[feat["key"]], label="Open →")


# ── "How it works" explainer (top-right dialog) ───────────────────────────────
@st.dialog("How it works", width="large")
def _how_it_works_dialog():
    st.markdown(
        "When you click **Run pipeline**, the backend **trains a model on the fly** from "
        "your history and predicts the new accounts — nothing is pre-trained. Confirmed "
        "mappings are appended to history and take effect on the next run.")
    st.code(
        "Streamlit UI ──HTTP──►  FastAPI backend\n"
        "                          │\n"
        "  ① Train + predict  ◄────┤  exact-lookup + char n-gram classifier\n"
        "  ② Split rows:           │\n"
        "     ✅ ready to approve   │  (confidence ≥ threshold)\n"
        "     🔍 to review          │  (low confidence)\n"
        "  ③ LLM agent analyses the review rows (background job) → best guess + reason\n"
        "  ④ Human approves / corrects  ──►  appended to history  ──►  improves next run ⟲",
        language="text")
    st.markdown(
        "**Confidence** 🟢 ≥70 · 🟡 50–69 · 🔴 <50 — the model's own probability for its top "
        "pick. **Ready to approve** = classifier-confident rows; **Review queue** = "
        "everything else, each carrying the LLM's prediction, confidence and explanation.")


# ── Page: Cost Allocation (thin client) ───────────────────────────────────────
def cost_allocation_page():
    with st.sidebar:
        st.header("Data source")
        st.caption("Data is pulled directly from Databricks and the AWS accounts API — "
                   "no upload needed.")
        if st.button("Run pipeline", type="primary", width="stretch"):
            with st.spinner("Pulling data, training & predicting on the server…"):
                try:
                    summary = api_client.run_batch()
                    st.session_state.batch_id = summary["batch_id"]
                    api_client.start_review(summary["batch_id"])  # LLM runs on the server
                except Exception as e:  # noqa: BLE001
                    st.error(f"Backend error: {e}")
        st.caption(f"Backend: `{api_client.API_URL}`")

    _, icol = st.columns([5, 1])
    if icol.button("ℹ️ How it works", width="stretch"):
        _how_it_works_dialog()
    st.markdown("<h1 style='text-align:center; margin-bottom:0'>🏷️ Cost Allocation</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#6b7280; font-size:1.05rem'>"
                "Map new cloud accounts to a Recharging_Item_ID.</p>", unsafe_allow_html=True)

    bid = st.session_state.batch_id
    if not bid:
        st.info("Click **Run pipeline** in the sidebar to start — data is pulled directly "
                "from Databricks and the AWS accounts API.")
        return

    try:
        summary = api_client.summary(bid)
        hist = api_client.history(bid)
    except Exception as e:  # noqa: BLE001
        st.error(f"Backend unavailable at `{api_client.API_URL}`: {e}")
        return

    tot_eur = summary["total_spend_eur"] or 0.0

    def pct(part):
        return f"{part / tot_eur * 100:.0f}%" if tot_eur else "—"

    m1, m2, m3 = st.columns(3)
    card(m1, "Total rows", big(summary["total_rows"]))
    card(m2, "✅ Ready to approve", big(summary["ready_to_approve"]),
         "High confidence — a human selects & batch-approves them.")
    card(m3, "🔍 To review", big(summary["to_review"]),
         "Everything not high-confidence — a human reviews it; the LLM is an optional assist.")
    st.write("")
    e1, e2, e3 = st.columns(3)
    card(e1, "Total spend", big(f"€{tot_eur:,.0f}"))
    card(e2, "Approve-ready spend",
         pct_eur(pct(summary["approve_ready_spend_eur"]), summary["approve_ready_spend_eur"],
                 "#c6efce", "#006100"))
    card(e3, "Spend to review",
         pct_eur(pct(summary["review_spend_eur"]), summary["review_spend_eur"],
                 "#ffeb9c", "#9c5700"))
    st.write("")

    tab_approve, tab_review, tab_history = st.tabs(
        [f"✅ Ready to approve ({summary['ready_to_approve']})",
         f"🔍 Review queue ({summary['to_review']})",
         f"🗂 History ({len(hist)})"])
    with tab_approve:
        _approve_tab(bid)
    with tab_review:
        _review_tab(bid, summary["review"])
    with tab_history:
        _history_tab(bid, hist)


def _selected(event, rows):
    return [rows[p] for p in (event.selection.rows if event and event.selection else [])]


def _download_csv(df: pd.DataFrame, filename: str, key: str) -> None:
    """A right-aligned 'Download CSV' button exporting the exact table shown above it."""
    _, right = st.columns([3, 1])
    right.download_button("⬇ Download CSV", df.to_csv(index=False).encode("utf-8"),
                          file_name=filename, mime="text/csv", key=key, width="stretch")


def _is_aws(r: dict) -> bool:
    return "aws" in str(r.get("provider", "")).lower()


def _provider_sections(rows: list[dict]):
    """Split rows into per-provider sections (label, emoji, is_aws, key-suffix, subset),
    each present only if it has rows — so AWS and Azure render as separate tables with
    the columns that apply to each."""
    aws = [r for r in rows if _is_aws(r)]
    azure = [r for r in rows if not _is_aws(r)]
    out = []
    if aws:
        out.append(("AWS", "☁️", True, "aws", aws))
    if azure:
        out.append(("Azure", "🔷", False, "azure", azure))
    return out


def _table_height(n: int) -> int:
    """Fit the table to its rows (up to a cap), so stacked provider tables don't each
    reserve a tall fixed pane."""
    return min(460, 70 + 35 * n)


def _approve_df(rows: list[dict], is_aws: bool) -> pd.DataFrame:
    """Provider-relevant columns: AWS shows its enrichment (owner/name/dcs/description);
    Azure shows ResourceGroup + axa tags."""
    if is_aws:
        return pd.DataFrame([{
            "€ Cost": r["cost_eur"], "Recharging_Item_ID": r["predicted_recharging_item_id"],
            "Confidence": r["confidence"], "Account": r["sub_account_name"],
            "AWS name": r["aws_name"], "Owner": r["aws_owner"],
            "dcs": r["aws_dcs"], "Description": r["aws_desc"],
        } for r in rows])
    return pd.DataFrame([{
        "€ Cost": r["cost_eur"], "Recharging_Item_ID": r["predicted_recharging_item_id"],
        "Confidence": r["confidence"], "Subscription": r["sub_account_name"],
        "ResourceGroup": r["resource_group"], "dcs": r["tag_dcs"], "app": r["tag_app"],
    } for r in rows])


def _approve_block(bid, rows: list[dict], is_aws: bool, suffix: str) -> None:
    df = _approve_df(rows, is_aws)
    event = st.dataframe(
        df.style.map(color_confidence, subset=["Confidence"]),
        width="stretch", height=_table_height(len(rows)), on_select="rerun",
        selection_mode="multi-row", key=f"approve_table_{suffix}",
        column_config={"€ Cost": st.column_config.NumberColumn(format="€%.0f"),
                       "Confidence": st.column_config.NumberColumn(format="%d")})
    _download_csv(df, f"ready_to_approve_{suffix}.csv", f"dl_approve_{suffix}")
    sel = _selected(event, rows)
    n = len(sel)
    b1, b2, b3 = st.columns([1, 1, 2.2])
    # Committing is disabled for now — the Approve button is inert (nothing is written).
    b1.button(f"✓ Approve selected ({n})", type="primary", disabled=n == 0,
              width="stretch", key=f"approve_btn_{suffix}")
    reject = b2.button(f"↩ Send to review ({n})", disabled=n == 0,
                       width="stretch", key=f"reject_btn_{suffix}")
    b3.caption("Approving is **disabled for now** — nothing is committed. Use **⬇ Download "
               "CSV** above to export. **Send to review** just moves rows to the Review tab.")
    if reject and sel:
        _do(lambda: api_client.reroute(bid, [r["row_id"] for r in sel]),
            f"Moved {n} row(s) to the Review queue.")


def _approve_tab(bid):
    rows = api_client.rows(bid, "approve")
    if not rows:
        st.success("Nothing awaiting approval right now. 🎉")
        return
    st.caption("Review the predictions below and **⬇ Download CSV** to export them. AWS "
               "and Azure are shown separately, each with the columns that apply. "
               "(Committing is disabled for now.)")
    for label, emoji, is_aws, suffix, subset in _provider_sections(rows):
        st.markdown(f"#### {emoji} {label} ({len(subset)})")
        _approve_block(bid, subset, is_aws, suffix)


@st.fragment(run_every=2)
def _review_progress(bid):
    """Poll the backend's review job; auto-refresh only this fragment."""
    try:
        status = api_client.review_status(bid)
    except Exception:  # noqa: BLE001
        status = {"running": False, "done": 0, "total": 1}
    if not status.get("running"):
        st.rerun()
        return
    done, total = status.get("done", 0), max(status.get("total", 1), 1)
    st.progress(min(done / total, 1.0),
                f"🤖 LLM analyzing the review queue… {done}/{total}")
    st.caption("Runs on the server — you can keep working in the other tabs.")


_REVIEW_COLCFG = {
    "€ Cost": st.column_config.NumberColumn(format="€%.0f"),
    "Agent confidence": st.column_config.NumberColumn(
        "Agent conf.", format="%d",
        help="The LLM's confidence after re-analyzing the row (blue scale)."),
    "Classifier prediction": st.column_config.TextColumn(
        "Classifier pred. (top-1)",
        help="The classifier's own best guess, for comparison with the agent's."),
    "Classifier confidence": st.column_config.NumberColumn(
        "Classifier conf.", format="%d",
        help="The classifier's score — this is what routed the row into Review "
             "(below the approve threshold)."),
    "Agent explanation": st.column_config.TextColumn(width="large"),
    "Tokens": st.column_config.NumberColumn(
        "Tokens", format="%d",
        help="LLM tokens consumed analyzing this row (0 = handled without an "
             "LLM call, e.g. no-signal rows)."),
}


def _review_df(rows: list[dict], is_aws: bool) -> pd.DataFrame:
    """Provider-relevant evidence columns beside the agent/classifier predictions:
    AWS shows owner/name/dcs/description; Azure shows ResourceGroup + axa tags."""
    def common(r):
        return {
            "Agent prediction": r["agent_prediction"], "Agent confidence": r["agent_confidence"],
            "Classifier prediction": r["predicted_recharging_item_id"],
            "Classifier confidence": r["confidence"],
            "Agent explanation": r["agent_explanation"], "Tokens": r.get("agent_tokens", 0),
        }
    if is_aws:
        return pd.DataFrame([{
            "€ Cost": r["cost_eur"], "Account": r["sub_account_name"],
            "AWS name": r["aws_name"], "Owner": r["aws_owner"], "Description": r["aws_desc"],
            "dcs": r["aws_dcs"], **common(r),
        } for r in rows])
    return pd.DataFrame([{
        "€ Cost": r["cost_eur"], "Subscription": r["sub_account_name"],
        "ResourceGroup": r["resource_group"], "dcs": r["tag_dcs"], "app": r["tag_app"],
        **common(r),
    } for r in rows])


def _review_block(bid, rows: list[dict], is_aws: bool, suffix: str) -> None:
    df = _review_df(rows, is_aws)
    styler = (df.style
              .map(color_agent, subset=["Agent confidence"])
              .map(color_confidence, subset=["Classifier confidence"]))
    event = st.dataframe(
        styler, width="stretch", height=_table_height(len(rows)), on_select="rerun",
        selection_mode="multi-row", key=f"review_table_{suffix}", column_config=_REVIEW_COLCFG)
    _download_csv(df, f"review_queue_{suffix}.csv", f"dl_review_{suffix}")
    sel = _selected(event, rows)
    n = len(sel)
    # Committing is disabled for now — the Approve button is inert (nothing is written).
    b1, b2 = st.columns([1.2, 3])
    b1.button(f"✓ Approve ({n})", type="primary", disabled=n == 0,
              width="stretch", key=f"rev_approve_{suffix}")
    b2.caption("Approving is **disabled for now** — nothing is committed. Use **⬇ Download "
               "CSV** above to export the review queue.")


def _review_tab(bid, review_status):
    if review_status.get("running"):
        _review_progress(bid)
        return
    if review_status.get("error"):
        st.warning(f"LLM analysis failed: {review_status['error']} — you can still enter "
                   "values manually below.")
    rows = api_client.rows(bid, "review")
    if not rows:
        st.success("No rows to review. 🎉")
        return
    st.caption("The LLM analyzed the review rows. Review the agent's prediction and "
               "**⬇ Download CSV** to export. AWS and Azure are shown separately, each "
               "with the columns that apply. (Committing is disabled for now.)")
    for label, emoji, is_aws, suffix, subset in _provider_sections(rows):
        st.markdown(f"#### {emoji} {label} ({len(subset)})")
        _review_block(bid, subset, is_aws, suffix)


def _history_tab(bid, hist):
    if not hist:
        st.info("No approvals committed yet for this batch. Committed decisions appear here "
                "and can be recalled (removed from the learning data, re-opened for review).")
        return
    st.caption("Decisions committed for this batch. **Recall** removes the row from the "
               "learning data and re-opens the account for review.")
    widths = [2.5, 2, 2, 1.3, 1]
    head = st.columns(widths)
    for h, label in zip(head, ["Account", "Recharging_Item_ID", "When (UTC)", "Source", ""]):
        h.markdown(f"**{label}**")
    for a in hist:
        c1, c2, c3, c4, c5 = st.columns(widths, vertical_alignment="center")
        c1.write(a["name"])
        c2.write(a["recharging_item_id"])
        c3.caption(a["reviewed_at"])
        c4.caption(a["source"])
        if c5.button("↩ Recall", key=f"recall_{a['approval_id']}"):
            _do(lambda aid=a["approval_id"]: api_client.recall(bid, aid),
                f"Recalled {a['name']}.")


# ── Placeholder factory for not-yet-built features ────────────────────────────
def _coming_soon(feat: dict):
    def page():
        st.title(f"{feat['icon']} {feat['title']}")
        st.caption("⚪ Coming soon")
        st.info(feat["desc"])
    return page


# ── Navigation (the multi-feature switcher) ───────────────────────────────────
PAGES_BY_KEY = {
    "home": st.Page(home_page, title="Home", icon="🏠", default=True),
    "cost": st.Page(cost_allocation_page, title="Cost Allocation", icon="🏷️"),
}
for _feat in FEATURES:
    if _feat["status"] == "soon":
        PAGES_BY_KEY[_feat["key"]] = st.Page(
            _coming_soon(_feat), title=_feat["title"], icon=_feat["icon"],
            url_path=_feat["key"])

nav = st.navigation({
    "FinOps Assistant": [PAGES_BY_KEY["home"]],
    "Features": [PAGES_BY_KEY["cost"]] + [PAGES_BY_KEY[f["key"]]
                                          for f in FEATURES if f["status"] == "soon"],
})
nav.run()
