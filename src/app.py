"""FinOps Assistant — multi-feature Streamlit app.

The Assistant is a hub for several FinOps tasks. Each task is a "feature" page,
selected from the sidebar. Today:

  • Cost Allocation (live) — map new cloud accounts to a Recharging_Item_ID:
      Results tab (matcher predictions + confidence + Excel export),
      Review Queue tab (enrichment agent proposes among the matcher's candidates;
      human Accepts/Overrides; the decision is appended to GO_MAPPING_LEARNING).

  • Anomaly Detection, Optimization, Tag Hygiene (coming soon) — placeholders that
      show where the Assistant is headed; swap them for real features as built.

New features = a new page function + one entry in the navigation below.
"""

import io
import sys

sys.path.insert(0, "src")

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from main import get_llm
from matcher import EMPTY_COLS, RechargingMatcher
from review import commit_decision, parse_candidates, run_review

# The canonical learning store. Uploaded files are batches to classify; confirmed
# decisions always append back here so the institutional knowledge accrues in one place.
LEARNING_WORKBOOK = "GO Report Extract LIGHT_V2.xlsx"

st.set_page_config(page_title="FinOps Assistant", layout="wide", page_icon="💸")


def init_state():
    for key, default in {"results": None, "matcher": None, "reviewed": False,
                         "chat_messages": []}.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_state()


# ── Shared helpers (cost allocation) ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _build_matcher(learning_bytes: bytes | None):
    """Train the matcher on the learning sheet (cached on the file contents)."""
    src = io.BytesIO(learning_bytes) if learning_bytes else LEARNING_WORKBOOK
    matcher = RechargingMatcher()
    matcher.build_index(pd.read_excel(src, sheet_name="GO_MAPPING_LEARNING"))
    return matcher


def predict(batch_bytes: bytes | None):
    matcher = _build_matcher(batch_bytes)
    src = io.BytesIO(batch_bytes) if batch_bytes else LEARNING_WORKBOOK
    results = matcher.predict(pd.read_excel(src, sheet_name="GO_MAPPING_EMPTY"))
    return matcher, results


def color_confidence(val):
    if val >= 70:
        return "background-color: #c6efce; color: #006100"
    if val >= 50:
        return "background-color: #ffeb9c; color: #9c5700"
    return "background-color: #ffc7ce; color: #9c0006"


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


# ── Catalogue of features (drives the Home cards + nav) ────────────────────────
# status: "live" features are clickable; "soon" features render a placeholder.
FEATURES = [
    {"key": "cost", "icon": "🏷️", "title": "Cost Allocation", "status": "live",
     "desc": "Map new cloud accounts to a Recharging_Item_ID using history + an "
             "investigation agent, with human review and a learning feedback loop."},
    {"key": "tags", "icon": "🧹", "title": "Tag Hygiene", "status": "soon",
     "desc": "Audit mandatory tags (owner, cost-centre) and propose the correct tag value "
             "for non-compliant resources — the judgment Cloudability's Tag Explorer leaves to you."},
]


# ── Page: Home ────────────────────────────────────────────────────────────────
def home_page():
    st.title("💸 FinOps Assistant")
    st.caption("A hub for FinOps tasks. Pick a feature below or from the sidebar.")
    st.divider()

    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        with cols[i % 2]:
            with st.container(border=True):
                live = feat["status"] == "live"
                badge = "🟢 Available" if live else "⚪ Coming soon"
                st.markdown(f"### {feat['icon']} {feat['title']}")
                st.caption(badge)
                st.write(feat["desc"])
                if live:
                    st.page_link(PAGES_BY_KEY[feat["key"]], label="Open →")


# ── Page: Cost Allocation (the live feature) ──────────────────────────────────
def cost_allocation_page():
    # Feature-specific sidebar: data source + batch chat.
    with st.sidebar:
        st.header("Data source")
        upload = st.file_uploader(
            "Upload a GO Report extract (.xlsx) with GO_MAPPING_EMPTY + GO_MAPPING_LEARNING",
            type=["xlsx"])
        st.caption(f"No file → uses the bundled **{LEARNING_WORKBOOK}**.")
        if st.button("Run pipeline", type="primary", use_container_width=True):
            with st.spinner("Building index & predicting..."):
                batch_bytes = upload.getvalue() if upload else None
                matcher, results = predict(batch_bytes)
                st.session_state.matcher = matcher
                st.session_state.results = results
                st.session_state.reviewed = False

        st.divider()
        st.header("Ask the Assistant")
        st.caption("High-level questions about this batch")
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("e.g. How many rows need review?"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        df = st.session_state.results
                        summary = "No batch run yet."
                        if df is not None:
                            summary = (
                                f"{len(df)} rows, {int(df['Needs_Review'].sum())} need review, "
                                f"avg confidence {df['Confidence'].mean():.0f}%. "
                                f"Top IDs: {df['Predicted_Recharging_Item_ID'].value_counts().head(8).to_dict()}")
                        sys_msg = SystemMessage(content=(
                            "You are a FinOps analyst assistant. Answer concisely from this context.\n"
                            f"Batch summary: {summary}"))
                        answer = get_llm().invoke([sys_msg, HumanMessage(content=prompt)]).content
                    except Exception as e:  # noqa: BLE001
                        answer = f"LLM unavailable: {e}"
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    st.title("🏷️ Cost Allocation")
    st.caption("Map new cloud accounts to a Recharging_Item_ID.")
    results = st.session_state.results
    if results is None:
        st.info("Pick a data source in the sidebar and click **Run pipeline** to start.")
        return

    tab_results, tab_review, tab_about = st.tabs(
        ["📊 Results", f"🔍 Review Queue ({int(results['Needs_Review'].sum())})", "ℹ️ About"])

    # Tab 1: Results
    with tab_results:
        total = len(results)
        review_count = int(results["Needs_Review"].sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total rows", total)
        m2.metric("Auto (high confidence)", total - review_count)
        m3.metric("Needs review", review_count)
        m4.metric("Avg confidence", f"{results['Confidence'].mean():.0f}%")

        st.divider()
        f1, f2 = st.columns(2)
        show = f1.selectbox("Show", ["All", "Needs Review", "High Confidence"])
        ids = ["All"] + sorted(results["Predicted_Recharging_Item_ID"].unique())
        fid = f2.selectbox("Filter by predicted ID", ids)

        filtered = results
        if show == "Needs Review":
            filtered = filtered[filtered["Needs_Review"]]
        elif show == "High Confidence":
            filtered = filtered[~filtered["Needs_Review"]]
        if fid != "All":
            filtered = filtered[filtered["Predicted_Recharging_Item_ID"] == fid]

        display_cols = [EMPTY_COLS["sub_account_name"], EMPTY_COLS["resource_group"],
                        EMPTY_COLS["tag_dcs"], EMPTY_COLS["tag_app"],
                        "Predicted_Recharging_Item_ID", "Confidence", "Top_Matches",
                        "Needs_Review"]
        cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[cols].style.map(color_confidence, subset=["Confidence"]),
                     use_container_width=True, height=460)

        st.download_button("⬇ Download results as Excel", data=to_excel_bytes(results),
                           file_name="GO_predictions.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Tab 2: Review Queue (agent proposals + human decision)
    with tab_review:
        review_idx = results.index[results["Needs_Review"]].tolist()
        if not review_idx:
            st.success("No rows need review in this batch. 🎉")
        else:
            c1, c2 = st.columns([1, 2])
            cap = c1.number_input("Rows to investigate", 1, len(review_idx),
                                  min(10, len(review_idx)))
            if c2.button(f"🤖 Run agent on {cap} review row(s)", type="primary"):
                bar = st.progress(0.0, "Investigating...")
                reviewed = run_review(results, max_rows=int(cap),
                                      progress=lambda d, t: bar.progress(d / t, f"Investigating {d}/{t}"))
                st.session_state.results = reviewed
                st.session_state.reviewed = True
                bar.empty()
                st.rerun()

            if st.session_state.reviewed:
                st.caption("Agent proposals are constrained to the matcher's candidate IDs. "
                           "Accept or override — your choice is appended to the learning data.")
                investigated = [i for i in review_idx if str(results.at[i, "Agent_Proposed_ID"]) != ""
                                or results.at[i, "Agent_Reasoning"] != ""]
                for i in investigated:
                    row = results.loc[i]
                    name = row.get(EMPTY_COLS["sub_account_name"], f"row {i}")
                    proposed = str(row["Agent_Proposed_ID"])
                    conf = row["Agent_Confidence"]
                    flag = "⚠️ needs human" if row["Agent_Needs_Human"] else f"proposes **{proposed}**"
                    with st.expander(f"**{name}**  —  {flag}  (conf {conf})"):
                        st.markdown(f"**Reasoning:** {row['Agent_Reasoning']}")
                        if row["Agent_Evidence"]:
                            st.markdown("**Evidence:**")
                            for ev in str(row["Agent_Evidence"]).split(" ; "):
                                st.markdown(f"- {ev}")
                        candidates = parse_candidates(row.get("Top_Matches", ""))
                        default = candidates.index(proposed) if proposed in candidates else 0
                        choice = st.radio("Confirm Recharging_Item_ID", candidates,
                                          index=default, horizontal=True, key=f"choice_{i}")
                        if st.button("✓ Accept & commit", key=f"accept_{i}"):
                            try:
                                commit_decision(row, choice, st.session_state.get("user", "ui"),
                                                LEARNING_WORKBOOK)
                                results.at[i, "Predicted_Recharging_Item_ID"] = choice
                                results.at[i, "Needs_Review"] = False
                                st.session_state.results = results
                                st.success(f"Committed {name} → {choice}. Retrains on next run.")
                                st.rerun()
                            except Exception as e:  # noqa: BLE001
                                st.error(f"Commit failed: {e}")

    # Tab 3: About
    with tab_about:
        st.markdown(
            "**Confidence** 🟢 ≥70 auto · 🟡 50–69 borderline · 🔴 <50 low.\n\n"
            "**Match methods:** `exact_name_rg` / `exact_name` (seen before, certain) · "
            "`classifier` (learned char-ngram model).\n\n"
            "**Review Queue:** rows the matcher flagged (low confidence, or opaque/name-only). "
            "The agent investigates with read-only tools and proposes one of the candidate IDs; "
            "it never invents an ID and never commits — you do.")


# ── Placeholder factory for not-yet-built features ────────────────────────────
def _coming_soon(feat: dict):
    def page():
        st.title(f"{feat['icon']} {feat['title']}")
        st.caption("⚪ Coming soon")
        st.info(feat["desc"])
        st.write("This feature isn't built yet. It's listed here to show where the "
                 "FinOps Assistant is headed — swap this placeholder for the real "
                 "implementation when ready.")
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
