"""FinOps Co-Pilot Dashboard — Streamlit UI with chat sidebar."""

import sys
sys.path.insert(0, "src")

import streamlit as st
import pandas as pd
from matcher import RechargingMatcher, LEARNING_COLS, EMPTY_COLS
from main import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

INPUT_FILE = "GO Report Extract LIGHT_V2.xlsx"

st.set_page_config(page_title="FinOps Co-Pilot", layout="wide")


# ------------------------------------------------------------------
# Session state init
# ------------------------------------------------------------------
def init_state():
    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.matcher = None
        st.session_state.overrides = {}
        st.session_state.chat_messages = []


init_state()


# ------------------------------------------------------------------
# Data loading & matching (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_and_predict():
    df_learning = pd.read_excel(INPUT_FILE, sheet_name="GO_MAPPING_LEARNING")
    df_empty = pd.read_excel(INPUT_FILE, sheet_name="GO_MAPPING_EMPTY")

    matcher = RechargingMatcher()
    matcher.build_index(df_learning)
    results = matcher.predict(df_empty)
    return matcher, results


# ------------------------------------------------------------------
# Confidence color coding
# ------------------------------------------------------------------
def color_confidence(val):
    if val >= 70:
        return "background-color: #c6efce; color: #006100"
    elif val >= 50:
        return "background-color: #ffeb9c; color: #9c5700"
    else:
        return "background-color: #ffc7ce; color: #9c0006"


# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
st.title("FinOps Co-Pilot")

# Sidebar: Chat
with st.sidebar:
    st.header("Ask the Co-Pilot")
    st.caption("Ask high-level questions about the predictions")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. Why did unmatched rows spike this month?"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm()
                    results_df = st.session_state.results
                    # Build a compact summary for the LLM — no raw rows in context
                    summary = ""
                    if results_df is not None:
                        total = len(results_df)
                        review = results_df["Needs_Review"].sum()
                        avg_conf = results_df["Confidence"].mean()
                        top_ids = results_df["Predicted_Recharging_Item_ID"].value_counts().head(10).to_dict()
                        summary = (
                            f"Dataset: {total} predictions, {review} need review, "
                            f"avg confidence {avg_conf:.0f}%. "
                            f"Top predicted IDs: {top_ids}"
                        )

                    sys_msg = SystemMessage(content=(
                        "You are a FinOps analyst co-pilot. Answer concisely based on this context.\n"
                        f"Prediction summary: {summary}\n"
                        "If the user asks to filter or show data, respond with the pandas query they should use, "
                        "e.g. df[df['Predicted_Recharging_Item_ID'] == 'PSO_ITM_530']"
                    ))
                    response = llm.invoke([sys_msg, HumanMessage(content=prompt)])
                    answer = response.content
                except Exception as e:
                    answer = f"LLM unavailable: {e}"

            st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# Main content
col1, col2 = st.columns([4, 1])

with col2:
    if st.button("Run Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running matcher..."):
            matcher, results = load_and_predict()
            st.session_state.results = results
            st.session_state.matcher = matcher

with col1:
    if st.session_state.results is not None:
        results = st.session_state.results

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        total = len(results)
        review_count = int(results["Needs_Review"].sum())
        m1.metric("Total Rows", total)
        m2.metric("High Confidence", total - review_count)
        m3.metric("Needs Review", review_count)
        m4.metric("Avg Confidence", f"{results['Confidence'].mean():.0f}%")

        # Filters
        st.divider()
        f1, f2 = st.columns(2)
        with f1:
            filter_review = st.selectbox("Show", ["All", "Needs Review", "High Confidence"])
        with f2:
            unique_ids = sorted(results["Predicted_Recharging_Item_ID"].unique())
            filter_id = st.selectbox("Filter by Predicted ID", ["All"] + unique_ids)

        filtered = results.copy()
        if filter_review == "Needs Review":
            filtered = filtered[filtered["Needs_Review"]]
        elif filter_review == "High Confidence":
            filtered = filtered[~filtered["Needs_Review"]]
        if filter_id != "All":
            filtered = filtered[filtered["Predicted_Recharging_Item_ID"] == filter_id]

        # Display columns (keep it readable)
        display_cols = [
            EMPTY_COLS["sub_account_name"],
            EMPTY_COLS["resource_group"],
            EMPTY_COLS["tag_dcs"],
            EMPTY_COLS["tag_app"],
            "Predicted_Recharging_Item_ID",
            "Confidence",
            "Top_Matches",
            "Needs_Review",
        ]
        available_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[available_cols].style.applymap(color_confidence, subset=["Confidence"]),
            use_container_width=True,
            height=500,
        )

        # Override section
        st.divider()
        st.subheader("Manual Override")
        st.caption("Select a row index and assign the correct Recharging_Item_ID")

        o1, o2, o3 = st.columns([1, 2, 1])
        with o1:
            row_idx = st.number_input("Row index", min_value=0, max_value=len(results) - 1, step=1)
        with o2:
            all_ids = sorted(set(st.session_state.matcher.ref_ids)) if st.session_state.matcher else []
            override_id = st.selectbox("Assign Recharging_Item_ID", all_ids)
        with o3:
            st.write("")
            st.write("")
            if st.button("Apply Override"):
                results.iloc[row_idx, results.columns.get_loc("Predicted_Recharging_Item_ID")] = override_id
                results.iloc[row_idx, results.columns.get_loc("Needs_Review")] = False
                st.session_state.overrides[row_idx] = override_id
                st.success(f"Row {row_idx} → {override_id}")
                st.rerun()

        # Export
        st.divider()
        if st.download_button(
            "Download Results as Excel",
            data=results.to_excel(index=False),
            file_name="GO_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ):
            st.success("Downloaded!")

    else:
        st.info("Click **Run Pipeline** to start the matching process.")
