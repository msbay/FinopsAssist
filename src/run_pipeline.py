"""Deterministic pipeline: load → validate → match → LLM checkpoint → output."""

import sys
sys.path.insert(0, "src")

import pandas as pd
from matcher import RechargingMatcher, LEARNING_COLS, EMPTY_COLS
from main import get_llm
from langchain_core.messages import HumanMessage

INPUT_FILE = "GO Report Extract LIGHT_V2.xlsx"
OUTPUT_FILE = "GO_predictions.xlsx"


# ------------------------------------------------------------------
# Step 1: Load & validate
# ------------------------------------------------------------------
def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[1/4] Loading data...")
    df_learning = pd.read_excel(path, sheet_name="GO_MAPPING_LEARNING")
    df_empty = pd.read_excel(path, sheet_name="GO_MAPPING_EMPTY")

    # Validate required columns exist
    for col in LEARNING_COLS.values():
        if col not in df_learning.columns:
            raise ValueError(f"Missing column in GO_MAPPING_LEARNING: {col}")
    for col in EMPTY_COLS.values():
        if col not in df_empty.columns:
            raise ValueError(f"Missing column in GO_MAPPING_EMPTY: {col}")

    missing_names = df_empty[EMPTY_COLS["sub_account_name"]].isna().sum()
    if missing_names > 0:
        print(f"  WARNING: {missing_names} rows have no SubAccountName")

    print(f"  Learning: {df_learning.shape[0]} rows | To predict: {df_empty.shape[0]} rows")
    return df_learning, df_empty


# ------------------------------------------------------------------
# Step 2: Run hybrid matcher (deterministic)
# ------------------------------------------------------------------
def run_matcher(
    df_learning: pd.DataFrame, df_empty: pd.DataFrame
) -> tuple[RechargingMatcher, pd.DataFrame]:
    print("[2/4] Building index & predicting...")
    matcher = RechargingMatcher()
    matcher.build_index(df_learning)
    results = matcher.predict(df_empty)
    print(f"  Predictions: {results.shape[0]} rows")
    return matcher, results


# ------------------------------------------------------------------
# Step 3: LLM checkpoint — reason over low-confidence rows only
# ------------------------------------------------------------------
def llm_review(results: pd.DataFrame, matcher: RechargingMatcher) -> pd.DataFrame:
    review_rows = results[results["Needs_Review"]].copy()
    if review_rows.empty:
        print("[3/4] LLM review — no low-confidence rows, skipping")
        results["LLM_Justification"] = ""
        return results

    print(f"[3/4] LLM review — reasoning over {len(review_rows)} ambiguous rows...")
    try:
        llm = get_llm()
    except Exception as e:
        print(f"  LLM unavailable ({e}), skipping justifications")
        results["LLM_Justification"] = ""
        return results

    justifications = {}
    for idx, row in review_rows.iterrows():
        fields = matcher._get_fields(row, EMPTY_COLS)
        item_desc = " | ".join(f for f in fields if f)
        prompt = (
            "You are a FinOps analyst. A new cloud account needs to be mapped to a Recharging_Item_ID.\n\n"
            f"New item: {item_desc}\n"
            f"Top candidates: {row['Top_Matches']}\n\n"
            "Pick the best Recharging_Item_ID from the candidates and explain why in one sentence."
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            justifications[idx] = resp.content
        except Exception as e:
            justifications[idx] = f"LLM error: {e}"

    results["LLM_Justification"] = results.index.map(lambda i: justifications.get(i, ""))
    print(f"  Generated {len(justifications)} justifications")
    return results


# ------------------------------------------------------------------
# Step 4: Output
# ------------------------------------------------------------------
def save_results(results: pd.DataFrame, path: str) -> None:
    print("[4/4] Saving results...")
    results.to_excel(path, index=False)

    needs_review = results["Needs_Review"].sum()
    total = results.shape[0]
    print(f"\n  SUMMARY")
    print(f"  Total predictions:  {total}")
    print(f"  High confidence:    {total - needs_review} ({(total - needs_review) / total * 100:.0f}%)")
    print(f"  Needs review:       {needs_review} ({needs_review / total * 100:.0f}%)")
    print(f"  Avg confidence:     {results['Confidence'].mean():.1f}")
    print(f"  Saved to:           {path}")


# ------------------------------------------------------------------
# Main pipeline — rigid, deterministic sequence
# ------------------------------------------------------------------
def run():
    df_learning, df_empty = load_data(INPUT_FILE)
    matcher, results = run_matcher(df_learning, df_empty)
    results = llm_review(results, matcher)
    save_results(results, OUTPUT_FILE)


if __name__ == "__main__":
    run()
