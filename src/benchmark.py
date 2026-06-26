"""Benchmark: split GO_MAPPING_LEARNING into train/test, measure prediction accuracy."""

import sys
sys.path.insert(0, "src")

from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from matcher import LEARNING_COLS

INPUT_FILE = "GO Report Extract LIGHT_V2.xlsx"
TEST_RATIO = 0.2


def get_fields(row):
    """Extract individual fields for separate matching."""
    name = str(row.get(LEARNING_COLS["sub_account_name"], "") or "").lower().strip()
    rg = str(row.get(LEARNING_COLS["resource_group"], "") or "").lower().strip()
    dcs = str(row.get(LEARNING_COLS["tag_dcs"], "") or "").lower().strip()
    app = str(row.get(LEARNING_COLS["tag_app"], "") or "").lower().strip()
    provider = str(row.get(LEARNING_COLS["provider"], "") or "").lower().strip()
    # Clean nan strings
    if rg == "nan": rg = ""
    if dcs == "nan": dcs = ""
    if app == "nan": app = ""
    if provider == "nan": provider = ""
    return name, rg, dcs, app, provider


def compute_score(q_fields, r_fields):
    """Weighted field-level fuzzy score."""
    q_name, q_rg, q_dcs, q_app, q_provider = q_fields
    r_name, r_rg, r_dcs, r_app, r_provider = r_fields

    # Provider mismatch penalty: if both have a provider and they differ, heavily penalize
    if q_provider and r_provider and q_provider != r_provider:
        return 0.0

    # Name: primary signal (weight 0.45)
    name_score = fuzz.token_set_ratio(q_name, r_name) / 100.0 if q_name and r_name else 0.0

    # Resource group: strong signal (weight 0.35)
    rg_score = fuzz.token_set_ratio(q_rg, r_rg) / 100.0 if q_rg and r_rg else 0.0

    # Tags: tiebreaker (weight 0.10 each)
    dcs_score = (1.0 if q_dcs == r_dcs else 0.0) if q_dcs and r_dcs else 0.0
    app_score = (1.0 if q_app == r_app else 0.0) if q_app and r_app else 0.0

    return 0.45 * name_score + 0.35 * rg_score + 0.10 * dcs_score + 0.10 * app_score


def predict_top_k(q_fields, ref_fields_list, ref_ids, k=5):
    """Predict using weighted field matching + top-k majority vote."""
    scores = np.array([compute_score(q_fields, rf) for rf in ref_fields_list])
    top_indices = np.argsort(scores)[::-1][:k]
    top_ids = [ref_ids[i] for i in top_indices]
    top_score = scores[top_indices[0]]

    # Majority vote among top-k
    counter = Counter(top_ids)
    best_id = counter.most_common(1)[0][0]

    return best_id, top_score


def run_benchmark():
    print("Loading GO_MAPPING_LEARNING...")
    df = pd.read_excel(INPUT_FILE, sheet_name="GO_MAPPING_LEARNING")

    id_col = LEARNING_COLS["recharging_item_id"]
    df_valid = df.dropna(subset=[id_col]).copy()
    print(f"  Valid rows: {len(df_valid)} (excluded {len(df) - len(df_valid)} with null target)")

    df_shuffled = df_valid.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * (1 - TEST_RATIO))
    df_train = df_shuffled.iloc[:split_idx].copy()
    df_test = df_shuffled.iloc[split_idx:].copy()

    print(f"  Train: {len(df_train)} rows")
    print(f"  Test:  {len(df_test)} rows")

    # Build reference
    print("\nBuilding reference index...")
    ref_fields_list = [get_fields(row) for _, row in df_train.iterrows()]
    ref_ids = df_train[id_col].astype(str).tolist()

    # Build exact-match lookup: (name, rg) → most common ID
    exact_lookup = {}
    for fields, rid in zip(ref_fields_list, ref_ids):
        key = (fields[0], fields[1])  # (name, resource_group)
        if key not in exact_lookup:
            exact_lookup[key] = []
        exact_lookup[key].append(rid)
    exact_lookup = {k: Counter(v).most_common(1)[0][0] for k, v in exact_lookup.items()}

    # Name-only lookup: only use when name ALWAYS maps to the same ID
    name_ids = {}
    for fields, rid in zip(ref_fields_list, ref_ids):
        name = fields[0]
        if name not in name_ids:
            name_ids[name] = set()
        name_ids[name].add(rid)
    name_lookup = {name: ids.pop() for name, ids in name_ids.items() if len(ids) == 1}

    # Predict
    print("Predicting (exact match → field-weighted fuzzy + top-5 majority)...")
    predictions = []
    scores_list = []
    exact_hits = 0
    total = len(df_test)
    for i, (_, row) in enumerate(df_test.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{total}...")
        q_fields = get_fields(row)

        # Try exact match first (name + RG)
        exact_key = (q_fields[0], q_fields[1])
        if exact_key in exact_lookup:
            predictions.append(exact_lookup[exact_key])
            scores_list.append(1.0)
            exact_hits += 1
        elif q_fields[0] in name_lookup:
            # Fallback: exact name match (ignoring RG)
            predictions.append(name_lookup[q_fields[0]])
            scores_list.append(0.95)
            exact_hits += 1
        else:
            # Fuzzy matching
            pred_id, score = predict_top_k(q_fields, ref_fields_list, ref_ids, k=5)
            predictions.append(pred_id)
            scores_list.append(score)

    actual = df_test[id_col].astype(str).tolist()

    correct = sum(p == a for p, a in zip(predictions, actual))
    accuracy = correct / total * 100
    scores_arr = np.array(scores_list)

    print(f"\n{'='*50}")
    print(f"  BENCHMARK RESULTS (Exact + Field-Fuzzy + Top-5)")
    print(f"{'='*50}")
    print(f"  Accuracy:         {accuracy:.1f}% ({correct}/{total})")
    print(f"  Exact match hits: {exact_hits} ({exact_hits / total * 100:.0f}%)")
    print(f"  Avg match score:  {scores_arr.mean():.3f}")
    print(f"  Median score:     {np.median(scores_arr):.3f}")

    print(f"\n  Accuracy by match score band:")
    pred_arr = np.array(predictions)
    act_arr = np.array(actual)
    for lo, hi, label in [(0.8, 1.01, "High (≥0.80)"), (0.6, 0.8, "Medium (0.60-0.79)"), (0.4, 0.6, "Medium-Low (0.40-0.59)"), (0.0, 0.4, "Low (<0.40)")]:
        mask = (scores_arr >= lo) & (scores_arr < hi)
        if mask.sum() > 0:
            band_acc = (pred_arr[mask] == act_arr[mask]).mean() * 100
            print(f"    {label:25s}: {band_acc:5.1f}% ({mask.sum()} rows)")
        else:
            print(f"    {label:25s}: no rows")

    wrong_mask = pred_arr != act_arr
    print(f"\n  Misclassified: {wrong_mask.sum()} rows")
    if wrong_mask.sum() > 0:
        wrong_indices = np.where(wrong_mask)[0]
        print(f"\n  Sample errors (first 10):")
        print(f"  {'Predicted':<20} {'Actual':<20} {'Score':>5}  Name | RG")
        print(f"  {'-'*85}")
        for i in wrong_indices[:10]:
            fields = get_fields(df_test.iloc[i])
            print(f"  {predictions[i]:<20} {actual[i]:<20} {scores_list[i]:5.3f}  {fields[0][:25]} | {fields[1][:25]}")


if __name__ == "__main__":
    run_benchmark()
