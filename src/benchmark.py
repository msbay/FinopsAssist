"""Benchmark: split GO_MAPPING_LEARNING into train/test, measure prediction accuracy."""

import sys
sys.path.insert(0, "src")

from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
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


def fields_to_text(fields):
    """Convert fields tuple to a single string for TF-IDF."""
    # Exclude provider (index 4) from text — it's used as a filter, not similarity signal
    return " | ".join(f for f in fields[:4] if f)


def apply_provider_filter(scores, q_provider, ref_fields_list):
    """Zero out scores where providers mismatch."""
    if not q_provider:
        return scores
    filtered = scores.copy()
    for j, rf in enumerate(ref_fields_list):
        r_provider = rf[4]
        if r_provider and q_provider != r_provider:
            filtered[j] = 0.0
    return filtered


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


def predict_hybrid(q_fields, q_tfidf_scores, ref_ids, k=5, ref_fields_list=None,
                   semantic_weight=0.4):
    """Predict using combined fuzzy + TF-IDF scores with top-k majority vote."""
    fuzzy_scores = np.array([compute_score(q_fields, rf) for rf in ref_fields_list])
    hybrid_scores = (1 - semantic_weight) * fuzzy_scores + semantic_weight * q_tfidf_scores

    top_indices = np.argsort(hybrid_scores)[::-1][:k]
    top_ids = [ref_ids[i] for i in top_indices]
    top_score = hybrid_scores[top_indices[0]]

    counter = Counter(top_ids)
    best_id = counter.most_common(1)[0][0]

    return best_id, top_score


def print_results(title, predictions, actual, scores_list, exact_hits, total, df_test):
    """Print benchmark results for a given strategy."""
    correct = sum(p == a for p, a in zip(predictions, actual))
    accuracy = correct / total * 100
    scores_arr = np.array(scores_list)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
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

    return accuracy


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

    # Build reference fields
    print("\nBuilding reference index...")
    ref_fields_list = [get_fields(row) for _, row in df_train.iterrows()]
    ref_ids = df_train[id_col].astype(str).tolist()

    # Build exact-match lookup: (name, rg) → most common ID
    exact_lookup = {}
    for fields, rid in zip(ref_fields_list, ref_ids):
        key = (fields[0], fields[1])
        if key not in exact_lookup:
            exact_lookup[key] = []
        exact_lookup[key].append(rid)
    exact_lookup = {k: Counter(v).most_common(1)[0][0] for k, v in exact_lookup.items()}

    # Name-only lookup
    name_ids = {}
    for fields, rid in zip(ref_fields_list, ref_ids):
        name = fields[0]
        if name not in name_ids:
            name_ids[name] = set()
        name_ids[name].add(rid)
    name_lookup = {name: ids.pop() for name, ids in name_ids.items() if len(ids) == 1}

    # Build provider array for fast filtering
    ref_providers = np.array([f[4] for f in ref_fields_list])

    # Build test fields
    test_fields_list = [get_fields(row) for _, row in df_test.iterrows()]
    actual = df_test[id_col].astype(str).tolist()
    total = len(df_test)

    # ── Pre-identify exact match rows (shared across all strategies) ──
    exact_results = {}  # index → (pred_id, score)
    for i, q_fields in enumerate(test_fields_list):
        exact_key = (q_fields[0], q_fields[1])
        if exact_key in exact_lookup:
            exact_results[i] = (exact_lookup[exact_key], 1.0)
        elif q_fields[0] in name_lookup:
            exact_results[i] = (name_lookup[q_fields[0]], 0.95)

    print(f"  Exact match rows: {len(exact_results)}/{total} ({len(exact_results)/total*100:.0f}%)")
    non_exact_indices = [i for i in range(total) if i not in exact_results]
    print(f"  Rows needing similarity matching: {len(non_exact_indices)}")

    # ══════════════════════════════════════════════════════════════
    #  EXPERIMENT 1: N-gram range sweep (TF-IDF only, with provider filter)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 1: N-gram range sweep (TF-IDF + provider filter)")
    print(f"{'='*60}")

    ngram_ranges = [(2, 4), (3, 5), (3, 6), (2, 5), (2, 6)]
    best_ngram = None
    best_ngram_acc = 0
    best_tfidf_matrix_ref = None
    best_tfidf = None

    for ngram_range in ngram_ranges:
        ref_texts = [fields_to_text(f) for f in ref_fields_list]
        test_texts = [fields_to_text(f) for f in test_fields_list]

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
        ref_matrix = vectorizer.fit_transform(ref_texts)
        test_matrix = vectorizer.transform(test_texts)
        sim_matrix = (test_matrix @ ref_matrix.T).toarray()

        preds, scores = [], []
        for i in range(total):
            if i in exact_results:
                preds.append(exact_results[i][0])
                scores.append(exact_results[i][1])
            else:
                q_provider = test_fields_list[i][4]
                sim_scores = sim_matrix[i].copy()
                # Provider filter
                if q_provider:
                    provider_mask = (ref_providers != "") & (ref_providers != q_provider)
                    sim_scores[provider_mask] = 0.0
                top_idx = np.argsort(sim_scores)[::-1][:5]
                top_ids = [ref_ids[j] for j in top_idx]
                best_id = Counter(top_ids).most_common(1)[0][0]
                preds.append(best_id)
                scores.append(sim_scores[top_idx[0]])

        correct = sum(p == a for p, a in zip(preds, actual))
        acc = correct / total * 100
        marker = ""
        if acc > best_ngram_acc:
            best_ngram_acc = acc
            best_ngram = ngram_range
            best_tfidf_matrix_ref = ref_matrix
            best_tfidf = vectorizer
            best_ngram_preds = preds
            best_ngram_scores = scores
            marker = " ◄ best"
        print(f"  ngram={ngram_range}  accuracy={acc:.1f}% ({correct}/{total}){marker}")

    print(f"\n  Best n-gram range: {best_ngram} → {best_ngram_acc:.1f}%")

    # ══════════════════════════════════════════════════════════════
    #  EXPERIMENT 2: Hybrid weight sweep (best n-gram + provider filter)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 2: Hybrid weight sweep (ngram={best_ngram})")
    print(f"{'='*60}")

    # Rebuild TF-IDF sim matrix with best n-gram
    test_texts = [fields_to_text(f) for f in test_fields_list]
    test_matrix = best_tfidf.transform(test_texts)
    tfidf_sim_matrix = (test_matrix @ best_tfidf_matrix_ref.T).toarray()

    # Precompute fuzzy scores for non-exact rows
    print("  Precomputing fuzzy scores for non-exact rows...")
    fuzzy_score_cache = {}
    for idx, i in enumerate(non_exact_indices):
        if (idx + 1) % 100 == 0:
            print(f"    {idx + 1}/{len(non_exact_indices)}...")
        q_fields = test_fields_list[i]
        fuzzy_scores = np.array([compute_score(q_fields, rf) for rf in ref_fields_list])
        fuzzy_score_cache[i] = fuzzy_scores

    tfidf_weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_weight = 0
    best_weight_acc = 0

    for tw in tfidf_weights:
        preds = []
        for i in range(total):
            if i in exact_results:
                preds.append(exact_results[i][0])
            else:
                q_provider = test_fields_list[i][4]
                tfidf_scores = tfidf_sim_matrix[i].copy()
                fuzzy_scores = fuzzy_score_cache[i].copy()
                # Provider filter on TF-IDF side too
                if q_provider:
                    provider_mask = (ref_providers != "") & (ref_providers != q_provider)
                    tfidf_scores[provider_mask] = 0.0

                hybrid = (1 - tw) * fuzzy_scores + tw * tfidf_scores
                top_idx = np.argsort(hybrid)[::-1][:5]
                top_ids = [ref_ids[j] for j in top_idx]
                preds.append(Counter(top_ids).most_common(1)[0][0])

        correct = sum(p == a for p, a in zip(preds, actual))
        acc = correct / total * 100
        marker = ""
        if acc > best_weight_acc:
            best_weight_acc = acc
            best_weight = tw
            best_weight_preds = preds
            marker = " ◄ best"
        label = f"fuzzy={1-tw:.0%} / tfidf={tw:.0%}"
        print(f"  {label:<30} accuracy={acc:.1f}% ({correct}/{total}){marker}")

    print(f"\n  Best weight: tfidf={best_weight:.0%} / fuzzy={1-best_weight:.0%} → {best_weight_acc:.1f}%")

    # ══════════════════════════════════════════════════════════════
    #  EXPERIMENT 3: Top-K sweep (best weight + best n-gram)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 3: Top-K sweep (tfidf={best_weight:.0%}, ngram={best_ngram})")
    print(f"{'='*60}")

    for k in [1, 3, 5, 7, 10]:
        preds = []
        for i in range(total):
            if i in exact_results:
                preds.append(exact_results[i][0])
            else:
                q_provider = test_fields_list[i][4]
                tfidf_scores = tfidf_sim_matrix[i].copy()
                fuzzy_scores = fuzzy_score_cache[i].copy()
                if q_provider:
                    provider_mask = (ref_providers != "") & (ref_providers != q_provider)
                    tfidf_scores[provider_mask] = 0.0

                hybrid = (1 - best_weight) * fuzzy_scores + best_weight * tfidf_scores
                top_idx = np.argsort(hybrid)[::-1][:k]
                top_ids = [ref_ids[j] for j in top_idx]
                preds.append(Counter(top_ids).most_common(1)[0][0])

        correct = sum(p == a for p, a in zip(preds, actual))
        acc = correct / total * 100
        print(f"  top_k={k:<3}  accuracy={acc:.1f}% ({correct}/{total})")

    # ══════════════════════════════════════════════════════════════
    #  ERROR ANALYSIS (best configuration)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  ERROR ANALYSIS (best config: tfidf={best_weight:.0%}, ngram={best_ngram})")
    print(f"{'='*60}")

    pred_arr = np.array(best_weight_preds)
    act_arr = np.array(actual)
    wrong_mask = pred_arr != act_arr
    wrong_indices = np.where(wrong_mask)[0]
    print(f"  Total errors: {wrong_mask.sum()}/{total}")

    # Errors by provider
    print(f"\n  Errors by provider:")
    provider_errors = Counter()
    provider_totals = Counter()
    for i in range(total):
        provider = test_fields_list[i][4] or "(empty)"
        provider_totals[provider] += 1
        if pred_arr[i] != act_arr[i]:
            provider_errors[provider] += 1
    for provider in sorted(provider_totals.keys()):
        errs = provider_errors.get(provider, 0)
        tot = provider_totals[provider]
        print(f"    {provider:<25} {errs}/{tot} errors ({errs/tot*100:.1f}%)")

    # Errors by actual ID (which IDs are hardest to predict)
    print(f"\n  Hardest Recharging_Item_IDs (most errors):")
    id_errors = Counter()
    id_totals = Counter()
    for i in range(total):
        id_totals[actual[i]] += 1
        if pred_arr[i] != act_arr[i]:
            id_errors[actual[i]] += 1
    for rid, errs in id_errors.most_common(15):
        tot = id_totals[rid]
        print(f"    {rid:<30} {errs}/{tot} errors ({errs/tot*100:.0f}%)")

    # Confusion pairs (predicted X but was Y)
    print(f"\n  Top confusion pairs (predicted → actual):")
    confusion = Counter()
    for i in wrong_indices:
        confusion[(pred_arr[i], act_arr[i])] += 1
    for (pred, act), count in confusion.most_common(10):
        print(f"    {pred:<20} → {act:<20} ({count}x)")

    # Error by match method (exact vs similarity)
    exact_errors = sum(1 for i in wrong_indices if i in exact_results)
    sim_errors = sum(1 for i in wrong_indices if i not in exact_results)
    print(f"\n  Errors from exact match: {exact_errors}")
    print(f"  Errors from similarity:  {sim_errors}/{len(non_exact_indices)} "
          f"({sim_errors/max(len(non_exact_indices),1)*100:.1f}% of similarity-matched rows)")

    # Sample errors with full detail
    print(f"\n  Sample errors (first 15):")
    print(f"  {'Predicted':<20} {'Actual':<20} Name | RG | DCS | Provider")
    print(f"  {'-'*100}")
    for i in wrong_indices[:15]:
        fields = test_fields_list[i]
        name, rg, dcs, app, provider = fields
        print(f"  {pred_arr[i]:<20} {act_arr[i]:<20} {name[:20]} | {rg[:20]} | {dcs[:10]} | {provider}")

    # ══════════════════════════════════════════════════════════════
    #  EXPERIMENT 4: Normalized tag lookup + fuzzy DCS matching
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 4: Normalized tag lookup (no freq penalty)")
    print(f"{'='*60}")

    # --- DCS normalization ---
    import re
    def normalize_dcs(dcs: str) -> str:
        """Normalize DCS tag variants: cloud_prod = cloud prod = cloud product."""
        if not dcs:
            return ""
        s = dcs.lower().strip()
        s = s.replace("_", " ").replace("-", " ")
        s = re.sub(r"\s+", " ", s)  # collapse whitespace
        s = s.rstrip("s")           # cloud products → cloud product
        return s

    # Build normalized DCS → ID lookup from training data
    print("  Building normalized tag_dcs → ID lookup...")
    dcs_id_counts: dict[str, Counter] = {}
    for fields, rid in zip(ref_fields_list, ref_ids):
        dcs_norm = normalize_dcs(fields[2])
        if dcs_norm:
            if dcs_norm not in dcs_id_counts:
                dcs_id_counts[dcs_norm] = Counter()
            dcs_id_counts[dcs_norm][rid] += 1

    dcs_lookup = {}
    for dcs_norm, id_counter in dcs_id_counts.items():
        total_for_dcs = sum(id_counter.values())
        best_id, best_count = id_counter.most_common(1)[0]
        purity = best_count / total_for_dcs
        if purity >= 0.85 and total_for_dcs >= 2:
            dcs_lookup[dcs_norm] = (best_id, purity, total_for_dcs)
    print(f"    Normalized DCS mappings (≥85% purity, ≥2 samples): {len(dcs_lookup)}")
    for dcs_norm, (rid, purity, cnt) in sorted(dcs_lookup.items()):
        print(f"      {dcs_norm:<30} → {rid:<20} (purity={purity:.0%}, n={cnt})")

    # Build normalized provider+DCS lookup
    provider_dcs_counts: dict[tuple, Counter] = {}
    for fields, rid in zip(ref_fields_list, ref_ids):
        provider = fields[4]
        dcs_norm = normalize_dcs(fields[2])
        if provider and dcs_norm:
            key = (provider, dcs_norm)
            if key not in provider_dcs_counts:
                provider_dcs_counts[key] = Counter()
            provider_dcs_counts[key][rid] += 1

    provider_dcs_lookup = {}
    for key, id_counter in provider_dcs_counts.items():
        total_for_key = sum(id_counter.values())
        best_id, best_count = id_counter.most_common(1)[0]
        purity = best_count / total_for_key
        if purity >= 0.80 and total_for_key >= 2:
            provider_dcs_lookup[key] = best_id
    print(f"    Provider+DCS mappings (≥80% purity, ≥2 samples): {len(provider_dcs_lookup)}")

    # Build fuzzy DCS matching for tags not found in exact lookup
    # Pre-compute all known DCS keys for fuzzy fallback
    dcs_keys = list(dcs_lookup.keys())

    # --- Run improved pipeline ---
    # Cascade: exact(name+RG) → exact(name) → provider+dcs(normalized) → dcs(normalized+fuzzy) → similarity
    print(f"\n  Running improved pipeline...")
    preds_improved = []
    match_methods = []

    for i in range(total):
        q_fields = test_fields_list[i]
        q_name, q_rg, q_dcs, q_app, q_provider = q_fields
        q_dcs_norm = normalize_dcs(q_dcs)

        # Layer 1: Exact name+RG
        exact_key = (q_name, q_rg)
        if exact_key in exact_lookup:
            preds_improved.append(exact_lookup[exact_key])
            match_methods.append("exact_name_rg")
            continue

        # Layer 2: Exact name only
        if q_name in name_lookup:
            preds_improved.append(name_lookup[q_name])
            match_methods.append("exact_name")
            continue

        # Layer 3: Provider+DCS normalized lookup
        if q_provider and q_dcs_norm:
            pd_key = (q_provider, q_dcs_norm)
            if pd_key in provider_dcs_lookup:
                preds_improved.append(provider_dcs_lookup[pd_key])
                match_methods.append("provider_dcs")
                continue

        # Layer 4: DCS normalized exact lookup
        if q_dcs_norm and q_dcs_norm in dcs_lookup:
            preds_improved.append(dcs_lookup[q_dcs_norm][0])
            match_methods.append("dcs_exact")
            continue

        # Layer 5: DCS fuzzy lookup (for tag variants not caught by normalization)
        if q_dcs_norm and dcs_keys:
            best_dcs_score = 0
            best_dcs_match = None
            for dk in dcs_keys:
                score = fuzz.ratio(q_dcs_norm, dk) / 100.0
                if score > best_dcs_score:
                    best_dcs_score = score
                    best_dcs_match = dk
            if best_dcs_score >= 0.80:
                preds_improved.append(dcs_lookup[best_dcs_match][0])
                match_methods.append("dcs_fuzzy")
                continue

        # Layer 6: Similarity (same as best config, no freq penalty)
        tfidf_scores = tfidf_sim_matrix[i].copy()
        fuzzy_scores = fuzzy_score_cache[i].copy()

        # Provider filter
        if q_provider:
            provider_mask = (ref_providers != "") & (ref_providers != q_provider)
            tfidf_scores[provider_mask] = 0.0

        hybrid = (1 - best_weight) * fuzzy_scores + best_weight * tfidf_scores
        top_idx = np.argsort(hybrid)[::-1][:5]
        top_ids = [ref_ids[j] for j in top_idx]
        best_id = Counter(top_ids).most_common(1)[0][0]
        preds_improved.append(best_id)
        match_methods.append("similarity")

    correct_improved = sum(p == a for p, a in zip(preds_improved, actual))
    acc_improved = correct_improved / total * 100

    # Count by match method
    method_counts = Counter(match_methods)
    method_correct = Counter()
    for i in range(total):
        if preds_improved[i] == actual[i]:
            method_correct[match_methods[i]] += 1

    print(f"\n  IMPROVED RESULTS")
    print(f"  {'='*55}")
    print(f"  Accuracy: {acc_improved:.1f}% ({correct_improved}/{total})")
    print(f"  Previous best: {best_weight_acc:.1f}%")
    improvement = acc_improved - best_weight_acc
    print(f"  Improvement: {'+' if improvement >= 0 else ''}{improvement:.1f}% ({int(improvement * total / 100):+d} rows)")

    print(f"\n  Accuracy by match method:")
    for method in ["exact_name_rg", "exact_name", "provider_dcs", "dcs_exact", "dcs_fuzzy", "similarity"]:
        cnt = method_counts.get(method, 0)
        corr = method_correct.get(method, 0)
        if cnt > 0:
            print(f"    {method:<20} {corr}/{cnt} correct ({corr/cnt*100:.1f}%)")

    # Remaining errors breakdown
    pred_imp_arr = np.array(preds_improved)
    wrong_improved = pred_imp_arr != act_arr
    wrong_imp_indices = np.where(wrong_improved)[0]

    print(f"\n  Remaining errors: {wrong_improved.sum()}")
    print(f"\n  Remaining errors by provider:")
    for provider in sorted(provider_totals.keys()):
        mask = np.array([test_fields_list[i][4] or "(empty)" for i in range(total)]) == provider
        errs = (wrong_improved & mask).sum()
        tot = mask.sum()
        if tot > 0:
            print(f"    {provider:<25} {errs}/{tot} errors ({errs/tot*100:.1f}%)")

    print(f"\n  Remaining confusion pairs:")
    confusion_imp = Counter()
    for i in wrong_imp_indices:
        confusion_imp[(pred_imp_arr[i], act_arr[i])] += 1
    for (pred, act), count in confusion_imp.most_common(10):
        print(f"    {pred:<20} → {act:<20} ({count}x)")

    print(f"\n  Sample remaining errors (first 15):")
    print(f"  {'Predicted':<20} {'Actual':<20} {'Method':<15} Name | RG | DCS | Provider")
    print(f"  {'-'*110}")
    for i in wrong_imp_indices[:15]:
        fields = test_fields_list[i]
        name, rg, dcs, app, provider = fields
        print(f"  {pred_imp_arr[i]:<20} {act_arr[i]:<20} {match_methods[i]:<15} {name[:18]} | {rg[:18]} | {dcs[:10]} | {provider}")


if __name__ == "__main__":
    run_benchmark()
