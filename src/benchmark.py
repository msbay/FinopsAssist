"""Benchmark the shipping matcher on GENUINELY NEW accounts.

Splits GO_MAPPING_LEARNING by SubAccountName (group split) so no account appears
in both train and test — this mirrors the ~56% of monthly rows whose account has
never been seen. A random row-level split leaks near-duplicate accounts across
the split and overstates accuracy by ~16 points, so it is not used here.

Reports: best char n-gram range, overall accuracy, confidence calibration
(reliability table + ECE), the production policy bands, and an error breakdown.
"""

import sys
sys.path.insert(0, "src")

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from matcher import CLF_C, LEARNING_COLS, RechargingMatcher, _evidence_reason

INPUT_FILE = "GO Report Extract LIGHT_V2.xlsx"
TEST_RATIO = 0.2
RANDOM_STATE = 42
# "" = benchmark on all codes. The XX_TOIDENTIFY placeholder is already removed
# from the data itself; the remaining XX_* codes are real categories, so we keep
# them. Set to e.g. "PSO" to restrict to PSO_* items only.
ID_PREFIX = ""
# A single group split swings ~±8pp by luck of which accounts land in test, so
# the headline accuracy is averaged over several splits. The detailed reliability
# / error sections below still use one split (RANDOM_STATE) for concrete examples.
REPEATS = 5

# Production confidence bands (matcher.py): ≥70 auto-accept, 50-69 review, <50 LLM.
AUTO_ACCEPT = 70.0
REVIEW_FLOOR = 50.0


def get_fields(row):
    name = str(row.get(LEARNING_COLS["sub_account_name"], "") or "").lower().strip()
    rg = str(row.get(LEARNING_COLS["resource_group"], "") or "").lower().strip()
    dcs = str(row.get(LEARNING_COLS["tag_dcs"], "") or "").lower().strip()
    app = str(row.get(LEARNING_COLS["tag_app"], "") or "").lower().strip()
    for v in ("nan",):
        rg = "" if rg == v else rg
        dcs = "" if dcs == v else dcs
        app = "" if app == v else app
    return name, rg, dcs, app


def to_text(fields):
    return " | ".join(f for f in fields if f)


def print_reliability(conf_arr, correct_arr, total):
    """Reliability table (predicted confidence vs. observed accuracy) + ECE."""
    print("\n  Reliability (predicted confidence vs. actual accuracy):")
    print(f"  {'conf bucket':<14} {'n':>5} {'avg conf':>9} {'accuracy':>9} {'gap':>7}")
    ece = 0.0
    for lo in range(0, 100, 10):
        hi = lo + 10
        m = (conf_arr >= lo) & (conf_arr < hi if hi < 100 else conf_arr <= 100)
        if m.sum() == 0:
            continue
        avg_conf, acc = conf_arr[m].mean(), correct_arr[m].mean() * 100
        gap = avg_conf - acc
        ece += (m.sum() / total) * abs(gap)
        print(f"  {f'[{lo:>2}-{hi:>3})':<14} {m.sum():>5} {avg_conf:>8.1f} {acc:>8.1f} {gap:>+7.1f}")
    print(f"\n  Expected Calibration Error (ECE): {ece:.1f} (0 = perfect; lower is better)")
    return ece


def print_policy_bands(conf_arr, correct_arr, total):
    """Coverage + accuracy in the three production confidence bands."""
    print("\n  Production policy bands:")
    print(f"  {'band':<28} {'rows':>6} {'coverage':>9} {'accuracy':>9}")
    for label, m in [
        (f"auto-accept (≥{AUTO_ACCEPT:.0f})", conf_arr >= AUTO_ACCEPT),
        (f"review ({REVIEW_FLOOR:.0f}-{AUTO_ACCEPT-1:.0f})",
         (conf_arr >= REVIEW_FLOOR) & (conf_arr < AUTO_ACCEPT)),
        (f"LLM (<{REVIEW_FLOOR:.0f})", conf_arr < REVIEW_FLOOR),
    ]:
        n = int(m.sum())
        acc = correct_arr[m].mean() * 100 if n else 0.0
        print(f"  {label:<28} {n:>6} {n/total*100:>8.0f}% {acc:>8.1f}%")


def stability_report(df_valid, name_col, id_col):
    """Average accuracy / calibration / auto-fill over REPEATS group splits.

    One split is noisy (~±8pp); this is the number to trust.
    """
    groups = df_valid[name_col].astype(str).str.lower().str.strip()
    texts = [to_text(get_fields(r)) for _, r in df_valid.iterrows()]
    y_all = df_valid[id_col].astype(str).values
    accs, eces, covs, precs = [], [], [], []
    for seed in range(REPEATS):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO,
                                        random_state=seed).split(df_valid, groups=groups))
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
        clf = LogisticRegression(max_iter=1000, C=CLF_C)
        clf.fit(vec.fit_transform([texts[i] for i in tr]), y_all[tr])
        proba = clf.predict_proba(vec.transform([texts[i] for i in te]))
        pred = clf.classes_[proba.argmax(1)]
        conf = proba.max(1) * 100
        corr = pred == y_all[te]
        accs.append(corr.mean() * 100)
        ece = sum((m.sum() / len(conf)) * abs(conf[m].mean() - corr[m].mean() * 100)
                  for lo in range(0, 100, 10)
                  for m in [(conf >= lo) & (conf < lo + 10 if lo < 90 else conf <= 100)]
                  if m.sum())
        eces.append(ece)
        auto = conf >= 70
        covs.append(auto.mean() * 100)
        precs.append(corr[auto].mean() * 100 if auto.sum() else 0.0)
    a = np.array(accs)
    print(f"\n{'='*60}\n  HEADLINE: averaged over {REPEATS} group splits\n{'='*60}")
    print(f"  Accuracy:            {a.mean():.1f}%  (range {a.min():.1f}–{a.max():.1f})")
    print(f"  Calibration (ECE):   {np.mean(eces):.1f}")
    print(f"  Auto-fill (conf≥70): {np.mean(covs):.0f}% of rows at {np.mean(precs):.1f}% accuracy")


def run_benchmark():
    print("Loading GO_MAPPING_LEARNING...")
    df = pd.read_excel(INPUT_FILE, sheet_name="GO_MAPPING_LEARNING")
    id_col = LEARNING_COLS["recharging_item_id"]
    name_col = LEARNING_COLS["sub_account_name"]
    df_valid = df.dropna(subset=[id_col]).copy()
    print(f"  Valid rows: {len(df_valid)} (excluded {len(df) - len(df_valid)} with null target)")
    if ID_PREFIX:
        before = len(df_valid)
        df_valid = df_valid[df_valid[id_col].astype(str).str.startswith(ID_PREFIX)].copy()
        print(f"  Filtered to IDs starting '{ID_PREFIX}': {len(df_valid)} rows "
              f"(excluded {before - len(df_valid)} XX_*/other), "
              f"{df_valid[id_col].nunique()} unique IDs")

    stability_report(df_valid, name_col, id_col)
    print(f"\n  (Detailed sections below use one split, seed {RANDOM_STATE}, for examples.)")

    # ── Group split by account: no SubAccountName in both train and test ──
    groups = df_valid[name_col].astype(str).str.lower().str.strip()
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df_valid, groups=groups))
    df_train, df_test = df_valid.iloc[train_idx].copy(), df_valid.iloc[test_idx].copy()
    leak = len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))
    print(f"  GROUP split by SubAccountName (name overlap={leak})")
    print(f"  Train: {len(df_train)} rows | Test: {len(df_test)} rows")

    # ── Reference data ──
    ref_fields = [get_fields(r) for _, r in df_train.iterrows()]
    ref_ids = df_train[id_col].astype(str).tolist()
    ref_texts = [to_text(f) for f in ref_fields]

    # Exact (name, rg) lookup — mirrors production layer 1. Near-zero under a
    # group split (that's the point: it forces the classifier to do the work).
    exact_lookup = {}
    for fields, rid in zip(ref_fields, ref_ids):
        exact_lookup.setdefault((fields[0], fields[1]), []).append(rid)
    exact_lookup = {k: Counter(v).most_common(1)[0][0] for k, v in exact_lookup.items()}

    test_fields = [get_fields(r) for _, r in df_test.iterrows()]
    test_texts = [to_text(f) for f in test_fields]
    actual = df_test[id_col].astype(str).tolist()
    total = len(df_test)
    exact_hits = sum((f[0], f[1]) in exact_lookup for f in test_fields)
    print(f"  Exact-match test rows: {exact_hits}/{total} ({exact_hits/total*100:.0f}%)")

    # ══════════════════════════════════════════════════════════════
    #  n-gram range sweep (the only tunable that moves accuracy here)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  n-gram range sweep (LogReg classifier)\n{'='*60}")
    best_ngram, best_acc = None, -1.0
    for ngram in [(2, 4), (2, 5), (2, 6), (3, 5), (3, 6)]:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram)
        clf = LogisticRegression(max_iter=2000, C=CLF_C)
        clf.fit(vec.fit_transform(ref_texts), np.array(ref_ids))
        pred = clf.classes_[clf.predict_proba(vec.transform(test_texts)).argmax(axis=1)]
        acc = sum(p == a for p, a in zip(pred, actual)) / total * 100
        marker = ""
        if acc > best_acc:
            best_acc, best_ngram = acc, ngram
            marker = " ◄ best"
        print(f"  ngram={ngram}  accuracy={acc:.1f}%{marker}")
    print(f"\n  Best n-gram range: {best_ngram} → {best_acc:.1f}%")

    # ══════════════════════════════════════════════════════════════
    #  Final evaluation at best n-gram (production cascade)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  Final evaluation (exact lookup → classifier)\n{'='*60}")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=best_ngram)
    clf = LogisticRegression(max_iter=2000, C=CLF_C)
    clf.fit(vec.fit_transform(ref_texts), np.array(ref_ids))
    proba = clf.predict_proba(vec.transform(test_texts))
    classes = clf.classes_

    preds, confs = [], []
    for fields, row_proba in zip(test_fields, proba):
        key = (fields[0], fields[1])
        if key in exact_lookup:
            preds.append(exact_lookup[key]); confs.append(100.0)
        else:
            j = int(row_proba.argmax())
            preds.append(classes[j]); confs.append(round(float(row_proba[j] * 100), 1))

    conf_arr = np.array(confs)
    correct_arr = np.array([p == a for p, a in zip(preds, actual)])
    print(f"  Overall accuracy: {correct_arr.mean()*100:.1f}% ({correct_arr.sum()}/{total})")
    print_reliability(conf_arr, correct_arr, total)
    print_policy_bands(conf_arr, correct_arr, total)

    # ══════════════════════════════════════════════════════════════
    #  Evidence guard: isolate rows the clues cannot determine
    # ══════════════════════════════════════════════════════════════
    # Classify each test row by available evidence and force the no-clue /
    # name-only rows to review, then show how much cleaner the auto-fill zone gets.
    reasons = [_evidence_reason(f) for f in test_fields]
    force_review = np.array([fr for _, fr in reasons])
    bucket = np.array(["no clue" if r.startswith("no clue") else
                       "weak (name only)" if r.startswith("weak") else "has clues"
                       for r, _ in reasons])

    print(f"\n{'='*60}\n  Evidence guard (isolate undeterminable rows)\n{'='*60}")
    print(f"  {'bucket':<20} {'rows':>5} {'errors':>7} {'accuracy':>9}")
    for b in ["has clues", "weak (name only)", "no clue"]:
        m = bucket == b
        if m.sum():
            print(f"  {b:<20} {int(m.sum()):>5} {int((~correct_arr[m]).sum()):>7} "
                  f"{correct_arr[m].mean()*100:>8.1f}%")

    auto = conf_arr >= AUTO_ACCEPT
    auto_guarded = auto & ~force_review
    print("\n  Auto-fill zone (confidence ≥70):")
    print(f"  {'':<22} {'rows':>5} {'wrong':>6} {'accuracy':>9}")
    print(f"  {'without guard':<22} {int(auto.sum()):>5} {int((~correct_arr[auto]).sum()):>6} "
          f"{correct_arr[auto].mean()*100:>8.1f}%")
    print(f"  {'with evidence guard':<22} {int(auto_guarded.sum()):>5} "
          f"{int((~correct_arr[auto_guarded]).sum()):>6} {correct_arr[auto_guarded].mean()*100:>8.1f}%")
    moved = int(auto.sum() - auto_guarded.sum())
    saved = int((~correct_arr[auto]).sum() - (~correct_arr[auto_guarded]).sum())
    print(f"  → guard moved {moved} rows out of auto-fill, removing {saved} wrong auto-fills")

    # ══════════════════════════════════════════════════════════════
    #  Error analysis
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  ERROR ANALYSIS\n{'='*60}")
    pred_arr, act_arr = np.array(preds), np.array(actual)
    wrong_idx = np.where(pred_arr != act_arr)[0]
    print(f"  Total errors: {len(wrong_idx)}/{total}")

    print("\n  Hardest Recharging_Item_IDs (most errors):")
    id_errors, id_totals = Counter(), Counter()
    for i in range(total):
        id_totals[actual[i]] += 1
        if pred_arr[i] != act_arr[i]:
            id_errors[actual[i]] += 1
    for rid, errs in id_errors.most_common(10):
        print(f"    {rid:<30} {errs}/{id_totals[rid]} errors")

    print("\n  Top confusion pairs (predicted → actual):")
    for (pred, act), count in Counter((pred_arr[i], act_arr[i]) for i in wrong_idx).most_common(10):
        print(f"    {pred:<20} → {act:<20} ({count}x)")

    print("\n  Sample errors (first 15):")
    print(f"  {'Predicted':<18} {'Actual':<18} {'Conf':>5}  Name | RG | DCS")
    print(f"  {'-'*90}")
    for i in wrong_idx[:15]:
        name, rg, dcs, app = test_fields[i]
        print(f"  {pred_arr[i]:<18} {act_arr[i]:<18} {conf_arr[i]:>5.0f}  "
              f"{name[:20]} | {rg[:20]} | {dcs[:12]}")

    print(f"\n{'='*60}")
    print("  NOTE: accuracy here is for GENUINELY NEW accounts (group split).")
    print("  In production ~44% of rows are seen accounts that hit the exact")
    print("  lookup at 100% confidence, so the real blended accuracy is higher.")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_benchmark()
