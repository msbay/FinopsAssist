"""Benchmark GO_MAPPING predictions on the GROUPED_V5 workbook.

V5 adds AWS-specific enrichment columns (AwsAccountTags.owner / .global.dcs /
.local.description / .global.app) that describe an AWS account in plain text —
exactly the signal missing for the many AWS accounts whose SubAccountName is an
opaque GUID. Per the requirement, those AWS-only columns are used ONLY for AWS
rows; Azure rows never see them.

What this script does
---------------------
For each provider (AWS, Azure) and each of the four targets
(Product_Family_information, Product_Name, Recharging_Item_Name,
Recharging_Item_ID) it:

  * splits GO_MAPPING_LEARNING by SubAccountName (a *group* split, so no account
    is in both train and test — this simulates genuinely new accounts, the only
    honest way to score generalisation), averaged over several seeds;
  * sweeps feature-column combinations to find which columns actually help;
  * compares predicting each target DIRECTLY vs. predicting the finest grain
    (Recharging_Item_ID) and mapping UP the deterministic tree
    (ID -> Name/Product/Family is 1:1 in the data).

Design choices, and why
-----------------------
  * Provider-split models. AWS (592 rows, 1 row/account) and Azure (3085 rows,
    174 accounts) have different feature availability and label distributions,
    and the AWS-only columns must not leak into Azure. Training one model per
    provider is the clean way to honour that and lets AWS fully exploit its tags.
    A global single-model baseline is also reported for comparison.
  * char_wb n-gram TF-IDF + multinomial LogisticRegression — the shipping
    representation; it beat fuzzy matching in the original benchmark and needs no
    tokenisation assumptions across GUIDs / emails / free text.
"""

import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore", category=ConvergenceWarning)

INPUT_FILE = "GO Report Extract GROUPED_V5.xlsx"
SHEET = "GO_MAPPING_LEARNING"
TEST_RATIO = 0.2
REPEATS = 5
NGRAM = (2, 4)
CLF_C = 50.0

NAME_COL = "SubAccountName"
PROVIDER_COL = "ProviderName"
TARGETS = [
    "Product_Family_information",
    "Product_Name",
    "Recharging_Item_Name",
    "Recharging_Item_ID",
]
ITEM_ID = "Recharging_Item_ID"
PLACEHOLDER_IDS = {"XX_TOIDENTIFY"}

# Feature columns, tagged by which provider may use them.
#   shared -> usable for both AWS and Azure
#   aws    -> AWS rows only (V5 enrichment); never shown to Azure
#   azure  -> Azure only (resource group does not exist for AWS)
SHARED_FEATURES = ["SubAccountName", "axa_tags_global_dcs", "axa_tags_global_app"]
AZURE_ONLY = ["axa_Azure_ResourceGroupName"]
AWS_ONLY = [
    "AwsAccountTags.owner",
    "AwsAccountTags.global.dcs",
    "AwsAccountTags.local.description",
    "AwsAccountTags.global.app",
]


def clean(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower().str.strip()
    return s.replace({"nan": "", "none": ""})


def build_text(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """One string per row: 'col_value | col_value | ...' over the given columns,
    skipping blanks. Columns absent from df are silently skipped."""
    present = [c for c in cols if c in df.columns]
    cleaned = {c: clean(df[c]) for c in present}
    out = []
    for i in range(len(df)):
        parts = [cleaned[c].iat[i] for c in present]
        out.append(" | ".join(p for p in parts if p))
    return out


def trainable(df: pd.DataFrame) -> pd.DataFrame:
    ids = df[ITEM_ID].astype(str).str.strip().str.upper()
    return df[df[ITEM_ID].notna() & ~ids.isin(PLACEHOLDER_IDS | {"", "NAN"})].copy()


def group_splits(df: pd.DataFrame, seeds: range):
    groups = df[NAME_COL].astype(str).str.lower().str.strip()
    for seed in seeds:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO,
                                        random_state=seed).split(df, groups=groups))
        yield tr, te


def fit_predict(train_text, y_train, test_text) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM)
    clf = LogisticRegression(max_iter=2000, C=CLF_C)
    clf.fit(vec.fit_transform(train_text), y_train)
    proba = clf.predict_proba(vec.transform(test_text))
    return clf.classes_[proba.argmax(1)]


def score_direct(df: pd.DataFrame, cols: list[str], target: str) -> tuple[float, float]:
    """Mean/std accuracy over REPEATS group splits, predicting `target` directly
    from the text of `cols`."""
    text = build_text(df, cols)
    y = df[target].astype(str).values
    accs = []
    for tr, te in group_splits(df, range(REPEATS)):
        pred = fit_predict([text[i] for i in tr], y[tr], [text[i] for i in te])
        accs.append((pred == y[te]).mean() * 100)
    return float(np.mean(accs)), float(np.std(accs))


def id_to_ancestor(df: pd.DataFrame, target: str) -> dict[str, str]:
    """Majority map Recharging_Item_ID -> value of an ancestor target."""
    m: dict[str, list[str]] = {}
    for rid, val in zip(df[ITEM_ID].astype(str), df[target].astype(str)):
        m.setdefault(rid, []).append(val)
    return {k: Counter(v).most_common(1)[0][0] for k, v in m.items()}


def score_via_tree(df: pd.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
    """Predict Recharging_Item_ID, then map up the tree to every target.
    Returns {target: (mean_acc, std)} for all four targets in one pass."""
    text = build_text(df, cols)
    y_id = df[ITEM_ID].astype(str).values
    per_target: dict[str, list[float]] = {t: [] for t in TARGETS}
    for tr, te in group_splits(df, range(REPEATS)):
        df_tr = df.iloc[tr]
        maps = {t: id_to_ancestor(df_tr, t) for t in TARGETS if t != ITEM_ID}
        pred_id = fit_predict([text[i] for i in tr], y_id[tr], [text[i] for i in te])
        for t in TARGETS:
            if t == ITEM_ID:
                pred = pred_id
            else:
                pred = np.array([maps[t].get(p, "") for p in pred_id])
            actual = df.iloc[te][t].astype(str).values
            per_target[t].append((pred == actual).mean() * 100)
    return {t: (float(np.mean(v)), float(np.std(v))) for t, v in per_target.items()}


def provider_features(provider: str) -> dict[str, list[str]]:
    """Named feature sets to sweep, per provider."""
    if provider == "AWS":
        return {
            "name only": ["SubAccountName"],
            "name + axa tags": SHARED_FEATURES,
            "name + AWS tags": ["SubAccountName"] + AWS_ONLY,
            "name + AWS desc only": ["SubAccountName", "AwsAccountTags.local.description"],
            "shared + AWS tags (ALL)": SHARED_FEATURES + AWS_ONLY,
        }
    return {
        "name only": ["SubAccountName"],
        "name + RG": ["SubAccountName", "axa_Azure_ResourceGroupName"],
        "name + axa tags": SHARED_FEATURES,
        "shared + RG (ALL)": SHARED_FEATURES + AZURE_ONLY,
    }


def run_provider(df_all: pd.DataFrame, provider: str) -> None:
    df = trainable(df_all[df_all[PROVIDER_COL] == provider])
    n_acc = df[NAME_COL].nunique()
    print(f"\n{'='*78}\n  {provider}   ({len(df)} rows, {n_acc} accounts, "
          f"{df[ITEM_ID].nunique()} distinct Recharging_Item_IDs)\n{'='*78}")

    feats = provider_features(provider)

    # ── Feature ablation: predict Recharging_Item_ID (the hardest target) directly ──
    print(f"\n  Feature ablation — accuracy on {ITEM_ID} (hardest target), "
          f"avg of {REPEATS} account-splits:")
    print(f"  {'feature set':<28} {'accuracy':>12}")
    ranked = []
    for label, cols in feats.items():
        m, s = score_direct(df, cols, ITEM_ID)
        ranked.append((m, s, label, cols))
        print(f"  {label:<28} {m:>7.1f}% ±{s:>3.1f}")
    ranked.sort(reverse=True)
    best_acc, best_std, best_label, best_cols = ranked[0]
    print(f"  → best feature set: '{best_label}'  ({best_acc:.1f}%)")

    # ── All four targets, with the best feature set: direct vs. via-ID-tree ──
    print(f"\n  All four targets with best features ('{best_label}'):")
    print(f"  {'target':<30} {'#cls':>5} {'direct':>12} {'via ID-tree':>14}")
    tree = score_via_tree(df, best_cols)
    for t in TARGETS:
        dm, ds = score_direct(df, best_cols, t)
        tm, ts = tree[t]
        ncls = df[t].nunique()
        star = " ◄" if dm >= tm else ""
        print(f"  {t:<30} {ncls:>5} {dm:>7.1f}% ±{ds:>3.1f} {tm:>8.1f}% ±{ts:>3.1f}{star}")
    print("  (◄ = direct wins; else predict ID once and map up the deterministic tree)")


def run_global_baseline(df_all: pd.DataFrame) -> None:
    """Single model over both providers. AWS-only columns are still gated to AWS
    rows (blank for Azure) so the requirement holds even in the shared model."""
    df = trainable(df_all).copy()
    # Blank out AWS-only columns on non-AWS rows so Azure never sees them.
    for c in AWS_ONLY:
        if c in df.columns:
            df.loc[df[PROVIDER_COL] != "AWS", c] = np.nan
    cols = SHARED_FEATURES + AZURE_ONLY + AWS_ONLY
    print(f"\n{'='*78}\n  GLOBAL baseline (one model, both providers, AWS cols gated to AWS)"
          f"\n{'='*78}")
    print(f"  {'target':<30} {'overall':>10} {'AWS':>8} {'Azure':>8}")
    text = build_text(df, cols)
    prov = df[PROVIDER_COL].values
    for t in TARGETS:
        y = df[t].astype(str).values
        acc_all, acc_aws, acc_az = [], [], []
        for tr, te in group_splits(df, range(REPEATS)):
            pred = fit_predict([text[i] for i in tr], y[tr], [text[i] for i in te])
            corr = pred == y[te]
            acc_all.append(corr.mean() * 100)
            pv = prov[te]
            acc_aws.append(corr[pv == "AWS"].mean() * 100 if (pv == "AWS").any() else np.nan)
            is_az = pv == "Microsoft"
            acc_az.append(corr[is_az].mean() * 100 if is_az.any() else np.nan)
        print(f"  {t:<30} {np.mean(acc_all):>9.1f}% {np.nanmean(acc_aws):>7.1f}% "
              f"{np.nanmean(acc_az):>7.1f}%")


def main() -> None:
    print(f"Loading {INPUT_FILE} :: {SHEET}")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET)
    print(f"  {len(df)} rows | providers: {dict(df[PROVIDER_COL].value_counts())}")
    print(f"  Split: group-by-{NAME_COL} (no account in both train/test), "
          f"{REPEATS} seeds, test={TEST_RATIO:.0%}")
    print("  Model: char_wb TF-IDF n-grams "
          f"{NGRAM} + LogisticRegression(C={CLF_C})")

    for provider in ["AWS", "Microsoft"]:
        run_provider(df, provider)

    run_global_baseline(df)


if __name__ == "__main__":
    sys.exit(main())
