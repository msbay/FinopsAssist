"""Recharging_Item_ID predictor.

Two layers, in order of cost:
  1. Exact lookup — a new row whose (SubAccountName, ResourceGroup) was seen in
     the learning data inherits that ID with full confidence. Cheap and certain.
  2. Learned classifier — multinomial logistic regression over character n-gram
     TF-IDF of the row text. Generalizes to genuinely new accounts and its
     predict_proba is a calibrated confidence.

This is the configuration that won the benchmark (see benchmark.py): on
genuinely new accounts (group split by account) it scores ~80% vs ~74% for the
old hand-weighted fuzzy matcher, with better-calibrated confidence.

Rows the available clues cannot determine (an opaque/short name with no resource
group or tags) are isolated up front via Review_Reason and always sent to review
— see the error analysis in benchmark.py for why.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Column name mappings for each sheet
LEARNING_COLS = {
    "sub_account_name": "Custom.focus_costs[SubAccountName]",
    "sub_account_id": "Custom.focus_costs[SubAccountId]",
    "resource_group": "Custom.focus_costs[axa_Azure_ResourceGroupName]",
    "tag_dcs": "Custom.focus_costs[axa_tags_global_dcs]",
    "tag_app": "Custom.focus_costs[axa_tags_global_app]",
    "recharging_item_id": "Custom.gld_referential_mdm_axagorechargingitem[Recharging_Item_ID]",
}

EMPTY_COLS = {
    "sub_account_name": "focus_costs[SubAccountName]",
    "sub_account_id": "focus_costs[SubAccountId]",
    "resource_group": "focus_costs[axa_Azure_ResourceGroupName]",
    "tag_dcs": "focus_costs[axa_tags_global_dcs]",
    "tag_app": "focus_costs[axa_tags_global_app]",
}

# Classifier hyper-parameters (tuned in benchmark.py)
NGRAM_RANGE = (2, 4)
CLF_C = 50.0  # C=50 gives a free ~+1pp over the original C=10 on the benchmark

# An opaque name = a GUID/hash or a very short token. Carries no learnable signal.
_OPAQUE_NAME = re.compile(r"^[0-9a-f]{16,}$")


def _evidence_reason(fields: tuple) -> tuple[str, bool]:
    """Classify a row by the evidence available, for review routing.

    Returns (reason, force_review). The error analysis (see benchmark.py) showed
    most misses come from rows the clues simply cannot determine — so we isolate
    those instead of letting them get a confident-looking wrong guess.
    """
    name, rg, dcs, app = fields
    has_clues = bool(rg) or bool(dcs) or bool(app)
    if not has_clues:
        if _OPAQUE_NAME.match(name) or len(name) <= 6:
            # Bucket 1: opaque/short name, nothing else — not inferable.
            return "no clue — opaque name, no resource group or tags; needs human", True
        # Bucket 2: a plain name and nothing else — weak, unreliable.
        return "weak — name only, no resource group or tags; low reliability", True
    # Bucket 3: has a resource group and/or tags — there is something to reason
    # over, so a borderline prediction is worth an LLM second opinion.
    return "check with LLM — has clues but name pattern is ambiguous", False


class RechargingMatcher:
    """Exact-lookup shortcut + char n-gram logistic-regression classifier."""

    def __init__(self):
        self.ref_ids: list[str] = []
        self.exact_lookup: dict[tuple, str] = {}
        self.name_lookup: dict[str, str] = {}
        self.vectorizer: TfidfVectorizer | None = None
        self.clf: LogisticRegression | None = None

    # ------------------------------------------------------------------
    # Field extraction / text representation
    # ------------------------------------------------------------------
    @staticmethod
    def _get_fields(row: pd.Series, cols: dict) -> tuple[str, str, str, str]:
        """Extract and normalize the four text fields."""
        name = str(row.get(cols["sub_account_name"], "") or "").lower().strip()
        rg = str(row.get(cols["resource_group"], "") or "").lower().strip()
        dcs = str(row.get(cols["tag_dcs"], "") or "").lower().strip()
        app = str(row.get(cols["tag_app"], "") or "").lower().strip()
        if rg == "nan": rg = ""
        if dcs == "nan": dcs = ""
        if app == "nan": app = ""
        return name, rg, dcs, app

    @staticmethod
    def _to_text(fields: tuple) -> str:
        """Single string for the classifier: name | rg | dcs | app (skip blanks)."""
        return " | ".join(f for f in fields if f)

    # ------------------------------------------------------------------
    # Index building (exact lookups + train classifier)
    # ------------------------------------------------------------------
    def build_index(self, df_learning: pd.DataFrame) -> None:
        id_col = LEARNING_COLS["recharging_item_id"]
        df_clean = df_learning.dropna(subset=[id_col]).copy()

        ref_fields = [self._get_fields(row, LEARNING_COLS) for _, row in df_clean.iterrows()]
        self.ref_ids = df_clean[id_col].astype(str).tolist()

        # Exact (name, rg) → majority ID
        key_ids: dict[tuple, list[str]] = {}
        for fields, rid in zip(ref_fields, self.ref_ids):
            key_ids.setdefault((fields[0], fields[1]), []).append(rid)
        self.exact_lookup = {k: Counter(v).most_common(1)[0][0] for k, v in key_ids.items()}

        # Name-only → ID, but only when the name is unambiguous
        name_ids: dict[str, set[str]] = {}
        for fields, rid in zip(ref_fields, self.ref_ids):
            name_ids.setdefault(fields[0], set()).add(rid)
        self.name_lookup = {n: ids.pop() for n, ids in name_ids.items() if len(ids) == 1}

        # Train the classifier on the row text
        texts = [self._to_text(f) for f in ref_fields]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE)
        x = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=2000, C=CLF_C)
        self.clf.fit(x, np.array(self.ref_ids))

        print(f"Index built: {len(ref_fields)} items, {len(set(self.ref_ids))} unique IDs")
        print(f"  Exact (name+RG) keys: {len(self.exact_lookup)}")
        print(f"  Safe name-only keys:  {len(self.name_lookup)}")
        print(f"  Classifier: LogReg over char {NGRAM_RANGE} n-grams")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, df_empty: pd.DataFrame, confidence_threshold: float = 50.0) -> pd.DataFrame:
        """Predict Recharging_Item_ID for each row in GO_MAPPING_EMPTY.

        Adds columns:
        - Predicted_Recharging_Item_ID
        - Confidence (0-100; classifier probability, or 100/95 for exact matches)
        - Top_Matches (top-3 candidates for audit)
        - Needs_Review (Confidence < threshold, OR a no-clue / name-only row)
        - Match_Method (exact_name_rg, exact_name, classifier)
        - Review_Reason (why a row is flagged: no clue / weak / check with LLM)
        """
        if self.clf is None or self.vectorizer is None:
            raise RuntimeError("build_index() must be called before predict()")

        empty_fields = [self._get_fields(row, EMPTY_COLS) for _, row in df_empty.iterrows()]
        # Batch the classifier over every row at once.
        proba = self.clf.predict_proba(self.vectorizer.transform([self._to_text(f) for f in empty_fields]))
        classes = self.clf.classes_

        results = []
        for q_fields, row_proba in zip(empty_fields, proba):
            exact_key = (q_fields[0], q_fields[1])
            if exact_key in self.exact_lookup:
                pred = self.exact_lookup[exact_key]
                results.append({
                    "Predicted_Recharging_Item_ID": pred,
                    "Confidence": 100.0,
                    "Top_Matches": f"{pred} (exact name+RG)",
                    "Needs_Review": False,
                    "Match_Method": "exact_name_rg",
                    "Review_Reason": "",
                })
                continue

            if q_fields[0] in self.name_lookup:
                pred = self.name_lookup[q_fields[0]]
                results.append({
                    "Predicted_Recharging_Item_ID": pred,
                    "Confidence": 95.0,
                    "Top_Matches": f"{pred} (exact name)",
                    "Needs_Review": False,
                    "Match_Method": "exact_name",
                    "Review_Reason": "",
                })
                continue

            order = np.argsort(row_proba)[::-1]
            pred = classes[order[0]]
            confidence = round(float(row_proba[order[0]] * 100), 1)
            top_matches = " | ".join(f"{classes[j]} ({row_proba[j]:.2f})" for j in order[:3])
            # Route by available evidence: isolate the rows the clues cannot
            # determine (no clue / name-only) even if the classifier looks confident.
            reason, force_review = _evidence_reason(q_fields)
            needs_review = bool(force_review or confidence < confidence_threshold)
            results.append({
                "Predicted_Recharging_Item_ID": pred,
                "Confidence": confidence,
                "Top_Matches": top_matches,
                "Needs_Review": needs_review,
                "Match_Method": "classifier",
                # Only tag a reason on rows actually routed to review — an
                # auto-filled row is not "to be checked by the LLM".
                "Review_Reason": reason if needs_review else "",
            })

        result_df = df_empty.copy()
        for col in ["Predicted_Recharging_Item_ID", "Confidence", "Top_Matches",
                    "Needs_Review", "Match_Method", "Review_Reason"]:
            result_df[col] = [r[col] for r in results]

        return result_df.sort_values("Confidence", ascending=True)
