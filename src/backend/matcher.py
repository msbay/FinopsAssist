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

# Canonical column names — the GROUPED_V3 schema, where both GO_MAPPING_LEARNING and
# GO_MAPPING_EMPTY share the same plain column names (no Custom.focus_costs[...] split).
# Hierarchy labels: each Recharging_Item belongs to exactly one Product, which belongs
# to (almost always) one Product_Family — predicted item -> Product/Family via the tree.
COLS = {
    "sub_account_name": "SubAccountName",
    "sub_account_id": "SubAccountId",
    "provider": "ProviderName",
    "resource_group": "axa_Azure_ResourceGroupName",
    "tag_dcs": "axa_tags_global_dcs",
    "tag_app": "axa_tags_global_app",
    "recharging_item_id": "Recharging_Item_ID",
    "product_family": "Product_Family_information",
    "product_name": "Product_Name",
    "recharging_item_name": "Recharging_Item_Name",
    "cost": "Sumaxa_EffectiveCost_EUR",  # € spend — drives review prioritisation
}
# Both sheets now use the same names; kept as separate aliases for backward-compatible
# imports (review.py, app.py, run_pipeline.py, benchmark.py).
LEARNING_COLS = COLS
EMPTY_COLS = COLS

# Legacy (LIGHT_V2) column names -> canonical, so older workbooks still load. The old
# LEARNING sheet used a "Custom.*" prefix and the EMPTY sheet a "focus_costs[...]" one;
# both map to the same canonical name here.
_LEGACY_RENAME = {
    "Custom.focus_costs[SubAccountName]": "SubAccountName",
    "focus_costs[SubAccountName]": "SubAccountName",
    "Custom.focus_costs[ProviderName]": "ProviderName",
    "focus_costs[ProviderName]": "ProviderName",
    "Custom.focus_costs[SubAccountId]": "SubAccountId",
    "focus_costs[SubAccountId]": "SubAccountId",
    "Custom.focus_costs[axa_Azure_ResourceGroupName]": "axa_Azure_ResourceGroupName",
    "focus_costs[axa_Azure_ResourceGroupName]": "axa_Azure_ResourceGroupName",
    "Custom.focus_costs[axa_tags_global_dcs]": "axa_tags_global_dcs",
    "focus_costs[axa_tags_global_dcs]": "axa_tags_global_dcs",
    "Custom.focus_costs[axa_tags_global_app]": "axa_tags_global_app",
    "focus_costs[axa_tags_global_app]": "axa_tags_global_app",
    "Custom.[Sumaxa_EffectiveCost_EUR]": "Sumaxa_EffectiveCost_EUR",
    "[Sumaxa_EffectiveCost_EUR]": "Sumaxa_EffectiveCost_EUR",
    "Custom.gld_referential_mdm_axagorechargingitem[Recharging_Item_ID]": "Recharging_Item_ID",
    "gld_referential_mdm_axagorechargingitem[Recharging_Item_ID]": "Recharging_Item_ID",
    "Custom.gld_referential_mdm_axagoproduct[Product_Family_information]":
        "Product_Family_information",  # noqa: E501 — legacy column name is inherently long
    "gld_referential_mdm_axagoproduct[Product_Family_information]": "Product_Family_information",
    "Custom.gld_referential_mdm_axagoproduct[Product_Name]": "Product_Name",
    "gld_referential_mdm_axagoproduct[Product_Name]": "Product_Name",
    "Custom.gld_referential_mdm_axagorechargingitem[Recharging_Item_Name]": "Recharging_Item_Name",
    "gld_referential_mdm_axagorechargingitem[Recharging_Item_Name]": "Recharging_Item_Name",
}


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy LIGHT_V2 columns to the canonical V3 names. No-op on V3 files.

    LIGHT_V2's LEARNING sheet carries redundant pairs (both `focus_costs[...]` and
    `Custom.focus_costs[...]`) that map to the same canonical name; after renaming we
    keep, for each duplicated name, the column with the most non-null values.
    """
    rename = {old: new for old, new in _LEGACY_RENAME.items() if old in df.columns}
    if not rename:
        return df
    df = df.rename(columns=rename)
    if df.columns.duplicated().any():
        keep: dict[str, int] = {}
        for i, c in enumerate(df.columns):
            if c not in keep or df.iloc[:, i].notna().sum() > df.iloc[:, keep[c]].notna().sum():
                keep[c] = i
        df = df.iloc[:, sorted(keep.values())]
    return df

# Classifier hyper-parameters (tuned in benchmark.py)
NGRAM_RANGE = (2, 4)
CLF_C = 50.0  # C=50 gives a free ~+1pp over the original C=10 on the benchmark

# How many candidate ids to surface in Top_Matches — the tidy shortlist shown to
# humans for audit/display.
TOP_K_CANDIDATES = 5

# The agent gets a WIDER, high-recall candidate set (Candidate_IDs): take items by
# descending probability until the cumulative mass reaches NUCLEUS_P, capped at
# CANDIDATE_CAP. Adaptive — confident rows stay small, ambiguous ones widen. This
# config lifts the review-row candidate ceiling to ~82% at ~8 candidates avg
# (vs 60% at a flat top-5); presented as the Family>Product>Item tree it stays
# navigable. Tuned in the candidate-set experiment.
NUCLEUS_P = 0.85
CANDIDATE_CAP = 10

# Placeholder target values that are NOT real categories — a row still "to identify".
# These must never be learned as a class (the model would predict "to identify"), so we
# drop them from training even if they reappear in the data after a manual clean-up.
PLACEHOLDER_IDS = {"XX_TOIDENTIFY"}

# Suggested next action for a predicted row (two buckets):
#   • high-confidence  -> ready for a human to batch-approve (we NEVER auto-commit,
#     even at 100% — a person always confirms).
#   • everything else  -> Review: a human decides, with the LLM agent's prediction as an
#     assist (and no-signal rows handled at zero token cost).
# The column drives which bucket a row lands in and which rows run_review sends to Bedrock.
ACTION_APPROVE = "Approve (human)"
ACTION_AGENT = "Review"


def suggested_action(match_method: str, needs_review: bool,
                     agent_proposed_id: str = "", agent_ran: bool = False) -> str:
    """Two buckets: high-confidence rows are ready to approve; everything else goes to
    Review (including no-signal rows — a human still handles them; the LLM just can't).
    Once the agent proposes a defensible candidate, the row becomes ready to approve."""
    if not needs_review:
        return ACTION_APPROVE
    if agent_proposed_id:            # agent found a defensible candidate -> human confirms
        return ACTION_APPROVE
    return ACTION_AGENT              # needs review (low-confidence, no-signal, or abstained)


def trainable_rows(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Rows usable for training: a real, non-blank, non-placeholder Recharging_Item_ID."""
    ids = df[id_col].astype(str).str.strip()
    keep = df[id_col].notna() & ~ids.str.upper().isin(PLACEHOLDER_IDS | {"", "NAN"})
    return df[keep].copy()

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


def _no_signal(fields: tuple) -> bool:
    """True when a row carries NO learnable signal at all: an opaque/GUID (or empty)
    SubAccountName AND no resource group AND no tags. The recharging id of such a row
    is not derivable from the data — it is isolated with *no prediction* (the answer
    lives in owner/cloud knowledge, not in these fields), rather than guessed.
    Applies only after exact-lookup fails: a previously-seen opaque id still resolves."""
    name, rg, dcs, app = fields
    if rg or dcs or app:
        return False
    return bool(_OPAQUE_NAME.match(name)) or name == ""


class RechargingMatcher:
    """Exact-lookup shortcut + char n-gram logistic-regression classifier."""

    def __init__(self):
        self.ref_ids: list[str] = []
        self.exact_lookup: dict[tuple, str] = {}
        self.name_lookup: dict[str, str] = {}
        self.vectorizer: TfidfVectorizer | None = None
        self.clf: LogisticRegression | None = None
        # Hierarchy: item id -> Product / Family / readable item name (majority vote),
        # plus the Family/Product of each classifier class, aligned to clf.classes_,
        # so we can marginalize the item probability vector up the tree.
        self.item_to_family: dict[str, str] = {}
        self.item_to_product: dict[str, str] = {}
        self.item_to_name: dict[str, str] = {}
        self.class_family: np.ndarray | None = None
        self.class_product: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Field extraction / text representation
    # ------------------------------------------------------------------
    @staticmethod
    def _get_fields(row: pd.Series, cols: dict) -> tuple[str, str, str, str]:
        """Extract and normalize the four text fields."""
        vals = [str(row.get(cols[k], "") or "").lower().strip()
                for k in ("sub_account_name", "resource_group", "tag_dcs", "tag_app")]
        name = vals[0]
        rg, dcs, app = ("" if v == "nan" else v for v in vals[1:])
        return name, rg, dcs, app

    @staticmethod
    def _to_text(fields: tuple) -> str:
        """Single string for the classifier: name | rg | dcs | app (skip blanks)."""
        return " | ".join(f for f in fields if f)

    # ------------------------------------------------------------------
    # Index building (exact lookups + train classifier)
    # ------------------------------------------------------------------
    def build_index(self, df_learning: pd.DataFrame) -> None:
        df_learning = normalize_schema(df_learning)
        id_col = LEARNING_COLS["recharging_item_id"]
        # Drop blank/placeholder (XX_TOIDENTIFY) targets — never learn them as a class.
        df_clean = trainable_rows(df_learning, id_col)

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

        # Learn the hierarchy: item id -> majority Product / Family / readable name.
        def majority(col_key: str) -> dict[str, str]:
            if col_key not in df_clean.columns:
                return {}
            tmp: dict[str, list[str]] = {}
            for rid, val in zip(self.ref_ids, df_clean[col_key].astype(str)):
                tmp.setdefault(rid, []).append(val)
            return {k: Counter(v).most_common(1)[0][0] for k, v in tmp.items()}

        self.item_to_family = majority(LEARNING_COLS["product_family"])
        self.item_to_product = majority(LEARNING_COLS["product_name"])
        self.item_to_name = majority(LEARNING_COLS["recharging_item_name"])

        # Train the classifier on the row text
        texts = [self._to_text(f) for f in ref_fields]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE)
        x = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=2000, C=CLF_C)
        self.clf.fit(x, np.array(self.ref_ids))

        # Family/Product of each class, aligned to clf.classes_, for marginalization.
        self.class_family = np.array([self.item_to_family.get(c, "") for c in self.clf.classes_])
        self.class_product = np.array([self.item_to_product.get(c, "") for c in self.clf.classes_])

        print(f"Index built: {len(ref_fields)} items, {len(set(self.ref_ids))} unique IDs")
        print(f"  Hierarchy: {len(set(self.item_to_family.values()))} families, "
              f"{len(set(self.item_to_product.values()))} products, {len(set(self.ref_ids))} items")
        print(f"  Exact (name+RG) keys: {len(self.exact_lookup)}")
        print(f"  Safe name-only keys:  {len(self.name_lookup)}")
        print(f"  Classifier: LogReg over char {NGRAM_RANGE} n-grams")

    # ------------------------------------------------------------------
    # Hierarchy helpers
    # ------------------------------------------------------------------
    def _ancestry(self, item_id: str) -> tuple[str, str, str]:
        """(Product_Family, Product_Name, Recharging_Item_Name) for an item id."""
        return (self.item_to_family.get(item_id, ""),
                self.item_to_product.get(item_id, ""),
                self.item_to_name.get(item_id, item_id))

    def _level_confidence(self, row_proba: np.ndarray, item_id: str) -> tuple[float, float]:
        """Marginalize the item probability vector up the tree:
        Family_conf = Σ P(item) over items in the predicted item's family; likewise Product.
        Coarser levels aggregate more mass, so confidence rises up the tree — and is
        always consistent with the chosen item."""
        fam, prod, _ = self._ancestry(item_id)
        fam_mask = self.class_family == fam
        # Nest the product strictly inside the predicted family. Most items map to a
        # single (product, family) pair, but the placeholder product
        # "(IT own Cloud consumption)" holds the XX_* catch-all items across many
        # families — nesting keeps Family_conf >= Product_conf >= Item_conf always.
        prod_mask = fam_mask & (self.class_product == prod)
        fam_conf = round(float(row_proba[fam_mask].sum() * 100), 1) if fam else 0.0
        prod_conf = round(float(row_proba[prod_mask].sum() * 100), 1) if prod else 0.0
        return fam_conf, prod_conf

    def _nucleus_candidates(self, row_proba: np.ndarray,
                            classes: np.ndarray) -> list[tuple[str, float]]:
        """High-recall candidate set for the agent, as (id, probability) pairs ordered
        by descending probability, until cumulative mass reaches NUCLEUS_P (capped at
        CANDIDATE_CAP). The order is the classifier's confidence ranking."""
        order = np.argsort(row_proba)[::-1]
        cum = np.cumsum(row_proba[order])
        n = min(int(np.searchsorted(cum, NUCLEUS_P)) + 1, CANDIDATE_CAP)
        return [(str(classes[j]), float(row_proba[j])) for j in order[:n]]

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, df_empty: pd.DataFrame, confidence_threshold: float = 50.0) -> pd.DataFrame:
        """Predict Recharging_Item_ID for each row in GO_MAPPING_EMPTY.

        Adds columns:
        - Predicted_Recharging_Item_ID
        - Confidence (0-100; classifier probability, or 100/95 for exact matches)
        - Top_Matches (top-K candidates for audit; K = TOP_K_CANDIDATES)
        - Needs_Review (Confidence < threshold, OR a no-clue / name-only row)
        - Match_Method (exact_name_rg, exact_name, classifier)
        - Review_Reason (why a row is flagged: no clue / weak / check with LLM)
        """
        if self.clf is None or self.vectorizer is None:
            raise RuntimeError("build_index() must be called before predict()")

        df_empty = normalize_schema(df_empty)
        empty_fields = [self._get_fields(row, EMPTY_COLS) for _, row in df_empty.iterrows()]
        # Batch the classifier over every row at once.
        proba = self.clf.predict_proba(
            self.vectorizer.transform([self._to_text(f) for f in empty_fields]))
        classes = self.clf.classes_

        results = []
        for q_fields, row_proba in zip(empty_fields, proba):
            exact_key = (q_fields[0], q_fields[1])
            if exact_key in self.exact_lookup:
                pred = self.exact_lookup[exact_key]
                fam, prod, item_name = self._ancestry(pred)
                results.append({
                    "Predicted_Product_Family": fam, "Family_Confidence": 100.0,
                    "Predicted_Product_Name": prod, "Product_Confidence": 100.0,
                    "Predicted_Recharging_Item_ID": pred,
                    "Predicted_Recharging_Item_Name": item_name,
                    "Confidence": 100.0,
                    "Top_Matches": f"{pred} (exact name+RG)",
                    "Candidate_IDs": pred,
                    "Needs_Review": False,
                    "Match_Method": "exact_name_rg",
                    "Review_Reason": "",
                })
                continue

            if q_fields[0] in self.name_lookup:
                pred = self.name_lookup[q_fields[0]]
                fam, prod, item_name = self._ancestry(pred)
                results.append({
                    "Predicted_Product_Family": fam, "Family_Confidence": 95.0,
                    "Predicted_Product_Name": prod, "Product_Confidence": 95.0,
                    "Predicted_Recharging_Item_ID": pred,
                    "Predicted_Recharging_Item_Name": item_name,
                    "Confidence": 95.0,
                    "Top_Matches": f"{pred} (exact name)",
                    "Candidate_IDs": pred,
                    "Needs_Review": False,
                    "Match_Method": "exact_name",
                    "Review_Reason": "",
                })
                continue

            # No signal at all (opaque/empty name, no rg, no tags) -> do NOT guess.
            # Isolate with an empty prediction; the answer needs owner/cloud knowledge.
            if _no_signal(q_fields):
                results.append({
                    "Predicted_Product_Family": "", "Family_Confidence": 0.0,
                    "Predicted_Product_Name": "", "Product_Confidence": 0.0,
                    "Predicted_Recharging_Item_ID": "", "Predicted_Recharging_Item_Name": "",
                    "Confidence": 0.0, "Top_Matches": "", "Candidate_IDs": "",
                    "Needs_Review": True, "Match_Method": "no_signal",
                    "Review_Reason": "no signal — opaque/empty name, no resource group or "
                                     "tags; not predictable from data (needs owner/enrichment)",
                })
                continue

            order = np.argsort(row_proba)[::-1]
            pred = classes[order[0]]
            confidence = round(float(row_proba[order[0]] * 100), 1)
            top_matches = " | ".join(f"{classes[j]} ({row_proba[j]:.2f})"
                                     for j in order[:TOP_K_CANDIDATES])
            candidate_ids = " | ".join(f"{cid} ({p:.2f})"
                                       for cid, p in self._nucleus_candidates(row_proba, classes))
            fam, prod, item_name = self._ancestry(pred)
            fam_conf, prod_conf = self._level_confidence(row_proba, pred)
            # Route by available evidence: isolate the rows the clues cannot
            # determine (no clue / name-only) even if the classifier looks confident.
            reason, force_review = _evidence_reason(q_fields)
            needs_review = bool(force_review or confidence < confidence_threshold)
            results.append({
                "Predicted_Product_Family": fam, "Family_Confidence": fam_conf,
                "Predicted_Product_Name": prod, "Product_Confidence": prod_conf,
                "Predicted_Recharging_Item_ID": pred,
                "Predicted_Recharging_Item_Name": item_name,
                "Confidence": confidence,
                "Top_Matches": top_matches,
                "Candidate_IDs": candidate_ids,
                "Needs_Review": needs_review,
                "Match_Method": "classifier",
                # Only tag a reason on rows actually routed to review — an
                # auto-filled row is not "to be checked by the LLM".
                "Review_Reason": reason if needs_review else "",
            })

        result_df = df_empty.copy()
        for col in ["Predicted_Product_Family", "Family_Confidence",
                    "Predicted_Product_Name", "Product_Confidence",
                    "Predicted_Recharging_Item_ID", "Predicted_Recharging_Item_Name",
                    "Confidence", "Top_Matches", "Candidate_IDs",
                    "Needs_Review", "Match_Method", "Review_Reason"]:
            result_df[col] = [r[col] for r in results]

        # The routing suggestion the dashboard shows and run_review acts on (agent not
        # run yet at prediction time, so this is the pre-investigation recommendation).
        result_df["Suggested_Action"] = [
            suggested_action(r["Match_Method"], r["Needs_Review"]) for r in results]

        return result_df.sort_values("Confidence", ascending=True)
