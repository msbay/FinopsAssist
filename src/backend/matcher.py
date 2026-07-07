"""Recharging_Item_ID predictor.

Two layers, in order of cost:
  1. Exact lookup — a new row whose (SubAccountName, ResourceGroup) was seen in
     the learning data inherits that ID with full confidence. Cheap and certain.
  2. Learned classifier — multinomial logistic regression over character n-gram
     TF-IDF of the row text. Generalizes to genuinely new accounts and its
     predict_proba is a calibrated confidence.

Per-provider models
-------------------
AWS and Azure accounts carry different clues, so we train ONE classifier per
provider (see benchmark_v5.py):
  * Azure rows read the ResourceGroup + axa tags. AWS has no ResourceGroup.
  * AWS rows additionally read the V5 enrichment columns (AwsAccountTags.owner /
    global.dcs / local.description / name). These describe the account in plain
    text and are the decisive signal for the many AWS accounts whose
    SubAccountName is an opaque GUID. Per the data-owner requirement they are fed
    ONLY to AWS rows; an Azure row never sees them.
On genuinely new accounts (group split by account) this scores ~87-88% on
Recharging_Item_ID for each provider and ~94% on Product_Family — a ~4pp lift
over one global model, and +18pp on AWS over name-only. The AWS tag columns use
different names in the LEARNING and EMPTY sheets; AWS_TAG_SOURCES maps both.

Rows the available clues cannot determine (an opaque/short name with no resource
group or tags) are isolated up front via Review_Reason and always sent to review
— see the error analysis for why.
"""

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Canonical column names — the GROUPED_V3+ schema, where both GO_MAPPING_LEARNING and
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

# AWS-only enrichment (GROUPED_V5). Each canonical slot lists the source columns to try
# in order — the LEARNING and EMPTY sheets name these differently:
#   slot     LEARNING sheet                     EMPTY sheet
#   aws_desc AwsAccountTags.local.description    AwsAccountTags.description
#   aws_name AwsAccountTags.global.app          AwsAccountTags.name  (deanonymises GUIDs)
# The first non-blank source wins; absent columns (e.g. V3 workbooks) are skipped.
AWS_TAG_SOURCES = {
    "aws_owner": ["AwsAccountTags.owner"],
    "aws_dcs": ["AwsAccountTags.global.dcs"],
    "aws_desc": ["AwsAccountTags.local.description", "AwsAccountTags.description"],
    "aws_name": ["AwsAccountTags.global.app", "AwsAccountTags.name"],
}

# Provider buckets. One classifier per bucket; anything not AWS is treated as Azure
# (Microsoft) — the shared/RG feature set has no AWS-only columns, so it never leaks them.
PROVIDER_AWS = "AWS"
PROVIDER_AZURE = "AZURE"

# Which fields (besides the name, which always leads the text) feed each bucket's
# classifier. AWS gets the enrichment slots; Azure gets the ResourceGroup.
_SHARED_CLUES = ("tag_dcs", "tag_app")
_AZURE_EXTRA = ("resource_group",)
_AWS_EXTRA = ("aws_owner", "aws_dcs", "aws_desc", "aws_name")


def provider_bucket(raw) -> str:
    """Map a raw ProviderName to a model bucket ('AWS' or 'AZURE')."""
    return PROVIDER_AWS if "aws" in str(raw or "").lower() else PROVIDER_AZURE


def _clue_keys(bucket: str) -> tuple[str, ...]:
    """Field keys that count as evidence (the name is judged separately)."""
    return _SHARED_CLUES + (_AWS_EXTRA if bucket == PROVIDER_AWS else _AZURE_EXTRA)

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
    """Rename legacy LIGHT_V2 columns to the canonical V3 names. No-op on V3+ files.

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

# Classifier hyper-parameters (tuned in benchmark_v5.py — char_wb(2,4)+LogReg C=50 is
# the sweet spot; word n-grams / other classifiers don't beat it, and the lift comes
# from the feature columns, not the model).
NGRAM_RANGE = (2, 4)
CLF_C = 50.0

# How many candidate ids to surface in Top_Matches — the tidy shortlist shown to
# humans for audit/display.
TOP_K_CANDIDATES = 5

# The agent gets a WIDER, high-recall candidate set (Candidate_IDs): take items by
# descending probability until the cumulative mass reaches NUCLEUS_P, capped at
# CANDIDATE_CAP. Adaptive — confident rows stay small, ambiguous ones widen.
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


def _evidence_reason(fields: dict, bucket: str) -> tuple[str, bool]:
    """Classify a row by the evidence available, for review routing.

    Returns (reason, force_review). Most misses come from rows the clues simply
    cannot determine — so we isolate those instead of letting them get a
    confident-looking wrong guess. AWS rows count their enrichment tags as clues.
    """
    name = fields["name"]
    has_clues = any(fields.get(k) for k in _clue_keys(bucket))
    if not has_clues:
        if _OPAQUE_NAME.match(name) or len(name) <= 6:
            # Bucket 1: opaque/short name, nothing else — not inferable.
            return "no clue — opaque name, no resource group or tags; needs human", True
        # Bucket 2: a plain name and nothing else — weak, unreliable.
        return "weak — name only, no resource group or tags; low reliability", True
    # Bucket 3: has a resource group and/or tags — there is something to reason
    # over, so a borderline prediction is worth an LLM second opinion.
    return "check with LLM — has clues but name pattern is ambiguous", False


def _no_signal(fields: dict, bucket: str) -> bool:
    """True when a row carries NO learnable signal at all: an opaque/GUID (or empty)
    SubAccountName AND no resource group / tags / (for AWS) enrichment. The recharging
    id of such a row is not derivable from the data — it is isolated with *no
    prediction*, rather than guessed. Applies only after exact-lookup fails."""
    if any(fields.get(k) for k in _clue_keys(bucket)):
        return False
    name = fields["name"]
    return bool(_OPAQUE_NAME.match(name)) or name == ""


class RechargingMatcher:
    """Exact-lookup shortcut + per-provider char n-gram logistic-regression classifiers."""

    def __init__(self):
        self.ref_ids: list[str] = []
        self.exact_lookup: dict[tuple, str] = {}
        self.name_lookup: dict[str, str] = {}
        # One trained model per provider bucket. Each entry:
        #   {"vec", "clf", "classes", "class_family", "class_product"}
        # where class_family/class_product are aligned to that clf's classes_ so the
        # item probability vector can be marginalized up the tree.
        self.models: dict[str, dict] = {}
        # Hierarchy: item id -> Product / Family / readable item name (majority vote).
        self.item_to_family: dict[str, str] = {}
        self.item_to_product: dict[str, str] = {}
        self.item_to_name: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Field extraction / text representation
    # ------------------------------------------------------------------
    @staticmethod
    def _extract(row: pd.Series, cols: dict) -> tuple[dict, str]:
        """Normalized text fields for a row (as a dict) plus its provider bucket.
        AWS enrichment slots read from whichever source column is present/non-blank."""
        def one(colname: str) -> str:
            v = str(row.get(colname, "") or "").lower().strip()
            return "" if v == "nan" else v

        fields = {
            "name": one(cols["sub_account_name"]),
            "resource_group": one(cols["resource_group"]),
            "tag_dcs": one(cols["tag_dcs"]),
            "tag_app": one(cols["tag_app"]),
        }
        for slot, sources in AWS_TAG_SOURCES.items():
            fields[slot] = next((v for v in (one(s) for s in sources) if v), "")
        return fields, provider_bucket(row.get(cols["provider"], ""))

    @staticmethod
    def _row_text(fields: dict, bucket: str) -> str:
        """Single classifier string for a row: name | shared tags | provider extras
        (blanks skipped). AWS rows include the enrichment slots; Azure the ResourceGroup."""
        keys = ("name",) + _SHARED_CLUES + (_AWS_EXTRA if bucket == PROVIDER_AWS else _AZURE_EXTRA)
        return " | ".join(fields[k] for k in keys if fields.get(k))

    def _resolve_bucket(self, bucket: str) -> str | None:
        """The bucket whose model serves this row: its own if trained, else any
        available model (fallback for a provider unseen in training)."""
        if bucket in self.models:
            return bucket
        return next(iter(self.models), None)

    # ------------------------------------------------------------------
    # Index building (exact lookups + train per-provider classifiers)
    # ------------------------------------------------------------------
    def build_index(self, df_learning: pd.DataFrame) -> None:
        df_learning = normalize_schema(df_learning)
        id_col = LEARNING_COLS["recharging_item_id"]
        # Drop blank/placeholder (XX_TOIDENTIFY) targets — never learn them as a class.
        df_clean = trainable_rows(df_learning, id_col)

        extracted = [self._extract(row, LEARNING_COLS) for _, row in df_clean.iterrows()]
        self.ref_ids = df_clean[id_col].astype(str).tolist()

        # Exact (name, rg) → majority ID  (global; rg is blank for AWS, so key is name-only there)
        key_ids: dict[tuple, list[str]] = {}
        for (fields, _), rid in zip(extracted, self.ref_ids):
            key_ids.setdefault((fields["name"], fields["resource_group"]), []).append(rid)
        self.exact_lookup = {k: Counter(v).most_common(1)[0][0] for k, v in key_ids.items()}

        # Name-only → ID, but only when the name is unambiguous
        name_ids: dict[str, set[str]] = {}
        for (fields, _), rid in zip(extracted, self.ref_ids):
            name_ids.setdefault(fields["name"], set()).add(rid)
        self.name_lookup = {n: ids.pop() for n, ids in name_ids.items() if len(ids) == 1}

        # Learn the hierarchy (global): item id -> majority Product / Family / readable name.
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

        # Train one classifier per provider bucket, each over its own feature layout.
        self.models = {}
        by_bucket: dict[str, list[int]] = defaultdict(list)
        for i, (_, bucket) in enumerate(extracted):
            by_bucket[bucket].append(i)
        for bucket, idxs in by_bucket.items():
            ids = np.array([self.ref_ids[i] for i in idxs])
            if len(set(ids)) < 2:  # LogisticRegression needs ≥2 classes; too sparse to model
                continue
            texts = [self._row_text(extracted[i][0], bucket) for i in idxs]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE)
            clf = LogisticRegression(max_iter=2000, C=CLF_C)
            clf.fit(vec.fit_transform(texts), ids)
            self.models[bucket] = {
                "vec": vec, "clf": clf, "classes": clf.classes_,
                "class_family": np.array([self.item_to_family.get(c, "") for c in clf.classes_]),
                "class_product": np.array([self.item_to_product.get(c, "") for c in clf.classes_]),
            }

        print(f"Index built: {len(extracted)} items, {len(set(self.ref_ids))} unique IDs")
        print(f"  Hierarchy: {len(set(self.item_to_family.values()))} families, "
              f"{len(set(self.item_to_product.values()))} products, {len(set(self.ref_ids))} items")
        print(f"  Exact (name+RG) keys: {len(self.exact_lookup)}")
        print(f"  Safe name-only keys:  {len(self.name_lookup)}")
        for bucket, m in self.models.items():
            print(f"  Classifier[{bucket}]: {len(by_bucket[bucket])} rows, "
                  f"{len(m['classes'])} classes, char {NGRAM_RANGE} n-grams")

    # ------------------------------------------------------------------
    # Hierarchy helpers
    # ------------------------------------------------------------------
    def _ancestry(self, item_id: str) -> tuple[str, str, str]:
        """(Product_Family, Product_Name, Recharging_Item_Name) for an item id."""
        return (self.item_to_family.get(item_id, ""),
                self.item_to_product.get(item_id, ""),
                self.item_to_name.get(item_id, item_id))

    def _level_confidence(self, row_proba: np.ndarray, item_id: str, class_family: np.ndarray,
                          class_product: np.ndarray) -> tuple[float, float]:
        """Marginalize the item probability vector (aligned to one model's classes) up the
        tree: Family_conf = Σ P(item) over items in the predicted item's family; likewise
        Product. Coarser levels aggregate more mass, so confidence rises up the tree — and
        is always consistent with the chosen item."""
        fam, prod, _ = self._ancestry(item_id)
        fam_mask = class_family == fam
        # Nest the product strictly inside the predicted family, so
        # Family_conf >= Product_conf >= Item_conf always (the placeholder product
        # "(IT own Cloud consumption)" holds XX_* catch-all items across many families).
        prod_mask = fam_mask & (class_product == prod)
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
    def predict(self, df_empty: pd.DataFrame, confidence_threshold: float = 60.0) -> pd.DataFrame:
        """Predict Recharging_Item_ID for each row in GO_MAPPING_EMPTY, routing each row
        to its provider's classifier.

        Adds columns:
        - Predicted_Recharging_Item_ID / _Name, Predicted_Product_Family / _Name
        - Confidence + Family_Confidence + Product_Confidence (0-100)
        - Top_Matches, Candidate_IDs (audit / agent shortlists)
        - Needs_Review, Match_Method, Review_Reason, Suggested_Action
        """
        if not self.models:
            raise RuntimeError("build_index() must be called before predict()")

        df_empty = normalize_schema(df_empty)
        extracted = [self._extract(row, EMPTY_COLS) for _, row in df_empty.iterrows()]

        # Batch the classifier per model: group row positions by the model that serves them.
        n = len(extracted)
        row_proba: list[np.ndarray | None] = [None] * n
        row_model: list[dict | None] = [None] * n
        groups: dict[str, list[int]] = defaultdict(list)
        for i, (_, bucket) in enumerate(extracted):
            rb = self._resolve_bucket(bucket)
            if rb is not None:
                groups[rb].append(i)
        for rb, positions in groups.items():
            model = self.models[rb]
            texts = [self._row_text(extracted[i][0], rb) for i in positions]
            proba = model["clf"].predict_proba(model["vec"].transform(texts))
            for j, i in enumerate(positions):
                row_proba[i] = proba[j]
                row_model[i] = model

        results = []
        for i, (fields, bucket) in enumerate(extracted):
            name, rg = fields["name"], fields["resource_group"]
            exact_key = (name, rg)
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

            if name in self.name_lookup:
                pred = self.name_lookup[name]
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

            # No signal at all (opaque/empty name, no rg, no tags/enrichment) -> do NOT
            # guess. Isolate with an empty prediction; the answer needs owner knowledge.
            if _no_signal(fields, bucket):
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

            model, proba = row_model[i], row_proba[i]
            classes = model["classes"]
            order = np.argsort(proba)[::-1]
            pred = classes[order[0]]
            confidence = round(float(proba[order[0]] * 100), 1)
            top_matches = " | ".join(f"{classes[j]} ({proba[j]:.2f})"
                                     for j in order[:TOP_K_CANDIDATES])
            candidate_ids = " | ".join(f"{cid} ({p:.2f})"
                                       for cid, p in self._nucleus_candidates(proba, classes))
            fam, prod, item_name = self._ancestry(pred)
            fam_conf, prod_conf = self._level_confidence(
                proba, pred, model["class_family"], model["class_product"])
            # Route by available evidence: isolate the rows the clues cannot
            # determine (no clue / name-only) even if the classifier looks confident.
            reason, force_review = _evidence_reason(fields, bucket)
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
