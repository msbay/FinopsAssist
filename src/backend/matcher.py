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
  * AWS rows read the V5 enrichment columns instead (AwsAccountTags.owner /
    global.dcs / local.description / name). These describe the account in plain
    text and are the decisive signal for the many AWS accounts whose
    SubAccountName is an opaque GUID. Per the data-owner requirement they are fed
    ONLY to AWS rows; an Azure row never sees them. The shared axa tags are NOT
    given to the AWS model — they are redundant with the AWS enrichment and add
    nothing (benchmark: ±0.5pp) — so AWS reads its enrichment alone.
On genuinely new accounts (group split by account) this scores ~87-88% on
Recharging_Item_ID for each provider and ~94% on Product_Family — a ~4pp lift
over one global model, and +18pp on AWS over name-only. The AWS tag columns use
different names in the LEARNING and EMPTY sheets; AWS_TAG_SOURCES maps both.

On Azure, rows the available clues cannot determine (an opaque/short name with no
resource group or tags) are isolated up front via Review_Reason and always sent to
review — name-only Azure rows are ~9pp less accurate, so the guard catches
over-confident errors. AWS is exempt: its SubAccountNames are descriptive, so
name-only AWS rows are as accurate as any (measured on V5), and AWS routes on
confidence alone.
"""

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import diags, hstack
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
# Canonical AWS enrichment column names the accounts source (aws_accounts.py) WRITES and
# these models READ — the single source of truth for that contract; aws_accounts imports
# them so a rename here can't silently desync the join.
AWS_OWNER_COL = "AwsAccountTags.owner"
AWS_DCS_COL = "AwsAccountTags.global.dcs"
AWS_DESC_COL = "AwsAccountTags.local.description"
AWS_NAME_COL = "AwsAccountTags.name"
AWS_TAG_SOURCES = {
    "aws_owner": [AWS_OWNER_COL],
    "aws_dcs": [AWS_DCS_COL],
    "aws_desc": [AWS_DESC_COL, "AwsAccountTags.description"],
    "aws_name": ["AwsAccountTags.global.app", AWS_NAME_COL],
}

# Provider buckets. One classifier per bucket; anything not AWS is treated as Azure
# (Microsoft) — the shared/RG feature set has no AWS-only columns, so it never leaks them.
PROVIDER_AWS = "AWS"
PROVIDER_AZURE = "AZURE"

# Fields (besides the name, which always leads the text) that feed each bucket's
# classifier and count as routing evidence.
_SHARED_CLUES = ("tag_dcs", "tag_app")      # axa_tags_global_dcs / _app
_AZURE_EXTRA = ("resource_group",)
_AWS_EXTRA = ("aws_owner", "aws_dcs", "aws_desc", "aws_name")

# Per-bucket feature/clue fields. Azure reads the axa tags + ResourceGroup. AWS reads
# ONLY its own enrichment — the shared axa tags are dropped for AWS because they are
# redundant with the AWS enrichment (axa_tags_global_dcs ≈ AwsAccountTags.global.dcs)
# and add nothing (benchmark: ±0.5pp, within noise). Azure has no enrichment, so it
# still depends on the axa tags.
_BUCKET_FIELDS = {
    PROVIDER_AWS: _AWS_EXTRA,
    PROVIDER_AZURE: _SHARED_CLUES + _AZURE_EXTRA,
}


def provider_bucket(raw) -> str:
    """Map a raw ProviderName to a model bucket ('AWS' or 'AZURE')."""
    return PROVIDER_AWS if "aws" in str(raw or "").lower() else PROVIDER_AZURE


def _clue_keys(bucket: str) -> tuple[str, ...]:
    """Field keys that count as evidence (the name is judged separately)."""
    return _BUCKET_FIELDS.get(bucket, _SHARED_CLUES + _AZURE_EXTRA)

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

# Azure ResourceGroup is the most specific identifier, but folded into one char-vector
# with the SubAccountName + tags its features get IDF-diluted. So the Azure model gives
# the RG its OWN vector, hstacked onto the context (name+tags) vector. This *separation*
# is the win — +2pp on Azure Recharging_Item_ID (87.5 -> ~89.6%) — and it targets the
# RG-discriminated confusions (OpenHosting 459/506, MPI 429/436, Shine 475/530). The
# weight is a scale on the RG block: a sweep showed a flat 0.75-1.0 plateau (up-weighting
# to ≥1.5 actually hurts, and repeating the RG text in one vector hurts more), so parity
# (1.0) is the natural choice. AWS has no RG, so it keeps the single-vector featuriser.
AZURE_RG_WEIGHT = 1.0

# Not every RG is discriminative: the auto-created 'networkwatcherrg' (in every
# subscription) maps to 13+ different items, and shared infra RGs are similar. Feeding
# those at full strength biases the RG vector toward whatever the RG was most often in
# training, overriding the informative SubAccountName. So instead of a hardcoded blocklist
# we SELF-TUNE: each RG's vector is scaled by its reliability = 1 / (#distinct items it
# mapped to in training). A clean RG (1 item) keeps weight 1.0; 'networkwatcherrg' (13
# items) fades to ~0.08, so the row leans on the name. Unseen RGs default to full trust; a
# small regex also zeroes per-instance auto-created RGs (e.g. AKS 'MC_*') that are each
# seen once so the data can't flag them. Recovers ~+8pp on the affected rows. Exact
# (name, RG) lookup is untouched — a *seen* generic-RG row still resolves exactly.
GENERIC_RG_RE = re.compile(r"^mc_|cloud-shell-storage|defaultresourcegroup|"
                           r"^databricks-rg|^az-ago-mgmt|log-analytics-default", re.I)

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

# Recharging_Item_ID values that are NOT a real, learnable mapping (the row is still "to
# identify"): blanks, null-like strings, and the placeholders above. Single source of truth
# for the mapped-vs-empty policy — used by both data_source._split (routing) and
# trainable_rows (training), so the two ends can't drift apart.
_NON_ID_VALUES = PLACEHOLDER_IDS | {"", "NAN", "NONE"}


def has_real_id(series: "pd.Series") -> "pd.Series":
    """Boolean mask: rows whose Recharging_Item_ID is a real, non-placeholder mapping."""
    ids = series.astype(str).str.strip().str.upper()
    return series.notna() & ~ids.isin(_NON_ID_VALUES)

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
    return df[has_real_id(df[id_col])].copy()

# An opaque name = a GUID/hash or a very short token. Carries no learnable signal.
_OPAQUE_NAME = re.compile(r"^[0-9a-f]{16,}$")


def _evidence_reason(fields: dict, bucket: str) -> tuple[str, bool]:
    """Classify a row by the evidence available, for review routing.

    Returns (reason, force_review). The evidence guard (force a name-only / no-clue row
    to review regardless of confidence) is AZURE-ONLY: on Azure, name-only rows are ~9pp
    less accurate than rows with clues, so forcing them catches over-confident errors. On
    AWS the SubAccountName is descriptive (e.g. go-sm-euc1-riskaissurance-prod) and
    name-only rows are actually *more* accurate than rows with clues — so AWS never forces
    review; confidence alone routes it (measured on V5, group split by account).
    """
    name = fields["name"]
    guard = bucket == PROVIDER_AZURE  # only Azure force-routes on weak evidence
    has_clues = any(fields.get(k) for k in _clue_keys(bucket))
    if not has_clues:
        if _OPAQUE_NAME.match(name) or len(name) <= 6:
            # Bucket 1: opaque/short name, nothing else — not inferable.
            return "no clue — opaque name, no resource group or tags; needs human", guard
        # Bucket 2: a plain name and nothing else — weak on Azure, reliable on AWS.
        return "weak — name only, no resource group or tags; low reliability", guard
    # Bucket 3: has a resource group and/or tags — there is something to reason
    # over, so a borderline prediction is worth an LLM second opinion.
    return "check with LLM — has clues but name pattern is ambiguous", False


def _no_signal(fields: dict, bucket: str) -> bool:
    """True when an AZURE row carries NO learnable signal at all: an opaque/GUID (or empty)
    SubAccountName AND no resource group / tags. Such a row's id is not derivable from the
    data, so it is isolated with *no prediction* rather than guessed. AWS is exempt — its
    names are descriptive enough that even name-only AWS rows predict reliably, so AWS rows
    are always scored (see _evidence_reason). Applies only after exact-lookup fails."""
    if bucket != PROVIDER_AZURE:
        return False
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
        """Single classifier string for a row: name | provider fields (blanks skipped).
        AWS uses its enrichment slots; Azure the axa tags + ResourceGroup."""
        keys = ("name",) + _BUCKET_FIELDS.get(bucket, _SHARED_CLUES + _AZURE_EXTRA)
        return " | ".join(fields[k] for k in keys if fields.get(k))

    @staticmethod
    def _context_text(fields: dict) -> str:
        """Azure context string WITHOUT the ResourceGroup (name + axa tags). The RG is
        vectorised separately and up-weighted — see _fit_featurizer."""
        keys = ("name",) + _SHARED_CLUES
        return " | ".join(fields[k] for k in keys if fields.get(k))

    # ------------------------------------------------------------------
    # Featurization — one char-vector for AWS; context + weighted-RG for Azure
    # ------------------------------------------------------------------
    @staticmethod
    def _rg_reliability(rg: str, rg_label_counts: dict) -> float:
        """How much to trust a RG as a signal: 1/(#distinct items it mapped to in training).
        Blank/regex-generic -> 0; unseen -> 1.0 (trust it); 'networkwatcherrg' -> ~0.08."""
        if not rg or GENERIC_RG_RE.search(rg):
            return 0.0
        k = rg_label_counts.get(rg)
        return 1.0 if k is None else 1.0 / k

    def _rg_block(self, vec_rg, fields_list, rg_label_counts):
        """RG char-vector with each row scaled by AZURE_RG_WEIGHT × its RG reliability, so
        non-discriminative RGs fade and the row leans on the SubAccountName."""
        rg = [f["resource_group"] for f in fields_list]
        w = np.array([AZURE_RG_WEIGHT * self._rg_reliability(r, rg_label_counts) for r in rg])
        return diags(w) @ vec_rg.transform(rg)

    def _fit_featurizer(self, bucket: str, fields_list: list[dict],
                        ids: np.ndarray) -> tuple[dict, "object"]:
        """Fit the vectoriser(s) for a bucket and return (model_parts, train_matrix). Azure
        gets two char vectors (context + reliability-weighted RG) hstacked; AWS one."""
        if bucket == PROVIDER_AZURE:
            # How many distinct items each RG maps to in training -> its reliability weight.
            rg_label_counts: dict[str, set] = defaultdict(set)
            for f, rid in zip(fields_list, ids):
                if f["resource_group"]:
                    rg_label_counts[f["resource_group"]].add(rid)
            rg_label_counts = {r: len(s) for r, s in rg_label_counts.items()}
            vec_ctx = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE)
            vec_rg = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE)
            vec_rg.fit([f["resource_group"] for f in fields_list])
            xc = vec_ctx.fit_transform([self._context_text(f) for f in fields_list])
            xr = self._rg_block(vec_rg, fields_list, rg_label_counts)
            model = {"vec_ctx": vec_ctx, "vec_rg": vec_rg, "rg_weight": AZURE_RG_WEIGHT,
                     "rg_label_counts": rg_label_counts}
            return model, hstack([xc, xr]).tocsr()
        texts = [self._row_text(f, bucket) for f in fields_list]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE)
        return {"vec": vec}, vec.fit_transform(texts)

    def _featurize(self, model: dict, bucket: str, fields_list: list[dict]):
        """Transform rows to the feature matrix using a model's fitted vectoriser(s)."""
        if "vec_rg" in model:  # Azure context + reliability-weighted RG model
            xc = model["vec_ctx"].transform([self._context_text(f) for f in fields_list])
            xr = self._rg_block(model["vec_rg"], fields_list, model["rg_label_counts"])
            return hstack([xc, xr]).tocsr()
        return model["vec"].transform([self._row_text(f, bucket) for f in fields_list])

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
            fields_list = [extracted[i][0] for i in idxs]
            model, x = self._fit_featurizer(bucket, fields_list, ids)
            clf = LogisticRegression(max_iter=2000, C=CLF_C)
            clf.fit(x, ids)
            model.update({
                "clf": clf, "classes": clf.classes_,
                "class_family": np.array([self.item_to_family.get(c, "") for c in clf.classes_]),
                "class_product": np.array([self.item_to_product.get(c, "") for c in clf.classes_]),
            })
            self.models[bucket] = model

        print(f"Index built: {len(extracted)} items, {len(set(self.ref_ids))} unique IDs")
        print(f"  Hierarchy: {len(set(self.item_to_family.values()))} families, "
              f"{len(set(self.item_to_product.values()))} products, {len(set(self.ref_ids))} items")
        print(f"  Exact (name+RG) keys: {len(self.exact_lookup)}")
        print(f"  Safe name-only keys:  {len(self.name_lookup)}")
        for bucket, m in self.models.items():
            feat = (f"context + reliability-weighted RG char {NGRAM_RANGE} n-grams"
                    if "vec_rg" in m else f"char {NGRAM_RANGE} n-grams")
            print(f"  Classifier[{bucket}]: {len(by_bucket[bucket])} rows, "
                  f"{len(m['classes'])} classes, {feat}")

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
            x = self._featurize(model, rb, [extracted[i][0] for i in positions])
            proba = model["clf"].predict_proba(x)
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
