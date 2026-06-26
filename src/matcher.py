from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# Column name mappings for each sheet
LEARNING_COLS = {
    "sub_account_name": "Custom.focus_costs[SubAccountName]",
    "sub_account_id": "Custom.focus_costs[SubAccountId]",
    "resource_group": "Custom.focus_costs[axa_Azure_ResourceGroupName]",
    "tag_dcs": "Custom.focus_costs[axa_tags_global_dcs]",
    "tag_app": "Custom.focus_costs[axa_tags_global_app]",
    "recharging_item_id": "Custom.gld_referential_mdm_axagorechargingitem[Recharging_Item_ID]",
    "provider": "Custom.focus_costs[ProviderName]",
}

EMPTY_COLS = {
    "sub_account_name": "focus_costs[SubAccountName]",
    "sub_account_id": "focus_costs[SubAccountId]",
    "resource_group": "focus_costs[axa_Azure_ResourceGroupName]",
    "tag_dcs": "focus_costs[axa_tags_global_dcs]",
    "tag_app": "focus_costs[axa_tags_global_app]",
    "provider": "focus_costs[ProviderName]",
}


class RechargingMatcher:
    """Field-weighted fuzzy matcher with exact-match shortcut for Recharging_Item_ID prediction."""

    def __init__(self):
        self.ref_fields: list[tuple] = []
        self.ref_ids: list[str] = []
        self.exact_lookup: dict[tuple, str] = {}
        self.name_lookup: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Field extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _get_fields(row: pd.Series, cols: dict) -> tuple[str, str, str, str]:
        """Extract and normalize individual fields."""
        name = str(row.get(cols["sub_account_name"], "") or "").lower().strip()
        rg = str(row.get(cols["resource_group"], "") or "").lower().strip()
        dcs = str(row.get(cols["tag_dcs"], "") or "").lower().strip()
        app = str(row.get(cols["tag_app"], "") or "").lower().strip()
        if rg == "nan": rg = ""
        if dcs == "nan": dcs = ""
        if app == "nan": app = ""
        return name, rg, dcs, app

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def build_index(self, df_learning: pd.DataFrame) -> None:
        """Build the reference index from GO_MAPPING_LEARNING."""
        id_col = LEARNING_COLS["recharging_item_id"]
        df_clean = df_learning.dropna(subset=[id_col]).copy()

        self.ref_fields = [self._get_fields(row, LEARNING_COLS) for _, row in df_clean.iterrows()]
        self.ref_ids = df_clean[id_col].astype(str).tolist()

        # Exact lookup: (name, rg) → majority ID
        key_ids: dict[tuple, list[str]] = {}
        for fields, rid in zip(self.ref_fields, self.ref_ids):
            key = (fields[0], fields[1])
            if key not in key_ids:
                key_ids[key] = []
            key_ids[key].append(rid)
        self.exact_lookup = {k: Counter(v).most_common(1)[0][0] for k, v in key_ids.items()}

        # Name-only lookup: only when name always maps to the same ID
        name_ids: dict[str, set[str]] = {}
        for fields, rid in zip(self.ref_fields, self.ref_ids):
            name = fields[0]
            if name not in name_ids:
                name_ids[name] = set()
            name_ids[name].add(rid)
        self.name_lookup = {name: ids.pop() for name, ids in name_ids.items() if len(ids) == 1}

        print(f"Index built: {len(self.ref_fields)} items, {len(set(self.ref_ids))} unique IDs")
        print(f"  Exact (name+RG) keys: {len(self.exact_lookup)}")
        print(f"  Safe name-only keys:  {len(self.name_lookup)}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_score(q_fields: tuple, r_fields: tuple) -> float:
        """Weighted field-level fuzzy score."""
        q_name, q_rg, q_dcs, q_app = q_fields
        r_name, r_rg, r_dcs, r_app = r_fields

        name_score = fuzz.token_set_ratio(q_name, r_name) / 100.0 if q_name and r_name else 0.0
        rg_score = fuzz.token_set_ratio(q_rg, r_rg) / 100.0 if q_rg and r_rg else 0.0
        dcs_score = (1.0 if q_dcs == r_dcs else 0.0) if q_dcs and r_dcs else 0.0
        app_score = (1.0 if q_app == r_app else 0.0) if q_app and r_app else 0.0

        return 0.45 * name_score + 0.35 * rg_score + 0.10 * dcs_score + 0.10 * app_score

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_confidence(scores: np.ndarray, ids: list[str], top_k: int = 5) -> float:
        """0-100 confidence from similarity, margin, and agreement."""
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        top_ids = [ids[i] for i in top_indices]

        best_score = top_scores[0]
        best_id = top_ids[0]

        other_scores = [s for s, rid in zip(top_scores, top_ids) if rid != best_id]
        margin = best_score - (other_scores[0] if other_scores else 0.0)
        agreement = sum(1 for rid in top_ids if rid == best_id) / top_k

        confidence = (0.5 * best_score + 0.3 * margin + 0.2 * agreement) * 100
        return round(min(max(confidence, 0), 100), 1)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self, df_empty: pd.DataFrame, top_k: int = 5, confidence_threshold: float = 50.0
    ) -> pd.DataFrame:
        """Predict Recharging_Item_ID for each row in GO_MAPPING_EMPTY.

        Returns a copy of df_empty with added columns:
        - Predicted_Recharging_Item_ID
        - Confidence
        - Top_Matches (top-3 for audit)
        - Needs_Review (confidence < threshold)
        - Match_Method (exact_name_rg, exact_name, fuzzy)
        """
        results = []
        for _, row in df_empty.iterrows():
            q_fields = self._get_fields(row, EMPTY_COLS)

            # Try exact match first
            exact_key = (q_fields[0], q_fields[1])
            if exact_key in self.exact_lookup:
                results.append({
                    "Predicted_Recharging_Item_ID": self.exact_lookup[exact_key],
                    "Confidence": 100.0,
                    "Top_Matches": f"{self.exact_lookup[exact_key]} (exact name+RG)",
                    "Needs_Review": False,
                    "Match_Method": "exact_name_rg",
                })
                continue

            if q_fields[0] in self.name_lookup:
                results.append({
                    "Predicted_Recharging_Item_ID": self.name_lookup[q_fields[0]],
                    "Confidence": 95.0,
                    "Top_Matches": f"{self.name_lookup[q_fields[0]]} (exact name)",
                    "Needs_Review": False,
                    "Match_Method": "exact_name",
                })
                continue

            # Fuzzy matching with top-k majority vote
            scores = np.array([self._compute_score(q_fields, rf) for rf in self.ref_fields])
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_ids = [self.ref_ids[i] for i in top_indices]

            best_id = Counter(top_ids).most_common(1)[0][0]
            confidence = self._compute_confidence(scores, self.ref_ids, top_k)

            top_matches = [
                f"{self.ref_ids[idx]} ({scores[idx]:.2f})" for idx in top_indices[:3]
            ]

            results.append({
                "Predicted_Recharging_Item_ID": best_id,
                "Confidence": confidence,
                "Top_Matches": " | ".join(top_matches),
                "Needs_Review": confidence < confidence_threshold,
                "Match_Method": "fuzzy",
            })

        result_df = df_empty.copy()
        for col in ["Predicted_Recharging_Item_ID", "Confidence", "Top_Matches", "Needs_Review", "Match_Method"]:
            result_df[col] = [r[col] for r in results]

        return result_df.sort_values("Confidence", ascending=True)
