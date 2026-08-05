import json
import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import LiteLLMModel, Tool

from datachat.output_normalizer import replace_nan

_MAX_UNIQUE_VALUES = 500
_MAX_CLUSTERS = 50
_MAX_TFIDF_FEATURES = 1000
_TOP_TERMS_PER_CLUSTER = 5

_CLASSIFY_SYSTEM_PROMPT = (
    "You are a text classification assistant. "
    "Assign each text to one of the provided categories. "
    "Return ONLY a valid JSON object mapping each input index to its category and confidence. "
    "No explanations, no extra text, no markdown."
)


def _to_json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


class ClassifyTool(Tool):
    """
    Classify text values in a column into topic categories.

    Two modes:
    - match:  assign to user-provided category labels using the LLM
    - cluster: automatically discover groups using TF-IDF + KMeans (scikit-learn)
    """

    name = "classify"
    description = (
        "Classify a text column into topic categories. Two modes:\n"
        "- method='match': assign each unique text value to one of the user-provided "
        "categories (requires 'categories' parameter). Uses the LLM.\n"
        "- method='cluster': automatically discover groups using TF-IDF + KMeans "
        "(requires 'n_clusters' parameter, default 5). Fast, no LLM needed.\n"
        "Do NOT use for sentiment analysis -- use 'sentiment_analysis' for that."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "column": {
            "type": "string",
            "description": "Name of the text column to classify.",
        },
        "method": {
            "type": "string",
            "description": "Classification mode: 'match' for predefined categories, 'cluster' for automatic grouping.",
            "enum": ["match", "cluster"],
            "nullable": True,
        },
        "categories": {
            "type": "array",
            "description": "List of category labels (required for method='match'). Example: ['Amministrativo', 'Sanitario', 'Sociale'].",
            "items": {"type": "string"},
            "nullable": True,
        },
        "n_clusters": {
            "type": "integer",
            "description": "Number of clusters to discover (required for method='cluster', default 5, max 50).",
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, the classification will run on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame, model: Optional[LiteLLMModel] = None) -> None:
        super().__init__()
        self._df = df
        self._model = model

    def forward(
        self,
        column: str,
        method: str = "match",
        categories: Optional[list[str]] = None,
        n_clusters: Optional[int] = 5,
        data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            if data is not None:
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")
                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}
                if len(data) == 0:
                    return {"kind": "table", "data": []}
                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {"kind": "error", "message": "Invalid data: could not build a table from records.", "code": "INVALID_DATA"}
            else:
                df = self._df

            col = (column or "").strip()
            if not col:
                return {"kind": "error", "message": "Missing column name.", "code": "MISSING_COLUMN"}
            if col not in df.columns:
                return {"kind": "error", "message": f"Column not found: {col}", "code": "INVALID_COLUMN"}

            method_clean = (method or "match").strip().lower()
            if method_clean not in {"match", "cluster"}:
                return {"kind": "error", "message": f"Invalid method '{method_clean}'. Use 'match' or 'cluster'.", "code": "INVALID_METHOD"}

            if method_clean == "match":
                return self._classify_match(df, col, categories)
            else:
                n = int(n_clusters) if n_clusters is not None else 5
                return self._classify_cluster(df, col, n)

        except Exception as e:
            logging.exception("[datachat][classify_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}

    # ------------------------------------------------------------------
    # Match mode: LLM-based classification into predefined categories
    # ------------------------------------------------------------------

    def _classify_match(
        self,
        df: pd.DataFrame,
        col: str,
        categories: Optional[list[str]],
    ) -> dict[str, Any]:
        cats = categories or []
        if not cats:
            return {"kind": "error", "message": "Missing 'categories' list for method='match'.", "code": "MISSING_CATEGORIES"}

        if self._model is None:
            return {"kind": "error", "message": "LLM model not available for method='match'.", "code": "MODEL_UNAVAILABLE"}

        s = df[col].dropna().astype(str)
        s = s[s.str.strip() != ""]
        unique_vals = s.unique().tolist()
        if len(unique_vals) > _MAX_UNIQUE_VALUES:
            unique_vals = unique_vals[:_MAX_UNIQUE_VALUES]

        if not unique_vals:
            return {"kind": "error", "message": "No non-empty text values found in column.", "code": "EMPTY_COLUMN"}

        cats_str = ", ".join(f"'{c}'" for c in cats)
        items_str = "\n".join(f'  "{i}": "{v}"' for i, v in enumerate(unique_vals))
        user_prompt = (
            f"Assign each text to one of the following categories: {cats_str}.\n\n"
            f"Texts:\n{{\n{items_str}\n}}\n\n"
            f"Respond with a JSON object mapping each index to "
            f'{{"category": "<category>", "confidence": <float 0-1>}}. '
            f"Example: {{\"0\": {{\"category\": \"{cats[0]}\", \"confidence\": 0.92}}, ...}}\n"
            f"Return ONLY valid JSON, no other text."
        )

        messages = [
            {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self._model(messages)
            raw_text = response.content.strip()
        except Exception as e:
            logging.exception("[datachat][classify_tool] LLM call failed")
            return {"kind": "error", "message": f"LLM call failed: {e}", "code": "LLM_FAILED"}

        parsed: dict[str, Any] = {}
        try:
            cleaned = raw_text
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            parsed = json.loads(cleaned)
        except Exception:
            pass

        if not isinstance(parsed, dict):
            return {"kind": "error", "message": "Failed to parse LLM response as JSON.", "code": "PARSE_FAILED"}

        cat_lower = {c.lower(): c for c in cats}
        lookup: dict[str, dict[str, Any]] = {}
        for key, val in parsed.items():
            if isinstance(val, dict) and "category" in val:
                idx_str = str(key).strip()
                raw_cat = str(val.get("category", cats[0])).strip()
                matched_cat = cat_lower.get(raw_cat.lower(), raw_cat)
                if matched_cat not in cats:
                    matched_cat = cats[0]
                try:
                    confidence = float(val.get("confidence", 0.8))
                except (ValueError, TypeError):
                    confidence = 0.8
                lookup[idx_str] = {"category": matched_cat, "confidence": round(confidence, 4)}

        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            raw = row.get(col)
            if pd.isna(raw) or not str(raw).strip():
                continue
            text_val = str(raw).strip()
            matched = None
            if text_val in unique_vals:
                val_idx = str(unique_vals.index(text_val))
                matched = lookup.get(val_idx)
            if matched is None:
                for key, val in lookup.items():
                    try:
                        lu_idx = int(key)
                        if lu_idx < len(unique_vals) and unique_vals[lu_idx] == text_val:
                            matched = val
                            break
                    except (ValueError, IndexError):
                        continue
            if matched is None:
                matched = {"category": cats[0], "confidence": 0.5}

            records.append({
                str(col): text_val,
                "category": matched["category"],
                "confidence": matched["confidence"],
            })

        records = replace_nan(records)
        logging.info("[datachat][classify_tool] mode=match col=%s rows=%d categories=%s", col, len(records), cats)
        return {"kind": "table", "data": records}

    # ------------------------------------------------------------------
    # Cluster mode: TF-IDF + KMeans (scikit-learn)
    # ------------------------------------------------------------------

    def _classify_cluster(
        self,
        df: pd.DataFrame,
        col: str,
        n_clusters: int,
    ) -> dict[str, Any]:
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return {
                "kind": "error",
                "message": "scikit-learn is required for method='cluster'. Install it: pip install scikit-learn",
                "code": "MISSING_SKLEARN",
            }

        n = max(2, min(int(n_clusters), _MAX_CLUSTERS))

        s = df[col].dropna().astype(str)
        s = s[s.str.strip() != ""]
        if s.empty:
            return {"kind": "error", "message": "No non-empty text values found in column.", "code": "EMPTY_COLUMN"}

        texts = s.tolist()
        try:
            vectorizer = TfidfVectorizer(
                max_features=_MAX_TFIDF_FEATURES,
                stop_words="english",
                lowercase=True,
            )
            X = vectorizer.fit_transform(texts)
        except Exception as e:
            logging.exception("[datachat][classify_tool] TF-IDF failed")
            return {"kind": "error", "message": f"TF-IDF vectorization failed: {e}", "code": "TFIDF_FAILED"}

        if X.shape[1] == 0:
            return {"kind": "error", "message": "No features extracted from text column.", "code": "NO_FEATURES"}

        # If fewer samples than requested clusters, reduce n
        n_actual = min(n, X.shape[0])
        if n_actual < 2:
            n_actual = 2

        try:
            km = KMeans(n_clusters=n_actual, random_state=42, n_init="auto")
            cluster_labels = km.fit_predict(X)
        except Exception as e:
            logging.exception("[datachat][classify_tool] KMeans failed")
            return {"kind": "error", "message": f"KMeans clustering failed: {e}", "code": "KMEANS_FAILED"}

        # Build cluster -> top terms
        feature_names = vectorizer.get_feature_names_out()
        cluster_top_terms: dict[int, list[str]] = {}
        for i in range(n_actual):
            mask = cluster_labels == i
            if mask.sum() == 0:
                cluster_top_terms[i] = []
                continue
            center = km.cluster_centers_[i]
            top_indices = center.argsort()[::-1][:_TOP_TERMS_PER_CLUSTER]
            terms = [str(feature_names[idx]) for idx in top_indices if idx < len(feature_names)]
            cluster_top_terms[i] = terms

        records: list[dict[str, Any]] = []
        for idx, row in df.iterrows():
            text_val = str(row.get(col, ""))
            row_cluster: Optional[int] = None
            # Find corresponding cluster index for this row's text value
            text_idx_in_s = s.index.get_loc(idx) if idx in s.index else None
            if text_idx_in_s is not None and isinstance(text_idx_in_s, int) and text_idx_in_s < len(cluster_labels):
                row_cluster = int(cluster_labels[text_idx_in_s])
            if row_cluster is None:
                continue
            top_terms = cluster_top_terms.get(row_cluster, [])
            records.append({
                str(col): text_val,
                "cluster": row_cluster,
                "label": ", ".join(top_terms) if top_terms else f"Cluster {row_cluster}",
                "top_terms": top_terms,
            })

        records = replace_nan(records)
        logging.info(
            "[datachat][classify_tool] mode=cluster col=%s rows=%d n_clusters=%d features=%d",
            col, len(records), n_actual, X.shape[1],
        )
        return {"kind": "table", "data": records}
