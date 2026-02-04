import os
import json
import textwrap
from dataclasses import dataclass, field
from typing import Any
import logging

import pandas as pd
from smolagents import CodeAgent, LiteLLMModel
from datachat.bootstrap import build_bootstrap_question
from datachat.engine_interface import DataChatEngine, EngineBootstrapResult
from datachat.tools.sample_rows_tool import SampleRowsTool
from datachat.tools.top_rows_tool import TopRowsTool
from datachat.tools.filter_rows_tool import FilterRowsTool
from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.describe_tool import DescribeTool
from datachat.tools.missing_values_tool import MissingValuesTool
from datachat.tools.plot_tool import PlotTool
from datachat.tools.trend_tool import TrendTool
from llm.litellm_factory import build_litellm_model
from prompt_utils import load_prompt, render_prompt

plots_dir = os.getenv("DATACHAT_PLOTS_DIR", "/tmp/datachat_plots")

@dataclass
class SmolagentsEngine(DataChatEngine):
    """
    Smolagents engine (initial wiring).

    NOTE: enabled via DATACHAT_ENGINE=smolagents. Tools and real behavior will be added step-by-step.
    """
    api_key: str
    user_name: str
    llm: Any  # kept for interface compatibility; not used here yet
    data: pd.DataFrame

    _agent: CodeAgent | None = field(default=None, init=False, repr=False)
    _model: LiteLLMModel | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        
        provider = os.getenv("DATACHAT_PROVIDER", "Deepinfra").strip()
        configured_model = os.getenv("DATACHAT_MODEL", "").strip()

        if not configured_model:
            self._model = None
            self._agent = None
            return

        try:
            model = build_litellm_model(
                provider=provider,
                configured_model=configured_model,
                temperature=0.0,
            )
        except Exception:
            
            self._model = None
            self._agent = None
            return

        self._model = model
        
        self._agent = CodeAgent(
            tools=[
                DescribeTool(self.data),
                MissingValuesTool(self.data),
                SampleRowsTool(self.data), 
                TopRowsTool(self.data),
                FilterRowsTool(self.data),
                AggregateTool(self.data),
                PlotTool(self.data, output_dir=plots_dir),
                TrendTool(self.data)
            ],
            model=model,
            max_steps=2,
            additional_authorized_imports=["json"],
        )


    def bootstrap(self, lang: str) -> EngineBootstrapResult:
        """
        1:1 con pandasai bootstrap:
        - usa build_bootstrap_question(data, lang)
        - chiama il model e restituisce HTML (o comunque testo) come suggested_questions_html
        """
        question = build_bootstrap_question(self.data, lang)

        if self._model is None:
            logging.error("[datachat][smolagents][bootstrap] model is not configured")
            return EngineBootstrapResult(suggested_questions_html=None)

        try:
            msg = [{"role": "user", "content": question}]
            out = self._model(msg)

            if isinstance(out, str):
                content = out
            else:
                content = getattr(out, "content", None)
                if content is None:
                    content = str(out)

            html = str(content).strip()
            if not html:
                logging.error("[datachat][smolagents][bootstrap] empty content")
                return EngineBootstrapResult(suggested_questions_html=None)

            return EngineBootstrapResult(suggested_questions_html=html)

        except Exception as e:
            logging.exception(f"[datachat][smolagents][bootstrap] model call failed: {e}")
            return EngineBootstrapResult(suggested_questions_html=None)

    def chat(self, message: str) -> Any:
        if self._agent is None:
            return {
                "kind": "error",
                "message": (
                    "SmolagentsEngine non è configurato correttamente: "
                    "verifica DATACHAT_PROVIDER/DATACHAT_MODEL e la relativa API key."
                ),
                "code": "MISSING_CONFIG",
            }

        # Contesto minimo (non serializziamo tutto il dataframe: troppo grande)
        cols = list(self.data.columns)
        default_context =  textwrap.dedent("""\
            You are a DataChat assistant. You help users understand a tabular dataset.
            Dataset columns: {columns}

            RULES:
            - Answer in the same language used by the user.
            - Use Python code ONLY if strictly necessary for data analysis (e.g., to compute statistics or describe subsets).
            - ALWAYS wrap any Python code in <code>...</code>.
            - If no analysis is needed, respond directly by executing <code>final_answer('your concise answer')</code>.
            - If the user asks for a plot or a table, you MAY return a structured table as described below, or describe what would be shown.
            - Keep answers concise and concrete.

            TOOLS:
            - You have access to eight tools: 'describe', 'missing_values', 'sample_rows', 'top_rows', 'filter_rows', 'aggregate', 'plot', and 'trend'.

            1) describe
            - Use it for a general overview or descriptive statistics of the dataset.
            - Call: describe(columns=["col1","col2",...], n=<int>).
            - Use when the user asks: 'riassumimi il dataset', 'statistiche descrittive', 'overview', 'summary'.

            2) missing_values
            - Use it to count missing/null values per column.
            - Call: missing_values(columns=["col1","col2",...], n=<int>).
            - Use when the user asks: 'valori mancanti', 'missing values', 'quanti null'.

            3) sample_rows
            - Use it for simple previews / examples.
            - Call: sample_rows(n=<int>, columns=["col1","col2",...]).
            - Use when the user asks: 'show me N rows', 'preview', 'example rows', 'first rows'.

            4) top_rows
            - Use it ONLY when the user asks for ordering/ranking such as: 'top', 'highest', 'lowest', 'best', 'worst', 'most', 'least', 'righe con X più alto/più basso'.
            - Call: top_rows(sort_by="<column>", n=<int>, ascending=<bool>, columns=[...]).
            - Example: top_rows(sort_by="Rate visita", n=5, ascending=False, columns=["Nome e Cognome","Rate visita"]).

            5) filter_rows
            - Use it when the user asks for rows matching a condition on a column.
            - Supported operations:
            * eq  → equals (default)
            * lt  → less than
            * lte → less than or equal
            * gt  → greater than
            * gte → greater than or equal
            - Call: filter_rows(where_col="<col>", value="<value>", op="eq|lt|lte|gt|gte", where_col2="<col or null>", value2="<value or null>", op2="eq|lt|lte|gt|gte", n=<int>, columns=[...]).
            - Examples:
            * Equality: filter_rows(where_col="Problemi", value="Lavoro", op="eq", n=5)
            * Boolean: filter_rows(where_col="MIgrante", value="true", op="eq", n=10)
            * Numeric: filter_rows(where_col="Rate visita", value="4", op="lt", n=10)
            * AND (two conditions): filter_rows(where_col="Problemi", value="Problemi di alloggio", op="eq", where_col2="Rate visita", value2="4", op2="lt", n=10)
            - IMPORTANT:
            * Use op="lt|lte|gt|gte" ONLY with numeric columns.
            * For filter requests you MUST use filter_rows (do NOT filter previews manually).

            6) aggregate
            - Use it ONLY when the user asks for summaries/aggregations, such as:
            * counts ("quanti", "conteggio", "numero di")
            * averages/means ("media")
            * sums ("somma", "totale")
            * group-by summaries ("per ogni", "raggruppa per")
            - Call: aggregate(group_by="<column>", op="count|mean|sum|min|max", metric="<column or null>", n=<int>, ascending=<bool>). 
            - Rules:
            * For op="count": metric MUST be null.
            * For op="mean" or op="sum": metric MUST be a numeric column.
            * group_by MUST be a single column name string (NOT a list, NOT null).
            - Examples:
            * "Quanti casi per Problemi?" -> aggregate(group_by="Problemi", op="count", metric=null, n=10, ascending=False)
            * "Media Rate visita per Problemi" -> aggregate(group_by="Problemi", op="mean", metric="Rate visita", n=10, ascending=False)

            7) plot
            - Use it ONLY when the user asks for a chart/graph/plot/istogramma.
            - Supported kinds: 'hist', 'bar', 'line', 'pie'.
            - Call: plot(kind="bar|hist|line|pie", x="<column>", y="<column or null>", agg="mean|sum", n=<int>, bins=<int>, title="<optional>").
            - PIE CHART RULES (IMPORTANT):
            * 'pie' is a DERIVED visualization of composition.
            * It does NOT introduce new analysis, only a different rendering of an aggregation.
            * It supports ONLY:
                - count by category (y=null)
                - sum(y) by category (agg='sum')
            * 'mean' is NOT supported for pie charts.
            * If y is provided for pie:
                - agg MUST be 'sum'
                - y MUST be an existing numeric column
            * If the requested column for y does NOT exist or has no numeric values,
                you MUST call plot(...) anyway and return the resulting error as-is.
                Do NOT invent or guess alternative columns (e.g. "costo", "peso", "valore").
            - Interpretation rules for vague requests:
            * If the user asks for "distribuzione", "composizione", "a colpo d’occhio",
                and does NOT mention a numeric measure,
                interpret this as count by category (y=null).
            * If the user explicitly mentions a numeric concept (e.g. "peso", "totale", "somma"),
                use pie ONLY if a suitable numeric column is explicitly named or already present.
                Otherwise, proceed with the tool call and return the error.
            - Canonical examples:
            * Histogram (distribution): plot(kind="hist", x="Rate visita", bins=20, title="Distribuzione Rate visita")
            * Bar counts by category: plot(kind="bar", x="Problemi", y=null, n=20, title="Casi per Problemi")
            - Canonical examples (pie):
            * Distribution of problems (counts): plot(kind="pie", x="Problemi", y=null, n=10, title="Distribuzione dei problemi")
            * Distribution by weight (sum): plot(kind="pie", x="Problemi", y="Costo", agg="sum", n=10, title="Peso dei problemi")
            - IMPORTANT:
            * If the user asks for a plot, you MUST call plot(...) at least once BEFORE answering.
            * If plot(...) returns {"kind":"error",...}, you MUST return THAT EXACT JSON as final_answer(json.dumps(result)). Do NOT rewrite it as text.
            * Always serialize tool outputs with json.dumps(...). NEVER use str(...).
            * For hist: x must be numeric.
            * For bar with y=null: it returns counts by x.
            * For line: x and y should be numeric.

            8) trend
            - Use it ONLY when the user asks for trends over time, time buckets, or evolution across days/weeks/months.
            - Typical user intents: 'nel tempo', 'per mese', 'per settimana', 'ogni giorno', 'da settembre a dicembre 2025', 'andamento', 'evoluzione'.
            - Supported frequencies (STOP HERE): day | week | month.
            - Output period is a STABLE string:
            * day   -> YYYY-MM-DD
            * week  -> YYYY-MM-DD→YYYY-MM-DD (Monday to Sunday)
            * month -> YYYY-MM
            - Note on weeks:
            * Weeks are bucketed using pandas weekly resampling.
            * Period labels show the full week range for human readability.
            - Call: trend(date_col="<date column>", freq="day|week|month", op="count|mean|sum", metric="<numeric column or null>", start="YYYY-MM-DD or null", end="YYYY-MM-DD or null", n=<int>, ascending=<bool>). 
            - Rules:
            * date_col MUST be a date/datetime column (or a column that can be parsed as dates).
            * op='count' -> metric MUST be null.
            * op='mean' or op='sum' -> metric MUST be a numeric column.
            * start/end are optional; use them when the user mentions a specific range.
            * ascending=True shows older periods first (default). Use ascending=False only if user asks for 'most recent'.
            - Examples:
            * "Quante visite per mese?" -> trend(date_col="created_at", freq="month", op="count", metric=null, start=null, end=null, n=24, ascending=True)
            * "Da 2025-09-01 a 2025-12-31 quante visite per settimana?" -> trend(date_col="created_at", freq="week", op="count", metric=null, start="2025-09-01", end="2025-12-31", n=20, ascending=True)
            * "Media Rate visita per mese" -> trend(date_col="created_at", freq="month", op="mean", metric="Rate visita", start=null, end=null, n=24, ascending=True)
            - IMPORTANT:
            * If the user asks for a trend, you MUST call trend(...) at least once BEFORE answering.
            * Always serialize tool outputs with json.dumps(...). NEVER use str(...).

            GENERAL TOOL RULES:
            - Do NOT invent rows. Do NOT fabricate values.
            - If the user requests a table/chart that requires operations NOT supported by these tools, answer in text explaining the limitation.
            - If the user asks 'what can you do' or 'what analyses are possible', answer with kind='text' and describe ONLY these tool-backed capabilities.
            - After calling a tool, you MUST return the final answer using final_answer(...).
            - IMPORTANT: All tools already return a FINAL payload dict with a 'kind' key (e.g. {'kind':'table',...}).
            - Therefore, after any tool call you MUST do: <code>import json
            result = tool_call(...)
            final_answer(json.dumps(result))</code> and NOTHING ELSE. Do NOT wrap the tool result inside another JSON object.
            - When using tool outputs, serialize using json.dumps(...).
            - NEVER use str(...) to serialize tool outputs.

            OUTPUT FORMAT (MANDATORY):
            - You MUST respond by calling <code>final_answer(...)</code>.
            - The argument of final_answer MUST be a JSON string.
            - The JSON string MUST start with '{' and end with '}'.
            - Return ONLY this code block, nothing else.

            - All string values MUST be valid JSON strings (escape inner quotes using \\" ).

            Valid JSON schemas:
            1) Text:
            <code>final_answer('{"kind":"text","text":"...","format":"plain"}')</code>
            2) Table (small result only):
            <code>final_answer('{"kind":"table","data":[{"col1":"value1","col2":"value2"}]}')</code>
            3) Image:
            <code>final_answer('{"kind":"image_path","path":"/tmp/.../plot_x.png"}')</code>

            TABLE RULES (MANDATORY):
            - 'data' MUST be a JSON array of objects (list of dicts).
            - Use at most 50 rows and at most 10 columns.
            - All cell values MUST be JSON scalars only: string, number, boolean, or null.
            - NEVER include '{' or '}' characters inside any string cell value.
            - If a value is structured (object/array/JSON), convert it to a flat string like "key1=value1; key2=value2" (no braces, no inner quotes).
            - If a cell value would be complex (object/array) or would require nested quotes, replace it with a short string summary.

            IMPORTANT:
            - The JSON must be valid.
            - Never include Python code inside the JSON.
        """)
        context_template = load_prompt(
            "data_chat_system",
            default_text=default_context,
        )
        context = render_prompt(context_template, columns=cols)

        try:
            out = self._agent.run(context + "\n\nUser question: " + str(message), reset=False)
        except Exception as e:
            return {
                "kind": "error",
                "message": f"SmolagentsEngine failed to run: {e}",
                "code": "RUN_FAILED",
            }

        raw = str(out).strip()


        def _unwrap_nested_table(obj: dict[str, Any]) -> dict[str, Any]:
            """
            Fix common LLM mistake:
            {"kind":"table","data": {"kind":"table","data":[...]}}
            -> {"kind":"table","data":[...]}
            """
            try:
                kind = str(obj.get("kind") or "").strip().lower()
                if kind != "table":
                    return obj

                data = obj.get("data")

                # Case A: data is already correct
                if isinstance(data, list):
                    return obj

                # Case B: data is a nested payload dict
                if isinstance(data, dict):
                    nested_kind = str(data.get("kind") or "").strip().lower()
                    nested_data = data.get("data")

                    # {"data": {"kind":"table","data":[...]}}
                    if nested_kind == "table" and isinstance(nested_data, list):
                        obj["data"] = nested_data
                        return obj

                    # {"data": {"data":[...]}}  (no nested kind)
                    if isinstance(nested_data, list):
                        obj["data"] = nested_data
                        return obj

                return obj
            except Exception:
                return obj


        def _parse_kind_payload(s: str) -> dict[str, Any] | None:
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "kind" in obj:
                    return obj
            except Exception:
                return None
            return None

        # 1) Tentativo diretto
        parsed = _parse_kind_payload(raw)
        if parsed is not None:
            return _unwrap_nested_table(parsed)

        # 2) Se è una stringa JSON "annidata" (es. "\"{...}\""), un json.loads in più la sblocca
        try:
            unwrapped = json.loads(raw)
            if isinstance(unwrapped, str):
                parsed = _parse_kind_payload(unwrapped)
                if parsed is not None:
                    return _unwrap_nested_table(parsed)
        except Exception:
            pass

        # 3) Estrazione conservativa tra prima { e ultima } (per output tipo final_answer('...'))
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            parsed = _parse_kind_payload(candidate)
            if parsed is not None:
                return _unwrap_nested_table(parsed)

        # 4) Ultimo fallback: testo
        return {"kind": "text", "text": raw, "format": "plain"}


    def close(self) -> None:
        return
