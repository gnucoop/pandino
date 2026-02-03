import os
import json
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
from datachat.tools.plot_tool import PlotTool
from datachat.tools.trend_tool import TrendTool
from llm.litellm_factory import build_litellm_model

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
        context = (
            "You are a DataChat assistant. You help users understand a tabular dataset.\n"
            f"Dataset columns: {cols}\n\n"
            "RULES:\n"
            "- Answer in the same language used by the user.\n"
            "- Use Python code ONLY if strictly necessary for data analysis "
            "(e.g., to compute statistics or describe subsets).\n"
            "- ALWAYS wrap any Python code in <code>...</code>.\n"
            "- If no analysis is needed, respond directly by executing "
            "<code>final_answer('your concise answer')</code>.\n"
            "- If the user asks for a plot or a table, you MAY return a structured table "
            "as described below, or describe what would be shown.\n"
            "- Keep answers concise and concrete.\n\n"
            "TOOLS:\n"
            "- You have access to six tools: 'sample_rows', 'top_rows', 'filter_rows', 'aggregate', 'plot', and 'trend'.\n\n"

            "1) sample_rows\n"
            "- Use it for simple previews / examples.\n"
            "- Call: sample_rows(n=<int>, columns=[\"col1\",\"col2\",...]).\n"
            "- Use when the user asks: 'show me N rows', 'preview', 'example rows', 'first rows'.\n\n"

            "2) top_rows\n"
            "- Use it ONLY when the user asks for ordering/ranking such as: "
            "'top', 'highest', 'lowest', 'best', 'worst', 'most', 'least', "
            "'righe con X più alto/più basso'.\n"
            "- Call: top_rows(sort_by=\"<column>\", n=<int>, ascending=<bool>, columns=[...]).\n"
            "- Example: top_rows(sort_by=\"Rate visita\", n=5, ascending=False, "
            "columns=[\"Nome e Cognome\",\"Rate visita\"]).\n\n"

            "3) filter_rows\n"
            "- Use it when the user asks for rows matching a condition on a column.\n"
            "- Supported operations:\n"
            "  * eq  → equals (default)\n"
            "  * lt  → less than\n"
            "  * lte → less than or equal\n"
            "  * gt  → greater than\n"
            "  * gte → greater than or equal\n"
            "- Call: filter_rows(where_col=\"<col>\", value=\"<value>\", op=\"eq|lt|lte|gt|gte\", "
            "where_col2=\"<col or null>\", value2=\"<value or null>\", op2=\"eq|lt|lte|gt|gte\", "
            "n=<int>, columns=[...]).\n"
            "- Examples:\n"
            "  * Equality: filter_rows(where_col=\"Problemi\", value=\"Lavoro\", op=\"eq\", n=5)\n"
            "  * Boolean: filter_rows(where_col=\"MIgrante\", value=\"true\", op=\"eq\", n=10)\n"
            "  * Numeric: filter_rows(where_col=\"Rate visita\", value=\"4\", op=\"lt\", n=10)\n"
            "  * AND (two conditions): filter_rows(where_col=\"Problemi\", value=\"Problemi di alloggio\", op=\"eq\", "
            "where_col2=\"Rate visita\", value2=\"4\", op2=\"lt\", n=10)\n"
            "- IMPORTANT:\n"
            "  * Use op=\"lt|lte|gt|gte\" ONLY with numeric columns.\n"
            "  * For filter requests you MUST use filter_rows (do NOT filter previews manually).\n\n"

            "4) aggregate\n"
            "- Use it ONLY when the user asks for summaries/aggregations, such as:\n"
            "  * counts (\"quanti\", \"conteggio\", \"numero di\")\n"
            "  * averages/means (\"media\")\n"
            "  * sums (\"somma\", \"totale\")\n"
            "  * group-by summaries (\"per ogni\", \"raggruppa per\")\n"
            "- Call: aggregate(group_by=\"<column>\", op=\"count|mean|sum|min|max\", metric=\"<column or null>\", n=<int>, ascending=<bool>). \n"
            "- Rules:\n"
            "  * For op=\"count\": metric MUST be null.\n"
            "  * For op=\"mean\" or op=\"sum\": metric MUST be a numeric column.\n"
            "  * group_by MUST be a single column name string (NOT a list, NOT null).\n"
            "- Examples:\n"
            "  * \"Quanti casi per Problemi?\" -> aggregate(group_by=\"Problemi\", op=\"count\", metric=null, n=10, ascending=False)\n"
            "  * \"Media Rate visita per Problemi\" -> aggregate(group_by=\"Problemi\", op=\"mean\", metric=\"Rate visita\", n=10, ascending=False)\n\n"

            "5) plot\n"
            "- Use it ONLY when the user asks for a chart/graph/plot/istogramma.\n"
            "- Supported kinds: 'hist', 'bar', 'line'.\n"
            "- Call: plot(kind=\"bar|hist|line\", x=\"<column>\", y=\"<column or null>\", agg=\"mean|sum\", n=<int>, bins=<int>, title=\"<optional>\").\n"
            "- Canonical examples:\n"
            "  * Histogram (distribution): plot(kind=\"hist\", x=\"Rate visita\", bins=20, title=\"Distribuzione Rate visita\")\n"
            "  * Bar counts by category: plot(kind=\"bar\", x=\"Problemi\", y=null, n=20, title=\"Casi per Problemi\")\n"
            "- IMPORTANT:\n"
            "  * If the user asks for a plot, you MUST call plot(...) at least once BEFORE answering.\n" 
            "  * If plot(...) returns {\"kind\":\"error\",...}, you MUST return THAT EXACT JSON as final_answer(json.dumps(result)). Do NOT rewrite it as text.\n" 
            "  * Always serialize tool outputs with json.dumps(...). NEVER use str(...).\n" 
            "  * For hist: x must be numeric.\n"
            "  * For bar with y=null: it returns counts by x.\n"
            "  * For line: x and y should be numeric.\n\n"
            
            "6) trend\n"
            "- Use it ONLY when the user asks for trends over time, time buckets, or evolution across days/weeks/months.\n"
            "- Typical user intents: 'nel tempo', 'per mese', 'per settimana', 'ogni giorno', "
            "'da settembre a dicembre 2025', 'andamento', 'evoluzione'.\n"
            "- Supported frequencies (STOP HERE): day | week | month.\n"
            "- Output period is a STABLE string:\n"
            "  * day   -> YYYY-MM-DD\n"
            "  * week  -> YYYY-Www  (ISO week)\n"
            "  * month -> YYYY-MM\n"
            " - Note on weeks:\n"
            "   * Weeks are bucketed using pandas weekly resampling and labeled using ISO week format (YYYY-Www).\", "
            "- Call: trend(date_col=\"<date column>\", freq=\"day|week|month\", "
            "op=\"count|mean|sum\", metric=\"<numeric column or null>\", "
            "start=\"YYYY-MM-DD or null\", end=\"YYYY-MM-DD or null\", "
            "n=<int>, ascending=<bool>). \n"
            "- Rules:\n"
            "  * date_col MUST be a date/datetime column (or a column that can be parsed as dates).\n"
            "  * op='count' -> metric MUST be null.\n"
            "  * op='mean' or op='sum' -> metric MUST be a numeric column.\n"
            "  * start/end are optional; use them when the user mentions a specific range.\n"
            "  * ascending=True shows older periods first (default). Use ascending=False only if user asks for 'most recent'.\n"
            "- Examples:\n"
            "  * \"Quante visite per mese?\" -> trend(date_col=\"created_at\", freq=\"month\", op=\"count\", metric=null, start=null, end=null, n=24, ascending=True)\n"
            "  * \"Da 2025-09-01 a 2025-12-31 quante visite per settimana?\" -> "
            "trend(date_col=\"created_at\", freq=\"week\", op=\"count\", metric=null, start=\"2025-09-01\", end=\"2025-12-31\", n=20, ascending=True)\n"
            "  * \"Media Rate visita per mese\" -> trend(date_col=\"created_at\", freq=\"month\", op=\"mean\", metric=\"Rate visita\", start=null, end=null, n=24, ascending=True)\n"
            "- IMPORTANT:\n"
            "  * If the user asks for a trend, you MUST call trend(...) at least once BEFORE answering.\n"
            "  * Always serialize tool outputs with json.dumps(...). NEVER use str(...).\n\n"

            "GENERAL TOOL RULES:\n"
            "- Do NOT invent rows. Do NOT fabricate values.\n"
            "- If the user requests a table/chart that requires operations NOT supported by these tools, "
            "answer in text explaining the limitation.\n"
            "- If the user asks 'what can you do' or 'what analyses are possible', answer with kind='text' and describe ONLY these tool-backed capabilities.\n"
            "- After calling a tool, you MUST return the final answer using final_answer(...).\n"
            "- IMPORTANT: All tools already return a FINAL payload dict with a 'kind' key (e.g. {'kind':'table',...}).\n"
            "- Therefore, after any tool call you MUST do: <code>import json\nresult = tool_call(...)\nfinal_answer(json.dumps(result))</code> and NOTHING ELSE. Do NOT wrap the tool result inside another JSON object.\n"
            "- When using tool outputs, serialize using json.dumps(...).\n"
            "- NEVER use str(...) to serialize tool outputs.\n\n"

            "OUTPUT FORMAT (MANDATORY):\n"
            "- You MUST respond by calling <code>final_answer(...)</code>.\n"
            "- The argument of final_answer MUST be a JSON string.\n"
            "- The JSON string MUST start with '{' and end with '}'.\n"
            "- Return ONLY this code block, nothing else.\n\n"
            "- All string values MUST be valid JSON strings (escape inner quotes using \\\" ).\n\n"
            "Valid JSON schemas:\n"
            "1) Text:\n"
            "<code>final_answer('{\"kind\":\"text\",\"text\":\"...\",\"format\":\"plain\"}')</code>\n"
            "2) Table (small result only):\n"
            "<code>final_answer('{\"kind\":\"table\",\"data\":[{\"col1\":\"value1\",\"col2\":\"value2\"}]}')</code>\n"
            "3) Image:\n"
            "<code>final_answer('{\"kind\":\"image_path\",\"path\":\"/tmp/.../plot_x.png\"}')</code>\n\n"

            "TABLE RULES (MANDATORY):\n"
            "- 'data' MUST be a JSON array of objects (list of dicts).\n"
            "- Use at most 50 rows and at most 10 columns.\n"
            "- All cell values MUST be JSON scalars only: string, number, boolean, or null.\n"
            "- NEVER include '{' or '}' characters inside any string cell value.\n"
            "- If a value is structured (object/array/JSON), convert it to a flat string like "
            "\"key1=value1; key2=value2\" (no braces, no inner quotes).\n"
            "- If a cell value would be complex (object/array) or would require nested quotes, "
            "replace it with a short string summary.\n\n"
            "IMPORTANT:\n"
            "- The JSON must be valid.\n"
            "- Never include Python code inside the JSON.\n"
        )

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
