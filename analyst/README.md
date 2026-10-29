# The Analyst — a guided tour

This guide walks you through the `analyst/` module step by step: what it does, how a request
flows through it, and what every file is for. Read it top to bottom once; afterwards the
*File map* (§4) and the *call graph* (§5) are the quick references.

---

## 0. In one minute

- The **interviewer** talks with the user and produces an **analysis brief** (JSON).
- When the user approves it, the brief is saved and handed to the **analyst**.
- The analyst works **in the background**, with four kinds of agents:
  1. a **planner** reads the brief — *whatever its structure* — probes the data, and writes
     a **plan**: the tasks, the charts, the report shape, plus a few **exploration tasks**
     of its own;
  2. one fresh, small **task agent** per task computes the results and records
     **insights** (notable facts it found on the way);
  3. a **chart agent** draws every planned chart from those results;
  4. a **report agent** writes the text and arranges the report.
- Code — not an LLM — decides the order, checks every agent's answer and renders the
  **HTML report** (Chart.js charts). A client follows the progress by polling.

**Mini glossary**

| Word | Meaning |
| --- | --- |
| **brief** | The JSON approved in the interview (`analysis_briefs` table). Written by a model: its keys and nesting are **not guaranteed**. |
| **plan** | What the analyst will do, in **one fixed shape** (`plan.py`). Everything after planning reads the plan, never the brief's keys. |
| **task** | One step of the plan: an `id`, a `mission`, the `outputs` it must save, the tasks it `depends_on`. *Exploratory* tasks are added by the planner. |
| **run** | One execution of one brief. It has an id like `20261014_101500_b42_a1b2c3` and its own folder. |
| **output** | A computed result a task saves under a key (`baseline_annual_revenue`, `margin_matrix`, …). |
| **insight** | A notable fact a task agent found beyond its outputs, with the computed evidence. |
| **dataset** | A pandas DataFrame loaded from the database, kept *outside* the model's context. |
| **event** | A small progress message (`task_started`, `step`, `insight`, …) a client reads to show what is happening. |

---

## 1. The big picture

```
            Client (Angular)
          │                    │
          │ approve            │ start / status / report / end
          ▼                    ▼
┌───────────────────────┐ ┌──────────────────────────┐
│ routes/interviewer.py │ │ routes/analyst.py        │
│   _approve()          │ │   4 endpoints            │
└──────────┬────────────┘ └────────────┬─────────────┘
           │ brief_id                  │
           ▼                           ▼
┌───────────────────────────────────────────────────────┐
│ analyst/service.py   start_analysis()                 │
│   checks → AnalysisRun → agent_manager → thread       │
└───────────────────────────┬───────────────────────────┘
                            │ background thread
                            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ analyst/orchestrator.py   Orchestrator.execute()                          │
│                                                                           │
│  1. plan          2. for each task    3. charts          4. report        │
│  ┌────────────┐   ┌──────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Planner    │──►│ TaskAgent    │──► │ ChartAgent  │──► │ ReportAgent │  │
│  │ (any brief │   │ (×N, fresh)  │    │             │    │             │  │
│  │  → plan)   │   └──────┬───────┘    └──────┬──────┘    └──────┬──────┘  │
│  └─────┬──────┘          │ tools             │ tools            │ tools   │
│        │ stats           ▼                   ▼                  ▼         │
│        ▼          ┌──────────────────────────────────────────────────┐    │
│    Database ◄─────│ RunState (state.py): plan · outputs · insights · │    │
│   (read-only)     │ datasets · charts · tables · metrics · html      │    │
│                   └──────────────────────────────────────────────────┘    │
│                                                                           │
│  AnalysisRun.progress ◄── events (plan_ready, task_started, step, …)      │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼
                analysis_reports (one row per run_id, saved as the run goes)
                  metadata · plan · state · html
```

Four ideas carry the whole design:

1. **Any brief, one plan.** The brief's shape changes from brief to brief; the plan's never
   does. The planner bridges the two, and if it fails, plain code (`plan_from_brief`) does —
   so no brief can stop a run.
2. **Code orchestrates, agents compute.** The order of the tasks, their dependencies, the
   required outputs, the chart coverage and the retries are plain Python.
3. **Small contexts.** Every task gets a brand-new agent that sees only what that task needs.
   Data lives in DataFrames, not in the prompt.
4. **Shared memory is a Python object.** Agents never talk to each other: they read and
   write the `RunState` through their tools.

---

## 2. The life of a run

```
 status:  queued ─► planning ─► running ─► charting ─► writing_report ─► completed
                       │           │          │              │
                       └───────────┴──────────┴──────────────┴──► failed / cancelled

 events:  run_started
          plan_started ─ step ─ … ─ plan_ready            {tasks, exploratory, charts, planner, planner_error}
          task_started ─ step ─ insight ─ … ─ task_completed        (task 1)
          task_started ─ step ─ … ─ task_retry ─ … ─ task_completed (task 2)
          task_failed                                                (a skipped/failed task)
          …
          charts_started ─ step ─ … ─ charts_saved                  (or charts_failed)
          report_started ─ step ─ … ─ report_saved
          run_completed          (or run_failed / run_cancelled)
```

`status` is what you show as the "big" state; `events` are the detail lines. `insight`
events carry the text of each finding as soon as it is recorded — nice for a live feed.

---

## 3. Step-by-step tutorial

We follow one real request: the user has just approved the brief
**"2027 Revenue and First-Margin Forecast"** (`analyst/reports/example-brief.json`:
6 sub-tasks, 6 charts, a `metrics_config`).

### Step 1 — The hand-off from the interviewer

📄 `routes/interviewer.py` → read `_approve()`

1. `save_approved_brief(...)` stores the brief in `analysis_briefs` and returns its `brief_id`.
2. `engine.approve()` closes the draft.
3. `start_analysis_payload(api_key, user_email, brief_id, config)` starts the analyst.
4. The response contains `brief_id` **and** `analysis`: `{run_id, status}` if the run started,
   `{error, code}` if it did not (e.g. `NOT_ENOUGH_TOKENS`). **The approval stands either
   way**: the brief can be run later with `POST /start-analysis`.
5. Then the interviewer session is deleted, as before.

### Step 2 — Starting a run

📄 `analyst/service.py` → read `start_analysis()`

It checks, in this order, and raises `AnalysisStartError(code, http_status)` on the first
problem:

| Check | Error code |
| --- | --- |
| The SQL datasource is enabled | `SQL_UNAVAILABLE` (400) |
| The brief exists **and belongs to the user** (`get_brief(id, user_email)`) | `BRIEF_NOT_FOUND` (404) |
| The user has `ANALYST_TOKEN_COST` tokens | `NOT_ENOUGH_TOKENS` (500) |
| No other run is going for this API key | `ANALYSIS_BUSY` (409) |

Then it:

1. builds an `AnalysisRun` with a new `run_id` (`new_run_id()`: `YYYYmmdd_HHMMSS_b<brief>_<hex>`);
2. registers it in `infrastructure/agent_manager.py` (`createAnalyst`, kind `"analyst"`);
   a *finished* previous run is replaced;
3. spends the tokens (`edit_tokens`);
4. calls `_spawn()`, which starts a daemon **thread** running `run_analysis()`
   (a greenlet under gunicorn/gevent). The HTTP request returns immediately.

> `start_analysis_payload()` is the same thing shaped for an HTTP response; both
> `routes/analyst.py` and `routes/interviewer.py` use it.

### Step 3 — The run object (what the client sees)

📄 `analyst/run.py`

- **`AnalysisRun`** — `run_id`, `brief_id`, owner, `status`, `current_task`,
  `tasks_done / tasks_total`, `error`, `report_ready`, and a **cancel flag**
  (`cancel()`, `close()`, `cancelled`).
- **`ProgressLog`** — an append-only list of numbered events (`seq`), protected by a lock
  because the orchestrator writes from its thread while routes read.
  `since(n)` returns the events after `n` plus the cursor for next time.
- **`status_payload(since)`** — exactly the JSON returned by `GET /analysis-status`.

Only the orchestrator writes here; the routes only read.

### Step 4 — The orchestrator (the conductor)

📄 `analyst/orchestrator.py` → read `Orchestrator.execute()` first, then each `_method`

```
execute()
 ├─ _load_brief()      get_brief()  — any shape is accepted
 ├─ _plan(brief)       PlannerAgentEngine.run()
 │                     └─ failed? ─► plan_from_brief()   (code, never fails)
 │                     ─► RunState.from_plan()  ─► _save(RUNNING)
 ├─ for each task (plan order):
 │     _run_task(task) ───────────────────────────────────────────┐
 │     _save(RUNNING)                                             │
 ├─ nothing completed? ─► fail                                     │
 ├─ _draw_charts()     ChartAgentEngine.run()  ─► _save(CHARTING)  │
 │                     └─ failed? ─► charts_failed, go on          │
 ├─ _write_report()    ReportAgentEngine.run()                     │
 │                     └─ failed? ─► _write_fallback_report()      │
 ├─ _save(COMPLETED)   the analysis_reports row, with the html     │
 └─ _mark_consumed()   analysis_briefs.status = 'consumed'         │
                                                                   │
_run_task(task)  ◄─────────────────────────────────────────────────┘
 ├─ a dependency failed?  ─► task failed ("Skipped…"), no agent runs
 └─ attempt 1 .. 1+ANALYST_TASK_RETRIES (exploratory tasks: one attempt only):
       check cancel
       engine = TaskAgentEngine(state, task, on_step=…, on_insight=…)
       result = engine.run()
       ├─ task_done        ─► completed, keep summary, stop
       ├─ MISSING_CONFIG   ─► abort the whole run
       ├─ AGENT_ERROR      ─► "impossible": failed, no retry
       └─ anything else    ─► record the error, keep the attempt's variables
                              (TaskRecord.carried), retry from them
```

Errors: anything that should stop the run raises `AnalysisAborted(message)`; `_finish()`
sets `failed` (or `cancelled`), saves the run and emits the last event. An
unexpected exception becomes "internal error" — the run never crashes the server.

`_save(status)` stores the whole run through `services/analysis_report_service.save_report`:
one `analysis_reports` row per `run_id`, inserted then updated in place, so the row shows
the run's progress. The orchestrator also counts tokens per agent run (`_count_tokens`:
`plan`, each task, `charts`, `report`) for its metadata.

### Step 5 — The plan: any brief, one shape

📄 `analyst/plan.py` — read the module docstring, then `validate_plan` and `plan_from_brief`

```json
{"title": "...", "language": "Italian", "audience": "...",
 "relations": ["Trasporti", "Ordini", "report_mensile"],
 "tasks":  [{"id": "baseline_2026", "name": "...", "mission": "...",
             "depends_on": ["data_prep_quality"], "outputs": ["euro_per_trip", "..."],
             "exploratory": false}],
 "charts": [{"id": "c1", "description": "Bars: monthly trips Jan–Aug 2026, Aug partial"}],
 "report": {"sections": ["..."], "tone": "...", "length": "..."},
 "metrics_config": null}
```

| Function | What it does |
| --- | --- |
| `validate_plan(plan, identifiers, brief_ids, max_extra)` | The planner's guard. Snake_case unique ids, dependencies on earlier tasks only, **no task of the brief depending on an exploratory one** (a failed task skips its dependents, and an optional task must not take a required one down), outputs present and unique, ≤ `ANALYST_MAX_EXTRA_TASKS` exploratory tasks, real relations, at least one chart, `metrics_config` keys = output keys, and **every task id the brief defined is kept**. Errors are worded for the model. |
| `plan_from_brief(brief, identifiers)` | The fallback, by code. Finds a task list under `sub_tasks / tasks / phases / steps / …` (or any list of task-like dicts, even one level down), slugifies ids, fills in missing order and dependencies, reads outputs as a list or `{required, optional}`, takes charts and sections wherever they are. Nothing usable? One generic task plus a summary. **It never fails.** |
| `brief_context_text(brief)` | How the brief itself reaches the task agents: a compact text of every key except tasks/charts/report/metrics, whatever those keys are called, capped. |
| `normalize_plan(plan)` | Fills harmless defaults (empty lists, `exploratory: false`) before validation. |

### Step 6 — The planner

📄 `analyst/planner_agent.py` → `PlannerAgentEngine`

A smolagents `CodeAgent` built on `SmolagentsEngineBase` (`datachat/smolagents_base.py`),
like every engine here. It gets:

- the **whole schema** (it decides which relations matter) and four statistics tools
  (`sql_stats`, `sql_group_stats`, `sql_correlation`, `sql_outliers`) plus `sql_engine`,
  to probe the data briefly (≤ 4 calls);
- the **brief as compact JSON**, as it is — no key is assumed.

It answers `{"kind":"plan","plan":{...}}`. The guard runs `validate_plan()` and replays every
error to the model until the plan is right. What it does with the brief:

- keeps the brief's tasks (same ids), making vague missions self-contained;
- adds up to `ANALYST_MAX_EXTRA_TASKS` **exploratory tasks** before the final synthesis —
  anomalies, concentrations, correlations, trend breaks, data quality — grounded in what
  its probing showed;
- lists every chart the brief asks for, **or designs 3–6** when it asks for none;
- maps any metric/KPI request to `metrics_config`;
- never makes a task of the brief depend on an exploratory one (`validate_plan` refuses it):
  a failed task skips its dependents, and an optional task must not take a required one down.

Its schema ends with the large-tables note (`sql_prompt.large_relations_note`), so missions
aggregate or sample the big relations. Its code runs in `GuardedPythonExecutor` like the
task and chart agents' (see Step 10). Should it fail, the run's metadata → `planner` keeps
its error message and last step errors, and the code plan (`plan_from_brief`) is used.

Model: `ANALYST_REPORT_MODEL` (the "lead analyst" model); steps: `ANALYST_PLAN_MAX_STEPS`.

### Step 7 — The shared memory

📄 `analyst/state.py`

```
RunState
 ├─ brief (raw, for context), plan, run_id, brief_id
 ├─ tasks:    {task_id: TaskRecord}      in plan order
 │              TaskRecord: mission, depends_on, required, exploratory,
 │                          status, outputs{}, summary, error, attempts
 ├─ datasets: {name: DataFrame}          shared by the whole run
 ├─ insights: [{task_id, text, evidence}]
 ├─ charts:   {name: {title, spec, task_id}}   spec = Chart.js config
 ├─ tables:   {name: {title, columns, rows, task_id}}
 ├─ metrics, delta                        KPI values and previous-run comparison
 └─ html, report_saved                    the rendered report, final once saved
```

Useful members: `dependency_outputs(task_id)` (what a task receives as `inputs`),
`merged_outputs()`, `missing_outputs()` on a task, the plan-backed properties `title`,
`language`, `audience`, `charts_requested`, `metrics_config`, and `snapshot()` (the
JSON-compatible part, stored as the run's `state`).

📄 `analyst/values.py` — small helpers used everywhere:
`to_native()` (numpy/pandas → plain JSON), `preview()` (JSON cut to N chars),
`describe()` ("DataFrame 8 rows x 2 cols: mese, viaggi").

### Step 8 — A task agent

📄 `analyst/task_agent.py` → `TaskAgentEngine`

| Hook | What it does here |
| --- | --- |
| `_init_config` | model = `ANALYST_PROVIDER/MODEL`, steps = `ANALYST_TASK_MAX_STEPS` |
| `_build_instructions` | system prompt + **schema of the plan's relations only** (`sql_prompt.py`) |
| `_extra_tools` | the 10 task tools, bound to *this* task |
| `_sql_tools` | `sql_engine`, capped at `ANALYST_SQL_MAX_ROWS` |
| `_build_agent` | `CodeAgent` with the **guarded executor**, the step callback and the final-answer check |

**What the agent sees and what it doesn't:**

```
      IN its context                               NOT in its context
 ┌────────────────────────────────────┐   ┌──────────────────────────────────┐
 │ system prompt (role, rules)        │   │ other tasks' missions            │
 │ schema of the plan's relations     │   │ the charts and report sections   │
 │ the brief as context text          │   │ previous tasks' history          │
 │   (brief_context_text)             │   │ rows of data                     │
 │ ITS task: mission, output keys,    │   │                                  │
 │   "exploratory" when it is one     │   │                                  │
 │ inputs/datasets: names + shapes    │   │                                  │
 └────────────────────────────────────┘   └──────────────────────────────────┘
      IN its Python namespace (usable by code, invisible to the model)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ datasets  = {name: DataFrame}       inputs = {dep_task: {key: value}}    │
 └──────────────────────────────────────────────────────────────────────────┘
```

`render_task()` builds the text; `run()` injects `datasets` and `inputs` into
`agent.state`, runs the agent, and returns either `{"kind":"task_done","summary"}` or
`{"kind":"error","code"}` (`RUN_FAILED`, `INVALID_OUTPUT`, `AGENT_ERROR`, `MISSING_CONFIG`).

**The guard:** `_check_final_answer()` refuses `task_done` while a required output is
unsaved, and tells the model which keys are missing. On a retry, the task text says what
the previous attempt saved and why it failed, so work is not redone.

**Not losing work at the step cap:** an agent's work lives in its variables until it saves
them, and a retry starts with a new executor. So three steps before `ANALYST_TASK_MAX_STEPS`
the agent is told, in the observation it reads next, to save every output it has computed
and answer (`_on_step`, only while outputs are missing). If the attempt still fails, its
variables (DataFrames, dicts, lists, numbers, strings: never tools or modules, at most 30)
are handed to the next attempt: placed in its namespace and listed in its task text
(`TaskAgentEngine.variables()`, `TaskRecord.carried`, not persisted), so the retry saves
them instead of recomputing. The agent still saves and answers itself; the guard is unchanged.

**Looking further:** the prompt asks the agent to use the statistics tools on the data it
touches and to record each verified notable fact with `record_insight(text, evidence)`.
An exploratory task exists only for this.

📄 `analyst/sql_prompt.py` — `build_scoped_sql_addendum()` renders only the plan's relations
in full and lists the others by name in one line.

**Large tables:** `large_relations_note()` reads the row estimates PostgreSQL keeps in
`pg_class.reltuples` (one catalog query, no table scanned) and names the relations above
`ANALYST_DATASET_MAX_ROWS`, e.g. `"Trasporti" ~15.2M`. Under them, fixed rules: aggregate in
SQL for totals and rankings, load a `TABLESAMPLE SYSTEM (p) REPEATABLE (42)` sample when a
method needs rows, filter before full-table statistics. The note closes the planner's, task
and chart agents' schema; it is built in code, so a prompt stored in the database cannot drop
it, and it is simply left out when the estimate cannot be read.
`load_dataset`'s truncation warning says the same: a truncated load is not the data.

### Step 9 — The tools

📁 `analyst/tools/` — one class per file, same format as `interviewer/tools/`:
class attributes `name`, `description`, `inputs`, `output_type`; a `forward()` that
**returns** `{"kind": ...}` (errors included) and never raises.

**Task agent tools**

| Tool | What it does |
| --- | --- |
| `load_dataset` | SELECT → `datasets[name]`; returns only shape, dtypes, 3 rows |
| `sql_stats` | count/mean/std/min/max/percentiles in the DB |
| `sql_group_stats` | aggregates per group in the DB, capped |
| `sql_correlation` | correlation matrix in the DB |
| `sql_outliers` | z-score / IQR outliers in the DB + sample |
| `detect_anomalies` | isolation forest / z-score / IQR on a dataset |
| `run_regression` | linear / ridge / lasso / random forest on a dataset |
| `run_forecast` | linear trend forecast + MAPE on a dataset |
| `save_sub_task_output` | save one output of *this* task (JSON, size-capped; a value that fits is saved as it is; a larger table goes to `datasets["<task>.<key>"]`, its first rows to the output) |
| `record_insight` | record a notable fact + its evidence (3 per task, `ANALYST_MAX_INSIGHTS` per run) |
| `sql_engine` | look at a few rows (`AnalystSqlEngineTool`, capped) |

**Planner tools:** `sql_stats`, `sql_group_stats`, `sql_correlation`, `sql_outliers`, `sql_engine`.

**Chart agent tools:** `save_chart`, `save_table`, `load_dataset`, `sql_engine`.

**Report agent tools**

| Tool | What it does |
| --- | --- |
| `read_sub_task_state` | status, summary, **previews** of the outputs, and the insights |
| `get_sub_task_status` | which tasks completed / failed |
| `save_run_metrics` | fills the KPI values from the outputs, by key |
| `get_delta_section` | compares with the previous completed run of the same brief (its stored metadata) |
| `get_all_saved_charts` | names and titles of charts and tables |
| `get_html_template` | renders the report from markdown sections (kept in `RunState`) |
| `save_html_report` | marks the rendered report as the run's report (the orchestrator stores it) |

**Shared helpers**

- `sql_common.py` — `SqlToolBase`: resolves relation/column names against the real schema,
  quotes them, wraps a query as a subquery, and passes the **final** SQL through
  `sql_guard.validate_select` before it runs. Also `SqlToolError`, `split_names`, `round_number`.
- `dataset_common.py` — `get_dataset()`, `numeric_columns()`, `tool_error()` for the
  DataFrame tools.

### Step 10 — The sandbox

📄 `analyst/sandbox.py`

Every analyst agent (and the interviewer) writes Python code. smolagents
already blocks dangerous imports, but a DataFrame can still write files
(`df.to_csv("/…")`), and code can reach a tool's private attributes
(`sql_engine._datasource`) to query past the SQL guard. So `GuardedPythonExecutor` runs
`check_code()` on every snippet **before** executing it and refuses:

- methods starting with `to_` / `read_` / `load` / `save` / `dump` (except an allowlist:
  `to_dict`, `tolist`, `to_numpy`, `json.dumps`, …), `tofile`, `eval`, `pipe`, `plot`, …;
- `to_string` / `to_markdown` with a file argument;
- private attributes (`_mgr`, …) and `getattr`, `open`, `eval`, `exec`, ….

Importable modules: only `json`, `math`, `statistics`, `datetime`. pandas, numpy and
scikit-learn are reached only through the given DataFrames and the tools.

**Escaped code steps:** the model sometimes writes a step as an escaped string (literal
`\n` for line breaks, `\"` for quotes, even `\]`), which Python refuses before running
anything. `unescape_code()` decodes such a step, only when it does not parse and its decoded
form does, so valid code is never touched. It also mends a `\n` split by a real line break
(backslash, newline, `n`): after a comment, or when the backslash stands on a line of its
own, that code parses with a stray `nprint(...)`, so those cases are always mended;
anywhere else it would be a valid continuation, so only code that does not parse is.
`GuardedPythonExecutor` applies it before the guard, so the guard checks the decoded code,
the code that actually runs. Every analyst agent and the interviewer run under it, which
matters most for the agents holding SQL tools: smolagents' own executor lets code reach a tool's
private attributes (`sql_engine._datasource`), and so query past the SQL guard or read
the database configuration.

**A code step every time:** every analyst agent's task text ends with `CODE_STEP_RULE`
(`prompts.py`, appended in code): every reply is a short Thought plus `<code>…</code>`, never an
announcement of what the agent is about to do. Models such as mistral-large otherwise end a
reply on "I'll start by…" with no code, a step smolagents rejects. It only restates the
CodeAgent contract, so models that follow it are unaffected.

**Outputs from variables:** the task text (built in code) also says that variables persist
between steps and that outputs are built from them, never from numbers copied out of printed
results: retyped literals are a transcription risk, and the code most often mangled.

### Step 11 — The chart agent

📄 `analyst/chart_agent.py` → `ChartAgentEngine`

It runs after the tasks, so every result exists. Its namespace holds `outputs`
(`outputs["<task>"]["<key>"]`, every completed task's outputs at full precision) and
`datasets`; its task text lists the planned charts and *describes* the outputs (keys, shapes,
short previews). It builds each Chart.js configuration **in code** from those variables —
no number goes through the model — and saves it with `save_chart` (or `save_table` for
heatmaps and matrices). It can `load_dataset` when a chart needs data no output holds.

It answers:

```json
{"kind":"charts_done","coverage":[
  {"chart":"c1","saved":"monthly_trips","reason":""},
  {"chart":"c5","saved":null,"reason":"No margin matrix in the outputs."}]}
```

**The guard** refuses the answer until every planned chart is listed, every `saved` name
really exists, and every skipped chart has a reason. Model: `ANALYST_MODEL`; steps:
`ANALYST_CHART_MAX_STEPS`. If it fails, the run goes on: the report has no charts.

### Step 12 — The report agent

📄 `analyst/report_agent.py` → `ReportAgentEngine`

It **never queries the database and never writes HTML**. Its task (`render_brief_report`)
carries, from the plan: title, audience, language, tone, length, sections, the requested
charts, the number of insights; one line per task with its status and summary; and a short
excerpt of the brief for the business context. It follows 6 steps:

```
read_sub_task_state ─► save_run_metrics ─► get_delta_section ─► get_all_saved_charts
        ─► get_html_template(sections=[{title, content(markdown), style, charts, tables}])
        ─► save_html_report ─► final_answer {"kind":"report_saved"}
```

Insights get their own **"additional findings"** section (titled in the report language),
after what the brief asked for and before the caveats. The guard refuses `report_saved`
until `save_html_report` really saved the rendered report.

### Step 13 — Rendering the report

📄 `analyst/report_template.py` + 📄 `analyst/templates/report.html`

`render_report(state, sections=…)` builds one self-contained HTML page:

- **KPI cards** from the plan's `metrics_config`, with values filled by
  `metrics.py::fill_metrics()` and the change vs the previous run
  (`analysis_report_service.previous_report_metadata()` + `compute_delta()`). A category saved as a dataset reference (a
  table above the output cap) counts its table's rows, not the reference's keys;
- **sections**: markdown rendered with HTML **disabled** (nothing an agent writes becomes
  raw HTML), plus the charts and tables placed by name;
- an **appendix** with any chart or table no section placed;
- **Chart.js** from a pinned CDN, colors assigned automatically, numbers formatted for the
  report language (`Intl.NumberFormat`, e.g. `8.331.251 €` in Italian).

If the report agent fails, `orchestrator._write_fallback_report()` renders a plain report
(one section per task with its summary, an "Additional findings" section, the charts in the
appendix) — no model involved.

📄 `analyst/prompts.py` — the in-code defaults of the 9 prompts, each overridable from the
`prompts` table: `analyst_plan_system`, `analyst_plan_turn`, `analyst_task_system`,
`analyst_task_sql_addendum` (shared by planner, task and chart agents), `analyst_task_turn`,
`analyst_chart_system`, `analyst_chart_turn`, `analyst_report_system`, `analyst_report_turn`.

### Step 14 — Following the progress

📄 `analyst/orchestrator.py` → `step_callback()`, 📄 `routes/analyst.py`

After every agent step, smolagents calls `step_callback`. It:

- emits a `step` event naming the **tools the step's code called** (never its data or SQL);
- if the run was cancelled, calls `agent.interrupt()`.

The client polls:

```
GET /analysis-status?since=0      → {status, current_task, tasks_done, events[…], next_since: 7}
GET /analysis-status?since=7      → only the new events
… until status is completed / failed / cancelled
GET /analysis-report?run_id=…       → the whole run as JSON (html, metadata, plan, state, …)
GET /analysis-report-html?run_id=…  → the HTML
GET /analysis-reports?brief_id=…    → the user's runs (no HTML), newest first
```

The report endpoints check the API key (`assert_valid_api_key`) and only ever return the
caller's own runs. The HTML comes with `X-Content-Type-Options: nosniff` and a
`Content-Security-Policy` allowing only its inline scripts and Chart.js from jsDelivr; the
JSON's `html` has no such header around it. Either way, a frontend should show it in a
sandboxed `<iframe sandbox="allow-scripts" [srcdoc]>`.

An Angular service can expose it as an Observable:

```ts
status$ = (since = 0) => timer(0, 3000).pipe(
  concatMap(() => this.http.get<AnalysisStatus>(`/analysis-status?since=${since}`, { headers })),
  tap(s => since = s.next_since),
  takeWhile(s => !['completed', 'failed', 'cancelled'].includes(s.status), true),
);
```

`POST /end-analysis` cancels the run (it stops at the next agent step) and forgets it.

### Step 15 — What is stored

📄 `services/analysis_report_service.py` → table `analysis_reports` (one row per `run_id`,
created by `database_pg.init`). Nothing is written to disk.

| Column | Content | Used by |
| --- | --- | --- |
| `run_id`, `brief_id`, `user_email`, `title`, `status` | who, what, where the run is (`planning` … `completed` / `failed` / `cancelled`) | `GET /analysis-reports`; owner check |
| `metadata` (JSONB) | timestamps, **metrics**, task statuses, tokens per agent, models; `planner` (plan used?, error, last step errors), `chart_agent` (done?, error, reasons for skipped charts, step errors), `report_agent` (fallback report?, error, step errors) | the **next run of the same brief** compares its metrics with the latest completed run's |
| `plan` (JSONB) | the plan the run followed | inspection: what the planner made of the brief |
| `state` (JSONB) | every task's status, summary, outputs, `step_errors` (the last 5 across attempts, first line each); insights; chart and table names | debugging / inspection: why a task failed |
| `html` (TEXT) | the report, once it exists | `GET /analysis-report-html`; `GET /analysis-report` returns it with every other column |
| `started_at`, `finished_at`, `created_at`, `updated_at` | timestamps | ordering, the previous-run lookup |

The row is saved after planning, after each task, after the charts and at the end, so it
also shows a run in progress. Runs live in memory (one per API key), but their rows stay
after the run is gone.

---

## 4. File map

| File | Contains | Used by |
| --- | --- | --- |
| `routes/analyst.py` | the 5 HTTP endpoints | client |
| `routes/interviewer.py` | `_approve()` → starts the analyst | client |
| `analyst/service.py` | `start_analysis`, `start_analysis_payload`, `end_analysis`, `_spawn` | routes |
| `analyst/run.py` | `AnalysisRun`, `ProgressLog`, the statuses | service, orchestrator, routes |
| `analyst/orchestrator.py` | `Orchestrator`, `run_analysis`, `step_callback` | service (thread) |
| `analyst/plan.py` | `validate_plan`, `plan_from_brief`, `brief_context_text`, `normalize_plan` | planner, orchestrator, agents |
| `analyst/state.py` | `RunState`, `TaskRecord` | orchestrator, agents, tools |
| `analyst/planner_agent.py` | `PlannerAgentEngine` | orchestrator |
| `analyst/task_agent.py` | `TaskAgentEngine` | orchestrator |
| `analyst/chart_agent.py` | `ChartAgentEngine` | orchestrator |
| `analyst/report_agent.py` | `ReportAgentEngine`, `render_brief_report` | orchestrator |
| `analyst/sandbox.py` | `GuardedPythonExecutor`, `check_code`, `unescape_code` | every analyst agent, interviewer |
| `analyst/sql_prompt.py` | `build_scoped_sql_addendum`, `scope_relations`, `large_relations_note` | planner, task and chart agents |
| `analyst/prompts.py` | prompt defaults and titles | all agents |
| `analyst/metrics.py` | `fill_metrics`, `compute_delta` | report tools, orchestrator |
| `analyst/model_options.py` | `AnalystModelOptions`: model calls cut at 180 s and re-sent up to 2 times by LiteLLM | the four analyst engines |
| `analyst/report_template.py` | `render_report` | `get_html_template`, fallback |
| `analyst/templates/report.html` | Jinja2 page: KPI cards, sections, Chart.js | `render_report` |
| `analyst/values.py` | `to_native`, `preview`, `describe`, `dataset_reference`, `is_dataset_reference` | tools, agents, `metrics.py` |
| `analyst/tools/sql_common.py` | `SqlToolBase` and SQL helpers | SQL tools |
| `analyst/tools/dataset_common.py` | dataset lookup helpers | DataFrame tools |
| `analyst/tools/*_tool.py` | one tool each (see Step 9) | agents |
| `config.py` | `AnalystConfig` (`ANALYST_*`) | everything |
| `infrastructure/agent_manager.py` | `getAnalyst` / `createAnalyst` / `deleteAnalyst` | service |
| `services/analysis_brief_service.py` | `get_brief`, `mark_brief_consumed` | service, orchestrator |
| `services/analysis_report_service.py` | `save_report`, `get_report`, `get_report_html`, `list_reports`, `previous_report_metadata` | orchestrator, `get_delta_section`, routes |
| `infrastructure/database_pg.py` / `database_methods.py` | the `analysis_reports` table and its queries | the report service |
| `main.py` | registers `analyst_bp` | Flask |

Reused without changes: `datachat/smolagents_base.py` (engine base), `datachat/sql_guard.py`,
`datachat/sql_identifiers.py`, `datachat/schema_snapshot_loader.py`, `datachat/sql_prompt.py`,
`datachat/final_answer_contract.py`, `datachat/tools/sql_engine_tool.py`.

---

## 5. Who calls whom

```
POST /start-analysis ─┐
_approve() ───────────┴─► start_analysis_payload() ─► start_analysis()
                                                        ├─ get_brief()
                                                        ├─ createAnalyst(run)
                                                        └─ _spawn() ══ thread ══╗
                                                                                ║
run_analysis() ─► Orchestrator.execute() ◄══════════════════════════════════════╝
  ├─ _load_brief() ─► get_brief()
  ├─ _plan() ─► PlannerAgentEngine.run() ─► CodeAgent.run()
  │               ├─ sql_stats() / sql_group_stats() / … ──► database
  │               └─ final_answer(plan) ─► validate_plan()      (guard)
  │           └─ failed? ─► plan_from_brief()
  │           ─► RunState.from_plan() ─► _save() ─► analysis_reports
  ├─ _run_task() ─► TaskAgentEngine.run() ─► CodeAgent.run()
  │                   ├─ load_dataset() ──► database ─► RunState.datasets
  │                   ├─ sql_*() ─────────► database
  │                   ├─ run_*() / detect_anomalies() ◄─ RunState.datasets
  │                   ├─ save_sub_task_output() ─► RunState.tasks[id].outputs
  │                   └─ record_insight() ──► RunState.insights ─► "insight" event
  ├─ _draw_charts() ─► ChartAgentEngine.run() ─► CodeAgent.run()
  │                   ├─ outputs / datasets (namespace)
  │                   └─ save_chart() / save_table() ─► RunState.charts / tables
  ├─ _write_report() ─► ReportAgentEngine.run() ─► CodeAgent.run()
  │                   ├─ read_sub_task_state() ◄─ RunState (outputs, insights)
  │                   ├─ save_run_metrics() ──► metrics.py
  │                   ├─ get_delta_section() ◄─ previous_report_metadata() ◄─ analysis_reports
  │                   ├─ get_html_template() ─► report_template.py ─► RunState.html
  │                   └─ save_html_report() ──► RunState.report_saved
  ├─ _save(COMPLETED) ─► save_report() ─► analysis_reports (metadata, plan, state, html)
  └─ _mark_consumed() ──► analysis_briefs.status = 'consumed'

GET  /analysis-status ─► get_run() ─► AnalysisRun.status_payload()  (events from step_callback)
GET  /analysis-report ─► get_report(run_id, user) ─► analysis_reports (every column)
GET  /analysis-report-html ─► get_report_html(run_id, user) ─► analysis_reports.html
GET  /analysis-reports ► list_reports(user, brief_id) ─► analysis_reports (no html)
POST /end-analysis ────► end_analysis() ─► deleteAnalyst() ─► AnalysisRun.close() ─► cancel
```

---

## 6. Where the tokens are saved — and where they are spent on purpose

| Technique | Where |
| --- | --- |
| No manager LLM: code runs the tasks | `orchestrator.py` |
| A fresh, short context per task | `task_agent.py` |
| Only the plan's relations in the task/chart schema | `sql_prompt.py` |
| Data in DataFrames, only shapes/previews in the context | `load_dataset`, `values.describe` |
| Charts built in code from the namespace, not from numbers in the prompt | `chart_agent.py` |
| Outputs capped; read back as previews | `ANALYST_OUTPUT_MAX_CHARS`, `ANALYST_PREVIEW_MAX_CHARS` |
| Oversized tables kept as datasets, not fought down to the cap | `save_sub_task_output` |
| Large tables named with estimated counts; sampling via TABLESAMPLE | `large_relations_note` |
| Retries told what is already saved | `render_task()` |
| Exploratory tasks never retried | `Orchestrator._run_task()` |
| Hung model calls cut at 180 s and re-sent up to twice by LiteLLM, not failing the whole attempt | `model_options.py` (interviewer and DataChat keep LiteLLM's defaults) |
| HTML never generated or echoed by a model | `get_html_template`, `report_template.py` |
| KPI values filled by code, not relayed | `save_run_metrics`, `metrics.py` |
| Cheaper model for tasks and charts, stronger for planning and writing | `ANALYST_MODEL` vs `ANALYST_REPORT_MODEL` |

Spent on purpose: the planner run and the exploration tasks (`ANALYST_MAX_EXTRA_TASKS`,
0 turns exploration off), which is what makes the analyst look beyond the brief.
Tokens actually used are recorded per agent run in the run's metadata → `tokens`.

---

## 7. Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANALYST_PROVIDER` / `ANALYST_MODEL` | DataChat's | model of the task and chart agents |
| `ANALYST_REPORT_PROVIDER` / `ANALYST_REPORT_MODEL` | the analyst's | model of the planner and report agents |
| `ANALYST_PLAN_MAX_STEPS` | 12 | steps for planning |
| `ANALYST_TASK_MAX_STEPS` | 20 | steps per task |
| `ANALYST_CHART_MAX_STEPS` | 15 | steps for the charts |
| `ANALYST_REPORT_MAX_STEPS` | 12 | steps for the report |
| `ANALYST_TASK_RETRIES` | 1 | extra attempts per task (0–3) |
| `ANALYST_MAX_EXTRA_TASKS` | 2 | exploration tasks the planner may add (0–5) |
| `ANALYST_MAX_INSIGHTS` | 12 | insights per run (0–50) |
| `ANALYST_SQL_MAX_ROWS` | 50 | rows `sql_engine` / group stats / outliers return to the model |
| `ANALYST_DATASET_MAX_ROWS` | 100000 | rows `load_dataset` loads into a DataFrame |
| `ANALYST_OUTPUT_MAX_CHARS` | 20000 | max size of one saved output; a larger table is kept as a dataset, its first rows in the output (1000–200000) |
| `ANALYST_PREVIEW_MAX_CHARS` | 1500 | size of one output as the report agent reads it |
| `ANALYST_MAX_CHARTS` | 20 | charts per report |
| `ANALYST_CHART_MAX_POINTS` | 500 | data points per chart |
| `ANALYST_TOKEN_COST` | 1 | user tokens spent when a run starts |

The SQL datasource settings (`DATACHAT_SQL_*`: timeout, allowed tables, …) apply to the
analyst too.

---

## 8. Try it, and read the tests

```bash
H=(-H "Content-Type: application/json" -H "X-API-KEY: $KEY" -H "X-USER-EMAIL: me@example.com")

curl -X POST http://127.0.0.1:5000/start-analysis "${H[@]}" -d '{"brief_id": 42}'
curl "http://127.0.0.1:5000/analysis-status?since=0" "${H[@]}"          # repeat with next_since
curl "http://127.0.0.1:5000/analysis-report?run_id=<run_id>" "${H[@]}"                    # the whole run
curl "http://127.0.0.1:5000/analysis-report-html?run_id=<run_id>" "${H[@]}" -o report.html
curl "http://127.0.0.1:5000/analysis-reports?brief_id=42" "${H[@]}"     # the runs of a brief
curl -X POST http://127.0.0.1:5000/end-analysis "${H[@]}"              # cancel / forget
```

The tests are a good second reading path — each one is small and shows one behaviour:

| Test file | Shows |
| --- | --- |
| `tests/test_analyst_plan.py` | briefs of many shapes → valid plans; what `validate_plan` refuses |
| `tests/test_analyst_orchestrator.py` | the whole flow with fake agents: planning and its fallback, order, retry, skip, charts, report fallback, cancel, what is saved and when |
| `tests/test_analyst_agents.py` | what each agent receives (tools, schema, task text) and its guards |
| `tests/test_analyst_sql_tools.py` | the SQL each tool builds, and what the guard refuses |
| `tests/test_analyst_sql_prompt.py` | the large tables note: catalog query, formatting, fallbacks |
| `tests/test_analyst_data_tools.py` | anomalies, regression, forecast, saving outputs/charts/tables, insights |
| `tests/test_analyst_sandbox.py` | which code runs and which is refused |
| `tests/test_analyst_report.py` | metrics, previous-run delta, rendering and its safety |
| `tests/test_analyst_state.py` | `RunState`, `ProgressLog`, value conversion |
| `tests/test_analyst_routes.py` | start checks, polling, report and list ownership, auth headers, cancel |
| `tests/test_analysis_report_service.py` | the `analysis_reports` queries, their wrappers and the service |

```bash
venv/bin/python -m pytest tests/test_analyst_*.py -q
```
