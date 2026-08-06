# Branch summary — `feature/new-datachat-tool`

Branched from `4038ca4` (2026-07-21, *docs: update development guide for admin tools*).
Everything below is work on the DataChat agent, in three parts: **6 small commits**, then one
large commit that dwarfs them, then a further body of **uncommitted** work on charts.

**Status:** 339 tests pass. Committed through `69d7e92`; Part 3 is uncommitted. A server
restart is required to pick up anything, and see *Open items* for two things that fail
silently until acted on.

---

## Part 1 — The 6 commits (2026-08-05)

Goal: give the agent text-analysis capability (sentiment, topic classification) and a way to
export full results.

| Commit | Time | What it did |
|---|---|---|
| `69c5272` | 12:51 | `sentiment_analysis` + `classify` tools; enabled `scikit-learn` in `requirements.txt` (was commented out) |
| `f89fa13` | 14:08 | Skip empty/NaN rows in both new tools |
| `f4a81de` | 14:18 | Sanitizer row limit 50 → 200 |
| `e57e79e` | 14:26 | `export_csv` tool; sentiment defaults to `aggregate=True` with a `max_rows` cap. **Also reverted `f4a81de`** back to 50 |
| `858cf93` | 14:32 | Removed the hard 50-row cap from sentiment |
| `5ca2494` | 14:36 | `export_csv(include_sentiment=True)` running its own full sentiment pass |

Net: 3 new tool files (~815 lines), `scikit-learn` promoted to a hard dependency.

### What the review of these commits found

The direction was right; the layering was not. Four problems, in the order they matter:

1. **The export never reached the user.** `export_csv` returned
   `{"kind":"text","text":"File CSV salvato: /tmp/…"}` — a raw server path. The normalizer
   had no file-serving branch (only `image_path` → base64), no route served the file, and
   `SmolagentsEngine.close()` `rmtree`s the directory on `/enddatachat`. The CSV was written
   to an ephemeral directory, never delivered, then deleted.
2. **Sentiment fabricated labels.** Both sentiment paths truncated unique values at
   `_MAX_UNIQUE_VALUES = 500`, then iterated *every* row and defaulted anything unmatched to
   `neutral`/`0.5`. On a column with more than 500 distinct texts, most rows carried a
   plausible label and confidence score the model never produced.
3. **`filename` allowed writes outside the export directory.** LLM-supplied, joined straight
   into a path, no `basename`, no extension check — unlike `plot_tool`, which always
   generates its own uuid name.
4. **`f4a81de` was a no-op in the final history**, silently undone by `e57e79e`. (Confirmed
   intentional: with `aggregate=True` the 200-row limit was no longer needed.)

Smaller: ~80 lines of sentiment logic duplicated between the two tools and already diverging
(different unmatched-default, different label validation, no `try/except` around the LLM call
in the copy); `unique_vals.index()` inside a per-row loop with a redundant fallback pass;
lost row identity in the export; `classify` dropping rows on a duplicated index; dead
imports; no tests.

---

## Part 2 — Commit `69d7e92`

*feat(datachat): honest table previews, CSV download, and analyst tools* — 38 files,
+4,738 / −395. Five phases, described below. Committed as one atomic change because
`limits.py` is imported by nine modified tools and single files carry changes from several
phases, so any split would have left an intermediate commit with a failing suite.

### 2.1 Table previews and CSV download

**The problem.** `_sanitize_table_records` did `data[:max_rows]` and discarded `len(data)` —
the one fact that mattered. `{"type":"dataframe","value":[…50 rows]}` was
indistinguishable from a query that genuinely matched 50, so users read truncated answers as
complete ones. The agent was never told either: its system prompt documented all four
`kind`s but never mentioned a row cap, and truncation happened *after* the agent finished.

**The fix.** The table branch of `normalize_datachat_response` is the single funnel every
table-returning tool passes through, and it receives the full record list *before*
truncation. `_build_table_response()` now does all three jobs there, once:

- previews 20 rows / 10 columns,
- writes the complete CSV via the engine,
- reports `total_rows`, `total_columns`, `preview_rows`, `truncated`, `download_url`,
  `download_filename`, and forwards any tool `note`.

`type`/`value` are unchanged, so a client ignoring the new fields renders exactly as before.

**Delivery.** `SmolagentsEngine` gained `register_export()` / `resolve_export()` and an
`_exports` map, and `GET /datachat/export/<token>` serves the file behind
`assert_valid_api_key` + `getAgent(api_key)`.

Security properties, by construction rather than by validation:

- the token is only ever a **dict key**, never a path component;
- the on-disk name **is** the token, so no caller-supplied string reaches the filesystem —
  this closes the `filename` traversal hole. Verified: a hint of
  `sentiment_my col/../../etc/passwd` lands as the flat suggested name
  `sentiment_my_col_.._.._etc_passwd.csv`, with the file inside the session directory;
- `_exports` lives on the engine, which `agent_manager` keys by API key, so a token is
  unresolvable without the key that made it;
- exports are **session-scoped** — `/enddatachat` deletes them, and only the 20 most recent
  are kept.

Also: DataFrame payloads used to bypass the row/column caps entirely and now take the same
path; `f4a81de`'s intent is superseded (the preview is 20 by design, with the full data one
click away).

### 2.2 Removing silent truncation everywhere

Three separate user reports of *"the preview shows 20 rows but the exported CSV has 50"*.
Each time the 50 came from a **different** tool's hardcoded ceiling leaking into the export,
because the CSV is built from whatever the tool returned. The caps predated the preview
layer, where they had been a reasonable payload guard.

New `datachat/tools/limits.py` encodes the rule: `resolve_limit()` (no implicit cap; 0 and
`None` both mean "everything"), `truncation_note()`, `sample_warning()`, `join_notes()`.

| Tool | Before | After |
|---|---|---|
| `aggregate` | 50 groups | all |
| `describe` | 50 columns | all |
| `missing_values` | 50 columns | all |
| `trend` | 50 periods | all |
| `unique_values` | 50 values | all |
| `filter_rows` | 50 rows, 10 columns | all |
| `top_rows` | ceiling 20, 10 columns | default 5, no ceiling, all columns |
| `sample_rows` | ceiling 20, 10 columns | default 5, no ceiling, all columns |
| `sentiment_analysis` | `max_rows` input | parameter removed |

`max_rows` was removed from `sentiment_analysis` outright: the model could cap the result,
which silently capped the CSV, and smolagents enforces schema/signature parity so it could
not be hidden. Sampling still works via `sample_rows` → `sentiment(data=…)`.

An explicit `n` is still honoured but never silent — it reports *"80 groups available; only
10 were returned because a limit of 10 was requested."*

Two caps remain deliberately: the KMeans cluster count in `classify`, and plot readability in
`plot_tool` (which returns an image, so no export is involved).

### 2.3 Filters that can express the question

**`filter_rows` could not find a null.** `_eq_mask` compared
`series.astype(str).str.strip().str.lower()` against `str(value).lower()`, so `value=None`
tested every cell against the literal string `"none"` while a NaN cell stringifies to
`"nan"`. Nothing ever matched.

Observed consequence, from the server log: asked *"quante righe contengono commenti?"* on a
column that was 70% blank, the agent tried `value=None`, then `value=""`, got 0 both times,
computed `804 − 0 = 804`, recognised the answer was impossible — *"è impossibile perché il
70.65% dei valori sono mancanti"* — and looped until it exhausted `DATACHAT_MAX_STEPS`,
resending a growing transcript each round against a 100k TPM budget. The true answer was 236.

Added: `is_empty` / `is_not_empty` (NaN, `""` and whitespace-only all count as empty; numeric
and bool columns use `isna()` only so `0.0` and `False` stay values), and `value=None` /
`value=""` under `eq` now mean "missing". Second conditions can use the empty ops, which
needed a fix since they carry no `value` to activate on.

**No text search existed at all** — no tool had a `contains` operator, on a dataset whose
analytical value is a few hundred open comments. Added `contains` / `not_contains` with
`regex=False`: the needle comes from an LLM, so `(2.000` matches literally instead of
raising, and `.*` matches nothing instead of everything. `not_contains` excludes blanks so it
is the exact complement of `contains` over real answers.

### 2.4 New analyst capabilities

A review of all 14 tools found conventional tabular coverage good and the gaps concentrated
in survey data. Three defects and four capabilities.

**Defect — two-dimensional grouping collapsed to one.** `aggregate` kept only the *first*
column of a list, so `group_by=["corso","ruolo"]` returned both courses at 3.0 while the real
cells were 5.0 / 1.0 / 3.0 / 4.0. Worse than the truncation bugs: the number was **wrong**,
not incomplete, and looked entirely plausible. Now groups by up to two columns and errors
above that, naming `crosstab`.

**Defect — Italian text tokenized with English stopwords.** `classify_tool` passed
`stop_words="english"` and sklearn ships no Italian list. Cluster labels came back as
`il, molto, poca, troppa, teoria` / `gli, orari, erano, comodi, meglio`. New
`datachat/tools/stopwords.py` (~270 Italian function words + sklearn's English list +
autodetection) turned those into `tempo, pratica, scarso, esercitarsi, teoria` and
`orari, pomeriggio, comodi, meglio, mattino`.

**Defect — markup in free text.** The real dataset carries 795 `&nbsp;` entities, which made
`nbsp` the single most frequent "word". `keywords_tool` now strips tags and unescapes
entities, including the U+00A0 that `&nbsp;` decodes to. Also added the missing `che` (one of
the most common Italian words) and accented forms `più`, `già`, `così`, `perché`, `però`,
`ciò`.

| New tool | Purpose |
|---|---|
| `crosstab` | Two-dimensional breakdown, counts or an aggregated metric, optional row/column percentages. Blank values become an explicit `(vuoto)` category rather than vanishing and misstating every percentage. |
| `keywords` | Word/bigram frequencies over free text, with `term`, `count`, `answers`, `share_of_answers`. Deterministic and free — no LLM call. |
| `compare_groups` | Welch t-test + Mann-Whitney (the ordinal-safe default for rating scales), 95% CI, Cohen's d, and a plain-language verdict. |

`correlation` extended: `spearman` alongside `pearson` (Pearson assumes evenly spaced values,
which a 1–4 rating scale is not), and `col_y` made optional — omit it to rank every numeric
column against one anchor, omit both for all pairs, sorted by absolute strength.

**Statistical honesty.** `sample_warning()` flags any group under 15 observations via the
existing `note` field. With ~800 responses over ~75 editions there are ~10 each, so ranking
means surfaces noise as signal. Deliberately a tool-level guard, not prompt guidance —
`load_prompt` reads the DB first, so prompt-only rules are inert wherever a
`data_chat_system` row exists.

`compare_groups` was restructured after testing revealed it returned 20 fields while the
client previews 10 — `verdict`, the entire point of the tool, was being cut along with the
difference and p-value. Now exactly 10 ordered fields, detail in `meta`, verdict in `note`
where nothing can trim it. Also floors tiny p-values so it never prints `p=0`.

Tool count 14 → 17. Definitions are re-sent every step: ~5,900 → ~7,600 tokens.

### 2.5 Prompt and observability

`_build_instructions` gained: `TABLE RESULTS ARE PREVIEWS` (never claim a table is complete,
never paginate), `CHOOSING COLUMNS` (pick one column, state which, ask when ambiguous),
`COUNTING FILLED OR EMPTY ROWS` (use `is_not_empty`, never subtract), `PICKING THE RIGHT
TOOL`, and `REPORTING RESULTS HONESTLY` (repeat any `note`).

**A pre-existing prompt bug, unrelated to this branch.** `render_prompt` uses `str.format`,
which raises `KeyError('"kind"')` on the 7 literal JSON brace blocks in the template; the
exception is swallowed and the raw template returned — so the model received the literal
`{columns}` and **never learned the dataset's column names from the prompt**. Now substituted
directly, which also covers a DB-stored prompt with the same placeholder.

`LITELLM_DEBUG` env var gates `litellm._turn_on_debug()` from `build_litellm_model()`, the
chokepoint both DataChat and agentchat share. Off by default because it logs full
request/response bodies — for DataChat that means users' dataset rows. Falls back to setting
the three loggers directly if the private helper moves. Registered in `.env.example` and the
admin allowlist.

`datachat_request_end` now carries `total_rows=`, `truncated=` and `export=`; the export
route logs its own line.

---

## Part 3 — Uncommitted: charts as data

12 files modified (+684 / −107), 5 new files. Prompted by a run where the user asked for a
detailed comment **plus two charts** and received the comment alone — while its prose described
*"Grafico 1 mostra…"*, narrating images that were never sent.

### 3.1 Why the charts vanished

Not a frontend bug. The contract allowed exactly one `kind` per answer
(`text|table|image_path|error`) and `image_path` holds a single path, so *prose plus a chart*
was inexpressible. The agent built both, had to choose, chose `text`, and the PNGs were
orphaned on disk and deleted at `/enddatachat`.

Charts were also matplotlib PNGs base64'd into the payload — ~200 KB each, fixed-size,
non-interactive, and styled by the backend so they could not follow the client's theme.

### 3.2 Charts became data

New `chart` tool emitting a **Chart.js-ready specification** the client renders:
`type`/`labels`/`datasets` (Chart.js's own `data` shape) plus semantic hints `title`,
`x_label`, `y_label`, `stacked`, `horizontal`. Eight kinds — `bar`, `line`, `area`, `pie`,
`doughnut`, `scatter`, `hist`, `kde` — with the backend computing histogram bins and the KDE
curve. Every emitted `type` is a real Chart.js type, so there is no translation step: an area
chart arrives as `line` + `fill: true`.

`plot` is untouched apart from its description and remains the PNG path for **`box` and
`hexbin`**, which have no data reduction. Client-side image rendering must therefore stay.

**No colours and no `options` are ever sent** — palette, fonts and light/dark theming are the
client's, which is the point of leaving server-rendered images. There is a test asserting it.

Charts never contradict the table beside them: tests assert counts equal `unique_values` and
aggregates equal `aggregate` on the same column. Missing series combinations stay `null`, never
`0` — on a 1–4 scale a zero is not a possible answer, so drawing one would invent data.

Measured: prose + two charts is **0.9 KB**, against ~400 KB for two PNGs.

### 3.3 The fix that actually delivers them

A `charts: [...]` array on `text`, `table` **and** `chart` payloads. A standalone `chart` kind
would still have forced a choice between a chart and prose.

Then the same failure recurred: the agent built two charts and called `final_answer` with a bare
text payload, leaving the specs in local variables. **That was a design defect, not model
error** — attaching them was bookkeeping the model gained nothing from, in a later step. So
`ChartTool` now reports every spec it builds to the engine (`collector=self`, the pattern
`ExportCsvTool` already uses for exports) and `chat()` attaches them when the model did not. A
model-supplied list still wins, order included. `chat_end` logs
`charts_attached=auto|model|none`.

Guards: identical specs recorded once, 8 per run, cleared before each run so nothing leaks
between requests, and a collector failure can never cost the user a chart.

### 3.4 Three defects a real run then exposed

1. **`agg="count"` was rejected** — `aggregate` accepts `op="count"`, the model carried that
   vocabulary across, and the call failed. `final_answer` handed the error to the user as the
   whole answer. Worse, `agg="count"` with no `y` describes *exactly* the tool's default. Now
   accepted, and `agg` is validated **only when a `y` makes it applicable** — a parameter that
   cannot affect the result must not be able to fail the call.
2. **Extra charts dropped on a `chart` answer.** The model returned one chart with the comment
   *"gli altri saranno disponibili nell'interfaccia"*; they were not. `chart` is now a chart
   host, attaching only the non-primary charts so none is rendered twice.
3. **The backend imposed Italian on every dataset.** Built against an Italian test file, the
   tools had grown Italian *output*: `"numero di risposte"` on every count chart, `"densità"`,
   `"(vuoto)"`, `"Export pronto: N righe."` — and the system prompt carried
   `Example: 'Nella colonna "...qualunque commento" ci sono 212 righe compilate.'`, hardcoding
   a column name and row count from the test data into every request.

### 3.5 Generalisation

Display language is the same kind of decision as colour, so the backend stopped making it.
Output is now **language-neutral tokens** for the client to localize: `count`,
`density(<column>)`, `(empty)`, `(not analyzed)`, `CSV ready: N rows`. The system prompt and its
examples are dataset- and language-neutral.

Stopwords extended from Italian+English to **all four languages the app declares**
(`bootstrap_static.py`: ITA/ENG/FRA/SPA), with detection by function-word share and a fallback
that excludes all four when none leads clearly — a mixed-language column is handled rather than
guessed at.

`tests/test_language_neutrality.py` guards the class: it walks every chart kind plus the
crosstab and export output asserting no Italian appears. **It found a real bug on its first
run** — `crosstab` ran `astype(str)` before replacing blanks, so a `None` category surfaced as
the literal string `"None"`. Both tools now collapse every flavour of missing (`nan`, `None`,
`<NA>`, `NaT`, `null`, blank) to one sentinel.

### 3.6 Bar orientation

Bar charts carry `horizontal`, decided automatically: horizontal above 10 categories or when
labels are long (80th percentile over 20 characters), which is the common case for survey
columns. Histograms stay vertical — bins belong on the x axis. With horizontal bars Chart.js
renders the first entry at the top and series arrive sorted largest-first, so descending order
and "largest at top" are the same thing; the spec tells the client not to re-sort.

The percentile rather than the max deliberately: one freak long label can be truncated, a whole
axis of them cannot.

---

## Files

### New — code

| File | Lines | Part |
|---|---|---|
| `datachat/tools/chart_tool.py` | 569 | 3 |
| `datachat/tools/compare_groups_tool.py` | 322 | 2 |
| `datachat/tools/crosstab_tool.py` | 267 | 2 |
| `datachat/tools/keywords_tool.py` | 215 | 2 |
| `datachat/tools/stopwords.py` | 214 | 2, extended in 3 |
| `datachat/tools/limits.py` | 83 | 2 |

### New — tests (2,811 lines, 240 tests)

**Part 2** — `test_filter_rows_empty_ops.py` (19) · `test_compare_groups_tool.py` (17) ·
`test_tool_row_limits.py` (16) · `test_crosstab_tool.py` (15) ·
`test_sentiment_tool_coverage.py` (14) · `test_keywords_tool.py` (14) ·
`test_correlation_methods.py` (13) · `test_filter_rows_contains.py` (12) ·
`test_aggregate_multi_group.py` (11) · `test_datachat_export_route.py` (10)

**Part 3** — `test_chart_tool.py` (41) · `test_chart_collection.py` (21) ·
`test_language_neutrality.py` (13) · `test_output_normalizer_preview.py` (24, extended)

### New — docs

- **`DINO_CLIENT_SPEC.md`** (678) — the single client specification, covering both additions:
  the response envelope with every `type` and its optional fields; table previews and the
  download endpoint; the chart specification, types, dataset shapes and orientation; the rules
  shared by both (absent vs `null`, tolerating unknown fields, `note` handling, reserved label
  tokens, defensive rendering); compatibility; known limitations; one acceptance checklist; and
  a question back to the Dino team about the base64 wrapper (see *Open items* #3).

  Started as two documents — `DINO_TABLE_PREVIEW_SPEC.md` (340) and `DINO_CHART_SPEC.md` (387)
  — and merged into one on request. The merge unified genuine duplication rather than
  concatenating: each had its own "absent vs null", "tolerate unknown fields" and `note`
  section, and each carried half the type list. Every payload example was re-validated as JSON
  and for internal consistency, and stale Italian labels (`numero di risposte`, `media (1-4)`)
  were corrected to the neutral tokens the backend now emits.
- `DATACHAT_TEST_PLAN.md` (249) — 48 questions against `Feedbacks.csv` with expected answers
  computed by running the tools

### Modified

**Part 2** — `datachat/output_normalizer.py` · `datachat/smolagents_engine.py` ·
`routes/datachat.py` · 9 tool files · `llm/litellm_factory.py` · `routes/admin.py` ·
`.env.example` · `DEVELOPMENT.md`

**Part 3** — `datachat/output_normalizer.py` · `datachat/smolagents_engine.py` ·
`datachat/tools/` (`plot_tool`, `crosstab_tool`, `export_csv_tool`, `sentiment_tool`,
`stopwords`) · `DEVELOPMENT.md`

### Removed

`DINO_TABLE_PREVIEW_SPEC.md` and `DINO_CHART_SPEC.md` — merged into `DINO_CLIENT_SPEC.md`.
All references across `DEVELOPMENT.md`, the tool docstrings and the tests were repointed.

---

## Open items

Ordered by how quietly each one fails.

1. **Prompt guidance may never reach the model.** `load_prompt`
   (`infrastructure/prompt_utils.py:21`) prefers the `prompts` DB table, so every prompt change
   on this branch is inert wherever a `data_chat_system` row exists. The in-code default is
   confirmed in use locally, but this must be checked per deployment — it fails with no error.
   A startup `WARNING Prompt 'data_chat_system' not found in DB` means the in-code text is live.
2. **Dino must ship the client side** — see `DINO_CLIENT_SPEC.md`. All new response fields are
   additive and safe to ignore, but ignoring them is now *worse* than before: tables show a
   20-row preview with no sign that more exists (the old preview was 50), and charts do not
   render at all. Two likeliest mistakes:
   - treating `download_url` as a plain link — it cannot send the required headers;
   - dropping `type: "image"` support, which is still the only path for `box` and `hexbin`.
3. **`fileToBase64` returns invalid base64.** `infrastructure/file_manager.py:15` returns
   `str(base64.b64encode(...))`, i.e. the Python repr wrapped in a literal `b'`…`'`;
   `b64decode(validate=True)` rejects it, and the correct line sits commented out beneath.
   Images render today, so **the client is compensating** — which means fixing the backend
   alone could break every image. `DINO_CLIENT_SPEC.md` §9 asks whether Dino's strip is
   conditional or unconditional; that answer decides whether the one-liner is safe.
4. **`note` prose is English.** Truncation warnings, small-sample cautions and the sentiment
   coverage note are English sentences, on a backend that otherwise emits language-neutral
   tokens. Localizing them needs structured note codes — a deliberate contract change, not a
   side effect.
5. **Two shapes of `Feedbacks.csv` exist.** The server export uses the question text as column
   headers with 804 rows; the local copy uses `q4`/`q5`… headers plus a **second header row**
   read as data, giving 805 rows for 804 responses and off-by-one
   `unique_values`/`crosstab`/`is_not_empty` counts. Documented in `DATACHAT_TEST_PLAN.md` §0;
   worth handling in `dataset_loader.py` if the local shape is ever uploaded.
6. **An uncaptured LiteLLM exception.** The `LLM Provider NOT provided` line in the logs is a
   benign DEBUG artifact of litellm's failure-logging path calling `get_api_base()` with an
   already-stripped model name (`mistral/mistral-small-latest` resolves correctly). The real
   exception above it was never captured. `LITELLM_DEBUG=true` will surface it.
7. **`exports/` in the project root** is empty, gitignored and referenced by no code — a
   leftover from manual testing, safe to delete.
8. **Part 3 is uncommitted.**

### Resolved since first writing

- **Token accounting.** Was `Cost not found for provider: Mistral and model:
  mistral-large-latest` → `db_log_ok=False`. Later logs show
  `token usage logged log_id=862`, `db_log_ok=True`. A costs row was added.
