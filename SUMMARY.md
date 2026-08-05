# Branch summary — `feature/new-datachat-tool`

Branched from `4038ca4` (2026-07-21, *docs: update development guide for admin tools*).
Everything below is work on the DataChat agent: **6 commits** plus a body of **uncommitted
changes** that is substantially larger than the commits.

**Status:** 221 tests pass. Nothing after the 6 commits is committed. A server restart is
required to pick up any of it.

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

## Part 2 — Uncommitted changes

19 files modified (+1,073 / −395), 18 new files. Five phases.

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

## Files

### New — code

| File | Lines |
|---|---|
| `datachat/tools/compare_groups_tool.py` | 322 |
| `datachat/tools/crosstab_tool.py` | 267 |
| `datachat/tools/keywords_tool.py` | 215 |
| `datachat/tools/stopwords.py` | 115 |
| `datachat/tools/limits.py` | 83 |

### New — tests (1,790 lines, 156 tests)

`test_tool_row_limits.py` (16) · `test_filter_rows_empty_ops.py` (19) ·
`test_compare_groups_tool.py` (17) · `test_crosstab_tool.py` (15) ·
`test_output_normalizer_preview.py` (15) · `test_sentiment_tool_coverage.py` (14) ·
`test_keywords_tool.py` (14) · `test_correlation_methods.py` (13) ·
`test_filter_rows_contains.py` (12) · `test_aggregate_multi_group.py` (11) ·
`test_datachat_export_route.py` (10)

### New — docs

- `DINO_TABLE_PREVIEW_SPEC.md` (340) — client spec: payloads (captured from real output),
  deserialization rules, required UI behaviour, download contract, acceptance checklist
- `DATACHAT_TEST_PLAN.md` (249) — 48 questions against `Feedbacks.csv` with expected answers
  computed by running the tools

### Modified

`datachat/output_normalizer.py` · `datachat/smolagents_engine.py` · `routes/datachat.py` ·
9 tool files · `llm/litellm_factory.py` · `routes/admin.py` · `.env.example` ·
`DEVELOPMENT.md`

---

## Open items

1. **Prompt guidance needs applying to the DB.** `load_prompt` prefers the `prompts` table,
   so every prompt change on this branch is inert wherever a `data_chat_system` row exists.
   Verify this first — it fails silently.
2. **Dino must ship the client side.** The new response fields are additive and safe to
   ignore, but a client that ignores them shows a 20-row preview with no indication that more
   exists — worse than before, when the preview was 50. See `DINO_TABLE_PREVIEW_SPEC.md`; the
   likeliest mistake is treating `download_url` as a plain link, which cannot send the
   required headers.
3. **Token accounting is failing.** `Cost not found for provider: Mistral and model:
   mistral-large-latest` → `db_log_ok=False`, `log_id=none`. The flat per-request deduction
   still happens; the per-request usage/cost audit row is lost. Needs a costs row for the
   model actually in use.
4. **`Feedbacks.csv` has two header rows.** Row 2 holds the question text and is read as
   data, so 805 rows load for 804 responses and `unique_values`/`crosstab`/`is_not_empty`
   counts are off by one. Consider handling in `dataset_loader.py`.
5. **An unresolved LiteLLM exception.** The `LLM Provider NOT provided` line in the logs is a
   benign DEBUG artifact of litellm's failure-logging path calling `get_api_base()` with an
   already-stripped model name (`mistral/mistral-small-latest` resolves correctly). The real
   exception above it has not been captured.
6. **`exports/` in the project root** is empty, gitignored and referenced by no code — a
   leftover from manual testing, safe to delete.
7. **Nothing after `5ca2494` is committed.**
