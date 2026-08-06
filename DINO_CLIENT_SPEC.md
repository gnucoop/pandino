# Dino client specification — DataChat responses

**Status:** backend implemented, awaiting Dino client support
**Backend branch:** `feature/new-datachat-tool`
**Audience:** Dino (MAUI) client developers

Covers everything the client needs for the new DataChat response contract: table previews and
CSV download, charts as data, and the rules shared by both.

---

## 1. Why this changed

Two independent problems, both of which made the backend claim more than it delivered.

**Tables were truncated silently.** DataChat caps table payloads to keep the JSON small. Until
now the cap was invisible: a query matching 340 rows returned 50, and the payload was
indistinguishable from one that genuinely matched 50. Users read truncated answers as complete
ones. The backend now states the real size and attaches a download for the full CSV.

**A response could carry only one thing.** The contract allowed exactly one `kind` per answer
(`text`, `table`, `image_path`, `error`) and `image_path` held a single path — so *prose plus a
chart*, or *two charts*, was inexpressible. Asked for a detailed comment **and** two charts, the
agent built both, had to choose, chose the prose, and the images were orphaned on disk — while
that prose said *"Grafico 1 mostra…"*, describing pictures the user never received.

**Charts were also server-rendered PNGs**: matplotlib output base64'd into the payload at
roughly 200 KB each, fixed-size, non-interactive, and styled by the backend so they could not
follow the client's theme. They are now **data** — a Chart.js-ready specification the client
draws. A spec is a few KB, renders crisp at any size, and is themed by Dino, which is what makes
several charts alongside prose affordable.

**The client is now the only remaining reason a user would be misled.** A client that ignores
these fields shows a 20-row preview with no sign that 320 more rows exist — *worse* than before,
because the preview got smaller — and renders no charts at all.

---

## 2. The response envelope

`POST /datachat` returns, as it always has:

```json
{ "response": { "type": "...", "value": ... }, "explanation": null, "log_id": 42 }
```

`type` and `value` are **unchanged**. Every field below is additive, so existing rendering code
keeps working untouched.

| `type` | `value` holds | Optional extra fields |
|---|---|---|
| `str` | string | `download_url`, `download_filename`, `charts`, `note` |
| `dataframe` | array of row objects | `total_rows`, `total_columns`, `preview_rows`, `truncated` (always present); `download_url`, `download_filename`, `charts`, `note` |
| `chart` | one chart specification | `charts` (further charts), `note` |
| `image` | base64 PNG | — |
| `dict` | object | — |

`text_and_image` also exists in the documented type list but is reachable only through a legacy
branch the backend annotates as dead. Treat it as won't-happen.

Note that **`str` covers both normal answers and errors** — see §8.

---

## 3. Tables — preview and download

### 3.1 Fields on `type: "dataframe"`

| Field | Type | Meaning |
|---|---|---|
| `total_rows` | int | Rows in the **complete** result. `value` may hold fewer. |
| `total_columns` | int | Columns in the complete result. `value` may hold fewer. |
| `preview_rows` | int | Rows actually present in `value`. |
| `truncated` | bool | `true` when `value` is a subset of the real result. |
| `download_url` | string \| null | Path to the complete CSV. `null` when nothing was cut. |
| `download_filename` | string \| null | Suggested filename, e.g. `sentiment_reviews.csv`. |

Current preview limits: **20 rows**, **10 columns**. Do not hardcode them — read
`total_rows`/`preview_rows` and `truncated`. They may be tuned server-side.

A `type: "str"` answer may also carry `download_url` + `download_filename`, when the user asks
for a file of raw data rather than an on-screen table.

### 3.2 Required UI

**When `truncated` is `true`**, show adjacent to the table, before the user can mistake it for
the whole result:

> Showing the first **{preview_rows}** of **{total_rows}** rows.

If `total_columns` exceeds the number of keys in a `value` row, also say columns were dropped —
the preview keeps only the first 10, so a wide result loses fields silently otherwise. Wording
is yours; the required content is **preview size, real size, and that this is not everything**.

**When `download_url` is set**, offer a download affordance labelled with `download_filename`.
Fetch it as described in §6 — **not** as a plain hyperlink, which cannot authenticate.

**Null cells are meaningful.** A `null` in a `sentiment`/`score` cell means *"not analyzed"* and
is deliberately not a default value. Render it blank or as an explicit "—", never as `0`,
`false`, `"null"`, or a plausible-looking category.

### 3.3 Payloads

Real outputs, not illustrations.

**A. Truncated table — banner and download button**

```json
{
  "response": {
    "type": "dataframe",
    "value": [ { "city": "c0", "n": 0 }, { "city": "c1", "n": 1 } ],
    "total_rows": 340,
    "total_columns": 2,
    "preview_rows": 20,
    "truncated": true,
    "download_url": "/datachat/export/bf80bb41d8214522b0d38bca61afbd26",
    "download_filename": "cities.csv"
  },
  "explanation": null
}
```

**B. Complete table — render exactly as today, no banner, no button**

```json
{
  "response": {
    "type": "dataframe",
    "value": [ { "city": "Roma", "n": 1 }, { "city": "Milano", "n": 2 } ],
    "total_rows": 2,
    "total_columns": 2,
    "preview_rows": 2,
    "truncated": false,
    "download_url": null,
    "download_filename": null
  },
  "explanation": null
}
```

**C. Truncated table with a caveat — banner, button **and** the note**

```json
{
  "response": {
    "type": "dataframe",
    "value": [
      { "txt": "ottimo servizio", "sentiment": "positive", "score": 0.95 },
      { "txt": "mai piu", "sentiment": null, "score": null }
    ],
    "total_rows": 530,
    "total_columns": 3,
    "preview_rows": 20,
    "truncated": true,
    "download_url": "/datachat/export/b3e2ed0fd6c44683858ef641542b108b",
    "download_filename": "sentiment_txt.csv",
    "note": "12 rows could not be analyzed: the column has 30 more distinct values than the 500-value analysis limit. Their sentiment is empty, not neutral."
  },
  "explanation": null
}
```

**D. Text answer carrying a file**

```json
{
  "response": {
    "type": "str",
    "value": "CSV ready: 340 rows.",
    "download_url": "/datachat/export/abc123",
    "download_filename": "dataset.csv"
  },
  "explanation": null
}
```

**E. Plain text — nothing new**

```json
{ "response": { "type": "str", "value": "This dataset contains..." }, "explanation": null }
```

---

## 4. Charts — as data

### 4.1 Where charts arrive

Two places:

- **`type: "chart"`** — the answer is a single chart. `value` holds the specification.
- **`charts: [...]`** on `str`, `dataframe` **or** `chart` — an array of specifications to
  render alongside `value`. This is the part that fixes the reported bug: a standalone `chart`
  type would give a chart *or* prose, never both.

Maximum **6** charts per response; beyond that a `note` says so.

**`charts` may appear unprompted.** The backend collects every chart the agent builds during a
run and attaches them when the answer does not carry them itself — the agent used to describe
charts in its prose and leave the specs behind. Do not assume `charts` only appears when the
text mentions a chart; render whatever arrives.

**On `type: "chart"`, `value` is the primary chart and `charts` holds *additional* ones** — the
primary is never repeated inside it. Render `value` first, then each entry of `charts`. This is
common: the agent tends to return one chart as the answer and assume the rest surface.

### 4.2 The specification

```json
{
  "type": "bar",
  "labels": ["1", "2", "3", "4"],
  "datasets": [
    { "label": "count", "data": [20, 71, 273, 440] }
  ],
  "title": "Overall satisfaction distribution",
  "x_label": "Nel complesso quanto sei soddisfatto/a della formazione?",
  "y_label": "count",
  "stacked": false,
  "horizontal": false
}
```

`type`, `labels` and `datasets` are **Chart.js's own `data` shape**, so they go straight in:

```csharp
new Chart(ctx, new { type = spec.Type, data = new { spec.Labels, spec.Datasets } })
```

The rest are **semantic hints, not styling**:

| Field | Type | Meaning |
|---|---|---|
| `type` | string | Chart.js chart type — see §4.3 |
| `labels` | string[] \| null | Category labels, one per point. `null` for `scatter`. |
| `datasets` | object[] | One entry per series; `label` names it, `data` holds the values |
| `title` | string \| null | Suggested chart title |
| `x_label` | string \| null | Suggested x-axis label. May be a full survey question — expect long strings. |
| `y_label` | string \| null | Suggested y-axis label — see reserved tokens in §5.4 |
| `stacked` | bool | Hint that a multi-series bar chart reads better stacked |
| `horizontal` | bool | Bar charts only: `indexAxis: 'y'`. Always `false` for other types. |

**There is deliberately no Chart.js `options` object, and no colours.** Palette, fonts, legend
placement, gridlines, animation and light/dark theming are Dino's — that is the whole reason for
leaving server-rendered images behind. Apply a consistent categorical palette of your own and
ensure both themes are legible; the backend will never send a colour.

### 4.3 Chart types, and the two exceptions

| `type` | Notes |
|---|---|
| `bar` | Counts by category, or an aggregate of a metric per category |
| `line` | `labels` are the x values, already sorted |
| `pie` | Single dataset; `labels` are the slices |
| `doughnut` | As `pie` |
| `scatter` | **`datasets[].data` is `{x, y}` objects; `labels` is `null`** — see §4.4 |
| `hist` | A `bar` chart of pre-computed bins; `labels` are bin ranges, e.g. `"1.00–1.75"` |
| `kde` | A `line` of a pre-computed density curve on an even x grid |

Every value of `type` is a **real Chart.js type** — there is no translation step. An *area* chart
arrives as `type: "line"` with `"fill": true` on the dataset, and a *histogram* as `type: "bar"`
with bin-range labels, because that is what Chart.js actually draws.

**`box` and `hexbin` have no native Chart.js equivalent** and continue to arrive as
**`type: "image"` with base64 in `value`**, from the existing `plot` tool. Dino must keep its
current image rendering — do not remove it as part of this work.

### 4.4 Per-type dataset shapes

Every type except `scatter` uses a flat `data` array **parallel to `labels`**:

```json
{ "type": "line", "labels": ["2025-07", "2025-08", "2025-09"],
  "datasets": [{ "label": "mean(soddisfazione)", "data": [3.2, 3.4, 3.5] }] }
```

`scatter` carries point objects and **no `labels`**:

```json
{ "type": "scatter", "labels": null,
  "datasets": [{ "label": "chiarezza / soddisfazione",
                 "data": [{ "x": 3, "y": 4 }, { "x": 2, "y": 2 }] }],
  "x_label": "chiarezza", "y_label": "soddisfazione" }
```

Multiple series arrive as multiple `datasets` sharing one `labels` array:

```json
{ "type": "bar", "labels": ["INTELLIGENZA ARTIFICIALE", "EXCEL", "SOFT SKILL"],
  "datasets": [
    { "label": "Operatore",    "data": [3.6, 3.3, 3.5] },
    { "label": "Coordinatore", "data": [3.5, 3.4, 3.6] }
  ],
  "stacked": true, "horizontal": true }
```

A dataset's `data` may contain `null` where a series has no value for that label — **render a
gap, not a zero**. On a 1–4 satisfaction scale a zero is not a possible answer, so drawing one
would invent data.

### 4.5 `horizontal` — bar orientation

`horizontal: true` means the bars run left-to-right:

```csharp
options = new { indexAxis = spec.Horizontal ? "y" : "x" }
```

**Categories must stay in the order given.** Datasets arrive sorted largest-first, and Chart.js
renders the first entry at the top of a horizontal axis, so honouring the array order produces
the largest bar at the top. Do not re-sort, and do not reverse the axis.

The backend chooses orientation automatically: horizontal when a bar chart has more than 10
categories, or when its labels are long (80th percentile over 20 characters) — the common case
for survey columns. Histograms are always vertical, because bins belong on the x axis. The agent
can force either orientation explicitly.

### 4.6 Chart payloads

**A. Prose plus two charts — the reported bug**

```json
{
  "response": {
    "type": "str",
    "value": "### Analisi del dataset\n\nLa soddisfazione complessiva…",
    "charts": [
      { "type": "bar",
        "labels": ["1", "2", "3", "4"],
        "datasets": [{ "label": "count", "data": [20, 71, 273, 440] }],
        "title": "Overall satisfaction distribution",
        "x_label": "q11", "y_label": "count",
        "stacked": false, "horizontal": false },
      { "type": "bar",
        "labels": ["INTELLIGENZA ARTIFICIALE", "DIGITAL FUNDRAISING", "SOFT SKILL"],
        "datasets": [{ "label": "mean(q11)", "data": [3.573, 3.552, 3.539] }],
        "title": "Mean satisfaction by programme",
        "x_label": "project_parent_name", "y_label": "mean(q11)",
        "stacked": false, "horizontal": true }
    ]
  },
  "explanation": null
}
```

**B. A chart on its own**

```json
{
  "response": {
    "type": "chart",
    "value": {
      "type": "pie",
      "labels": ["probabile", "molto_probabile", "poco_probabile", "improbabile"],
      "datasets": [{ "label": "count", "data": [409, 309, 64, 22] }],
      "title": "Likelihood to recommend",
      "x_label": "q16", "y_label": "count", "stacked": false, "horizontal": false
    }
  },
  "explanation": null
}
```

**C. A chart answer carrying a second chart**

```json
{
  "response": {
    "type": "chart",
    "value": { "type": "bar", "labels": ["1", "2"],
               "datasets": [{ "label": "count", "data": [20, 71] }],
               "title": null, "x_label": "q4", "y_label": "count",
               "stacked": false, "horizontal": false },
    "charts": [
      { "type": "bar", "labels": ["1", "2"],
        "datasets": [{ "label": "count", "data": [11, 72] }],
        "title": null, "x_label": "q5", "y_label": "count",
        "stacked": false, "horizontal": false }
    ]
  },
  "explanation": null
}
```

**D. A table with a chart beside it**

```json
{
  "response": {
    "type": "dataframe",
    "value": [ { "project_parent_name": "INTELLIGENZA ARTIFICIALE", "mean_q11": 3.573 } ],
    "total_rows": 8, "total_columns": 2, "preview_rows": 8,
    "truncated": false, "download_url": null, "download_filename": null,
    "charts": [
      { "type": "bar", "labels": ["INTELLIGENZA ARTIFICIALE"],
        "datasets": [{ "label": "mean(q11)", "data": [3.573] }],
        "title": "Mean satisfaction by programme",
        "x_label": "project_parent_name", "y_label": "mean(q11)",
        "stacked": false, "horizontal": true }
    ]
  },
  "explanation": null
}
```

**E. Box plot — still an image**

```json
{ "response": { "type": "image", "value": "<base64 PNG>" }, "explanation": null }
```

---

## 5. Rules that apply to everything

### 5.1 Absent and `null` mean the same thing

The payload is **not** symmetric across types, by design of the existing normalizer:

- on `dataframe`, `download_url`/`download_filename` are always present, `null` when unused;
- on `str`, they are **absent entirely** when unused;
- `charts` is **absent** when there are none — never `null` or `[]`;
- `note` is **absent** when there is none — never `null`;
- `title`, `x_label`, `y_label` and `labels` may each be `null` inside a chart spec.

Treat "missing key" and "null value" as identical. Never branch on key presence.

### 5.2 Tolerate unknown fields

Use a lenient deserializer. Further additive fields may appear without a client release; a
strict `MissingMemberHandling.Error`-style setting would break on the next backend change.

Suggested model:

```csharp
public sealed class DataChatResponse
{
    public string Type { get; set; }
    public JsonElement Value { get; set; }

    // tables
    public int? TotalRows { get; set; }
    public int? TotalColumns { get; set; }
    public int? PreviewRows { get; set; }
    public bool Truncated { get; set; }          // absent -> false, correct default
    public string? DownloadUrl { get; set; }
    public string? DownloadFilename { get; set; }

    // charts
    public List<ChartSpec>? Charts { get; set; }

    // both
    public string? Note { get; set; }

    public bool HasDownload => !string.IsNullOrEmpty(DownloadUrl);
    public bool HasCharts   => Charts is { Count: > 0 };
}
```

Do **not** compute `truncated` yourself. Read the flag: `value.Count < total_rows` is equivalent
today but will not be if a future limit applies per-column only.

### 5.3 `note` must always be displayed

`note` appears on tables, text answers and charts alike, and reports caveats about the result
itself: rows that could not be analyzed, groups too small to trust, charts or series trimmed.
Display it verbatim, near the content, visually distinct from the data — it is a caveat, not a
value. Do not truncate it, and do not suppress it when `truncated` is `false`: it can accompany
a complete result.

Suppressing it is how a partial answer becomes an apparently complete one. Example: *"12 rows
could not be analyzed… Their sentiment is empty, not neutral."* Hide that and the user reads
empty cells as neutral sentiment.

### 5.4 Reserved label values

The backend never chooses a display language, exactly as it never chooses a colour. Labels are
**semantic tokens** for the client to localize:

| Token | Meaning |
|---|---|
| `"count"` | `y_label`/dataset label of a chart counting rows per category |
| `"density"` / `"density(<column>)"` | a KDE curve |
| `"<agg>(<column>)"` e.g. `mean(q11)` | an aggregate; embeds a user column name, so translate only the function part if at all |
| `"(empty)"` | a category standing for missing/blank values — a placeholder, **not data** |
| `"(not analyzed)"` | in a `sentiment_analysis` summary, rows the model could not score — again a placeholder, not a sentiment |

`(empty)` and `(not analyzed)` should be visually distinguishable from real categories.

### 5.5 Validate before drawing

Charts are generated from user questions, so be defensive:

- `datasets[].data` should be the same length as `labels` for every non-`scatter` type. If it is
  not, show the `title` plus a short "chart could not be rendered" rather than throwing.
- Empty `datasets`, or every value `null`, should render as "no data", not a blank canvas.
- Expect up to a few thousand points in a `scatter`.

### 5.6 Long labels are normal

`x_label` and dataset labels come from column names, which in survey exports are entire
questions — *"Quanto ritieni che i temi trattati siano coerenti con il fabbisogno formativo
dell'organizzazione per cui lavori?"* is a real one. Wrap, truncate with a tooltip, or shrink;
do not let it break the layout.

---

## 6. Download endpoint

```
GET {baseUrl}{download_url}
```

`download_url` is a **server-relative path** (`/datachat/export/<token>`). Concatenate it with
the same base URL used for `/datachat`. Do not parse or rebuild the token.

### 6.1 Required headers

| Header | Value |
|---|---|
| `X-API-KEY` | the same key used for `/datachat` |
| `X-USER-EMAIL` | the same email used for `/datachat` |

**This is the single most likely integration mistake:** the endpoint is not publicly reachable,
so opening `download_url` in a browser or a plain `<a href>` will not authenticate. It must be
fetched with headers (`HttpClient`) and the body saved or handed to the platform share sheet.

### 6.2 Success

- `200`, `Content-Type: text/csv`
- `Content-Disposition: attachment; filename=cities.csv`

The body is the complete result — **all rows and all columns**, not the preview. Prefer
`download_filename` from the JSON over parsing `Content-Disposition`.

### 6.3 Failures

| Status | Body | Cause | Suggested client behaviour |
|---|---|---|---|
| `400` | `{"error":"Missing X-API-KEY header"}` | header omitted | bug — fix the request |
| `400` | `{"error":"Missing X-USER-EMAIL header"}` | header omitted | bug — fix the request |
| `400` | `{"error":"Agent not active for this Api Key"}` | session ended / server restarted | "the chat session has ended; re-run the query" |
| `403` | **HTML**, not JSON | key invalid or expired | same handling as a 403 from `/datachat` |
| `404` | `{"error":"Export not found or expired"}` | token unknown, or session closed | "this download is no longer available; re-run the query" |

The `403` returns an HTML error page — the app registers no JSON error handler, so this is
consistent with every other endpoint. Do not attempt to parse it as JSON.

### 6.4 Lifetime — exports are session-scoped

A token resolves **only** through the engine registered for the API key that created it, and
only while that chat session is alive. It becomes a `404` when `/enddatachat` is called, the
session expires, or the server restarts.

Consequences:

- Do not persist `download_url` across app restarts or offer it in chat history after the
  session closes. Treat a `404` as expected, not as an error worth a crash report.
- Download during the conversation; save locally at download time if the file must outlive the
  session.
- Only the most recent **20** exports per session are retained; older tokens 404 as well.

---

## 7. Backward and forward compatibility

- A client that ignores every new field renders exactly as today (with a 20-row instead of
  50-row table). No flag day, no coordinated deploy.
- **`type: "chart"` is the one exception** — it is a new value, so until Dino handles it a
  chart-only answer hits the default branch. Handle it in the same release that adds the
  renderer.
- **`type: "image"` must keep working** — it remains the path for `box` and `hexbin`.
- Existing `type` values are unchanged: `str`, `dataframe`, `image`, `dict`, `text_and_image`.
- Old client, new backend: safe but misleading — a smaller preview with no indication it is one,
  and no charts. Ship §3.2 and §4 promptly.
- New client, old backend: safe. `truncated` deserializes to `false`, `download_url` and
  `charts` to null/absent, so no banner, no button, no charts.

---

## 8. Known limitations

**Errors are indistinguishable from answers.** `kind: "error"` and `kind: "text"` both normalize
to `type: "str"`, so the client cannot style failures differently. Pre-existing behaviour,
unchanged by this work. If Dino wants the distinction, it needs a backend discriminator — raise
it and we will add one.

**`note` text is English prose.** Truncation warnings, small-sample cautions and the sentiment
coverage note are English sentences, on a backend that otherwise emits language-neutral tokens
(§5.4). Localizing them needs structured note codes — a deliberate future contract change.

---

## 9. Question back to the Dino team

**How does Dino currently strip the `b'…'` wrapper from image base64?**

`fileToBase64` (`infrastructure/file_manager.py:15`) returns
`str(base64.b64encode(file.read()))` — the Python *repr* of a bytes object, so every image value
is wrapped in a literal `b'` … `'`. Verified: `base64.b64decode(value, validate=True)` rejects
it. The correct line sits commented out directly beneath.

Images render in Dino today, so the client must be compensating. Which way?

- **Conditionally** (trim only when the value starts with `b'`) → the backend one-liner is safe
  to fix on its own, and correct base64 passes through untouched.
- **Unconditionally** (always drop the first two and last characters) → fixing the backend would
  corrupt every image, and the two changes must ship together.

Please confirm before the backend fix is scheduled. Independent of this work, but it touches the
same rendering path.

---

## 10. Acceptance checklist

Against a dataset with more than 20 rows and more than 10 columns.

**Tables**

- [ ] Query returning >20 rows shows a preview banner with the correct `total_rows`
- [ ] Query returning ≤20 rows shows **no** banner and **no** download button
- [ ] Query returning >10 columns indicates that columns were dropped
- [ ] Download button fetches with both headers and yields a CSV whose row count equals
      `total_rows` — this is the end-to-end proof
- [ ] Null `sentiment`/`score` cells render blank or "—", never `0`/`false`/`"null"`
- [ ] A download attempted after `/enddatachat` shows a friendly "no longer available", not a
      crash or a raw 404
- [ ] A text response carrying `download_url` shows a download button alongside the message

**Charts**

- [ ] A `str` response with `charts` of length 2 renders the prose **and both charts** — the
      reported bug, and the primary test
- [ ] A `type: "chart"` response renders a single chart
- [ ] A `type: "chart"` response with `charts` renders the primary **plus** the extras, with
      nothing duplicated
- [ ] A `dataframe` response with `charts` renders table and chart together
- [ ] A response with no `charts` key renders exactly as before
- [ ] `bar`, `line`, `pie`, `doughnut`, `hist`, `kde` all draw from `labels` + flat `data`
- [ ] `scatter` draws from `{x, y}` points with `labels` null
- [ ] An area chart (`type: "line"` with `fill: true`) renders filled
- [ ] A multi-dataset chart shows a legend; `stacked: true` stacks it
- [ ] `horizontal: true` draws bars left-to-right, largest at the top, order unchanged
- [ ] `null` inside `data` renders as a gap, never as zero
- [ ] Charts are legible in **both** light and dark themes
- [ ] A question-length `x_label` does not break the layout
- [ ] A malformed spec (data/labels length mismatch, empty datasets) degrades gracefully
- [ ] `type: "image"` still renders — ask for a boxplot to confirm

**Both**

- [ ] `note` is displayed whenever present, visually distinct from the data
- [ ] Reserved tokens `(empty)` and `(not analyzed)` are distinguishable from real categories

---

## 11. Backend reference

| Concern | Location |
|---|---|
| Response assembly | `datachat/output_normalizer.py` → `normalize_datachat_response` |
| Table preview + export | same file → `_build_table_response()` |
| Preview limits | same file → `_PREVIEW_ROWS`, `_PREVIEW_COLUMNS`, `_MAX_CHARTS` |
| Chart specs | `datachat/tools/chart_tool.py` |
| Chart collection / attachment | `datachat/smolagents_engine.py` → `record_chart()`, `_attach_run_charts()` |
| PNG charts (`box`, `hexbin`) | `datachat/tools/plot_tool.py` |
| Image handling | `output_normalizer.py` — `kind == "image_path"` → `{"type": "image", "value": fileToBase64(path)}` |
| Download endpoint | `routes/datachat.py` → `downloadDatachatExport()` |
| Token issuing / lifetime | `datachat/smolagents_engine.py` → `register_export()`, `resolve_export()`, `close()` |
| Allowed final answer kinds | `datachat/smolagents_engine.py` → `_ALLOWED_FINAL_KINDS` |
| Base64 helper | `infrastructure/file_manager.py` → `fileToBase64` (see §9) |
| Response schema docs | `DEVELOPMENT.md` → "Response shape — DataChat `response`" |

Server-side logs carry `total_rows=`, `truncated=`, `export=` and `charts_attached=` on every
`datachat_request_end` / `chat_end` line, and `datachat_export status=` on each download —
useful when reconciling a client-side report.
