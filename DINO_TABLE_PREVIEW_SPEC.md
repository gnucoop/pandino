# Dino client specification — DataChat table previews & CSV download

**Status:** backend implemented, awaiting Dino client support
**Backend branch:** `feature/new-datachat-tool`
**Audience:** Dino (MAUI) client developers

---

## 1. Why this change

DataChat caps the size of table responses to keep the JSON payload small. Until now that
cap was **silent**: a query matching 340 rows returned 50 rows, and the payload was
indistinguishable from a query that genuinely matched 50. Users read truncated answers as
complete ones.

The backend now says explicitly how large a result really is, and attaches a download for
the complete CSV whenever it had to cut something. The Dino side needs to surface both.

**The client is now the only reason a user would still be misled.** If Dino ignores these
fields it renders 20 rows with no indication that 320 more exist — worse than before,
because the preview is smaller. This is the change that makes the feature real.

---

## 2. What changed in the response

`POST /datachat` still returns:

```json
{ "response": { "type": "...", "value": ... }, "explanation": null, "log_id": 42 }
```

`type` and `value` are **unchanged** — existing rendering code keeps working untouched.
Everything below is additive.

### 2.1 New fields on `type: "dataframe"`

| Field | Type | Meaning |
|---|---|---|
| `total_rows` | int | Rows in the **complete** result. `value` may hold fewer. |
| `total_columns` | int | Columns in the complete result. `value` may hold fewer. |
| `preview_rows` | int | Rows actually present in `value`. |
| `truncated` | bool | `true` when `value` is a subset of the real result. |
| `download_url` | string \| null | Path to the complete CSV. `null` when nothing was cut. |
| `download_filename` | string \| null | Suggested filename, e.g. `sentiment_reviews.csv`. |
| `note` | string | **Optional.** A caveat about the result itself — see §5. |

Current preview limits: **20 rows**, **10 columns**. Do not hardcode them — read
`total_rows`/`preview_rows` and `truncated`. The limits may be tuned server-side.

### 2.2 New fields on `type: "str"`

A text answer may also carry `download_url` + `download_filename` (same meaning). This
happens when the user asks for a file of raw data rather than an on-screen table.

---

## 3. Exact payloads

These are real outputs, not illustrations.

### A. Truncated table — show the banner and the download button

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

### B. Complete table — render exactly as today, no banner, no button

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

### C. Truncated table with a caveat — banner, button, **and** the note

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

### D. Text answer carrying a file

```json
{
  "response": {
    "type": "str",
    "value": "Export pronto: 340 righe.",
    "download_url": "/datachat/export/abc123",
    "download_filename": "dataset.csv"
  },
  "explanation": null
}
```

### E. Plain text — nothing new

```json
{ "response": { "type": "str", "value": "Questo dataset contiene..." }, "explanation": null }
```

---

## 4. Deserialization requirements

### 4.1 Absent and `null` mean the same thing

The two response types are **not** symmetric, by design of the existing normalizer:

- on `dataframe`, `download_url`/`download_filename` are always present, `null` when unused;
- on `str`, they are **absent entirely** when unused;
- `note` is **absent** when there is none — it is never `null`.

Treat "missing key" and "null value" as identical. Do not branch on key presence.

### 4.2 Tolerate unknown fields

Use a tolerant deserializer. Further additive fields may appear without a client release;
a strict `MissingMemberHandling.Error`-style setting would break on the next backend change.

### 4.3 Do not compute `truncated` yourself

Read the flag. `value.Count < total_rows` happens to be equivalent today but will not be
if a future limit applies per-column only.

Suggested model:

```csharp
public sealed class DataChatResponse
{
    public string Type { get; set; }
    public JsonElement Value { get; set; }

    public int? TotalRows { get; set; }
    public int? TotalColumns { get; set; }
    public int? PreviewRows { get; set; }
    public bool Truncated { get; set; }          // absent -> false, correct default
    public string? DownloadUrl { get; set; }
    public string? DownloadFilename { get; set; }
    public string? Note { get; set; }

    public bool HasDownload => !string.IsNullOrEmpty(DownloadUrl);
}
```

---

## 5. Required UI behaviour

### 5.1 When `truncated` is `true`

Show, adjacent to the table and before the user can mistake it for the whole result:

> Showing the first **{preview_rows}** of **{total_rows}** rows.

If `total_columns > ` the number of keys in a `value` row, also say columns were dropped —
the preview keeps only the first 10 columns, so a wide result loses fields silently
otherwise.

Wording is yours; the required content is **preview size, real size, and that this is not
everything**.

### 5.2 When `download_url` is set

Offer a download affordance (button/link) labelled with `download_filename`. Fetch it as
described in §6 — **not** as a plain hyperlink, which will fail to authenticate.

Applies identically to `type: "str"` responses (case D): a text bubble plus a download
button.

### 5.3 When `note` is present

Display it verbatim, near the table, visually distinct from the data (it is a caveat, not
a value). Do not truncate it and do not suppress it when `truncated` is `false` — it can
appear on a complete table.

Notes report things like *"12 rows could not be analyzed: the column has 340 more distinct
values than the 500-value analysis limit. Their sentiment is empty, not neutral."* If this
is hidden, the user reads empty cells as neutral sentiment — the exact class of silent
misreading this whole change exists to remove.

### 5.4 Null cells are meaningful

A `null` in a `sentiment`/`score` (or similar) cell now means **"not analyzed"**, and is
deliberately not a default value. Render it as blank or an explicit "—", never as `0`,
`false`, `"null"`, or a plausible-looking category.

---

## 6. Download endpoint

```
GET {baseUrl}{download_url}
```

`download_url` is a **server-relative path** (`/datachat/export/<token>`). Concatenate it
with the same base URL used for `/datachat`. Do not parse or rebuild the token.

### 6.1 Required headers

| Header | Value |
|---|---|
| `X-API-KEY` | the same key used for `/datachat` |
| `X-USER-EMAIL` | the same email used for `/datachat` |

**This is the single most likely integration mistake:** the endpoint is not publicly
reachable, so opening `download_url` in a browser or a plain `<a href>` will not
authenticate. It must be fetched with headers (`HttpClient`) and the response body saved
or handed to the platform share/save sheet.

### 6.2 Success

- `200`, `Content-Type: text/csv`
- `Content-Disposition: attachment; filename=cities.csv`

Body is the complete result: **all rows and all columns**, not just the preview. Prefer
`download_filename` from the JSON over parsing `Content-Disposition`.

### 6.3 Failures

| Status | Body | Cause | Suggested client behaviour |
|---|---|---|---|
| `400` | `{"error":"Missing X-API-KEY header"}` | header omitted | bug — fix the request |
| `400` | `{"error":"Missing X-USER-EMAIL header"}` | header omitted | bug — fix the request |
| `400` | `{"error":"Agent not active for this Api Key"}` | session ended / server restarted | "the chat session has ended; re-run the query" |
| `403` | **HTML**, not JSON | key invalid or expired | same handling as a 403 from `/datachat` |
| `404` | `{"error":"Export not found or expired"}` | token unknown, or session closed | "this download is no longer available; re-run the query" |

Note the `403` returns an HTML error page — the app registers no JSON error handler, so
this is consistent with every other endpoint. Do not attempt to parse it as JSON.

### 6.4 Lifetime — exports are session-scoped

A token resolves **only** through the engine registered for the API key that created it,
and only while that chat session is alive. It becomes a `404` when:

- `POST /enddatachat` is called,
- the session expires, or
- the server restarts.

Consequences for the client:

- Do not persist `download_url` across app restarts or offer it in chat history after the
  session closes; treat a `404` as expected, not as an error worth a crash report.
- Download during the conversation. If you want the file to outlive the session, save it
  locally at download time.
- Only the most recent **20** exports per session are retained; older tokens 404 as well.

---

## 7. Backward and forward compatibility

- A client that ignores every new field renders exactly as it does today (with a 20-row
  instead of 50-row table). No flag day, no coordinated deploy.
- `type` values are unchanged: `str`, `dataframe`, `image`, `dict`, `text_and_image`.
- Old clients pointed at the new backend: safe, but misleading — they show a smaller
  preview with no indication it is one. Ship §5.1 promptly.
- New clients pointed at an old backend: safe. `truncated` deserializes to `false` and
  `download_url` to `null`, so no banner and no button.

---

## 8. Known limitation, not addressed here

`kind: "error"` and `kind: "text"` both normalize to `type: "str"`, so the client cannot
currently distinguish a failure message from a normal answer, and cannot style errors
differently. Pre-existing behaviour, unchanged by this work. If Dino wants that
distinction, it needs a separate backend change — raise it and we will add a discriminator.

---

## 9. Acceptance checklist

Client-side verification, against a dataset with more than 20 rows and more than 10 columns:

- [ ] Query returning >20 rows shows a preview banner with the correct `total_rows`
- [ ] Query returning ≤20 rows shows **no** banner and **no** download button
- [ ] Query returning >10 columns indicates that columns were dropped
- [ ] Download button fetches with both headers and yields a CSV whose row count equals
      `total_rows` (this is the end-to-end proof)
- [ ] `note` is displayed when present, visually distinct from the table data
- [ ] Null `sentiment`/`score` cells render blank or "—", never `0`/`false`/`"null"`
- [ ] A download attempted after `/enddatachat` shows a friendly "no longer available",
      not a crash or a raw 404
- [ ] Text response carrying `download_url` (ask "scarica tutto il dataset") shows a
      download button alongside the message

---

## 10. Backend reference

| Concern | Location |
|---|---|
| Preview + export construction | `datachat/output_normalizer.py` → `_build_table_response()` |
| Preview limits | `datachat/output_normalizer.py` → `_PREVIEW_ROWS`, `_PREVIEW_COLUMNS` |
| Download endpoint | `routes/datachat.py` → `downloadDatachatExport()` |
| Token issuing / lifetime | `datachat/smolagents_engine.py` → `register_export()`, `resolve_export()`, `close()` |
| Response schema docs | `DEVELOPMENT.md` → "Response shape — DataChat `response`" |

Server-side logs carry `total_rows=`, `truncated=` and `export=` on every
`datachat_request_end` line, and `datachat_export status=` on each download — useful when
reconciling a client-side report.
