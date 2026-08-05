# DataChat test plan — `Feedbacks.csv`

Questions to exercise all 17 tools, with the expected answer for each so you can tell a
working tool from a plausible-looking wrong one. Ask them in Italian, one at a time.

---

## 0. Read this first: the file has two header rows

Row 1 holds the machine names (`id`, `q4`, `q5`, …) and pandas uses it as the header.
**Row 2 holds the question text and is read as data.** So `load_csv_to_dataframe` returns
**805 rows for 804 real responses**, and the extra row carries a long Italian sentence in
every rating column.

Confirmed effects on real tool output:

| Tool | What you will see |
|---|---|
| `row_count` | **805**, not 804 |
| `unique_values("q11")` | a 5th value: `"Nel complesso, quanto sei soddisfatto/a…"` with count 1 |
| `crosstab` on `q16` | an extra column named with the whole question, and a `(vuoto)` row |
| `filter_rows(q15, is_not_empty)` | **237**, because the question text counts as an answer |
| ratings `q4`–`q14` | dtype `object` instead of numeric (means still compute correctly — `to_numeric` coerces the label row to NaN) |

**Every count below is given as "as loaded / real".** Averages are unaffected. Decide
whether to fix this at ingestion (`datachat/dataset_loader.py`, e.g. drop a row whose `id`
equals `"id"`) before or after testing — but do not mistake the off-by-one for a tool bug.

---

## 1. What is actually in the dataset

**805 rows (804 responses) × 32 columns.** Ratings are **1–4**, not 1–5.

| Column | Meaning | Mean |
|---|---|---|
| `q4` | pertinenza dell'argomento | 3.27 |
| `q5` | adeguatezza dei materiali | 3.46 |
| `q6` | chiarezza della presentazione | **3.54** (best) |
| `q7` | livello di interazione | 3.44 |
| `q8` | durata della formazione | 3.16 |
| `q9` | in linea col fabbisogno personale | 3.11 |
| `q10` | coerenza col fabbisogno dell'organizzazione | 3.24 |
| `q11` | **soddisfazione complessiva** | 3.41 |
| `q13` | competenze acquisite | 3.30 |
| `q14` | capacità di applicare quanto appreso | **2.83** (worst) |

- `q16` — recommendation, 4 levels: `probabile` 409, `molto_probabile` 309,
  `poco_probabile` 64, `improbabile` 22
- `q15` — suggerimenti (free text): **237 / 236 filled**, 568 empty (70.6%)
- `q17` — commenti (free text): **213 / 212 filled**, 592 empty (73.6%)
- `project_parent_name` — 8 programmes; `project_name` — 75 editions
- `created_at` — 2025-07-16 → 2026-07-13, 134 distinct days

**Unusable columns** (100% empty): `dinoinvalid`, `project_metric_data`,
`project_sectors_of_intervention`, `project_donors`, `project_start_date`,
`project_end_date`. `user_data_full_name` is `"Admin Dino"` for every row.

**Not a measure:** `project_code` and `project_code_auto` are numeric IDs. They will appear
in a correlation matrix (correctly at ≈0) — the agent should not treat them as scores.

---

## 2. Questions by tool

### `row_count`
1. *Quante risposte ci sono in totale?* → **805** (real: 804 — see §0)

### `describe`
2. *Descrivi il dataset.* → 32 columns covered, **not** capped at 50
3. *Di cosa parla questo dataset?* → prose answer, no raw table dump

### `missing_values`
4. *Quali colonne hanno valori mancanti?* → all 32 columns; the 6 fully-empty ones at 100%
5. *Quante persone non hanno lasciato suggerimenti?* → **568** empty for `q15`

### `unique_values`
6. *Quali sono i possibili valori di q16?* → the 4 levels with counts 409 / 309 / 64 / 22
7. *Quante edizioni diverse ci sono?* → **75** distinct `project_name`

### `filter_rows` — empty / not empty
8. *Quante righe contengono suggerimenti?* → **237 / 236**, and **not** 804 minus something
9. *Elenca le risposte che hanno lasciato un commento.* → 213 rows, 20 previewed, CSV download

### `filter_rows` — text search (new)
10. *Mostrami i suggerimenti che parlano di "ore".* → **33** matches
11. *Quali commenti citano il "formatore"?* → **15** matches
12. *Ci sono suggerimenti che menzionano Excel?* → **5** matches
13. *Quanti commenti contengono "grazie"?* → **120** matches

### `aggregate` — one dimension
14. *Qual è la soddisfazione media per programma?* → 8 rows, ranked:

    ```
    3.573  INTELLIGENZA ARTIFICIALE
    3.552  DIGITAL FUNDRAISING
    3.539  SOFT SKILL
    3.390  FOGLI DI CALCOLO
    3.359  EXCEL
    3.297  RACCOLTA E ANALISI DATI
    3.189  CASE MANAGEMENT
    3.043  MISURAZIONE E RENDICONTAZIONE
    ```

15. *Classifica le 75 edizioni per soddisfazione complessiva.* → **all 75 groups**, not 50,
    plus a small-sample caution (many editions have ~10 responses)

### `crosstab` — two dimensions (new)
16. *Mostrami la soddisfazione media per programma e per livello di raccomandazione.*
    → a wide table, **both** dimensions present
17. *Come si distribuiscono le raccomandazioni in ogni programma, in percentuale?*
    → each row sums to 100
18. *Incrocia la soddisfazione complessiva con la capacità di applicare quanto appreso.*
    → 4×4 table

### `compare_groups` — significance (new)
19. *Il programma INTELLIGENZA ARTIFICIALE è migliore di MISURAZIONE E RENDICONTAZIONE?*
    → diff **+0.529**, p **<1e-06**, **significant** — a real difference
20. *E rispetto a DIGITAL FUNDRAISING?*
    → diff **+0.020**, p **0.5725**, **not significant**. The answer must say the two are
    indistinguishable and refuse to rank them. **This is the most important test in the
    list** — a plain average would have called IA the winner.

### `correlation` — Spearman and matrix (new)
21. *Quali domande sono più correlate con la soddisfazione complessiva?* → one call, ranked:

    ```
    +0.654 q13 (competenze acquisite)      strong
    +0.646 q4  (pertinenza)                strong
    +0.643 q5  (materiali)                 strong
    +0.642 q9  (fabbisogno personale)      strong
    +0.623 q6  (chiarezza)                 strong
    +0.612 q10 (fabbisogno organizzazione) strong
    +0.560 q7  (interazione)               strong
    +0.557 q8  (durata)                    strong
    +0.468 q14 (applicabilità)             moderate  <- weakest link
    ```

22. *C'è correlazione tra la chiarezza del docente e la soddisfazione?* → ≈ **+0.623**
23. *Usa Spearman invece di Pearson.* → `method` reported as `spearman`

### `keywords` — themes in free text (new)
24. *Di cosa parlano i suggerimenti?* → `formazione` 35, `ore` 27, `corso` 22, `tempo` 18,
    `lezioni` 14 … **no** `il`/`che`/`molto`/`più`, and **no `nbsp`** (the file contains 795
    `&nbsp;` entities; seeing `nbsp` as the top term means the markup stripping regressed)
25. *Quali sono le coppie di parole più frequenti nei suggerimenti?* → `ore formazione`,
    `avrei preferito`, `esercizi pratici`, `corso excel`, `impegni lavorativi`
26. *Di cosa parlano i commenti?* → `corso`, `formazione`, `mille`, `formatore`, `utile`

### `sentiment_analysis`
27. *Fai la sentiment analysis dei commenti.* → aggregate counts by default
28. *Elenca ogni commento con sentiment e punteggio.* → **212 rows**, 20 previewed,
    CSV download of all 212. If the export has 50 rows, an upstream cap is back.
29. *Ci sono suggerimenti negativi?* → negatives, with any unscored rows reported as
    empty rather than "neutral"

### `classify`
30. *Raggruppa i suggerimenti in 4 temi.* → cluster labels of **meaningful** words. Labels
    like `il, molto, poca` mean the Italian stopword list regressed.
31. *Classifica i commenti nelle categorie: docente, contenuti, organizzazione, durata.*

### `trend`
32. *Come è andata la soddisfazione mese per mese?* → a series across 2025-07 → 2026-07,
    **all** periods, not capped at 50
33. *Quante risposte abbiamo raccolto per mese?*

### `top_rows` / `sample_rows`
34. *Mostrami 5 risposte di esempio.* → 5 rows, **all 32 columns** available (10 previewed)
35. *Quali sono le 30 risposte con soddisfazione più alta?* → **30**, not silently 20

### `plot`
36. *Fai un grafico a barre della soddisfazione media per programma.*
37. *Mostra la distribuzione delle risposte alla domanda sulla durata.*

### `export_csv`
38. *Esportami tutto il dataset in CSV.* → download link, 805 rows
39. *Dammi un file con solo i commenti.*

---

## 3. Trap questions

These probe the failures we actually hit. Each has a specific wrong answer to watch for.

40. *Quante sono le righe che contengono commenti?*
    → **213 / 212**. Wrong answer to watch for: **328**, which is `804 − 476` (rows where
    *both* `q15` and `q17` are empty) — i.e. the agent silently answering about *either*
    free-text field. It must say **which column** it used.

41. *Elenca tutte le righe che riportano dei suggerimenti.*
    → 237 rows previewed at 20 with a full CSV. Wrong answer: **804**, from filtering
    `value=None` under the old broken `eq` semantics.

42. *Qual è l'edizione migliore?*
    → must name the edition **and** warn that editions have ~10 responses each, so the
    ranking is unreliable. A confident single winner with no caveat is a fail.

43. *Fai la sentiment analysis su tutte le righe elencando commento, score e risultato.*
    → 212 rows in the export. **50 rows is the regression signal** — it means a capped
    upstream tool fed `data=` into `sentiment_analysis`.

44. *La soddisfazione media per programma e per edizione.*
    → two dimensions, so `crosstab` (or `aggregate` with two columns). Wrong answer: a
    one-dimensional table with the second dimension silently dropped.

45. *Quanti hanno risposto "molto_probabile" alla domanda sulla raccomandazione?*
    → **309**

46. *Qual è la domanda con il punteggio più basso?*
    → **q14** (applicabilità), 2.83. The real insight in this dataset: people are satisfied
    (`q11` 3.41) but least able to apply what they learned.

47. *Confronta la soddisfazione tra chi ha lasciato un suggerimento e chi non l'ha fatto.*
    → chains `filter_rows` → `compare_groups`; watch that neither side gets truncated.

48. *Quanti record ci sono nella colonna dinoinvalid?*
    → must report the column as entirely empty, not invent a number.

---

## 4. Quick pass — 8 questions

If you only have ten minutes, these cover every new capability and every past regression:

1. *Quante risposte ci sono in totale?* → 805
2. *Quante sono le righe che contengono commenti?* → 213, naming the column
3. *Mostrami i suggerimenti che parlano di "ore".* → 33
4. *Qual è la soddisfazione media per programma?* → 8 rows, IA top at 3.573
5. *La soddisfazione media per programma e per livello di raccomandazione.* → 2-D table
6. *INTELLIGENZA ARTIFICIALE è migliore di DIGITAL FUNDRAISING?* → **no**, p=0.5725
7. *Di cosa parlano i suggerimenti?* → `formazione`, `ore`, `corso` — no `nbsp`, no `che`
8. *Elenca ogni commento con sentiment e punteggio.* → 212 in the CSV, not 50

---

## 5. What to watch in the server log

Every request ends with:

```
datachat_request_end … total_rows=213 truncated=True export=yes
```

- `total_rows` is what the tool handed over — if it is 50, an upstream tool capped the
  result and the preview layer is innocent.
- Each tool logs its own counts, e.g. `[datachat][filter_rows_tool] … rows=213`, and
  `[datachat][keywords_tool] … answers=213 terms=…`.
- `[datachat] Failed to log token usage: Cost not found for provider: Mistral and model: …`
  is a separate known issue — token accounting is not being written for this model.
