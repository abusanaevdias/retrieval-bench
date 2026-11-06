# Retrieval Bench

Offline retrieval evaluation with an inspectable BM25 lexical baseline. Give it a document corpus, queries and explicit relevance judgments; compare configurations and export stable JSON plus a self-contained HTML evidence report.

**Scope:** a local demonstrator built with the Python 3.11+ standard library. It evaluates document retrieval against supplied labels. It does not generate answers, assess LLM factual correctness, call providers or benchmark a model.

## Try it

No installation or API keys are needed. From this directory:

```bash
python3.11 cli.py sample.json --top-k 2 \
  --config standard:1.2:0.75 \
  --config no-length:1.2:0 \
  --json report.json --html report.html
```

Open `report.html` locally. Each query shows its judgment, retrieved document text, matched tokens, scores, missed relevant IDs and no-answer outcome. The HTML uses inline CSS, native expandable sections and no scripts, fonts or remote assets.

Omit `--json` to print the complete JSON report to stdout. Output directories must already exist. Output files are replaced individually; input and output paths must be distinct.

`--top-k` is a positive integer, defaulting to 3. Repeat `--config NAME:K1:B` to compare BM25 settings at the same cutoff. Names must be unique; `k1` must be positive and finite, and `b` must be in `[0, 1]`. The first configuration is the comparison baseline. Without `--config`, the defaults are `k1=1.2`, `b=0.75`. The Python API also allows a separate cutoff per configuration.

## Computed synthetic example

The bundled [`sample.json`](sample.json) is an invented operations handbook: eight documents and eight queries, with six answerable and two explicitly no-answer queries. All content and labels are synthetic.

The command above produces these results for both configurations:

| Metric | Measured value |
| --- | --- |
| Macro recall@2, answerable queries | 0.8333333333333334 |
| MRR@2, answerable queries | 0.8333333333333334 |
| No-answer false-positive rate | 0.5 |

The two configurations have equal aggregate metrics on this tiny fixture. This does not establish that either setting is better on another corpus.

Two deliberate failures remain visible: `privileged employees` misses the relevant staff document because it uses different words, while the unsupported Bitcoin refund query retrieves a document that explicitly says it has no cryptocurrency procedure. Exact token overlap cannot resolve either semantic issue.

[`sample-report.json`](sample-report.json) and [`sample-report.html`](sample-report.html) are actual generated outputs. `test_sample.py` regenerates them in memory and checks equality with the committed artifacts.

## Dataset contract

Use UTF-8 JSON, at most 10 MiB, with exactly these root fields:

```json
{
  "schema_version": 1,
  "name": "My small corpus",
  "corpus": [
    {"id": "doc-a", "title": "Retry policy", "text": "Retry a transient 503 response."}
  ],
  "queries": [
    {"id": "q-a", "text": "retry 503"},
    {"id": "q-b", "text": "orbital telescope"}
  ],
  "qrels": {"q-a": ["doc-a"], "q-b": []}
}
```

Every query requires one explicit `qrels` entry. An empty array means **judged to have no relevant document in this corpus**, rather than “not labeled yet.” Nonlisted documents are treated as irrelevant; labels therefore need to be exhaustive for this interpretation.

IDs are unique within their collection, 1–80 ASCII characters, start with a letter or digit, and then use letters, digits, `_`, `.` or `-`. All titles and texts are strings; empty texts are permitted. Unknown fields, malformed JSON, duplicate object keys, duplicate IDs, repeated judgments, unsupported schema versions and relevance IDs absent from the corpus are rejected. Empty corpora and query lists are supported.

## Metric definitions

- **Recall@k:** retrieved relevant documents divided by all judged relevant documents for that query. The report averages per-query recall across answerable queries.
- **MRR@k:** average of `1 / first_relevant_rank` across answerable queries; a relevant document outside the returned top-k contributes zero.
- **No-answer false-positive rate:** fraction of explicitly no-answer queries returning any positive-overlap document. Lower is better. No-answer queries are excluded from recall and MRR denominators.
- **Missing denominator:** JSON `null`, displayed as `n/a` in HTML. An empty corpus or empty query set never produces NaN or a misleading perfect score.

The retrieval engine returns only documents sharing a query token; it does not fill unused ranks with zero-score documents. Abstention means zero lexical overlap, not a calibrated confidence decision. BM25 scores are ranking weights, not probabilities.

## Verify

```bash
python3.11 -m unittest discover -v
```

Tests cover a known ranking and independently calculated score, Unicode tokenization, deterministic ties, metric arithmetic, explicit no-answer cases, empty inputs, invalid datasets and relevance IDs, escaped HTML, output-path collisions, shipped report regeneration, and CLI byte reproducibility across separate processes with different hash seeds.

To regenerate the committed examples, use the first command with output names `sample-report.json` and `sample-report.html`.

## Read the implementation

| File | Responsibility |
| --- | --- |
| `dataset.py` | Strict schema validation, immutable records and canonical serialization |
| `retrieval.py` | Unicode tokenization, inverted index and BM25 ranking |
| `evaluation.py` | Query metrics, configuration runs and comparison deltas |
| `reporting.py` | Deterministic JSON and escaped standalone HTML |
| `cli.py` | Argument parsing and local file output |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for scoring details, assumptions and limitations. This is an original small implementation of an established retrieval algorithm; BM25 itself is not an original algorithm. Licensed under [MIT](LICENSE).
