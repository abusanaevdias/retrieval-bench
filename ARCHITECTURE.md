# Architecture and limits

## Data flow

```text
UTF-8 dataset
   → strict schema and reference checks
   → canonical document/query order
   → one BM25 inverted index per configuration
   → top-k rankings and binary-judgment metrics
   → comparison deltas + JSON / HTML evidence
```

The project deliberately has flat modules and no dependencies. `dataset.py` owns input validation, `retrieval.py` owns matching, `evaluation.py` owns metric semantics, and `reporting.py` owns presentation. The CLI coordinates these pieces without accessing the network.

## Retrieval choices

Tokenization applies Unicode NFKC normalization and case folding, then extracts Unicode alphanumeric runs. Underscores, hyphens and punctuation split terms. Accents are preserved; there is no stemming, language detection or stop-word list. This makes behavior explicit, but languages without whitespace segmentation and morphological variants are poorly served.

Document titles and bodies are concatenated with equal weight. Each document contributes a token-frequency map, a length, and postings for its terms. Duplicate query terms are ignored; the query is a set of normalized terms.

For each matching term, the score contribution is:

```text
idf = log(1 + (N - df + 0.5) / (df + 0.5))
length_factor = 1 - b + b * document_length / average_document_length
contribution = idf * tf * (k1 + 1) / (tf + k1 * length_factor)
```

`N` includes empty documents. If every document is empty, there are no postings and no division by zero. `k1` controls term-frequency saturation and `b` controls length normalization. For large `k1`, the algebra is rearranged to avoid overflowing an intermediate multiplication. Contributions are summed in sorted term order with `math.fsum`.

Only matching postings are scored. Results are sorted by descending unrounded score, then ascending document ID for exact ties. The report serializes scores with 12 significant digits after ranking. It does not use scores as confidence probabilities or infer factual support from token overlap.

## Evaluation choices

Relevance is binary and query-specific. Each query must have a label list, including an explicit empty list for no-answer cases. A nonempty list defines an answerable query. Its recall and reciprocal rank contribute equally to macro aggregates; queries with many relevant documents do not receive a larger aggregate weight.

An empty list contributes only to no-answer false-positive accounting. Any returned document is a false positive for that query. The program does not assume that “no labeled positive” means “not annotated.” This strict distinction avoids silently converting missing labels into no-answer evidence, but requires carefully authored exhaustive judgments.

Comparisons subtract the first configuration's metrics from each later configuration. Positive recall/MRR deltas are better; negative no-answer false-positive deltas are better. No statistical significance, confidence interval or automatic winner is reported. The synthetic fixture is too small for such conclusions.

The Python API supports different top-k values in separate `Configuration` objects; the CLI uses one common cutoff for parameter comparisons. A larger cutoff can improve recall while introducing irrelevant results. Precision, nDCG, latency and memory benchmarks are outside the current report.

## Reproducibility and report handling

Validated documents, queries and relevance lists are sorted before fingerprinting and evaluation. The dataset SHA-256 covers canonical content, including judgments and the dataset name; it is not a hash of the original file bytes. Configuration order is retained so the requested baseline remains explicit.

Reports contain no timestamps, absolute input paths, random identifiers or execution durations. CLI tests compare JSON and HTML bytes across fresh processes and different hash seeds. This verifies repeatability in the tested Python environment. Floating-point behavior and Unicode tables may differ between Python versions; near ties across other runtimes are not guaranteed to remain identical.

All dataset strings are HTML-escaped. The HTML has no JavaScript or remote resources and includes a restrictive Content Security Policy. Returned document text is included in full. Treat generated reports as containing the input corpus excerpts and query text; this tool does not redact private content.

Each output uses a temporary file in the destination directory followed by `os.replace`. A single file is replaced atomically on the same filesystem. Writing JSON and HTML is not a multi-file transaction: an I/O error on the second output can leave the first successfully written. Parent directories are not created implicitly.

## Intended scale and boundaries

- Small offline corpora fit in memory. Input is capped at 10 MiB; this is an input-size bound, not a throughput claim.
- There is no persistent index, service API, embedding store, vector search, model integration or production deployment.
- Exhaustive labels are assumed. Unjudged-but-relevant documents depress measured quality.
- Synonyms, negation, context, authoritativeness and answer entailment are not modeled. The bundled fixture exposes a synonym miss and a no-answer false positive rather than hiding them.
- No provider credentials are read, and no external calls occur during evaluation or tests.
- Test success establishes the implemented contracts and fixture behavior. It does not establish real-world retrieval quality or LLM answer correctness.
