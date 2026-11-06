"""Stable JSON and escaped, self-contained HTML reports without remote assets."""

from html import escape
import json


def render_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_html(report: dict) -> str:
    dataset = report["dataset"]
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">',
        "<title>Retrieval Bench — " + escape(dataset["name"]) + "</title>",
        """<style>
        :root{color-scheme:light;background:#f4f3ef;color:#172b34;font:16px/1.55 system-ui,sans-serif}
        body{max-width:1100px;margin:0 auto;padding:40px 22px}h1{font-size:clamp(2rem,5vw,3.5rem);line-height:1.1}
        h2{margin-top:2.5rem}h3{margin-bottom:.5rem}a{color:#005b64}code{overflow-wrap:anywhere}
        .eyebrow{letter-spacing:.12em;font-size:.8rem;text-transform:uppercase;color:#005b64}
        .scope,details{background:white;border:1px solid #d7dedc;border-radius:10px;padding:18px;margin:16px 0}
        .scope{border-left:5px solid #007b83}.muted{color:#53636a;font-size:.9rem}.table-wrap{overflow:auto}
        table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #d7dedc}
        th{font-size:.85rem}summary{cursor:pointer;font-weight:650}li{margin:14px 0}.badge{display:inline-block;padding:2px 8px;border-radius:5px;background:#edf2f1;font-size:.8rem}
        .relevant{background:#d9f0e8;color:#07573e}.missed{color:#8c3d1f}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;font-size:.9rem}
        nav a{display:inline-block;margin-right:16px}footer{margin-top:3rem;border-top:1px solid #d7dedc;padding-top:20px}
        @media print{body{padding:0}details{break-inside:avoid}summary{list-style:none}}
        </style></head><body>""",
        '<p class="eyebrow">Offline · inspectable · reproducible</p><h1>Retrieval Bench</h1>',
        "<p>" + escape(dataset["name"]) + "</p>",
        '<p class="scope">' + escape(report["claim_scope"]) + "</p>",
        f'<p>{dataset["document_count"]} documents · {dataset["query_count"]} queries · '
        f'{dataset["answerable_query_count"]} answerable · {dataset["no_answer_query_count"]} explicitly no-answer</p>',
        '<p class="muted">Dataset SHA-256: <code>' + escape(dataset["sha256"]) + "</code></p>",
        "<nav aria-label=\"Configurations\">",
    ]
    for run in report["runs"]:
        name = escape(run["configuration"]["name"])
        parts.append(f'<a href="#config-{name}">{name}</a>')
    parts.extend([
        "</nav><h2>Configuration comparison</h2>",
        '<p class="muted">Recall and MRR use answerable queries only. The cutoff is each configuration\'s top-k. '
        'No-answer false-positive rate counts no-answer queries that returned any document; lower is better. '
        'A missing denominator is shown as n/a.</p>',
        '<div class="table-wrap"><table><thead><tr><th>Configuration</th><th>k / k1 / b</th>'
        '<th>Recall@k</th><th>MRR@k</th><th>No-answer FP rate</th></tr></thead><tbody>',
    ])
    for run in report["runs"]:
        config, metrics = run["configuration"], run["metrics"]
        parts.append(
            f'<tr><td>{escape(config["name"])}</td><td>{config["top_k"]} / {config["k1"]} / {config["b"]}</td>'
            f'<td>{_number(metrics["macro_recall_at_k"])}</td><td>{_number(metrics["mrr_at_k"])}</td>'
            f'<td>{_number(metrics["no_answer_false_positive_rate"])}</td></tr>'
        )
    parts.append("</tbody></table></div>")
    for comparison in report["comparisons"]:
        parts.append(
            '<p class="muted">' + escape(comparison["configuration"]) + " versus " + escape(comparison["against"])
            + f': Δ recall {_number(comparison["macro_recall_at_k_delta"])}, '
            + f'Δ MRR {_number(comparison["mrr_at_k_delta"])}, '
            + f'Δ no-answer FP {_number(comparison["no_answer_false_positive_rate_delta"])}.</p>'
        )
    for run in report["runs"]:
        name = escape(run["configuration"]["name"])
        parts.append(f'<section id="config-{name}"><h2>{name}: per-query evidence</h2>')
        for query in run["queries"]:
            metrics = query["metrics"]
            parts.append('<details open><summary>' + escape(query["id"]) + " · " + escape(query["status"].replace("_", " ")) + "</summary>")
            parts.append("<h3>" + escape(query["text"] or "(empty query)") + "</h3>")
            parts.append('<p class="muted">Judged relevant: ' + escape(", ".join(query["relevant_ids"]) or "none (explicit no-answer)") + "</p>")
            parts.append(f'<p>Recall@k: {_number(metrics["recall_at_k"])} · RR@k: {_number(metrics["reciprocal_rank_at_k"])}</p>')
            if metrics["missed_relevant_ids"]:
                parts.append('<p class="missed">Missed relevant documents: ' + escape(", ".join(metrics["missed_relevant_ids"])) + "</p>")
            if not query["hits"]:
                parts.append("<p>No documents share an indexed query token.</p>")
            else:
                parts.append("<ol>")
                for hit in query["hits"]:
                    tone, label = ("badge relevant", "judged relevant") if hit["is_relevant"] else ("badge", "not judged relevant")
                    parts.append(
                        "<li><strong>" + escape(hit["doc_id"]) + " — " + escape(hit["title"]) + "</strong> "
                        + f'<span class="{tone}">{label}</span>'
                        + f'<p class="muted">BM25 score: {hit["score"]} · matched tokens: '
                        + escape(", ".join(hit["matched_terms"])) + "</p><pre>" + escape(hit["text"]) + "</pre></li>"
                    )
                parts.append("</ol>")
            parts.append("</details>")
        parts.append("</section>")
    parts.append(
        "<footer><p>BM25 lexical baseline. Scores are ranking weights, not probabilities. "
        "Judgments must be exhaustive for this evaluation's interpretation of irrelevant results and no-answer queries. "
        "No generated answers or LLM providers are involved.</p></footer></body></html>\n"
    )
    return "\n".join(parts)
