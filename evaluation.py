"""Compare retrieval configurations using explicit, binary relevance labels."""

from dataclasses import asdict, dataclass
import hashlib
import json
from statistics import fmean
from typing import Sequence

from dataset import Dataset, ID_PATTERN, dataset_to_dict
from retrieval import BM25Index


@dataclass(frozen=True)
class Configuration:
    name: str
    top_k: int = 3
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self):
        if not isinstance(self.name, str) or not ID_PATTERN.fullmatch(self.name):
            raise ValueError("configuration name must be a 1–80 character ASCII identifier")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        BM25Index((), k1=self.k1, b=self.b)


def score_ranking(retrieved_ids: Sequence[str], relevant_ids: set[str] | frozenset[str]) -> dict:
    """Recall and reciprocal rank at the supplied ranking's cutoff."""
    if len(set(retrieved_ids)) != len(retrieved_ids):
        raise ValueError("ranking contains duplicate document ids")
    relevant = set(relevant_ids)
    found = relevant.intersection(retrieved_ids)
    first = next((rank for rank, doc_id in enumerate(retrieved_ids, 1) if doc_id in relevant), None)
    return {
        "recall_at_k": len(found) / len(relevant) if relevant else None,
        "reciprocal_rank_at_k": (1 / first if first else 0.0) if relevant else None,
        "first_relevant_rank": first,
        "relevant_retrieved": len(found),
        "missed_relevant_ids": sorted(relevant - found),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _delta(value: float | None, baseline: float | None) -> float | None:
    return value - baseline if value is not None and baseline is not None else None


def evaluate(dataset: Dataset, configurations: Sequence[Configuration]) -> dict:
    if not configurations:
        raise ValueError("at least one configuration is required")
    if len({config.name for config in configurations}) != len(configurations):
        raise ValueError("configuration names must be unique")
    canonical = json.dumps(dataset_to_dict(dataset), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    answerable_count = sum(bool(dataset.qrels[query.id]) for query in dataset.queries)
    report = {
        "schema_version": 1,
        "tool": "retrieval-bench",
        "method": "BM25 lexical baseline",
        "claim_scope": "Offline retrieval against supplied judgments; not LLM factual correctness or a model benchmark.",
        "dataset": {
            "name": dataset.name,
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "document_count": len(dataset.corpus),
            "query_count": len(dataset.queries),
            "answerable_query_count": answerable_count,
            "no_answer_query_count": len(dataset.queries) - answerable_count,
        },
        "baseline_configuration": configurations[0].name,
        "runs": [],
        "comparisons": [],
    }
    for config in configurations:
        index = BM25Index(dataset.corpus, k1=config.k1, b=config.b)
        queries = []
        recalls, reciprocal_ranks, no_answer_false_positives = [], [], []
        for query in dataset.queries:
            relevant = dataset.qrels[query.id]
            hits = index.search(query.text, top_k=config.top_k)
            metrics = score_ranking([hit.document.id for hit in hits], relevant)
            if relevant:
                recalls.append(metrics["recall_at_k"])
                reciprocal_ranks.append(metrics["reciprocal_rank_at_k"])
                status = "answerable_hit" if metrics["relevant_retrieved"] else "answerable_missed"
            else:
                no_answer_false_positives.append(float(bool(hits)))
                status = "no_answer_false_positive" if hits else "no_answer_abstained"
            queries.append({
                "id": query.id,
                "text": query.text,
                "status": status,
                "relevant_ids": sorted(relevant),
                "metrics": metrics,
                "hits": [{
                    "rank": rank,
                    "doc_id": hit.document.id,
                    "title": hit.document.title,
                    "text": hit.document.text,
                    "score": float(f"{hit.score:.12g}"),
                    "matched_terms": list(hit.matched_terms),
                    "is_relevant": hit.document.id in relevant,
                } for rank, hit in enumerate(hits, 1)],
            })
        report["runs"].append({
            "configuration": asdict(config),
            "metrics": {
                "macro_recall_at_k": _mean_or_none(recalls),
                "mrr_at_k": _mean_or_none(reciprocal_ranks),
                "no_answer_false_positive_rate": _mean_or_none(no_answer_false_positives),
                "queries_with_hits": sum(bool(query["hits"]) for query in queries),
            },
            "queries": queries,
        })
    baseline = report["runs"][0]
    for run in report["runs"][1:]:
        report["comparisons"].append({
            "configuration": run["configuration"]["name"],
            "against": baseline["configuration"]["name"],
            **{
                metric + "_delta": _delta(run["metrics"][metric], baseline["metrics"][metric])
                for metric in ("macro_recall_at_k", "mrr_at_k", "no_answer_false_positive_rate")
            },
        })
    return report
