"""Metric arithmetic, no-answer handling, reporting, and CLI contracts."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dataset import validate_dataset
from evaluation import Configuration, evaluate, score_ranking
from reporting import render_html, render_json
from test_core import example_data


ROOT = Path(__file__).resolve().parent


def evaluation_data():
    raw = example_data()
    raw["queries"] += [
        {"id": "q2", "text": "unseenword"},
        {"id": "q3", "text": "apple"},
        {"id": "q4", "text": "unseenword"},
        {"id": "q5", "text": ""},
    ]
    raw["qrels"].update(q2=["d3"], q3=[], q4=[], q5=[])
    return raw


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_first_relevant_rank_and_all_judged_relevant_ids(self):
        score = score_ranking(["irrelevant", "a", "b"], {"a", "missing"})
        self.assertEqual(score["recall_at_k"], 0.5)
        self.assertEqual(score["reciprocal_rank_at_k"], 0.5)
        self.assertEqual(score["first_relevant_rank"], 2)
        self.assertEqual(score["missed_relevant_ids"], ["missing"])

    def test_empty_retrieval_scores_zero_for_answerable_query(self):
        score = score_ranking([], {"a"})
        self.assertEqual(score["recall_at_k"], 0)
        self.assertEqual(score["reciprocal_rank_at_k"], 0)
        self.assertIsNone(score["first_relevant_rank"])

    def test_no_answer_metrics_are_undefined_and_duplicates_are_rejected(self):
        score = score_ranking(["a"], set())
        self.assertIsNone(score["recall_at_k"])
        self.assertIsNone(score["reciprocal_rank_at_k"])
        with self.assertRaises(ValueError):
            score_ranking(["a", "a"], {"a"})

    def test_macro_metrics_exclude_no_answer_queries_and_report_false_positives(self):
        report = evaluate(validate_dataset(evaluation_data()), [Configuration("one", top_k=1)])
        metrics = report["runs"][0]["metrics"]
        self.assertEqual(metrics["macro_recall_at_k"], 0.25)
        self.assertEqual(metrics["mrr_at_k"], 0.5)
        self.assertEqual(metrics["no_answer_false_positive_rate"], 1 / 3)
        queries = {q["id"]: q for q in report["runs"][0]["queries"]}
        self.assertEqual(queries["q1"]["hits"][0]["doc_id"], "d1")
        self.assertTrue(queries["q1"]["hits"][0]["is_relevant"])
        self.assertEqual(queries["q3"]["status"], "no_answer_false_positive")
        self.assertEqual(queries["q4"]["status"], "no_answer_abstained")

    def test_compare_configurations_reports_metric_deltas(self):
        report = evaluate(validate_dataset(example_data()), [
            Configuration("one", top_k=1), Configuration("two", top_k=2, b=0)
        ])
        self.assertEqual(report["comparisons"][0]["against"], "one")
        self.assertEqual(report["comparisons"][0]["macro_recall_at_k_delta"], 0.5)
        self.assertEqual(report["comparisons"][0]["mrr_at_k_delta"], 0)

    def test_empty_corpus_and_empty_query_set_have_explicit_null_denominators(self):
        raw = example_data()
        raw["corpus"], raw["queries"], raw["qrels"] = [], [], {}
        report = evaluate(validate_dataset(raw), [Configuration("empty")])
        self.assertIsNone(report["runs"][0]["metrics"]["macro_recall_at_k"])
        self.assertIsNone(report["runs"][0]["metrics"]["mrr_at_k"])
        self.assertIsNone(report["runs"][0]["metrics"]["no_answer_false_positive_rate"])
        raw["queries"] = [{"id": "q", "text": "any"}]
        raw["qrels"] = {"q": []}
        report = evaluate(validate_dataset(raw), [Configuration("empty")])
        self.assertEqual(report["runs"][0]["metrics"]["no_answer_false_positive_rate"], 0)

    def test_canonical_fingerprint_ignores_input_record_order_but_tracks_judgments(self):
        original = evaluation_data()
        reordered = copy.deepcopy(original)
        reordered["corpus"].reverse()
        reordered["queries"].reverse()
        reordered["qrels"]["q1"].reverse()
        config = [Configuration("same")]
        first = evaluate(validate_dataset(original), config)
        self.assertEqual(render_json(first), render_json(evaluate(validate_dataset(reordered), config)))
        reordered["qrels"]["q1"] = ["d1"]
        self.assertNotEqual(first["dataset"]["sha256"], evaluate(validate_dataset(reordered), config)["dataset"]["sha256"])

    def test_configuration_validation(self):
        for options in ({"name": "bad name"}, {"name": "ok", "top_k": 0}, {"name": "ok", "b": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Configuration(**options)
        dataset = validate_dataset(example_data())
        with self.assertRaises(ValueError):
            evaluate(dataset, [])
        with self.assertRaises(ValueError):
            evaluate(dataset, [Configuration("same"), Configuration("same")])

    def test_html_escapes_dataset_content_and_is_self_contained(self):
        raw = example_data()
        raw["name"] = '<script>alert("x")</script>'
        raw["corpus"][0]["title"] = "<img src=x onerror=alert(1)>"
        raw["corpus"][0]["text"] += " </details><script>bad()</script>"
        report = evaluate(validate_dataset(raw), [Configuration("safe")])
        html = render_html(report)
        self.assertNotIn("<script", html)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Content-Security-Policy", html)
        self.assertNotIn('href="http', html)
        self.assertIn("apple", html)
        hits = json.loads(render_json(report))["runs"][0]["queries"][0]["hits"]
        self.assertEqual({hit["doc_id"] for hit in hits}, {"d1", "d2"})


class CLITests(unittest.TestCase):
    def run_cli(self, *arguments, hash_seed="1"):
        return subprocess.run(
            [sys.executable, str(ROOT / "cli.py"), *map(str, arguments)],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )

    def test_cli_outputs_are_reproducible_across_processes_and_hash_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "input.json"
            data.write_text(json.dumps(evaluation_data()), encoding="utf-8")
            paths = []
            for seed in ("1", "999"):
                output, html = base / f"out-{seed}.json", base / f"out-{seed}.html"
                result = self.run_cli(data, "--top-k", 2, "--config", "standard:1.2:0.75", "--config", "no-length:1.2:0", "--json", output, "--html", html, hash_seed=seed)
                self.assertEqual(result.returncode, 0, result.stderr)
                paths.append((output, html))
            self.assertEqual(paths[0][0].read_bytes(), paths[1][0].read_bytes())
            self.assertEqual(paths[0][1].read_bytes(), paths[1][1].read_bytes())
            self.assertEqual(len(json.loads(paths[0][0].read_text())["runs"]), 2)

    def test_cli_stdout_is_json_when_no_json_path_is_supplied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(example_data()), encoding="utf-8")
            result = self.run_cli(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["method"], "BM25 lexical baseline")

    def test_cli_rejects_malformed_inputs_and_dangerous_output_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "input.json"
            data.write_text(json.dumps(example_data()), encoding="utf-8")
            for args in (("--top-k", "0"), ("--config", "bad"), ("--config", "name:nan:0"), ("--json", data), ("--json", data.parent / "same", "--html", data.parent / "same")):
                with self.subTest(args=args):
                    result = self.run_cli(data, *args)
                    self.assertEqual(result.returncode, 2)
                    self.assertNotIn("Traceback", result.stderr)
            data.write_text("{bad-json", encoding="utf-8")
            self.assertEqual(self.run_cli(data).returncode, 2)


if __name__ == "__main__":
    unittest.main()
