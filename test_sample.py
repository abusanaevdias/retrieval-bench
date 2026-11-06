"""The shipped synthetic example is evaluated, not a hand-authored metric table."""

from pathlib import Path
import unittest

from dataset import load_dataset
from evaluation import Configuration, evaluate
from reporting import render_html, render_json


ROOT = Path(__file__).resolve().parent


def sample_report():
    return evaluate(load_dataset(ROOT / "sample.json"), [
        Configuration("standard", top_k=2),
        Configuration("no-length", top_k=2, b=0.0),
    ])


class SampleTests(unittest.TestCase):
    def test_bundled_sample_has_a_visible_synonym_miss_and_no_answer_false_positive(self):
        report = sample_report()
        self.assertEqual(report["dataset"]["document_count"], 8)
        self.assertEqual(report["dataset"]["query_count"], 8)
        for run in report["runs"]:
            self.assertAlmostEqual(run["metrics"]["macro_recall_at_k"], 5 / 6)
            self.assertAlmostEqual(run["metrics"]["mrr_at_k"], 5 / 6)
            self.assertEqual(run["metrics"]["no_answer_false_positive_rate"], 0.5)
            queries = {query["id"]: query for query in run["queries"]}
            self.assertEqual(queries["q-synonym"]["hits"], [])
            self.assertEqual(queries["q-bitcoin"]["status"], "no_answer_false_positive")
            self.assertEqual(queries["q-astronomy"]["status"], "no_answer_abstained")

    def test_committed_sample_reports_match_a_fresh_evaluation(self):
        report = sample_report()
        self.assertEqual((ROOT / "sample-report.json").read_text(encoding="utf-8"), render_json(report))
        self.assertEqual((ROOT / "sample-report.html").read_text(encoding="utf-8"), render_html(report))


if __name__ == "__main__":
    unittest.main()
