"""Behavior checks for dataset validation and deterministic lexical retrieval."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from dataset import DatasetError, Document, dataset_from_json, load_dataset, validate_dataset
from retrieval import BM25Index, tokenize


def example_data():
    return {
        "schema_version": 1,
        "name": "test corpus",
        "corpus": [
            {"id": "d1", "title": "", "text": "apple apple pear"},
            {"id": "d2", "title": "", "text": "apple peach"},
            {"id": "d3", "title": "", "text": "banana"},
        ],
        "queries": [{"id": "q1", "text": "apple"}],
        "qrels": {"q1": ["d1", "d2"]},
    }


class DatasetTests(unittest.TestCase):
    def test_valid_data_is_canonical_and_relevance_is_immutable(self):
        raw = example_data()
        raw["corpus"].reverse()
        dataset = validate_dataset(raw)
        self.assertEqual([d.id for d in dataset.corpus], ["d1", "d2", "d3"])
        self.assertEqual(dataset.qrels["q1"], frozenset({"d1", "d2"}))
        with self.assertRaises(TypeError):
            dataset.qrels["q1"] = frozenset()

    def test_empty_corpus_and_explicit_no_answer_are_valid(self):
        raw = example_data()
        raw["corpus"] = []
        raw["qrels"]["q1"] = []
        self.assertEqual(validate_dataset(raw).corpus, ())

    def test_empty_queries_are_valid(self):
        raw = example_data()
        raw.update(queries=[], qrels={})
        self.assertEqual(validate_dataset(raw).queries, ())

    def test_invalid_relevance_reference_is_rejected(self):
        raw = example_data()
        raw["qrels"]["q1"] = ["missing-document"]
        with self.assertRaisesRegex(DatasetError, "unknown document"):
            validate_dataset(raw)

    def test_every_query_requires_an_explicit_judgment(self):
        raw = example_data()
        raw["qrels"] = {}
        with self.assertRaisesRegex(DatasetError, "exactly"):
            validate_dataset(raw)

    def test_unknown_query_judgment_is_rejected(self):
        raw = example_data()
        raw["qrels"]["unknown-query"] = []
        with self.assertRaises(DatasetError):
            validate_dataset(raw)

    def test_duplicate_ids_and_relevance_are_rejected(self):
        for kind in ("corpus", "queries", "qrels"):
            with self.subTest(kind=kind):
                raw = example_data()
                if kind == "qrels":
                    raw[kind]["q1"] = ["d1", "d1"]
                else:
                    raw[kind].append(copy.deepcopy(raw[kind][0]))
                with self.assertRaisesRegex(DatasetError, "duplicate"):
                    validate_dataset(raw)

    def test_malformed_shapes_types_and_unknown_fields_are_rejected(self):
        changes = [
            {"schema_version": True}, {"schema_version": 2},
            {"name": " "}, {"corpus": {}}, {"queries": None},
            {"qrels": []}, {"unexpected": "field"},
        ]
        for change in changes:
            with self.subTest(change=change):
                raw = example_data()
                raw.update(change)
                with self.assertRaises(DatasetError):
                    validate_dataset(raw)
        for field, value in (("id", "bad id"), ("title", 3), ("text", None)):
            with self.subTest(field=field):
                raw = example_data()
                raw["corpus"][0][field] = value
                with self.assertRaises(DatasetError):
                    validate_dataset(raw)

    def test_json_syntax_duplicate_keys_and_nonfinite_values_are_rejected(self):
        for text in ('{"broken":', '{"name":"a","name":"b"}', '{"value":NaN}'):
            with self.subTest(text=text):
                with self.assertRaises(DatasetError):
                    dataset_from_json(text)

    def test_invalid_unicode_and_excessive_json_nesting_are_rejected(self):
        raw = example_data()
        raw["queries"][0]["text"] = "\ud800"
        with self.assertRaises(DatasetError):
            validate_dataset(raw)
        with self.assertRaises(DatasetError):
            dataset_from_json("[" * 2000 + "0" + "]" * 2000)

    def test_utf8_loader_and_missing_file_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            raw = example_data()
            raw["name"] = "Синтетический набор"
            path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(load_dataset(path).name, raw["name"])
            path.write_bytes(b"\xff")
            with self.assertRaises(DatasetError):
                load_dataset(path)
            with self.assertRaises(DatasetError):
                load_dataset(Path(directory) / "missing.json")


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.documents = validate_dataset(example_data()).corpus
        self.index = BM25Index(self.documents)

    def test_known_ranking_and_independently_calculated_score(self):
        hits = self.index.search("apple", top_k=2)
        self.assertEqual([h.document.id for h in hits], ["d1", "d2"])
        expected = math.log(1 + 1.5 / 2.5) * (2 * 2.2) / (
            2 + 1.2 * (0.25 + 0.75 * 3 / 2)
        )
        self.assertAlmostEqual(hits[0].score, expected, places=12)
        self.assertEqual(hits[0].matched_terms, ("apple",))

    def test_unicode_casefold_punctuation_and_compatibility_normalization(self):
        self.assertEqual(
            tokenize("CAFÉ Straße ４２ hello_world—Привет!"),
            ("café", "strasse", "42", "hello", "world", "привет"),
        )
        self.assertEqual(tokenize("cafe\u0301"), tokenize("café"))

    def test_equal_scores_use_document_id_not_corpus_order(self):
        documents = [Document("z", "", "same"), Document("a", "", "same")]
        for values in (documents, list(reversed(documents))):
            with self.subTest(values=values):
                hits = BM25Index(values).search("same", top_k=1)
                self.assertEqual(hits[0].document.id, "a")

    def test_query_repetition_and_word_order_do_not_change_scores(self):
        self.assertEqual(self.index.search("pear apple", 3), self.index.search("apple pear apple", 3))

    def test_title_is_indexed(self):
        index = BM25Index([Document("d", "titleword", "bodyword")])
        self.assertEqual(index.search("titleword", 1)[0].document.id, "d")

    def test_empty_unknown_and_punctuation_queries_return_no_hits(self):
        for text in ("", "!?", "unseenword"):
            self.assertEqual(self.index.search(text, 3), [])
        self.assertEqual(BM25Index([]).search("apple", 3), [])
        self.assertEqual(BM25Index([Document("d", "", "")]).search("apple", 3), [])

    def test_zero_overlap_documents_do_not_fill_top_k(self):
        self.assertEqual(len(self.index.search("banana", 100)), 1)

    def test_extreme_finite_k1_produces_finite_positive_scores(self):
        for value in (1e308, 5e-324):
            with self.subTest(value=value):
                hits = BM25Index(self.documents, k1=value).search("apple", 3)
                self.assertTrue(all(math.isfinite(hit.score) and hit.score > 0 for hit in hits))

    def test_invalid_top_k_is_rejected_even_for_empty_index(self):
        for value in (0, -1, 1.2, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BM25Index([]).search("", value)

    def test_invalid_parameters_and_duplicate_document_ids_are_rejected(self):
        for kwargs in ({"k1": 0}, {"k1": math.inf}, {"k1": True}, {"b": -0.1}, {"b": 1.1}, {"b": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BM25Index(self.documents, **kwargs)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            BM25Index([Document("d", "", "a"), Document("d", "", "b")])


if __name__ == "__main__":
    unittest.main()
