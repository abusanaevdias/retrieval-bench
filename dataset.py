"""Strict, small JSON datasets with explicit binary relevance judgments."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


MAX_DATASET_BYTES = 10 * 1024 * 1024
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


class DatasetError(ValueError):
    """The dataset cannot be read or violates the versioned input contract."""


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str


@dataclass(frozen=True)
class Dataset:
    name: str
    corpus: tuple[Document, ...]
    queries: tuple[Query, ...]
    qrels: Mapping[str, frozenset[str]]


def _object(value: Any, fields: set[str], location: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise DatasetError(f"{location} must contain exactly: {', '.join(sorted(fields))}")
    return value


def _identifier(value: Any, location: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise DatasetError(f"{location} must be a 1–80 character ASCII identifier")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise DatasetError(f"{location} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise DatasetError(f"{location} contains invalid Unicode") from error
    return value


def validate_dataset(raw: Any) -> Dataset:
    """Validate all cross references; an empty judgment list means no answer."""
    raw = _object(raw, {"schema_version", "name", "corpus", "queries", "qrels"}, "dataset")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise DatasetError("schema_version must be integer 1")
    name = _text(raw["name"], "name")
    if not name.strip():
        raise DatasetError("name cannot be blank")

    collections = {}
    for key, fields, constructor in (
        ("corpus", {"id", "title", "text"}, Document),
        ("queries", {"id", "text"}, Query),
    ):
        if not isinstance(raw[key], list):
            raise DatasetError(f"{key} must be an array")
        records = {}
        for position, record in enumerate(raw[key]):
            where = f"{key}[{position}]"
            record = _object(record, fields, where)
            identifier = _identifier(record["id"], f"{where}.id")
            if identifier in records:
                raise DatasetError(f"duplicate {key} id: {identifier}")
            values = {field: _text(record[field], f"{where}.{field}") for field in fields}
            records[identifier] = constructor(**values)
        collections[key] = records

    corpus, queries = collections["corpus"], collections["queries"]
    if not isinstance(raw["qrels"], dict) or set(raw["qrels"]) != set(queries):
        raise DatasetError("qrels must contain exactly one entry for every query id")
    qrels = {}
    for query_id, judgments in raw["qrels"].items():
        if not isinstance(judgments, list):
            raise DatasetError(f"qrels.{query_id} must be an array of document ids")
        relevant = set()
        for document_id in judgments:
            document_id = _identifier(document_id, f"qrels.{query_id}")
            if document_id not in corpus:
                raise DatasetError(f"qrels.{query_id} references unknown document: {document_id}")
            if document_id in relevant:
                raise DatasetError(f"duplicate relevance id in qrels.{query_id}: {document_id}")
            relevant.add(document_id)
        qrels[query_id] = frozenset(relevant)

    return Dataset(
        name=name,
        corpus=tuple(corpus[key] for key in sorted(corpus)),
        queries=tuple(queries[key] for key in sorted(queries)),
        qrels=MappingProxyType({key: qrels[key] for key in sorted(qrels)}),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise DatasetError(f"non-finite JSON value is not permitted: {value}")


def dataset_from_json(text: str) -> Dataset:
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except json.JSONDecodeError as error:
        raise DatasetError(f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}") from error
    except RecursionError as error:
        raise DatasetError("JSON nesting is too deep") from error
    return validate_dataset(raw)


def load_dataset(path: str | Path) -> Dataset:
    try:
        with Path(path).open("rb") as stream:
            content = stream.read(MAX_DATASET_BYTES + 1)
        if len(content) > MAX_DATASET_BYTES:
            raise DatasetError("dataset exceeds the 10 MiB input limit")
        return dataset_from_json(content.decode("utf-8"))
    except (OSError, UnicodeError) as error:
        raise DatasetError(f"cannot read UTF-8 dataset: {error}") from error


def dataset_to_dict(dataset: Dataset) -> dict:
    """A canonical representation for fingerprints and portable exports."""
    return {
        "schema_version": 1,
        "name": dataset.name,
        "corpus": [{"id": doc.id, "title": doc.title, "text": doc.text} for doc in dataset.corpus],
        "queries": [{"id": query.id, "text": query.text} for query in dataset.queries],
        "qrels": {key: sorted(value) for key, value in dataset.qrels.items()},
    }
