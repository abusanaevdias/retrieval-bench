"""Run a lexical retrieval evaluation with Python 3.11+ and no dependencies."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

from dataset import DatasetError, load_dataset
from evaluation import Configuration, evaluate
from reporting import render_html, render_json


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _configuration(value: str, top_k: int) -> Configuration:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("--config must have the form NAME:K1:B, for example standard:1.2:0.75")
    try:
        return Configuration(name=parts[0], k1=float(parts[1]), b=float(parts[2]), top_k=top_k)
    except ValueError as error:
        raise ValueError(f"invalid --config {value!r}: {error}") from error


def _write_atomic(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline BM25 retrieval evaluation; no LLM or provider calls.")
    parser.add_argument("dataset", type=Path, help="UTF-8 JSON dataset with corpus, queries and qrels")
    parser.add_argument("--top-k", type=_positive_integer, default=3, help="positive retrieval cutoff (default: 3)")
    parser.add_argument("--config", action="append", default=[], metavar="NAME:K1:B", help="repeat to compare BM25 parameters; first is the baseline")
    parser.add_argument("--json", dest="json_path", type=Path, help="save JSON; otherwise print JSON to stdout")
    parser.add_argument("--html", dest="html_path", type=Path, help="save a self-contained HTML evidence report")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required")
    try:
        outputs = [path.resolve() for path in (args.json_path, args.html_path) if path is not None]
        if args.dataset.resolve() in outputs or len(outputs) != len(set(outputs)):
            raise ValueError("dataset and output paths must be distinct")
        configs = [_configuration(value, args.top_k) for value in args.config] or [Configuration("bm25", top_k=args.top_k)]
        report = evaluate(load_dataset(args.dataset), configs)
        json_text = render_json(report)
        html_text = render_html(report) if args.html_path is not None else None
        if args.json_path is not None:
            _write_atomic(args.json_path, json_text)
        if args.html_path is not None:
            _write_atomic(args.html_path, html_text)
    except (DatasetError, ValueError, OSError) as error:
        parser.error(str(error))
    if args.json_path is None:
        print(json_text, end="")
    else:
        counts = report["dataset"]
        print(f'Evaluated {counts["query_count"]} queries against {counts["document_count"]} documents across {len(configs)} configurations.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
