"""A deterministic, in-memory BM25 lexical retrieval baseline."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Sequence
import unicodedata

from dataset import Document


TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """NFKC + casefold; Unicode alphanumeric runs, no stemming/stop words."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(TOKEN_PATTERN.findall(normalized))


def _finite_number(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float
    matched_terms: tuple[str, ...]


class BM25Index:
    """Index title + body equally; only positive-overlap documents are returned."""

    def __init__(self, documents: Sequence[Document], k1: float = 1.2, b: float = 0.75):
        _finite_number(k1, "k1")
        _finite_number(b, "b")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be > 0 and b must be between 0 and 1")
        self.k1, self.b = k1, b
        self.documents = tuple(sorted(documents, key=lambda document: document.id))
        if len({document.id for document in self.documents}) != len(self.documents):
            raise ValueError("duplicate document id")
        self.lengths = []
        self.postings = defaultdict(list)
        for position, document in enumerate(self.documents):
            counts = Counter(tokenize(document.title + " " + document.text))
            self.lengths.append(sum(counts.values()))
            for term in sorted(counts):
                self.postings[term].append((position, counts[term]))
        self.average_length = sum(self.lengths) / len(self.documents) if self.documents else 0.0

    def search(self, text: str, top_k: int = 3) -> list[Hit]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        contributions = defaultdict(list)
        matched = defaultdict(list)
        count = len(self.documents)
        for term in sorted(set(tokenize(text))):
            posting = self.postings.get(term, ())
            if not posting:
                continue
            document_frequency = len(posting)
            idf = math.log1p((count - document_frequency + 0.5) / (document_frequency + 0.5))
            for position, frequency in posting:
                length_factor = 1 - self.b + self.b * self.lengths[position] / self.average_length
                if self.k1 <= 1:
                    weight = frequency * (self.k1 + 1) / (frequency + self.k1 * length_factor)
                else:
                    # Divide through by k1 before multiplication can overflow.
                    inverse = 1 / self.k1
                    weight = (1 + inverse) / (frequency * inverse + length_factor) * frequency
                contributions[position].append(idf * weight)
                matched[position].append(term)
        hits = [
            Hit(self.documents[position], math.fsum(values), tuple(matched[position]))
            for position, values in contributions.items()
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.document.id))
        return hits[:top_k]
