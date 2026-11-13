import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def _terms(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def search_bm25(query: str, passages: list[dict], *, limit: int = 5) -> list[dict]:
    """Rank matching CMS passages with Okapi BM25."""
    query_terms = set(_terms(query))
    if not passages or not query_terms or limit <= 0:
        return []

    documents = [
        Counter(_terms(f"{passage['heading']} {passage['text']}"))
        for passage in passages
    ]
    average_length = sum(document.total() for document in documents) / len(documents)
    if not average_length:
        return []

    document_frequency = Counter(term for document in documents for term in document)
    k1, b = 1.2, 0.75
    ranked = []
    for passage, document in zip(passages, documents):
        normalization = k1 * (1 - b + b * document.total() / average_length)
        score = 0.0
        for term in sorted(query_terms & document.keys()):
            count = document[term]
            frequency = document_frequency[term]
            weight = math.log1p((len(documents) - frequency + 0.5) / (frequency + 0.5))
            score += weight * count * (k1 + 1) / (count + normalization)
        if score > 0:
            ranked.append({**passage, "score": score})

    return sorted(ranked, key=lambda hit: (-hit["score"], hit["id"]))[:limit]


_CLAUSE_BOUNDARY = re.compile(r"(?:\s*,\s*|\s+)and\s+(?=(?:what|how|where|why|which|when|tell)\b)", re.IGNORECASE)


def search_clause_bm25(query: str, passages: list[dict], *, limit: int = 5) -> list[dict]:
    """Reserve room for the top BM25 result of each explicit question clause."""
    if limit <= 0:
        return []
    clauses = _CLAUSE_BOUNDARY.split(query)
    full_ranking = search_bm25(query, passages, limit=limit)
    if len(clauses) == 1:
        return full_ranking

    by_id = {}
    for clause in clauses[:limit]:
        # A literal HCPCS code in a clause is a stronger signal than its filler words.
        codes = re.findall(r"\b[a-z]\d{4}\b", clause, flags=re.IGNORECASE)
        candidates = [
            passage for passage in passages
            if any(code.casefold() in _terms(passage["text"]) for code in codes)
        ] if codes else passages
        for hit in search_bm25(clause, candidates or passages, limit=1):
            by_id.setdefault(hit["id"], hit)
    for hit in full_ranking:
        by_id.setdefault(hit["id"], hit)
    return list(by_id.values())[:limit]


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class HybridRetriever:
    """Fuse BM25 and cosine similarity on the same versioned passages."""

    def __init__(self, passages: list[dict], *, cache_directory: str):
        from sentence_transformers import SentenceTransformer

        self.passages = {passage["id"]: passage for passage in passages}
        self.encoder = SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            cache_folder=cache_directory,
            device="cpu",
        )
        self.vectors = self.encoder.encode(
            [f"{passage['heading']}\n{passage['text']}" for passage in passages],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, *, limit: int = 5) -> list[dict]:
        if not _terms(query) or limit <= 0:
            return []

        vector = self.encoder.encode(query, normalize_embeddings=True, show_progress_bar=False)
        similarities = self.vectors @ vector
        dense_order = sorted(zip(self.passages, similarities), key=lambda hit: (-hit[1], hit[0]))
        sparse_order = search_bm25(query, list(self.passages.values()), limit=len(self.passages))

        scores = Counter()
        for ranking in ([hit["id"] for hit in sparse_order], [key for key, _ in dense_order]):
            for rank, key in enumerate(ranking, start=1):
                scores[key] += 1 / (60 + rank)

        order = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
        return [{**self.passages[key], "score": scores[key]} for key in order]
