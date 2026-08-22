
# Dense/embedding-based retriever ("Adobe-style" half of the hybrid design).

import re
import math
from collections import Counter
from typing import List, Tuple

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class TfidfProjectionBackend:

    def __init__(self, texts: List[str], dim: int = 128, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self._fit(texts)

    def _fit(self, texts: List[str]):
        tokenized = [tokenize(t) for t in texts]
        df = Counter()
        for toks in tokenized:
            for term in set(toks):
                df[term] += 1
        n_docs = len(texts)
        self.vocab = {term: i for i, term in enumerate(df.keys())}
        self.idf = np.zeros(len(self.vocab))
        for term, i in self.vocab.items():
            self.idf[i] = math.log((1 + n_docs) / (1 + df[term])) + 1

        rng = np.random.default_rng(self.seed)
        # Random projection matrix, fixed seed => reproducible "embeddings"
        self.projection = rng.normal(
            loc=0.0, scale=1.0 / math.sqrt(self.dim), size=(len(self.vocab), self.dim)
        )
        self.doc_vectors = np.vstack([self._tfidf_vector(toks) for toks in tokenized])
        self.doc_embeddings = self.doc_vectors @ self.projection
        self._normalize(self.doc_embeddings)

    def _tfidf_vector(self, tokens: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        if not tokens:
            return vec
        tf = Counter(tokens)
        max_tf = max(tf.values())
        for term, count in tf.items():
            i = self.vocab.get(term)
            if i is not None:
                vec[i] = (count / max_tf) * self.idf[i]
        return vec

    @staticmethod
    def _normalize(mat: np.ndarray):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        mat /= norms

    def embed_query(self, query: str) -> np.ndarray:
        vec = self._tfidf_vector(tokenize(query))
        emb = vec @ self.projection
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    def embed_corpus(self) -> np.ndarray:
        return self.doc_embeddings


class SentenceTransformerBackend:

    def __init__(self, texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)[0]

    def embed_corpus(self) -> np.ndarray:
        return self.doc_embeddings


def get_dense_backend(texts: List[str], dim: int = 128):
    try:
        return SentenceTransformerBackend(texts), "sentence-transformers (all-MiniLM-L6-v2)"
    except Exception:
        return TfidfProjectionBackend(texts, dim=dim), "tfidf-random-projection (fallback, no download)"


class DenseRetriever:
    def __init__(self, chunks, dim: int = 128):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.backend, self.backend_name = get_dense_backend(texts, dim=dim)
        self.doc_embeddings = self.backend.embed_corpus()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_emb = self.backend.embed_query(query)
        sims = self.doc_embeddings @ q_emb
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
