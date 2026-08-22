
# Combines the BM25 sparse retriever and the dense retriever using
# Reciprocal Rank Fusion (RRF), then applies a lightweight reranking pass.


from typing import List, Dict
from collections import defaultdict


def reciprocal_rank_fusion(
    sparse_ranked: List[tuple], dense_ranked: List[tuple], k: int = 60
) -> List[tuple]:

    fused_scores: Dict[int, float] = defaultdict(float)

    for rank, (idx, _score) in enumerate(sparse_ranked):
        fused_scores[idx] += 1.0 / (k + rank + 1)

    for rank, (idx, _score) in enumerate(dense_ranked):
        fused_scores[idx] += 1.0 / (k + rank + 1)

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "much", "my", "of", "on",
    "or", "that", "the", "this", "to", "used", "what", "when", "where",
    "which", "who", "will", "with", "you", "your",
}


def lexical_overlap_rerank(query: str, chunks, candidate_idxs: List[int], top_k: int):
   
# earlier it was favouring the long  the text , but we fixed it by restricting the overlap count to content words.
    import re

    q_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - _STOPWORDS

    def overlap_score(idx):
        c_terms = set(re.findall(r"[a-z0-9]+", chunks[idx].text.lower())) - _STOPWORDS
        return len(q_terms & c_terms)

    scored = list(enumerate(candidate_idxs))  # (fusion_rank, chunk_idx)
    reranked = sorted(scored, key=lambda pair: (overlap_score(pair[1]), -pair[0]), reverse=True)
    return [idx for _, idx in reranked[:top_k]]


class HybridRetriever:
    def __init__(self, sparse_retriever, dense_retriever, chunks):
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.chunks = chunks

    def search(
        self,
        query: str,
        sparse_k: int = 10,
        dense_k: int = 10,
        fusion_k: int = 60,
        final_k: int = 5,
        rerank: bool = True,
    ):
        sparse_ranked = self.sparse.search(query, top_k=sparse_k)
        dense_ranked = self.dense.search(query, top_k=dense_k)
        fused = reciprocal_rank_fusion(sparse_ranked, dense_ranked, k=fusion_k)
        candidate_idxs = [idx for idx, _ in fused][: max(final_k * 3, final_k)]

        if rerank:
            final_idxs = lexical_overlap_rerank(query, self.chunks, candidate_idxs, final_k)
        else:
            final_idxs = candidate_idxs[:final_k]

        return [self.chunks[i] for i in final_idxs]
