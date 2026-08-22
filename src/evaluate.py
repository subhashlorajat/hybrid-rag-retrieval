
# Quantitative retrieval evaluation of three conditions on the 22-query 
# hand-labelled set (data/eval_queries.py):


import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

from chunking import build_chunk_corpus
from sparse_retriever import SparseRetriever
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever
from eval_queries import EVAL_SET

TOP_K = 3


def reciprocal_rank(ranked_doc_ids, expected_doc_id):
    for rank, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id == expected_doc_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_doc_ids, expected_doc_id, k):
    return 1.0 if expected_doc_id in ranked_doc_ids[:k] else 0.0


def evaluate_condition(name, results_by_query):
    recalls, rrs = [], []
    for query, expected_doc_id, ranked_doc_ids in results_by_query:
        recalls.append(recall_at_k(ranked_doc_ids, expected_doc_id, TOP_K))
        rrs.append(reciprocal_rank(ranked_doc_ids, expected_doc_id))
    return {
        "condition": name,
        "n_queries": len(results_by_query),
        f"recall@{TOP_K}": round(sum(recalls) / len(recalls), 3),
        "mrr": round(sum(rrs) / len(rrs), 3),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(here, "..", "data", "raw_corpus")
    results_dir = os.path.join(here, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    chunks = build_chunk_corpus(raw_dir, chunk_size=150, overlap=30)
    sparse = SparseRetriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(sparse, dense, chunks)

    print(f"Dense backend in use: {dense.backend_name}")
    print(f"Corpus: {len(chunks)} chunks across "
          f"{len(set(c.doc_id for c in chunks))} documents")
    print(f"Evaluation set: {len(EVAL_SET)} queries\n")

    bm25_results, dense_results, hybrid_results = [], [], []
    per_query_log = []

    for query, expected_doc_id in EVAL_SET:
        # BM25-only
        sparse_hits = sparse.search(query, top_k=TOP_K)
        sparse_doc_ids = [chunks[i].doc_id for i, _ in sparse_hits]

        # Dense-only
        dense_hits = dense.search(query, top_k=TOP_K)
        dense_doc_ids = [chunks[i].doc_id for i, _ in dense_hits]

        # Hybrid
        hybrid_chunks = hybrid.search(query, final_k=TOP_K)
        hybrid_doc_ids = [c.doc_id for c in hybrid_chunks]

        bm25_results.append((query, expected_doc_id, sparse_doc_ids))
        dense_results.append((query, expected_doc_id, dense_doc_ids))
        hybrid_results.append((query, expected_doc_id, hybrid_doc_ids))

        per_query_log.append({
            "query": query,
            "expected_doc_id": expected_doc_id,
            "bm25_top3_docs": sparse_doc_ids,
            "dense_top3_docs": dense_doc_ids,
            "hybrid_top3_docs": hybrid_doc_ids,
        })

    summary = [
        evaluate_condition("bm25_only", bm25_results),
        evaluate_condition("dense_only", dense_results),
        evaluate_condition("hybrid_rrf_rerank", hybrid_results),
    ]

    print(f"{'Condition':<20}{'Recall@' + str(TOP_K):<12}{'MRR':<8}")
    for row in summary:
        print(f"{row['condition']:<20}{row['recall@'+str(TOP_K)]:<12}{row['mrr']:<8}")

    out_file = os.path.join(results_dir, "evaluation_results.json")
    with open(out_file, "w") as f:
        json.dump(
            {"dense_backend": dense.backend_name, "summary": summary, "per_query": per_query_log},
            f,
            indent=2,
        )
    print(f"\nSaved detailed results to {out_file}")


if __name__ == "__main__":
    main()
