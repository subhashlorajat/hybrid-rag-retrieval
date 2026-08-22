
# Small hyperparameter comparison for the Hyperparameter Optimization 
# section of the report. Varies:
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

from chunking import build_chunk_corpus, load_corpus, corpus_stats
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


def run_condition(chunks, sparse, dense, hybrid_kwargs=None):
    hybrid = HybridRetriever(sparse, dense, chunks)
    results = {"bm25": [], "dense": [], "hybrid": []}
    for query, expected_doc_id in EVAL_SET:
        sparse_hits = sparse.search(query, top_k=10)
        sparse_doc_ids = [chunks[i].doc_id for i, _ in sparse_hits]
        dense_hits = dense.search(query, top_k=10)
        dense_doc_ids = [chunks[i].doc_id for i, _ in dense_hits]
        kwargs = hybrid_kwargs or {}
        hybrid_chunks = hybrid.search(query, final_k=TOP_K, **kwargs)
        hybrid_doc_ids = [c.doc_id for c in hybrid_chunks]

        results["bm25"].append((expected_doc_id, sparse_doc_ids))
        results["dense"].append((expected_doc_id, dense_doc_ids))
        results["hybrid"].append((expected_doc_id, hybrid_doc_ids))

    summary = {}
    for cond, rows in results.items():
        recalls = [recall_at_k(doc_ids, exp, TOP_K) for exp, doc_ids in rows]
        rrs = [reciprocal_rank(doc_ids, exp) for exp, doc_ids in rows]
        summary[cond] = {
            f"recall@{TOP_K}": round(sum(recalls) / len(recalls), 3),
            "mrr": round(sum(rrs) / len(rrs), 3),
        }
    return summary


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(here, "..", "data", "raw_corpus")
    results_dir = os.path.join(here, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    docs = load_corpus(raw_dir)
    all_rows = []

    print("=" * 78)
    print("PART 1: chunk_size / chunk_overlap sweep")
    print("=" * 78)
    chunk_configs = [(100, 20), (150, 30), (200, 40), (150, 0)]

    for chunk_size, overlap in chunk_configs:
        chunks = build_chunk_corpus(raw_dir, chunk_size=chunk_size, overlap=overlap)
        stats = corpus_stats(docs, chunks)
        sparse = SparseRetriever(chunks)
        dense = DenseRetriever(chunks)
        summary = run_condition(chunks, sparse, dense)

        row = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": stats["num_chunks"],
            "avg_chunk_words": stats["avg_chunk_words"],
            **{f"{cond}_{metric}": val for cond, m in summary.items() for metric, val in m.items()},
        }
        all_rows.append(row)
        print(
            f"chunk_size={chunk_size:<5} overlap={overlap:<4} "
            f"num_chunks={stats['num_chunks']:<5} "
            f"bm25(R/MRR)={summary['bm25']['recall@3']}/{summary['bm25']['mrr']}  "
            f"dense(R/MRR)={summary['dense']['recall@3']}/{summary['dense']['mrr']}  "
            f"hybrid(R/MRR)={summary['hybrid']['recall@3']}/{summary['hybrid']['mrr']}"
        )

    best = max(all_rows, key=lambda r: (r["hybrid_recall@3"], r["hybrid_mrr"]))
    print(f"\nBest chunk config by hybrid Recall@3/MRR: "
          f"chunk_size={best['chunk_size']}, overlap={best['overlap']}")

    print("\n" + "=" * 78)
    print(f"PART 2: RRF fusion_k sweep (at chunk_size={best['chunk_size']}, "
          f"overlap={best['overlap']})")
    print("=" * 78)
    chunks = build_chunk_corpus(raw_dir, chunk_size=best["chunk_size"], overlap=best["overlap"])
    sparse = SparseRetriever(chunks)
    dense = DenseRetriever(chunks)

    fusion_rows = []
    for fusion_k in [10, 60, 100]:
        summary = run_condition(chunks, sparse, dense, hybrid_kwargs={"fusion_k": fusion_k})
        fusion_rows.append({"fusion_k": fusion_k, **summary["hybrid"]})
        print(f"fusion_k={fusion_k:<5} hybrid Recall@3={summary['hybrid']['recall@3']} "
              f"MRR={summary['hybrid']['mrr']}")

    out_file = os.path.join(results_dir, "hyperparam_sweep_results.json")
    with open(out_file, "w") as f:
        json.dump(
            {
                "chunk_sweep": all_rows,
                "best_chunk_config": {"chunk_size": best["chunk_size"], "overlap": best["overlap"]},
                "fusion_k_sweep": fusion_rows,
            },
            f,
            indent=2,
        )
    print(f"\nSaved full sweep results to {out_file}")


if __name__ == "__main__":
    main()
