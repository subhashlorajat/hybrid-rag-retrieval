'''
End-to-end POC demo:
  1. Load + clean the NCI corpus (data/raw_corpus/*.txt)
  2. Chunk it (word-based sliding window)
  3. Build BM25 (sparse) + embedding (dense) indexes
  4. Run hybrid retrieval (RRF fusion + lightweight rerank) on sample queries
  5. Print corpus/chunk statistics and retrieval results
'''
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunking import load_corpus, build_chunk_corpus, corpus_stats
from sparse_retriever import SparseRetriever
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever


SAMPLE_QUERIES = [
    "How much is the Student Contribution Fee and when is it due?",
    "What do I need to do if I failed a module in Semester 2?",
    "How do I get my student card photo approved?",
    "What is the Harvard referencing style used for in-text citations?",
    "Who is eligible for DARE and HEAR supplementary orientation?",
    "What happens if I do not pay my fees on time?",
]

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
SPARSE_K = 10
DENSE_K = 10
FUSION_K = 60
FINAL_K = 3


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(here, "..", "data", "raw_corpus")
    results_path = os.path.join(here, "..", "results")
    os.makedirs(results_path, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Loading + cleaning corpus")
    print("=" * 70)
    docs = load_corpus(raw_dir)
    for d in docs:
        print(f"  - {d['doc_id']}: {len(d['text'].split())} words | {d['title']}")

    print("\n" + "=" * 70)
    print(f"STEP 2: Chunking (chunk_size={CHUNK_SIZE} words, overlap={CHUNK_OVERLAP} words)")
    print("=" * 70)
    chunks = build_chunk_corpus(raw_dir, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    stats = corpus_stats(docs, chunks)
    print(json.dumps(stats, indent=2))

    print("\n" + "=" * 70)
    print("STEP 3: Building sparse (BM25) index")
    print("=" * 70)
    t0 = time.time()
    sparse = SparseRetriever(chunks)
    print(f"  BM25 index built over {len(chunks)} chunks in {time.time()-t0:.3f}s")

    print("\n" + "=" * 70)
    print("STEP 4: Building dense index")
    print("=" * 70)
    t0 = time.time()
    dense = DenseRetriever(chunks)
    print(f"  Dense backend: {dense.backend_name}")
    print(f"  Dense index built over {len(chunks)} chunks in {time.time()-t0:.3f}s")

    hybrid = HybridRetriever(sparse, dense, chunks)

    print("\n" + "=" * 70)
    print("STEP 5: Hybrid retrieval demo on sample queries")
    print("=" * 70)

    all_results = []
    for query in SAMPLE_QUERIES:
        t0 = time.time()
        results = hybrid.search(
            query,
            sparse_k=SPARSE_K,
            dense_k=DENSE_K,
            fusion_k=FUSION_K,
            final_k=FINAL_K,
        )
        latency = time.time() - t0
        print(f"\nQuery: {query}")
        print(f"  Retrieval latency: {latency*1000:.1f} ms")
        for rank, c in enumerate(results, 1):
            snippet = c.text[:160].replace("\n", " ")
            print(f"  [{rank}] {c.doc_id} ({c.chunk_id}): {snippet}...")
        all_results.append(
            {
                "query": query,
                "latency_ms": round(latency * 1000, 2),
                "top_chunks": [
                    {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "snippet": c.text[:200]}
                    for c in results
                ],
            }
        )

    out_file = os.path.join(results_path, "retrieval_demo_results.json")
    with open(out_file, "w") as f:
        json.dump(
            {
                "corpus_stats": stats,
                "dense_backend": dense.backend_name,
                "hyperparameters": {
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "sparse_k": SPARSE_K,
                    "dense_k": DENSE_K,
                    "fusion_k": FUSION_K,
                    "final_k": FINAL_K,
                },
                "queries": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved full results to {out_file}")


if __name__ == "__main__":
    main()
