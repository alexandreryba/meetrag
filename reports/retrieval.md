# Retrieval evaluation

33 labelled questions, 98 chunks, embeddings `BAAI/bge-small-en-v1.5`. Global search (no source filter), chunk-level ranking mapped to documents.

| Mode | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR@10 | nDCG@5 | Latency (ms) |
|---|---|---|---|---|---|---|---|
| bm25 | 0.879 | 0.970 | 0.970 | 0.970 | 0.923 | 0.930 | 0.1 |
| dense | 0.970 | 1.000 | 1.000 | 0.985 | 0.985 | 0.969 | 13.7 |
| hybrid | 0.939 | 0.970 | 1.000 | 1.000 | 0.961 | 0.960 | 12.8 |
| hybrid+rerank | 0.909 | 1.000 | 1.000 | 0.985 | 0.949 | 0.951 | 1057.2 |
