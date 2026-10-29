# Retrieval evaluation

48 labelled questions, 123 chunks, embeddings `BAAI/bge-small-en-v1.5`. Global search (no source filter), chunk-level ranking mapped to documents.

| Mode | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR@10 | nDCG@5 | Latency (ms) |
|---|---|---|---|---|---|---|---|
| bm25 | 0.833 | 0.917 | 0.917 | 0.899 | 0.875 | 0.860 | 0.1 |
| dense | 0.875 | 0.979 | 0.979 | 0.934 | 0.923 | 0.896 | 11.9 |
| hybrid | 0.896 | 0.938 | 0.958 | 0.931 | 0.918 | 0.896 | 15.3 |
| hybrid+rerank | 0.854 | 0.958 | 0.958 | 0.913 | 0.906 | 0.880 | 15649.8 |
