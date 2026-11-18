# Controlled efficiency benchmark (CPU, 8 threads, one model at a time)

| Model | Trainable params | Train samples/s | Est. min / epoch | Inference tx/s | Latency median (ms) | Latency p95 (ms) | Peak memory (MB) |
|---|---|---|---|---|---|---|---|
| LSTM | 42,945 | 1,311 | 1.9 | 15,775 | 6.1 | 11.3 | 1,178 |
| GRU | 36,801 | 7,286 | 0.3 | 22,134 | 3.4 | 3.8 | 1,155 |
| GCN | 27,713 | 754 | 3.4 | 2,200 | 8.7 | 27.2 | 2,614 |
| GraphSAGE | 35,905 | 2,473 | 1.0 | 2,744 | 2.9 | 6.2 | 2,404 |
| GAT | 31,809 | 1,147 | 2.2 | 1,459 | 6.3 | 7.7 | 2,814 |
| GAT-LSTM | 64,577 | 1,004 | 2.5 | 1,356 | 8.8 | 11.5 | 2,953 |
| TRACE-X (proposed) | 238,369 | 1,176 | 2.2 | 1,568 | 11.3 | 17.3 | 3,070 |

Latency includes causal feature/subgraph construction for the transaction. Hardware: AMD Ryzen 7 5800H, 16 GB RAM, no CUDA GPU.