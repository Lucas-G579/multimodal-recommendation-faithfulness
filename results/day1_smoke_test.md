# Day 1 Smoke Test

- Status: COMPLETED
- Date: 2026-07-29
- Command: `.\.venv\Scripts\python.exe .\scripts\run_mmrec_smoke.py`
- Model: LightGCN
- Dataset: MMRec Baby
- Device: NVIDIA GeForce RTX 4050 Laptop GPU
- Python: 3.11.3
- PyTorch: 2.3.0+cu121
- Seed: 2026
- Epochs: 1
- Train loss: 20.0867
- Train time: 0.68 s

| Split | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| Validation | 0.0126 | 0.0203 | 0.0067 | 0.0087 |
| Test | 0.0124 | 0.0191 | 0.0066 | 0.0084 |

These values verify execution only. They are not a reproduced baseline and must
not be used as research results.
