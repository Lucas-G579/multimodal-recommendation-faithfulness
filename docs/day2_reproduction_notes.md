# Day 2 Reproduction Notes

## BM3 / Baby 正式复现

- 日期：2026-07-29
- 配置：MMRec 官方 BM3 / Baby 配置，seed 999，batch size 2048，最多 1000 epochs，early stopping 20
- 发布日志：Recall@20 0.0883，NDCG@20 0.0383
- 本地运行 1：最佳 epoch 112，Recall@20 0.0865，NDCG@20 0.0369
- 本地运行 2：最佳 epoch 112，Recall@20 0.0862，NDCG@20 0.0369
- 复现判定：两个核心指标相对发布日志误差均小于 5%，通过
- checkpoint：`outputs/checkpoints/mmrec/BM3-baby-best.pth`
- checkpoint SHA-256：`D38601C7343D1C77A90E0F7D3F62EEFA9E0F44DCE542A70BFFAD1043AD96119B`
- checkpoint 参数量：33,570,688

同 seed 两次运行的 Recall@20 相对差异约 0.35%。当前 CUDA 路径未启用严格确定性，
因此记录为“指标稳定复现”，不声称逐位一致。

