# FaithRec-MM

面向多模态推荐解释的模态归因错配审计项目。

当前阶段：Day 3，已验证 BM3 checkpoint 可精确恢复，并完成推理路径审计。

## 研究问题

多模态推荐模型实际依赖的图片、文本和协同信号，与 LLM 解释声称使用的证据是否一致？

## 环境

- Windows 10/11
- Python 3.11
- PyTorch 2.3.0+cu121
- NVIDIA RTX 4050 Laptop GPU（6 GB VRAM）

## 目录

- `docs/`：研究协议、复现笔记与失败日志
- `external/`：外部研究代码，不纳入当前仓库
- `scripts/`：环境检查和可复现实验入口
- `configs/`：后续固定的数据、模型和审计配置
- `results/`：可提交的小型汇总结果
- `outputs/`：训练输出与 checkpoint，不提交
- `patches/`：对上游代码的最小可审计补丁

完整路线见 [暑期两个月执行方案.md](./暑期两个月执行方案.md)。

## 本地快速验证

```powershell
.\.venv\Scripts\python.exe .\scripts\check_environment.py
.\.venv\Scripts\python.exe .\scripts\summarize_interactions.py .\external\MMRec\data\baby\baby.inter
.\.venv\Scripts\python.exe .\scripts\inspect_mmrec_features.py `
  --interactions .\external\MMRec\data\baby\baby.inter `
  --image .\external\MMRec\data\baby\image_feat.npy `
  --text .\external\MMRec\data\baby\text_feat.npy
.\.venv\Scripts\python.exe .\scripts\run_mmrec_smoke.py --model LightGCN --profile smoke
.\.venv\Scripts\python.exe .\scripts\run_mmrec_smoke.py --model BM3 --profile smoke
```

LightGCN 和 BM3 的单 epoch 结果只用于管线 smoke test，不得作为正式基线引用。

BM3 / Baby 正式复现结果：Recall@20 0.0862、NDCG@20 0.0369，相对 MMRec
发布日志的差异均小于 5%。详见 [Day 2 复现报告](./results/day2_bm3_reproduction.md)。

BM3 checkpoint 恢复后的 32 个验证/测试指标与存档完全一致。推理路径审计同时确认：
BM3 的图像和文本只在训练损失中发挥作用，训练后直接清零模态不是有效的依赖测量方法。
详见 [Day 3 审计报告](./results/day3_checkpoint_and_path_audit.md)。
