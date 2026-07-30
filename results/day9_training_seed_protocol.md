## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PLANNED
- Version Label: mgcn_training_seed_protocol_v1

# Day 9：MGCN 跨训练种子复验协议

## 给初学者的解释

目前全部审计都来自同一个 seed=999 模型。训练神经网络时，初始参数和样本顺序带有
随机性；换一个训练种子，相当于让模型从另一个随机起点重新学习一次。

本阶段先训练两个新模型，不替换 seed=999，也不根据结果挑选“最好看的模型”。

## 预先固定的设计

- 已有参考模型：seed=999
- 新训练种子：seed=2026、seed=3407
- 模型与数据：MGCN / Baby
- 最大 epoch：1000
- early stopping patience：20
- 选择标准：验证集 Recall@20
- `cl_loss`：0.01
- `knn_k`：20
- 每个种子使用独立 checkpoint 目录，禁止静默覆盖

选择 2026 和 3407 发生在查看新训练结果之前。失败的训练如实记录，不更换种子。

## 训练成功标准

每个训练种子必须同时满足：

1. 进程正常退出；
2. 独立 checkpoint 与 `run_manifest.json` 均存在；
3. checkpoint 可以严格加载，缺失键和多余键均为 0；
4. 从 checkpoint 重新计算的验证/测试指标与存档一致；
5. Recall@20 与 NDCG@20 均报告，不以论文数字决定是否保留该种子。

论文结果接近度只作为诊断信息。不能因为某个种子结果较低就删除它。

## 后续审计顺序

对 seed=2026 和 seed=3407 分别执行：

1. checkpoint 恢复验收；
2. zero、mean 和 5 个固定 permutation 种子的全量干预；
3. 生成各自的 A/B/unstable 标签；
4. 与 seed=999 比较逐样本标签；
5. 报告交集覆盖率、模态构成和一致率。

最终主分析样本必须跨三个训练种子都得到一致的 A 级非并列标签。若覆盖率过低，
应如实报告并调整研究问题，不能放宽规则来追求样本量。

## 资源和停止规则

- 设备：本地 RTX 4050 Laptop GPU（6 GB）
- 预期单次训练：约数分钟，实际由 early stopping 决定
- 训练期间监控：进程存活、GPU 显存、日志更新时间
- 硬超时：每个种子 30 分钟
- 除硬超时外，不因指标不理想自动终止

## 计划命令

```powershell
.\.venv\Scripts\python.exe .\scripts\run_mmrec_smoke.py `
  --model MGCN --profile official --seed 2026

.\.venv\Scripts\python.exe .\scripts\run_mmrec_smoke.py `
  --model MGCN --profile official --seed 3407
```
