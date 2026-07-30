## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
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

## 训练与恢复结果

两个预注册的新种子均正常完成 early stopping，没有更换种子或因结果高低删除运行。

| 训练 seed | 最佳 epoch | 测试 Recall@20 | 测试 NDCG@20 | checkpoint SHA-256 |
|---:|---:|---:|---:|---|
| 999（已有） | 32 | 0.0933 | 0.0421 | `8607638F89E68B3ED395018B605EAD86C4FE14310599C67A3EAFBB981080497E` |
| 2026 | 37 | 0.0934 | 0.0418 | `B355E39B86A34DDB7189D492269DFF68D70BD7DD8AF362DF17AEAE54852CE26A` |
| 3407 | 35 | 0.0926 | 0.0414 | `23FBF81CF277886A9E4E3287F71EDF6807F96426AA6E9A00D3C4EC3599C26EFF` |

三个种子的 Recall@20 范围为 0.0926–0.0934，NDCG@20 范围为
0.0414–0.0421。两个新种子相对论文 Baby/All 的 Recall@20=0.0964、
NDCG@20=0.0427 均在 5% 相对误差内。

每个新 checkpoint 均被重新构建模型、严格加载参数并重新计算 16 个验证指标和
16 个测试指标。两次验收的全部 32 项指标与 checkpoint 存档值完全一致，
绝对差均为 0。

| 产物 | SHA-256 |
|---|---|
| seed=2026 `run_manifest.json` | `709631A1CFA923D89F4FA41CC5322E4AB8CF721B090FEDF49FC7534B5A0A0C11` |
| seed=2026 恢复验收 JSON | `72886FB681D22F11079E68FC4023350E163B45144A4ACCF95B0702D1FB697113` |
| seed=3407 `run_manifest.json` | `12448504C00C9CAE5CBAAD47571D22403D948F0BF6E8208EEDB1F7BC188726D1` |
| seed=3407 恢复验收 JSON | `3510878E29DF63D8BF863F5B12E82B41ED63EAEE5A16A5C3D53C7529BC97621B` |

## 当前结论边界

可以确认：

- 三个训练种子的整体推荐性能接近；
- 两个新 checkpoint 均可严格恢复；
- 新模型具备进入跨训练种子干预审计的资格。

尚不能确认：

- 同一条推荐在三个训练种子下具有相同的图像/文本依赖标签；
- seed=999 的 3,815 个 A 级样本仍会在另外两个模型中保持 A 级；
- 三个种子的整体指标接近，等价于内部决策机制接近。

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
