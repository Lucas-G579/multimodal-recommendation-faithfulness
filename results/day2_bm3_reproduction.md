# Day 2：BM3 / Baby 基线复现报告

日期：2026-07-29  
硬件：NVIDIA GeForce RTX 4050 Laptop GPU（6 GB）  
环境：Python 3.11.3，PyTorch 2.3.0+cu121，NumPy 1.26.4

## 1. 结论

BM3 在 MMRec 的 Baby 官方配置下完成端到端复现。第二次正式运行的最佳模型出现在第 112 轮：

| 来源 | Recall@20 | NDCG@20 |
|---|---:|---:|
| MMRec 发布日志 | 0.0883 | 0.0383 |
| 本地正式运行 1 | 0.0865 | 0.0369 |
| 本地正式运行 2（已保存 checkpoint） | 0.0862 | 0.0369 |
| 本地运行 2 相对发布日志差异 | -2.38% | -3.66% |

两个核心指标均处于预先采用的 5% 相对误差容忍范围内，因此判定复现通过。该容忍范围用于工程复现验收，不代表统计等价性。

## 2. 数据与特征完整性

Baby 交互数据包含 160,792 条交互、19,445 个用户和 7,050 个商品；train / valid / test 为
118,551 / 20,559 / 21,682。数据无重复 user-item 行、无缺失值。

| 特征 | shape | dtype | SHA-256 |
|---|---|---|---|
| image | 7050 × 4096 | float64 | `36C3BE592B98506189A7D5DE71B21577CF626F0293B539D861534673B3E9FD70` |
| text | 7050 × 384 | float32 | `6667F2AD655C9ECC97CB3383F58988864EF51EC0B39C158B15986C66769F2DC4` |

两种特征均无 NaN、Inf 或全零行，并与商品编号 `0..7049` 精确对齐。

## 3. 正式配置与结果

- model：BM3
- dataset：Baby
- seed：999
- epochs：1000
- early stopping patience：20
- train batch size：2048
- embedding size：64
- n_layers：1
- dropout：0.5
- reg_weight：0.1

第二次运行在第 133 轮触发 early stopping，总耗时约 718.5 秒。最佳 epoch 112 的测试结果：

正式运行日志：`external/MMRec/src/log/BM3-baby-Jul-29-2026-23-27-55.log`，
SHA-256 为 `425F6324D2355676EAB3F1191A6C49598DACE8FA2A04A184B52CD1CE55D1A945`。

| k | Recall | NDCG | Precision | MAP |
|---:|---:|---:|---:|---:|
| 5 | 0.0321 | 0.0214 | 0.0072 | 0.0173 |
| 10 | 0.0534 | 0.0285 | 0.0060 | 0.0201 |
| 20 | 0.0862 | 0.0369 | 0.0048 | 0.0224 |
| 50 | 0.1489 | 0.0496 | 0.0033 | 0.0244 |

## 4. Checkpoint 验收

- 路径：`outputs/checkpoints/mmrec/BM3-baby-best.pth`
- 大小：134,288,312 bytes
- SHA-256：`D38601C7343D1C77A90E0F7D3F62EEFA9E0F44DCE542A70BFFAD1043AD96119B`
- epoch：112
- state tensors：10
- 参数量：33,570,688
- 元数据：model、dataset、epoch、state_dict、valid_result、test_result

上游 Trainer 接收 `saved=True`，但当前快照没有实现保存逻辑。本项目使用
`patches/MMRec-save-best.patch` 做最小修改：仅当验证集 Recall 改善时保存最佳模型及对应指标。

## 5. 可重复性边界

相同 seed 的两次正式运行分别得到 Recall@20 0.0865 和 0.0862，相对差异约 0.35%；
NDCG@20 均为 0.0369（四位小数）。这说明基线在研究决策所需精度上稳定，但 CUDA
路径没有配置为严格确定性，因此不能声称逐位可重复。

后续因果干预实验必须从本报告验收的 checkpoint 出发，固定数据切分、候选集和评估代码；
核心结论至少运行 3 个随机种子，并报告均值与离散程度。
