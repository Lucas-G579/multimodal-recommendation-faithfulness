# FaithRec-MM

面向多模态推荐解释的模态归因错配审计项目。

当前阶段：Day 9，已完成 seed=999、2026、3407 三个 MGCN 训练种子的 checkpoint
资格验证，准备进行跨训练种子的逐样本干预审计。

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

MGCN / Baby 正式结果为 Recall@20 0.0933、NDCG@20 0.0421，与论文结果的相对差异
均小于 5%；checkpoint 的 32 个验证/测试指标恢复后完全一致。固定 32 位用户的资格探针
确认图像和文本分支均会改变推荐分数及排名。详见
[Day 4 MGCN 资格报告](./results/day4_mgcn_qualification.md)。

全量行为审计覆盖 19,445 位用户和 21,682 个测试用户—商品对，可以从输出表精确还原
Recall@20 0.0933 与 NDCG@20 0.0421。CPU 审计连续两次得到完全相同的 CSV SHA-256。
详见 [Day 5 全量行为审计](./results/day5_full_behavior_audit.md)。

以用户为单位进行 10,000 次聚类 Bootstrap 和 20,000 次配对符号翻转后，关闭文本与同时
关闭图文对 NDCG@20 的下降在 Holm 校正后仍稳定，但配对标准化效应均小于 0.04。
详见 [Day 6 统计完整性检查](./results/day6_clustered_uncertainty.md)。

三种表示层干预均支持文本分支对平均 NDCG 的小幅稳定影响；约 54.34% 的测试样本在三种
干预下得到一致的非并列模态标签，其余保留为 unstable，不强制分类。详见
[Day 7 干预稳健性报告](./results/day7_intervention_robustness.md)。

进一步使用 5 个预先固定的 permutation 种子复验后，逐样本置换标签的两两一致率仅为
59.77%–60.94%。因此主分析收紧为 zero、mean 和全部置换种子均一致的 A 级样本：
3,815 对，占 17.60%；A+B 共 8,521 对，只用于敏感性分析，其余拒绝判断。详见
[Day 8 多置换种子稳定性报告](./results/day8_permutation_seed_stability.md)。

两个预注册的新训练种子 2026、3407 均正常完成并通过 checkpoint 严格恢复，32 项
验证/测试指标与存档值零误差。三个种子的测试 Recall@20 为 0.0926–0.0934，
NDCG@20 为 0.0414–0.0421。详见
[Day 9 跨训练种子协议与结果](./results/day9_training_seed_protocol.md)。
