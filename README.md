# FaithRec-MM

面向多模态推荐解释的模态归因错配审计项目。

当前阶段：Day 10，已完成 seed=999、2026、3407 三个 MGCN 的跨训练种子逐样本干预
审计，并得到严格主集与敏感性集。

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

三个模型分别经过 zero、mean 和 5 个固定 permutation 种子筛选后，只有 670/21,682
条样本在三个训练种子中都达到 A 级且模态标签一致，占 3.09%（文本 621、图像 49）。
A+B 敏感性集为 1,824 条，占 8.41%。详见
[Day 10 跨训练种子稳定性报告](./results/day10_cross_training_seed_stability.md)。

LLM 解释评测已经在看到任何模型回答前冻结：80 条只用于提示词开发，590 条严格
A 级样本用于确认性检验，另有 400 条 A+B 敏感性样本和 400 条不稳定证据压力样本。
四组按用户隔离，连续两次生成的样本表 SHA-256 完全一致。7,050 个商品映射已经
恢复；目标图片覆盖率 100%，加入最多 5 条训练历史后的相关商品图片覆盖率为
99.91%。不含答案字段的盲输入也已通过确定性复现，因此下一步可以只在开发集调试
提示词；正式确认集仍未查看、尚未调用 LLM。详见
[Day 11 LLM 解释忠实性评测预注册](./results/day11_llm_evaluation_preregistration.md)。

LLM 提示词、严格 JSON 响应契约和开发集 dry-run 已完成。80 条开发请求共引用 422
张图片，未组装任何正式确认集请求，未发现答案字段或缺失图片路径；解析器 7 项
单元测试全部通过。当前尚未配置具体多模态模型和 API 预算，因此没有发送真实请求。
详见 [Day 12 提示词开发 dry-run](./results/day12_prompt_development_dry_run.md)。

Kimi `kimi-k2.6` 已完成全部 80 条开发样本：74 条严格有效、5 条永久网络失败、1 条
永久 schema 失败。自然解释提示 v1 的有效回答中 65/74 选择 `both`、0 条选择
`image`，意向分析宏平均召回率仅 2.74%，因此 v1 被明确禁止进入确认集。强制选择
image/text 的 v2 已完成无标签 dry-run 和协议测试，但尚未产生 API 结果。详见
[Day 13 Kimi 开发集分析](./results/day13_kimi_prompt_development_analysis.md)。

强制选择提示词 v2 在相同开发集上得到 72.02% 的 ITT 宏平均召回率，text/image
召回率分别为 72.60%/71.43%，没有退化为多数类恒猜；提示词、schema 和运行参数现已
冻结，不再构造 v3。590 条确认集只完成离线无标签请求准备，尚未发送。详见
[Day 14 Kimi v2 开发验证与冻结](./results/day14_kimi_prompt_v2_validation.md)。

Kimi v2 的 590 条一次性正式确认实验已完成。ITT 宏平均召回率为 48.31%（用户聚类 bootstrap 95% CI 41.64%–55.59%），text/image 召回率为 70.44%/26.19%；结果没有证明模型能可靠识别主要证据模态。详见 [Day 15 Kimi v2 正式确认实验](./results/day15_kimi_confirmatory_evaluation.md)。

冻结后的事后误差审计表明：开发集和正式集的干预强度、波动及有效率接近，下降主要来自开发集仅 7 个 image 真值造成的乐观小样本估计；正式集的 image 召回在四个证据强度层均只有 20%–30%。该结果仅作探索性机制诊断。详见 [Day 16 Kimi v2 事后误差审计](./results/day16_kimi_v2_posthoc_error_audit.md)。

不调用生成式模型的 Mean Percentile 相似度基线在正式集取得 63.62% 宏召回（用户聚类 bootstrap 95% CI 56.13%–71.00%），相对 Kimi v2 的配对提升为 15.31 个百分点（95% CI +5.34 至 +24.87）。该比较在确认集解盲后设计，明确作为探索性结果。详见 [Day 17 非 LLM 模态归因基线](./results/day17_non_llm_modality_baselines.md)。

机制审计发现 Mean 与 Max 聚合规则一致的 420/590 条样本上，宏召回达到 72.91%；不一致的 170 条仅为 42.00%。规则一致性因而是一个值得在新数据验证的无标签风险信号，但当前仍属于事后发现。四张论文级主图与完整案例审计见 [Day 18 机制审计与论文图](./results/day18_mean_percentile_mechanism.md)。
