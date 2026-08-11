# Day 18：非 LLM 基线机制审计与论文图协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Origin Date：2026-08-11
- Verification Status：FROZEN BEFORE EXECUTION
- Version Label：`mean_percentile_mechanism_v1`
- Epistemic Status：POSTHOC_EXPLORATORY
- Parent Protocol：Day 17，commit `eefcfca`

## Experiment Overview

- **Title**：Mean Percentile 机制审计与论文级可视化
- **Objective**：解释冻结主基线何时正确、何时失败，以及它相对 Kimi 的优势来自哪里
- **Hypothesis**：模态百分位差值绝对值越大，基线判断越稳定；主要优势来自 image 召回改善，而非多数类 accuracy
- **Type**：analysis

## Setup

- **Language/Framework**：Python、NumPy、pandas、Matplotlib
- **Entry Command**：`python scripts/analyze_mean_percentile_mechanism.py`
- **Working Directory**：项目根目录
- **Environment**：CPU；不调用外部 API

## Inputs

| Input | Path | Description |
|---|---|---|
| 样本 | `data/manifests/llm_evaluation_samples.csv` | cohort、用户、跨种子真值 |
| 盲输入 | `data/manifests/llm_blind_inputs.jsonl` | 目标与历史商品，不含答案 |
| 图像特征 | `external/MMRec/data/baby/image_feat.npy` | MGCN 图像特征 |
| 文本特征 | `external/MMRec/data/baby/text_feat.npy` | MGCN 文本特征 |
| Kimi 响应 | `outputs/llm_confirmatory_v2/kimi_responses.jsonl` | 冻结正式回答 |
| Day 17 汇总 | `data/manifests/non_llm_modality_baselines.json` | 指标与 bootstrap 区间 |

## 固定分析

1. 对 590 条正式集重算 `image_mean_percentile`、`text_mean_percentile` 与有符号 margin=`image-text`。
2. 固定绝对 margin 区间：`[0,.025)`、`[.025,.05)`、`[.05,.10)`、`[.10,.20)`、`[.20,1]`；每层完整报告 n、accuracy、text/image n 与各类召回。空层保留。
3. 完整报告 Mean 与 Max 规则的一致/不一致样本数，以及每组的 accuracy、text/image recall、macro recall。
4. 对 Mean 与 Kimi 形成四格：both correct、Mean only、Kimi only、neither；同时按真值类别报告，不进行事后子群选择。
5. 案例选择规则固定为：每个真值类别选绝对 margin 最大的 5 个错误案例，以及绝对 margin 最小的 5 个正确案例；并列按 `sample_id` 排序。只展示公开商品标题、样本 ID、两种分数和预测，不人工挑选。
6. 不手工阅读案例后创建新的错误类别，不拟合阈值，不修改 Mean Percentile 规则。

## 固定论文图

- Figure 1：Kimi、Majority Text、Mean Percentile、Max Percentile 的正式集 macro recall 与用户聚类 95% CI；
- Figure 2：四种方法的 text/image recall 分组柱状图；
- Figure 3：590 条样本的 text/image mean percentile 散点图，按真值着色、按 Mean 正误区分点型，并画 `y=x` 决策线；
- Figure 4：固定绝对 margin 区间的样本数与 accuracy，不隐藏空层。

所有图输出 PNG（300 dpi）与 SVG；使用英文标签、色盲友好配色、0–1 固定坐标，并在图注中标记 `Post-hoc exploratory`。

## Expected Outputs

| Output | Path | Success Criterion |
|---|---|---|
| 脚本 | `scripts/analyze_mean_percentile_mechanism.py` | 可编译并退出码 0 |
| 汇总 | `data/manifests/mean_percentile_mechanism.json` | 严格 JSON，590 条覆盖 |
| 报告 | `results/day18_mean_percentile_mechanism.md` | 明确探索性边界 |
| 图 1–4 | `results/figures/day18_*` | PNG 和 SVG 均存在且非空 |

## Verification

- 连续两次运行的 JSON 与 SVG 必须逐文件 SHA-256 一致；PNG 若含固定元数据也必须一致；
- 13 个既有协议测试继续通过；
- Day 17 的主指标必须原样复现；
- 不上传数据、不调用 API、不覆盖 Day 15 确认性结论。
