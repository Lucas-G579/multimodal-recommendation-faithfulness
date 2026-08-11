<div align="center">

# FaithRec-MM

### 多模态推荐解释忠实性审计

**模型解释说它依赖了图片或文字——行为干预证据真的支持这个说法吗？**

[![Research Stage](https://img.shields.io/badge/stage-Day%2019-2155d6)](./results/day19_selective_agreement_ab_validation.md)
[![Protocol Tests](https://img.shields.io/badge/tests-13%2F13%20passing-1d8062)](./tests)
[![Pages](https://github.com/Lucas-G579/multimodal-recommendation-faithfulness/actions/workflows/pages.yml/badge.svg)](https://lucas-g579.github.io/multimodal-recommendation-faithfulness/)
[![Python](https://img.shields.io/badge/python-3.11-3776ab)](https://www.python.org/)

[项目主页](https://lucas-g579.github.io/multimodal-recommendation-faithfulness/) ·
[正式确认报告](./results/day15_kimi_confirmatory_evaluation.md) ·
[非 LLM 基线](./results/day17_non_llm_modality_baselines.md) ·
[独立敏感性验证](./results/day19_selective_agreement_ab_validation.md)

</div>

---

## 当前状态

项目已推进至 **Day 19**，完成了从推荐模型复现、跨训练种子行为干预、LLM 盲评预注册，到一次性正式确认实验及独立敏感性验证的完整链路。

当前最重要的结论是：

> Kimi v2 的流畅、高置信多模态解释，没有可靠识别 MGCN 实际依赖的主要模态；一个无需生成文本、只比较图文历史相似度的确定性基线，反而取得更好的类别平衡忠实性。

严格确认实验的结论保持冻结；Day 16 之后的方法比较均明确标记为事后探索或独立敏感性验证，不反向修改正式结果。

## 核心结果

### 1. 一次性正式确认实验

590 条严格 A 级样本、583 位用户；提示词、模型、schema、重试和评分规则均在运行前冻结。

| 方法 | Accuracy | Text Recall | Image Recall | Macro Recall |
|---|---:|---:|---:|---:|
| Majority Text | **92.88%** | 100.00% | 0.00% | 50.00% |
| Kimi v2 | 67.29% | 70.44% | 26.19% | **48.31%** |
| Mean Percentile¹ | 63.05% | 62.96% | 64.29% | **63.62%** |
| Max Percentile¹ | 75.59% | 77.19% | 54.76% | **65.98%** |

¹ 非 LLM 基线在确认集解盲后设计，属于事后探索，不能替代 Kimi v2 的一次性确认结论。

- Kimi v2 宏召回：**48.31%**，用户级 Bootstrap 95% CI **41.64%–55.59%**。
- Kimi v2 只识别出 **11/42** 个 image 真值，并产生 **133** 个置信度 ≥ 0.8 的错误回答。
- Mean Percentile 相对 Kimi 的宏召回配对提升：**+15.31 个百分点**，95% CI **+5.34 至 +24.87**。

<p align="center">
  <img src="./results/figures/day18_macro_recall_comparison.png" width="820" alt="Kimi 与非 LLM 基线的宏平均召回比较">
</p>

### 2. 独立 A+B 敏感性验证

400 条此前未使用的样本，text/image 各 200 条；400 位用户与其他 cohort 完全隔离。

| 方法 | 覆盖率 | Macro Recall | 用户 Bootstrap 95% CI |
|---|---:|---:|---:|
| Full Mean | 100% | 52.50% | 47.62%–57.38% |
| Mean/Max 一致才回答 | 65.25% | 57.76% | 51.95%–63.58% |
| Full Max | 100% | **57.25%** | **52.50%–62.06%** |

预先冻结的选择预测通过了“覆盖率 ≥ 50% 且宏召回区间下界 > 50%”双重门槛；但 Full Max 无需拒答即可获得相近结果，因此一致性是有效风险信号，尚未证明是更实用的最终策略。

## 研究问题

多模态推荐模型实际依赖的图像、文本和协同信号，与 LLM 解释声称使用的证据是否一致？

本项目将“解释忠实性”拆成两个可审计对象：

1. **行为真值**：关闭、均值替换或置换图像/文本分支，观察推荐分数与排名变化。
2. **解释声明**：让多模态 LLM 在看不到干预标签的条件下，判断主要证据来自图像还是文本。

两者一致才构成模态归因忠实性。

## 方法概览

```text
MGCN / Baby 复现
        ↓
zero · mean · 5× permutation 干预
        ↓
3 个训练种子交叉稳定性筛选
        ↓
严格 A 级行为真值 + A+B 敏感性集
        ↓
无答案字段的图文盲输入
        ↓
Kimi v2 一次性确认 + 非 LLM 基线审计
```

关键防泄漏措施：

- 提示词开发集、正式确认集、A+B 敏感性集和 unstable 压力集按用户完全隔离。
- 正式请求不包含 label、cohort、contrast、rank change、训练种子等答案字段。
- 永久 API/schema 失败在主分析中按错误计入，不做选择性删除。
- 主要协议在查看相应模型结果前提交并记录哈希。

## 可复现性

### 环境

- Windows 10/11
- Python 3.11
- PyTorch 2.3.0 + CUDA 12.1
- NVIDIA RTX 4050 Laptop GPU（6 GB VRAM）

### 协议测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -p "test_*.py" -v
```

当前结果：**13/13 tests passing**。

### 关键分析入口

```powershell
# Kimi v2 正式确认结果分析
python .\scripts\analyze_kimi_confirmatory_v2.py

# Day 16 事后误差审计
python .\scripts\analyze_kimi_v2_posthoc_errors.py

# Day 17 非 LLM 基线
python .\scripts\evaluate_non_llm_modality_baselines.py

# Day 18 机制审计与论文图
python .\scripts\analyze_mean_percentile_mechanism.py

# Day 19 独立 A+B 敏感性验证
python .\scripts\validate_selective_agreement_ab.py
```

原始训练输出、checkpoint 和 API 响应位于 `outputs/`，默认不提交 Git；仓库只保留协议、代码、聚合统计、图表与可审计哈希。

## 实验进度与报告

| 阶段 | 内容 | 报告 |
|---|---|---|
| Day 1–3 | 环境、BM3 复现、checkpoint 与推理路径审计 | [Day 1](./results/day1_smoke_test.md) · [Day 2](./results/day2_bm3_reproduction.md) · [Day 3](./results/day3_checkpoint_and_path_audit.md) |
| Day 4–6 | MGCN 资格、全量行为审计、聚类不确定性 | [Day 4](./results/day4_mgcn_qualification.md) · [Day 5](./results/day5_full_behavior_audit.md) · [Day 6](./results/day6_clustered_uncertainty.md) |
| Day 7–10 | 多干预、多置换种子、跨训练种子稳定性 | [Day 7](./results/day7_intervention_robustness.md) · [Day 8](./results/day8_permutation_seed_stability.md) · [Day 9](./results/day9_training_seed_protocol.md) · [Day 10](./results/day10_cross_training_seed_stability.md) |
| Day 11–14 | LLM 预注册、盲输入、提示词开发与冻结 | [Day 11](./results/day11_llm_evaluation_preregistration.md) · [Day 12](./results/day12_prompt_development_dry_run.md) · [Day 13](./results/day13_kimi_prompt_development_analysis.md) · [Day 14](./results/day14_kimi_prompt_v2_validation.md) |
| Day 15 | Kimi v2 一次性正式确认实验 | [正式报告](./results/day15_kimi_confirmatory_evaluation.md) |
| Day 16 | 冻结后的失败机制审计 | [误差审计](./results/day16_kimi_v2_posthoc_error_audit.md) |
| Day 17 | 确定性非 LLM 模态归因基线 | [基线报告](./results/day17_non_llm_modality_baselines.md) |
| Day 18 | Mean/Max 机制审计与论文图 | [机制报告](./results/day18_mean_percentile_mechanism.md) |
| Day 19 | 用户隔离的独立 A+B 敏感性验证 | [验证报告](./results/day19_selective_agreement_ab_validation.md) |

## 仓库结构

```text
configs/          固定实验配置
data/manifests/   样本、协议与机器可读聚合结果
docs/             预注册协议、复现笔记与失败日志
patches/          对上游 MMRec 的最小可审计补丁
results/          Day 1–19 报告与论文图
scripts/          数据、训练、审计与统计入口
site/             GitHub Pages 项目展示页
tests/            LLM 响应协议测试
```

## 结果解释边界

- Day 15 是当前唯一的严格 LLM 确认性结论。
- Day 16–18 使用已经解盲的正式集，均为事后探索。
- Day 19 是独立 A+B 敏感性验证，但标签稳定性低于严格 A 级。
- 当前结果来自 MGCN / Baby，尚不能直接外推到其他推荐模型或数据集。
- Full Max 是下一阶段候选方法，不是已经完成严格 A 级独立确认的最终方法。

## 致谢与上游

推荐模型复现基于 [MMRec](https://github.com/enoche/MMRec)。项目保留上游代码快照、最小补丁、数据哈希和失败日志，以支持独立审计。
