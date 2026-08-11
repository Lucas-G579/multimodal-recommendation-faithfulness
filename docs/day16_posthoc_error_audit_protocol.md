# Day 16：Kimi v2 事后误差审计协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Origin Date：2026-08-11
- Verification Status：FROZEN BEFORE DIAGNOSTIC EXECUTION
- Version Label：`kimi_v2_posthoc_audit_v1`
- Scope：既有 80 条开发集与 590 条正式确认集；不产生新的 API 请求
- Epistemic Status：EXPLORATORY，不能覆盖或改写 Day 15 的确认性结论

## 1. 目的

解释为什么 Kimi v2 的 ITT 宏平均召回率从开发集的 72.02% 降至正式确认集的 48.31%，并区分三类可能来源：

1. 开发集与正式集的用户分桶或类别组成差异；
2. Kimi 对 image/text 的系统性预测偏差；
3. schema/API 失败与某类样本集中相关。

本审计只描述失败结构，不选择新提示词、不重新评分、不把事后发现包装成确认性假设。

## 2. 冻结输入

- `data/manifests/llm_evaluation_samples.csv`
- `outputs/llm_prompt_development_v2/kimi_responses.jsonl`
- `outputs/llm_confirmatory_v2/kimi_responses.jsonl`
- `data/manifests/llm_blind_inputs.jsonl`（仅使用非答案元数据）
- Day 14 与 Day 15 已冻结的分析汇总

原始响应只读；每个样本使用最后一次尝试，永久失败按 ITT 计错。

## 3. 固定诊断项目

### A. 开发集—正式集分布对照

- 样本数、唯一用户数、user bucket；
- text/image 真值比例；
- `contrast_median`、绝对 contrast、`contrast_mad` 的中位数与四分位数；
- 最终有效率、预测类别比例、ITT accuracy 与 macro recall。

### B. 正式集按 user bucket 的完整分层

对 bucket 1–9 全部报告：样本数、类别数、有效率、text/image 召回、宏召回、预测 image 比例。不得只挑表现最好或最差的 bucket。

### C. 按真值与证据强度的完整分层

在每个真值类别内部，按绝对 `contrast_median` 的预先定义四分位数切分。每层报告样本数、有效率、召回率和平均自报置信度。若边界重复，保留 pandas `qcut(..., duplicates="drop")` 的实际层数并如实记录。

### D. 失败机制

- 首次失败数、重试恢复数、永久 schema/API 失败数；
- 永久失败按真值和 user bucket 的完整分布；
- 不读取或手工编码解释文本来创建新的有利分组。

### E. 置信度诊断

固定区间 `[0.5,0.6)`、`[0.6,0.7)`、`[0.7,0.8)`、`[0.8,0.9)`、`[0.9,1.0]`，报告有效样本数、准确率、类别构成；另报告高置信错误数。置信度分析仅为诊断，不解释为概率校准证明。

### F. 开发—正式差值的不确定性

以 `user_id` 为聚类单位做 10,000 次 bootstrap，seed=`20260811`，报告正式集减开发集的 accuracy 和 macro recall 差值及 95% 百分位区间。两组独立重采样；不新增显著性检验。

## 4. 多重比较与停止规则

- 所有分层结果完整输出，不以 p 值筛选；因此不运行一组可被选择性汇报的显著性检验。
- 不因诊断结果修改本协议，也不创建 v2 的“修正版正式结果”。
- 完成本协议的 A–F 后停止。若要设计 v3，必须另建探索协议和新的未见验证集。

## 5. 预期产物

- `scripts/analyze_kimi_v2_posthoc_errors.py`
- `data/manifests/kimi_v2_posthoc_error_audit.json`
- `results/day16_kimi_v2_posthoc_error_audit.md`

所有产物必须显式标注 `POSTHOC_EXPLORATORY`。
