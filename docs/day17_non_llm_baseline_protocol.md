# Day 17：非 LLM 模态归因基线协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Origin Date：2026-08-11
- Verification Status：FROZEN BEFORE BASELINE EXECUTION
- Version Label：`non_llm_modality_baselines_v1`
- Scope：80 条开发集与 590 条正式确认集；外部 API 请求 0 条
- Epistemic Status：confirmatory set 上的事后比较，必须标记 `POSTHOC_EXPLORATORY`

## 1. 研究问题

不借助生成式 LLM，仅使用目标商品与用户历史商品的既有图像/文本特征相似度，能否比 Kimi v2 更接近 MGCN 行为干预产生的模态真值？

## 2. 冻结输入

- 图像特征：`external/MMRec/data/baby/image_feat.npy`（7050 × 4096）
- 文本特征：`external/MMRec/data/baby/text_feat.npy`（7050 × 384）
- 无答案盲输入：`data/manifests/llm_blind_inputs.jsonl`
- 样本与真值：`data/manifests/llm_evaluation_samples.csv`
- Kimi v2 响应：开发集与正式确认集既有 JSONL

不训练新推荐模型，不重新生成标签，不调用 API，不读取 Kimi 自然语言解释来构造基线。

## 3. 特征归一化

图像和文本嵌入空间不能直接比较原始 cosine。对每个目标商品、每种模态：

1. L2 归一化目录中所有 7050 个商品向量；
2. 计算目标商品与全目录商品的 cosine 分布；
3. 将每个历史商品 cosine 转成其在该目标—目录分布中的经验百分位；
4. 排除目标自身；若历史条目缺失对应模态，只在该模态的可用历史上聚合。

该转换使两个模态都处于 `[0,1]` 的相对稀有度尺度，避免直接比较不同嵌入空间的数值。

## 4. 冻结基线

### 主基线：Mean Percentile

分别计算图像与文本历史相似度百分位的均值；图像均值更高则预测 `image`，否则预测 `text`。完全相等时预测 `text`，并单独报告平局数。

### 次要基线：Max Percentile

分别取两个模态历史相似度百分位的最大值；图像更高则预测 `image`，否则预测 `text`。完全相等时预测 `text`。

### 参照线

- Majority Text：所有样本预测 `text`；
- Seed-999 Intervention Reference：使用单个训练种子 `seed_999_strict_label` 预测跨三个训练种子的共识标签。它使用行为干预信息，不是可部署解释器，只作为标签稳定性参照；
- Cross-seed Label Identity：直接读取目标标签，固定为 100%，只作实现完整性上界，不作为方法比较结果。

不根据正式集结果选择 mean/max，不拟合阈值；主基线始终是 Mean Percentile。

## 5. 冻结指标

对开发集和正式集完整报告：accuracy、text recall、image recall、macro recall、预测 image 比例、混淆矩阵。主要比较使用正式集 ITT；确定性基线没有 API/schema 失败。

对正式集使用 `user_id` 聚类 bootstrap 10,000 次、seed=`20260811`，报告：

- 各基线 accuracy 与 macro recall 的 95% 百分位区间；
- Mean Percentile 相对 Kimi v2 的配对 accuracy 与 macro recall 差值区间；
- Mean Percentile 相对 Majority Text 的配对 macro recall 差值区间。

不新增 p 值筛选；所有冻结基线均完整呈现。

## 6. 完整性检查

- 验证两个特征矩阵均为 7050 行且无非有限值；
- 验证 670 个目标样本全部可解析目标 item ID；
- 报告缺失历史模态条目数、零范数向量数、百分位平局数；
- 两次独立运行的机器结果必须字节级一致；
- 结果只能标记 `POSTHOC_EXPLORATORY`，不能替代 Day 15。

## 7. 预期产物

- `scripts/evaluate_non_llm_modality_baselines.py`
- `data/manifests/non_llm_modality_baselines.json`
- `results/day17_non_llm_modality_baselines.md`
