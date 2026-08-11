# Day 19：Mean/Max 一致性选择预测的独立 A+B 验证协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Origin Date：2026-08-11
- Verification Status：FROZEN BEFORE PREDICTION EXECUTION
- Version Label：`mean_max_selective_ab_validation_v1`
- Method Origin：Day 18 事后发现，commit `102834d`
- Validation Cohort：`A_or_B_sensitivity`，此前未用于方法设计
- Epistemic Status：INDEPENDENT SENSITIVITY VALIDATION；不是严格 A 级主确认实验

## 1. 研究问题

在 400 条用户完全隔离、text/image 平衡、但标签稳定性低于严格 A 级的 A+B 样本上，Day 18 冻结的“Mean 与 Max 一致才输出，否则拒答”规则能否保持高于平衡机会水平的选择性模态归因能力？

## 2. 数据边界

- 样本：`cohort == A_or_B_sensitivity`，固定 400 条、400 位唯一用户；
- 标签：`cross_seed_A_or_B_label`，固定 text=200、image=200；
- 与 prompt development、primary confirmatory、unstable overconfidence 三组用户交集均为 0；
- 盲输入覆盖 400/400；
- Day 19 不读取或计算 `unstable_overconfidence` 的特征预测，保留其后续压力测试用途；
- A+B 标签允许 B 级证据，不能与 Day 15 的严格跨种子 A 级真值等同。

## 3. 冻结方法

完全复用 Day 17 的目录百分位定义：

- Mean Percentile：比较图像/文本历史相似度百分位均值；
- Max Percentile：比较图像/文本历史相似度百分位最大值；
- Selective Agreement：Mean 与 Max 相同则输出该类别，不同则 `abstain`；
- Majority Text：全部预测 text，仅作平衡数据上的基线；
- 不拟合阈值，不修改聚合，不查看结果后改变拒答规则。

## 4. 主终点与成功门槛

### 主终点

Selective Agreement 在已回答样本上的宏平均召回率（分别在已回答 text/image 中计算后平均）。

使用按 `user_id` 聚类的 10,000 次 bootstrap、seed=`20260811`。由于每位用户恰好一条样本，聚类仍保留统一分析口径。

### 双重成功门槛

1. 总覆盖率至少 50%；
2. 已回答样本宏召回率的 95% bootstrap CI 下界高于 50%。

两项必须同时满足，才判定选择预测在 A+B 分布外敏感性集上通过。

## 5. 固定次要指标

- Selective 总覆盖率及 text/image 类别覆盖率；
- Selective 已回答 accuracy、text/image recall、macro recall；
- 把 abstain 计错的端到端 text/image recall 与 macro recall；
- 全量 Mean、Max、Majority Text 的 accuracy、分类别 recall、macro recall；
- Mean/Max 一致组与不一致组中，两套规则的完整表现；
- 已回答预测类别比例与拒答类别构成；
- 400 条完整混淆计数。

## 6. 冻结不确定性分析

10,000 次用户 bootstrap 报告：

- coverage 95% CI；
- selective answered macro recall 95% CI；
- selective end-to-end macro recall 95% CI；
- full Mean 与 full Max macro recall 95% CI；
- selective answered macro recall − full Mean macro recall 的配对差值及 95% CI。

不新增 p 值，不对多个指标选择性宣布成功；主门槛只使用第 4 节的两个条件。

## 7. 完整性与停止规则

- 400/400 样本必须完成；
- 特征和盲输入哈希写入结果；
- 输出严格 JSON；连续两次运行 SHA-256 必须一致；
- 结果无论成功或失败均停止，不修改规则、不将 unstable cohort 纳入调参；
- 若未来使用 unstable cohort，必须另行冻结压力测试协议。

## 8. 预期产物

- `scripts/validate_selective_agreement_ab.py`
- `data/manifests/selective_agreement_ab_validation.json`
- `results/day19_selective_agreement_ab_validation.md`
