# Day 14：Kimi 强制选择提示词 v2 开发验证与冻结

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-08-11
- Verification Status：ANALYZED
- Version Label：`kimi_prompt_v2_development_frozen`
- Scope：80 条开发集；正式确认集 API 请求 0 条
- Overall Confidence：CAUTION（开发调参集，不是确认性结果）

## 1. 执行结果

| 项目 | v1 | v2 |
|---|---:|---:|
| 唯一开发样本 | 80 | 80 |
| API 尝试 | 92 | 96 |
| 最终严格有效 | 74 | 75 |
| 最终 API 失败 | 5 | 0 |
| 最终 schema 失败 | 1 | 5 |
| 发生重试的样本 | 12 | 16 |
| 重试后恢复 | 6 | 11 |

v2 的五个永久失败均为解释超过冻结的 1200 字符上限。原始运行日志因异常类绑定问题
把它们标为 `api_error`；分析层根据 `error_type=ResponseValidationError` 纠正为
`schema_invalid`。原始日志不修改，后续运行器的异常绑定已经修正。

v2 全部尝试的 API usage 合计 152,825 token，其中缓存 44,535。响应日志 SHA-256：

`4F4CF509D9C79D763F2A9A7D5649D9FA3198089274A82120BF522D714695D47C`

## 2. v2 开发指标

75 个有效回答中，text 55、image 20，没有 `both` 或 `history` 逃避项。

| 指标 | v2 |
|---|---:|
| ITT 准确率 | 72.50% |
| ITT 准确率用户聚类 bootstrap 95% CI | 62.50%–82.28% |
| text 召回率 | 72.60% |
| image 召回率 | 71.43% |
| ITT 宏平均召回率 | 72.02% |
| 宏平均召回率用户聚类 bootstrap 95% CI | 51.54%–88.53% |
| 仅有效回答准确率 | 77.33% |
| 平均自报置信度 | 0.836 |
| 高置信错误 | 14/80 |
| 文字 share 与连续干预强度 Spearman | 0.186 |
| Spearman 用户聚类 bootstrap 95% CI | −0.046–0.397 |

v2 相对 v1 的宏平均召回率配对提升为 0.693，用户聚类 bootstrap 95% 区间为
0.486–0.862。此区间只描述开发集内的提示词修订效果，不能当成独立测试结果。

## 3. 判断

v2 解决了 v1 的 `both` 坍缩，也没有退化成永远猜多数类 text：开发集 text 与 image
召回率分别为 72.60% 和 71.43%。因此冻结 v2，用于一次性确认性评测。

但连续相关区间跨 0，不能声称模型已经准确估计依赖强度。开发集只有 7 个 image
真值，宏平均区间很宽。真正的论文结论只能来自 590 条从未用于调提示词的确认集。

从此不再基于开发集构造 v3，也不再修改 v2 的提示词、schema、模型、thinking 模式、
最大输出或重试次数。

## 4. 冻结配置

- 模型：`kimi-k2.6`
- thinking：disabled
- response format：JSON Object
- max completion tokens：512
- temperature / seed：提供方不支持
- 每样本最多两次完全相同请求
- 第二次仍失败：保留失败并继续
- system SHA-256：`174BAE1BC178B1A1B253377848898688439667EE02CE41FB6ECD937785900FB4`
- user SHA-256：`D6C62EED6077CDFBD147532ADAF2545D4666ACA59894385DC5007355C044B4C7`
- schema SHA-256：`A1DF34AADBBECD507CCF91D821FC321D7A08B3634EBC22FEF42C8DE51BDFCA04`

## 5. 确认集准备状态

590 条确认集请求将由独立脚本从盲输入生成。生成器只选择
`primary_confirmatory` 的 sample ID，并在序列化内容中拒绝 label、cohort、contrast、
rank change、训练种子等答案字段。生成与哈希验收不等于授权外传；在用户另行明确
同意前，确认集不会发送到 Kimi。

离线生成已验收：590 个唯一请求、3152 张图片路径、0 条开发请求、0 个答案字段；
连续两次生成 SHA-256 均为：

`24022DB9515B24E52453A474F11F5B98FD58CF575862E42E450ABF6D2A6EAADB`

11 类统计谬误检查沿用 Day 13，并新增两点：

- Garden of forking paths：v2 是开发后修订，已通过版本化与停止创建 v3 控制；
- Survivorship bias：正式 ITT 必须把 API/schema 永久失败计错，不能只报 75 个有效项。
