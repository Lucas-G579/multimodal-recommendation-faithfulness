# Day 13：Kimi 开发集结果与强制选择提示词 v2

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run + validate
- Origin Date：2026-08-11
- Verification Status：ANALYZED
- Version Label：`kimi_prompt_development_v1_analysis`
- Scope：80 条 `prompt_development`；正式确认集 0 条
- Overall Confidence：CAUTION（开发集、类别极不平衡、外部 API 不可确定性复跑）

## 1. 实验执行结果

- 模型：`kimi-k2.6`
- 提示词：v1，自由选择 image/text/both/history/collaborative/insufficient
- 唯一样本：80/80
- API 尝试：92
- 每样本最多尝试：2
- 最终严格有效：74
- 最终 API 失败：5
- 最终 schema 失败：1
- 发生重试的样本：12
- 第二次恢复成功：6（50%）
- 正式确认集请求：0
- 全部有返回的尝试合计：115,738 token（其中缓存 27,342）

五个永久 API 失败均为 `URLError`。永久 schema 失败是非拒答时图片与文字 share 之和
不等于 1。失败样本没有删除或换题，在意向分析中均计为错误。

响应日志 SHA-256：

`6394A8E0A27E53695405406C17E2A17E1E05BEA830EDC6FBFF7796CB5DBB8DF9`

## 2. 开发集统计

开发集真实标签为 text 73 条、image 7 条，严重不平衡。因此总准确率可能被多数类
欺骗，宏平均召回率才是方向识别的核心描述量。

| 指标 | 结果 |
|---|---:|
| 意向分析准确率 | 5.00% |
| text 召回率 | 5.48% |
| image 召回率 | 0.00% |
| 宏平均召回率 | 2.74% |
| 仅有效回答准确率 | 5.41% |
| 平均自报置信度 | 0.808 |
| 文字 share 与连续干预强度 Spearman 相关 | −0.045 |
| 错误且 confidence ≥ 0.8 | 51/80（63.75%） |

74 个有效回答的主证据分布：

| primary_evidence | 数量 |
|---|---:|
| both | 65 |
| history | 5 |
| text | 4 |
| image | 0 |

## 3. 如何解释

v1 的失败不是“模型不会生成 JSON”。74 个回答通过了严格解析，但 87.84% 的有效
回答选择 `both`，没有任何回答选择 `image`。模型把“图文都存在”当成了安全答案，
没有完成我们真正要测的“哪一种模态更强”。

相关系数接近零说明自报的文字占比没有追随连续干预强度。但这是开发性描述，不做
显著性或泛化声明；80 条样本、尤其仅 7 条 image，不足以精确估计总体效果。

因此 v1 **不得进入正式确认集**。这个负结果应保留为自然解释条件的发现，而不是
删除：自然的事后解释会高置信度地把两种证据都写进去，却不一定反映推荐模型的实际
模态敏感方向。

## 4. 提示词 v2：强制方向选择

v2 保留相同图文输入和解释字段，只修改输出决策：

- `primary_evidence` 只允许 `image`、`text`、`insufficient`；
- 证据充分时必须选择更强模态，不能回答 `both`；
- 解释仍可同时讨论图文，share 仍可表达两者贡献；
- image 标签必须满足 image share > text share，text 标签反之；
- 只有真正证据不足时才能拒答并把两个 share 设为 0。

这样做不是把真实干预标签告诉模型，也没有按单题答案修改输入；它只是让输出空间与
预注册主指标一致。v1 作为自然解释条件保留，v2 作为强制归因条件候选。

v2 离线验收：80 条开发请求、422 张图片、0 条确认集、0 个答案字段；13/13 个 v1+v2
协议测试通过，连续两次 dry-run 哈希一致。

固定哈希：

- system：`174BAE1BC178B1A1B253377848898688439667EE02CE41FB6ECD937785900FB4`
- user template：`D6C62EED6077CDFBD147532ADAF2545D4666ACA59894385DC5007355C044B4C7`
- schema：`A1DF34AADBBECD507CCF91D821FC321D7A08B3634EBC22FEF42C8DE51BDFCA04`
- requests：`25BC43741ACB48ED86BD8F85394EAB4AF81F86063C65A68BA44DDFDB71711FFF`

## 5. 统计谬误检查

覆盖：11/11。

| 类型 | 严重度 | 判断 |
|---|---|---|
| Simpson's paradox | NOTE | 尚无分层反转检验；正式分析必须同时报告 text/image 分层，不能只报总体准确率。 |
| Ecological fallacy | NOTE | 分析单位与推断单位均为用户—商品对；不把聚合比例解释为每个用户行为。 |
| Berkson's paradox | CAUTION | 样本经过严格 A 级稳定性筛选，不代表全部推荐样本。 |
| Collider bias | NOTE | 当前没有回归控制变量；未发现由 IV/DV 共同导致的控制项。 |
| Base-rate neglect | RED_FLAG（若只报准确率） | 标签为 73:7；必须报告宏平均及分类别召回率。 |
| Regression to mean | NOTE | 非前后测、未按极端 LLM 输出选择样本，不适用。 |
| Survivorship bias | CAUTION | 6 个最终失败必须保留在 ITT；仅有效结果另列，不能替代 ITT。 |
| Look-elsewhere effect | CAUTION | 多个开发指标只用于诊断；不得挑最好指标冒充主结果。 |
| Garden of forking paths | CAUTION | v2 是观察 v1 开发结果后的修订；必须版本化并在确认集前冻结。 |
| Correlation ≠ causation | CAUTION | LLM share 与干预强度相关只能说明一致性，不能证明 LLM 读取了 MGCN 内部机制。 |
| Reverse causality | NOTE | 不作方向因果声明；两套输出来自不同系统，只有配对关联。 |

## 6. 下一道门

尚不能运行正式确认集。必须先在同一 80 条开发集运行 v2，检查：

1. 严格有效率；
2. image/text/insufficient 分布；
3. 宏平均召回率与连续相关；
4. 是否从 `both` 退化成永远猜多数类 `text`；
5. 高置信错误率。

开发集已经用于设计 v2，所以 v2 的开发指标也不是无偏论文结果。v2 一旦通过预先
规定的最低门槛，就冻结其哈希，并只在正式确认集运行一次。
