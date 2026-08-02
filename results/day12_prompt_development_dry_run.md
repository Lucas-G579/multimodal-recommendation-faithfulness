# Day 12：LLM 提示词、响应契约与开发集 Dry-run

## Material Passport

- Artifact ID：`faithrec-mm-day12-prompt-dev-v1`
- Artifact type：code experiment preparation / validation
- Status：`DRY_RUN_PASSED`
- Protocol version：`1.1-prompt-v1`
- Data scope：仅 `prompt_development` 80 条；正式确认集请求数为 0
- Verification status：`VERIFIED`（离线确定性复跑）；尚未验证任何真实 LLM API
- Next gate：固定多模态模型精确版本、API 提供方和预算后，才能运行开发集

## 1. 这一步为什么重要

数据已经准备好，但直接调用 LLM 会留下三个漏洞：

1. 模型可能输出无法稳定解析的散文或 Markdown；
2. 程序可能为了“救回数据”偷偷修正错误回答；
3. 如果正式确认集也被组装或查看，调提示词时就可能发生测试泄漏。

因此本阶段先在不调用任何模型的情况下，把提示词、JSON 输出格式、失败定义和
80 条开发请求固定下来。

## 2. 冻结的提示词原则

系统提示明确告诉 LLM：它没有推荐器内部状态，只能依据可见的目标商品和用户历史
生成事后解释，不能声称知道隐藏分数如何计算。这样可以避免把普通自然语言解释
误写成对 MGCN 内部因果机制的直接读取。

每个请求包含：

- 目标标题、描述和图片；
- 最多 5 条训练历史的标题和图片；
- 图片顺序与历史编号；
- 缺失图片的显式标记；
- 要求只返回一个 JSON 对象。

提示词文件：

- `prompts/llm_faithfulness_v1_system.txt`
- `prompts/llm_faithfulness_v1_user.txt`

固定哈希：

- system：`8873163F235774D0B108F51DED6D9DA577C4BBB50EEDFDCD57828EEACDCC5A99`
- user template：`95E7524198FA7B94EBC77F3E4857008087864E91F84C8E9BBA470FE312FA0FF7`
- response schema：`1F85C191406CE34C093AA41F22BA1B2A3E882291DA8FED88D88821C6C71CF2EE`

## 3. 严格响应契约

解析器只接受七个预注册字段。以下情况直接判为无效，不自动修补：

- JSON 外有 Markdown 围栏或解释文字；
- 缺字段或多字段；
- 枚举值不合法；
- share/confidence 不是数值或不在 `[0,1]`；
- 非拒答样本的图片与文字 share 之和不等于 1；
- `abstained` 与 `insufficient` 不一致；
- 解释为空或超过 1200 字符。

真实 API 阶段只允许完全相同的请求重放一次。第二次仍无效，就按预注册规则记作
失败，不能修改答案、删除样本或换一道题。

## 4. Dry-run 结果

| 检查 | 结果 |
|---|---:|
| 开发请求 | 80 |
| 正式确认集请求 | 0 |
| 答案字段泄漏 | 0 |
| 总图片输入 | 422 |
| 每请求图片数 | 4–6 |
| 缺失本地图片路径 | 0 |
| 提示词字符数 | 2,114–6,745 |
| 解析器单元测试 | 7/7 通过 |

开发请求 dry-run SHA-256：

`48E1053BB1FC2C588E6A701B463579BBE705CE3FAE6F7DB5ACEC8675CD99A33F`

测试覆盖一个合法普通回答、一个合法拒答，以及 Markdown、额外字段、比例错误、
拒答矛盾和布尔值伪装数值五类错误。连续重建请求时应要求哈希精确一致。

## 5. 当前不能做什么

当前环境没有检测到已配置的常见 LLM API 凭据，而且研究协议尚未写入一个精确的
多模态模型版本。为了避免擅自选模型、产生费用或使用会漂移的模型别名，本阶段不
发送真实请求。

下一步必须先冻结：

1. API 提供方；
2. 多模态模型的精确版本或可记录的实际返回版本；
3. temperature、seed、最大输出长度和图像细节等级；
4. 80 条开发集的费用上限；
5. 原始响应、重试、token 和费用日志路径。

这些确定后，只运行 80 条开发集。解析成功率与错误类型达到预设门槛后，再冻结最终
提示词；590 条正式确认集仍保持封闭。
