# FaithRec-MM

面向多模态推荐解释的模态归因错配审计项目。

当前阶段：Day 1，建立可复现环境并跑通 MMRec 最小示例。

## 研究问题

多模态推荐模型实际依赖的图片、文本和协同信号，与 LLM 解释声称使用的证据是否一致？

## 环境

- Windows 10/11
- Python 3.11
- PyTorch（版本将在 MMRec 兼容性验证后锁定）
- NVIDIA RTX 4050 Laptop GPU，6GB VRAM

## 目录

- `docs/`：研究协议、复现笔记与失败日志
- `external/`：外部研究代码，不纳入当前仓库
- `scripts/`：环境检查和可复现实验入口
- `configs/`：后续固定的数据、模型和审计配置
- `results/`：可提交的小型汇总结果
- `outputs/`：训练输出与 checkpoint，不提交

完整路线见 [暑期两个月执行方案.md](./暑期两个月执行方案.md)。

