# Reproduction Notes

## 2026-07-29：本机环境基线

- OS：Windows / PowerShell
- GPU：NVIDIA GeForce RTX 4050 Laptop GPU
- VRAM：6141 MiB
- NVIDIA driver：536.45
- `nvidia-smi` CUDA：12.2
- 默认 Python：3.13.0（不用于本项目）
- 项目 Python：3.11
- Git：2.54.0.windows.1
- Conda：未检测到
- uv：未检测到

## 约定

- 使用 `py -3.11` 创建虚拟环境；
- 完整训练前先在本地小切分 smoke test；
- 外部代码放入 `external/`，记录上游 URL 和 commit；
- 每次训练保存配置、seed、环境、commit 和输出路径；
- 不直接将不同数据切分下的指标与论文数字比较。

## 待验证

- MMRec 在 Python 3.11 下的依赖兼容性；
- 当前驱动支持的 PyTorch CUDA wheel；
- MMRec 示例数据和最小模型命令；
- 本地训练时可用显存。

