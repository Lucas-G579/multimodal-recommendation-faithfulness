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

- MMRec 在 Python 3.11 / PyTorch 2.3 下的运行兼容性；
- 当前驱动支持的 PyTorch CUDA wheel；
- MMRec 示例数据和最小模型命令；
- 本地训练时可用显存。

## MMRec 上游快照

- 上游：`https://github.com/enoche/MMRec`
- 获取方式：GitHub codeload `master` 分支 ZIP
- 获取日期：2026-07-29
- ZIP SHA-256：`AF335CC31B4FDBDB39DD5E8CCB0CF01BA254639110970964AA9A9281D79EF76D`
- ZIP 大小：465617 bytes
- 许可证：GPL-3.0
- 说明：原始 Git clone 超时，因此当前快照没有 commit metadata；GitHub API 又因共享出口匿名
  rate limit 无法返回 `master` commit。后续通过远端引用补记，当前以 ZIP SHA-256 固定快照。

## 依赖兼容策略

上游依赖固定为 Python 3.7.11、PyTorch 1.11、NumPy 1.21.5 和
Pandas 1.3.5，无法直接安装到 Python 3.11。项目先采用：

- Python 3.11.3；
- PyTorch 2.3.0 + torchvision 0.18.0，官方 CUDA 12.1 wheel；
- NumPy 1.26.4；
- Pandas 2.0.3；
- SciPy 1.11.4。

只复现 LightGCN 与 BM3 所需路径；暂不安装 `torch_geometric`，避免为未使用模型
引入额外二进制依赖。若 smoke test 出现 API 不兼容，先记录并做最小补丁，不修改
上游目录中的原始快照。

## Baby 交互数据

- 来源：MMRec README 指向的官方 Google Drive 公共文件夹
- 文件 ID：`1i2IB2bdxu_jMSxr2IvZ04MG54ySgI2JN`
- SHA-256：`E0ABB033EA5CC538BB2BECD8C3DC50B619F28F7974BA61F5ED11CE27CF405940`
- 文件大小：4362239 bytes
- 行数：160792
- 用户数：19445
- 商品数：7050
- 重复 user-item 行：0
- 缺失值：0
- 训练/验证/测试：118551 / 20559 / 21682

## LightGCN smoke test

- 日期：2026-07-29
- 命令：`.\.venv\Scripts\python.exe .\scripts\run_mmrec_smoke.py`
- 模型：LightGCN
- 数据：Baby
- seed：2026
- epoch：1
- n_layers：1
- reg_weight：1e-4
- train batch：4096
- GPU：RTX 4050
- 训练时间：约 0.68 秒
- 验证：Recall@20 0.0203，NDCG@20 0.0087
- 测试：Recall@20 0.0191，NDCG@20 0.0084
- 结论：训练、验证、测试管线完整通过；指标仅用于 smoke test。

兼容层：

- 将 Matplotlib 缓存重定向到项目内；
- 在入口脚本提供 `np.float = float`，不修改 MMRec 上游源码。

仍存在但不阻塞：

- `torch.LongTensor([numpy arrays])` 性能警告；
- `torch.sparse.FloatTensor` 弃用警告。
