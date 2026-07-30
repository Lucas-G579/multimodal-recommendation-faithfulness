# Failure Log

所有环境、数据和实验失败都在这里记录，不静默重试。

| 日期 | 阶段 | 命令/操作 | 症状 | 根因 | 处理 | 状态 |
|---|---|---|---|---|---|---|
| 2026-07-29 | 环境检查 | `python --version` | 默认 Python 为 3.13 | 系统 PATH 优先指向 Python 3.13 | 项目固定使用已有 Python 3.11 | 已规避 |
| 2026-07-29 | 获取上游 | `git clone --depth 1 https://github.com/enoche/MMRec.git external/MMRec` | 124 秒硬超时，仅留下不完整 `.git` | GitHub HTTPS 连接无数据返回，具体网络原因未验证 | 按硬超时终止残留 Git 子进程；改用官方 codeload ZIP 并记录 SHA-256 | 已绕过 |
| 2026-07-29 | Smoke test | 从 `external/MMRec/src` 调用虚拟环境 | Python 可执行文件未找到，测试未启动 | 相对路径少一层 `..` | 修正为 `../../../.venv/Scripts/python.exe` | 已修正 |
| 2026-07-29 | LightGCN smoke | 单 epoch、Baby、seed 2026 | 模型成功构建后，Matplotlib 字体缓存写入用户目录被拒绝 | 默认 `MPLCONFIGDIR` 位于工作区外 | 在入口脚本中固定为 `outputs/cache/matplotlib` | 已修正并复跑通过 |
| 2026-07-29 | LightGCN smoke | 完成 epoch 0 后验证 | `AttributeError: numpy has no attribute float` | MMRec 评估器仍使用 NumPy 已删除别名 `np.float` | 保持上游快照不变，在项目入口提供 `np.float = float` 兼容层；扫描确认仅 4 处同类引用 | 已修正并复跑通过 |
| 2026-07-29 | 上游版本记录 | GitHub commits API | 匿名 API rate limit exceeded | 共享出口已耗尽匿名额度 | 保留 ZIP SHA-256；commit 标记待补记 | 待补记 |
| 2026-07-29 | Baby 图像特征 | gdown 默认 cookies | Google CDN TLS EOF，未留下文件 | cookies 路径获得的 CDN 连接不稳定 | 保持 TLS 校验，改用 `--no-cookies`；下载及 SHA-256 校验成功 | 已绕过 |
| 2026-07-29 | BM3 checkpoint | 正式训练，`saved=True` | 训练成功但没有 `saved/` 或 checkpoint | 当前 MMRec `Trainer.fit` 接收 `saved` 却没有保存实现 | 添加可审计补丁，仅在验证集刷新时保存最佳 state_dict | 已修正并复跑通过 |
| 2026-07-30 | BM3 checkpoint 恢复 | 独立构建 `TrainDataLoader` | `RecDataset` 缺少 `inter_num` | MMRec 把 `inter_num` 的初始化隐藏在 `RecDataset.__str__()`；`quick_start` 因日志输出偶然触发它 | 恢复脚本显式初始化三个数据切分的统计字段，并保留注释说明上游副作用 | 已修正并复跑通过 |
| 2026-07-30 | BM3 推理路径探针 | 用“任意非零差异”判断干预是否改变分数 | 图文清零出现最大约 `2.38e-7` 的差异 | GPU 稀疏矩阵重复计算存在末位浮点抖动，二进制零差异标准不适用 | 增加不做修改的重复计算对照；以 `1e-6` 为数值容差，并保留商品向量清零正对照 | 已修正并复跑通过 |
| 2026-07-30 | MGCN smoke | 构建图像 KNN 图 | `ModuleNotFoundError: torch_scatter` | 上游仅为度数聚合调用未安装的二进制扩展 | 用 PyTorch 2.3 原生 `Tensor.scatter_add_` 做语义等价替换，并保存独立补丁 | 已修正并复跑通过 |
