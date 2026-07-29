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
