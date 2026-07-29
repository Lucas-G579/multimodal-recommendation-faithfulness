# Failure Log

所有环境、数据和实验失败都在这里记录，不静默重试。

| 日期 | 阶段 | 命令/操作 | 症状 | 根因 | 处理 | 状态 |
|---|---|---|---|---|---|---|
| 2026-07-29 | 环境检查 | `python --version` | 默认 Python 为 3.13 | 系统 PATH 优先指向 Python 3.13 | 项目固定使用已有 Python 3.11 | 已规避 |
| 2026-07-29 | 获取上游 | `git clone --depth 1 https://github.com/enoche/MMRec.git external/MMRec` | 124 秒硬超时，仅留下不完整 `.git` | GitHub HTTPS 连接无数据返回，具体网络原因未验证 | 按硬超时终止残留 Git 子进程；未删除目录；等待决定重试方式 | 待处理 |
