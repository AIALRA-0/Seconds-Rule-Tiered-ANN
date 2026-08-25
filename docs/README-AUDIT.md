# README 与公开资源审计记录

## 1 审计范围

本记录覆盖中英文 README、Python 源码、实验配置、Shell 脚本、测试、Markdown 报告、PDF 报告和新发布的本地图片

<div align="center">

表 1.1 审计结论

| 范围 | 结论 | 证据 |
| --- | --- | --- |
| README 结构 | 中文优先，英文备份，章节、表格与图片结构一致 | `README.md`、`README.en.md` |
| 源码与配置 | 44 个受 Git 跟踪文件，实验核心位于 `src/` 与 `configs/` | Git 文件清单 |
| 测试 | 隔离环境中 6 项测试通过 | Python 3.12.7、Faiss 1.15.0、Pytest 9.1.1 |
| 报告 | Markdown 报告和 76 页 PDF 均保留 | `FINAL_REPORT.md`、`FINAL_REPORT.pdf` |
| 结果图片 | 从 PDF 无损提取 3 张默认工作负载图 | `docs/assets/readme/` |
| 已知秘密模式 | 未发现凭据、私有部署地址或用户账号硬匹配 | 完整仓库文本与 PDF 文本扫描 |

</div>

## 2 身份与 PDF 脱敏

原 Markdown 报告和 PDF 首页包含个人姓名
本轮统一替换为 `AIALRA-0 Contributors`，同时清理 PDF 作者元数据，报告正文、76 页结构和实验图表保持不变

原始 PDF 仍可从 Git 历史恢复
公开镜像如需彻底移除历史身份字段，需要另行评估历史重写对提交摘要和外部引用的影响

## 3 事实校正

原 README 的仓库树列出 `data/` 与 `results/`，这两个目录实际被 `.gitignore` 排除，当前提交没有包含数据集、原始 CSV 或完整结果目录

`scripts/get_sift1m.sh` 只验证数据文件是否存在，不负责下载
`configs/default.yaml` 把工作负载写成 `default`，当前工作负载工厂不接受该名称，因此 README 不把默认配置描述成可直接完成全流程

## 4 测试边界

现有 6 项测试覆盖策略预算、滑动窗口行为、Seconds Rule 间隔更新、IOPS 延迟方向和 IVF 列表编号范围
它们不覆盖完整 SIFT1M 数据流、全部参数扫掠、图表再生成或报告数值复算

历史实验原始目录没有纳入 Git
README 中的数值和三张图来自 2025-12-15 报告快照，属于模型实验结果，不是生产 SSD 或真实在线服务测量值

## 5 发布门槛

公开新结果前需要保留配置快照、依赖锁定、原始 CSV、聚合 CSV、图表生成日志和可复算校验结果
`scripts/one_click.sh` 会把本地根路径、环境与提交摘要写入忽略目录，任何日志进入公开提交前都要重新扫描并脱敏
