# Part C GitHub交付与原始文件

本次同步由用户明确授权。全部保留的34条运行已完成；8条取消任务不再启动。没有为这次解读新增训练或选择更有利的seed。

- [原图对照与差异解释](results/INTERPRETATION_ZH.md)
- [可粘贴Check-in中文总结](results/CHECKIN_SUMMARY_ZH.md)
- [数值报告](results/REPORT.md)与[逐运行CSV](results/runs.csv)
- [执行计划快照](docs/EXECUTION_PLAN_ZH.md)与[工作日志快照](logs/WORK_LOG.md)
- [Release下载入口](https://github.com/PatriRUC/CSCI567_Grokking/releases/tag/part-c-checkin-2026-10-27)

普通Git保存源码、全部配置、结果CSV、五组PNG/PDF、验证记录、原始文件索引、计划与日志。大型文件放在同一仓库的Release中：

| 资产 | 内容 |
| --- | --- |
| `part-c-review-34.zip` | 可独立审阅的代码、图表、结果、配置、运行日志、34条逐步范数、参考fit检查点及离线校验器 |
| `part-c-full-runs-34.zip` | 34条完成运行的全部原始文件，含136个init/fit/latest/final检查点、34条范数轨迹、数据索引、配置、命令、环境和日志；逐文件SHA256清单内置 |
| `SHA256SUMS.txt` | 两个ZIP的SHA256，供下载后独立检查 |

MNIST公共缓存与Python虚拟环境不打包，按README重建；取消的部分运行不属于完成结果，仍保留本地并在`metadata/cancelled_artifacts_index.json`留有索引。早先本地审阅收据是历史阶段记录；本次远端资产信息见`metadata/publication.json`。

从本仓库克隆后，下载资产并检查哈希。PowerShell示例：

```powershell
Get-FileHash -Algorithm SHA256 part-c-review-34.zip
Get-FileHash -Algorithm SHA256 part-c-full-runs-34.zip
# 在mnist_dynamics目录执行，校验器仅用Python标准库：
python code/verify_bundle.py <下载目录>/part-c-review-34.zip
python code/package_full_runs.py --verify <下载目录>/part-c-full-runs-34.zip
```

完整原始包以`mnist_dynamics/`为顶层目录。需要重新分析或恢复检查点时，从仓库根目录解压到`Project check-in/实验部分`，再按README执行分析；解压前保存自己同名目录中的修改。审阅包可解压到独立空目录阅读，其顶层另有归档时的log.md与执行计划。

发布记录包含资产大小、SHA256、下载URL及打包时的代码提交。`manifest_sha256.json`与`review_context_sha256.json`专门描述审阅ZIP的快照，不是随后仓库文档的实时校验清单。发布收据因包含ZIP自身哈希而单独保存在Git中。归档内日志为打包时快照，仓库日志延续到发布核验前的提交边界；最终远端核验仍追加在原工作区log.md。此边界避免日志记录自己的提交哈希而产生循环。
