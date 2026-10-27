# MNIST Figure 3d-e：本地动力学复现

状态：按用户侧对话修订，34条保留运行均完成100000步并通过独立数据核验，136个检查点通过CPU审计。3d为18条；3e主实验为N=200/500/1000/2000/4000各3个seed，另保留N=8000 seed0作补充。原42条计划中后续8条已取消，不恢复训练。

[原图对照解读](results/INTERPRETATION_ZH.md) · [Check-in中文总结](results/CHECKIN_SUMMARY_ZH.md) · [结果说明](results/REPORT.md) · [A/B/D交接](results/HANDOFF.md) · [逐运行数据](results/runs.csv) · [完整产物下载](PUBLICATION.md)

本工作从官方 Omnigrok 的动力学 notebook 提取串行 MLP/AdamW 训练协议，增加独立训练随机数流、可恢复批次、完整评估、每步范数和检查点。所有训练在本机 Windows + RTX 4060 Laptop 8 GB 上运行；按用户新授权同步至团队仓库，大型原始文件随GitHub Release提供，见[交付说明](PUBLICATION.md)。

## 配置与来源

| 项目 | 本地配置 | 来源或差异 |
| --- | --- | --- |
| 网络 | 784→200→200→10，ReLU，含 bias | 官方 notebook/附录 A |
| 输入 | CPU FP64 /255，再传 GPU，无额外标准化 | 与官方 ToTensor 数值一致验证通过 |
| 损失 | one-hot element-mean MSE | 同时记录十倍的 per-sample sum-output loss |
| 优化器 | AdamW，lr=.001，betas=(.9,.999)，eps=1e-8，foreach=False，fused=False | 全局默认参数显式记录；无范数投影 |
| batch | 200，保留尾批，每 epoch 重新 shuffle | 自定义可恢复 sampler；训练 RNG 与评估分离 |
| 精度 | FP64，TF32 关闭，确定性算法 | 官方 notebook 默认 FP64；本机实测 |
| 数据 | 固定前 N 个训练样本，完整 10000 测试样本 | 对齐 notebook；论文文字“随机子集”存在差别 |
| Figure 3d | N=1000，wd=1e-4；alpha=1/2/4/6/8/10 | N 和 wd 来自原图；alpha 网格为项目选择 |
| Figure 3e | wd=5e-3；alpha=8；主实验N=200/500/1000/2000/4000；N8000仅seed0补充 | wd来自原图；alpha、N网格为项目选择，后续大N任务按用户要求取消 |
| 重复与预算 | 主比较各3 seeds，每条100000更新；33条主实验+1条补充 | 原42条完整计划保存在manifest_original_42.json；取消不当作测试右删失 |

因此本工作定位为**披露协议差异的定性复现**。不能将通用 notebook 的 wd=0.01 默认运行冒充 3d/3e，也不能宣称原论文逐点数值复刻。

用户取消记录在`metadata/user_cancellation.json`；`project_scope.py`从原始42条及取消名单交叉推导当前34条，避免按已有结果任意缩小验收范围。N8000单seed不计算跨seed均值/SD，也不纳入主趋势统计。

固定官方版本：`3a0a57346230a7255e5390fcd565812d00f1a281`。
[官方 notebook](https://github.com/KindXiaoming/Omnigrok/blob/3a0a57346230a7255e5390fcd565812d00f1a281/mnist/grokking/mnist-grokking.ipynb)。完整来源与未知项见 `metadata/provenance.json`。

## 指标和检查点

Step t 始终表示已经完成 t 次更新的模型。`norms.npy` 每行对应 step=0..100000，列为 global、layer1、layer2、layer3 的 L2 范数，全部含 bias。每步范数真实计算，不由稀疏点插值。

3d 评估：t=0..29、30/40/.../140、667 的倍数，以及最终 100000。参考运行另外每 100 步检查训练拟合，直到找到检查点。3e 评估：0/10/50/100/200，以及每 500 步至 100000。训练和测试全量评估，所有尾样本都计入。

3e 事件定义为首次完整测试准确率 **>0.60**，以正确个数的整数比较实现。记录前一次未越阈评估 step。预算内未达到时 t60=null，标为右删失；不能把 100000 当达到时间。仅当同 N 的三个 seed 全部达到时报告普通均值/样本 SD；失败和未完成运行不当作删失。

拟合 checkpoint 仅由训练指标选择：accuracy>=.99 且 element-mean MSE<=.001。保存模型、同一步 AdamW 状态、Python/NumPy/CPU/CUDA RNG、sampler permutation/cursor/generator、已提交的评估行和配置 hash。init/latest/final 与 fit（若达到）分别保存。中断时用 latest 恢复；日志和范数在恢复点后重算。

## 本地运行

重建环境需要Python 3.12、支持CUDA 12.4运行时的NVIDIA驱动及可用GPU。`metadata/requirements-freeze.txt`保存本次完整依赖版本。新机器可在本目录创建独立环境（已有本次环境可直接跳过）：

```powershell
python -m venv .venv
$python = '.venv/Scripts/python.exe'
& $python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
& $python -m pip install numpy==2.1.3 matplotlib==3.9.2
& $python code/train_dynamics.py --download-only
```

以上安装和数据下载命令需要网络；本次实际安装与下载来源已记录在工作区log.md。训练脚本在数据缺失时不会静默改用别的数据或设备。

本次解释器为工作区 `.venv-partc/Scripts/python.exe`。Windows PowerShell 从本目录执行：

```powershell
$python = 'D:\USC 作业\CSCI567\.venv-partc\Scripts\python.exe'
& $python code/analyze_dynamics.py
& $python code/audit_checkpoints.py
& $python code/collect_process_outcomes.py
$env:MPLCONFIGDIR = 'D:\USC 作业\CSCI567\tmp\mpl-partc'
& $python code/make_figures.py
& $python code/validate_pdf_exports.py
& $python code/write_report.py
& $python code/write_handoff.py
```

以上是现有文件的核验/重绘流程。PDF验收需本地Poppler的pdfinfo/pdftoppm；脚本只验证可渲染，实际查看渲染PNG后才能标记视觉验收完成。

重新训练接口见下例。验证目录必须为空；保留已有记录时使用`--output metadata/validation_new`。新训练使用新的checkout/输出目录，避免覆盖原始证据。当前默认队列仅含34条完成项，sweep会跳过它们；加载原42条备份也会先过滤8条用户取消ID。

```powershell
& $python -u code/validate_dynamics.py --output metadata/validation_new
& $python -u code/run_sweep.py --dry-run
& $python -u code/run_sweep.py --max-concurrent 1
# 对确实未完成且仍获授权的运行，可从其latest检查点恢复：
# & $python code/train_dynamics.py --resume runs/<run_id>/checkpoints/latest.pt
```

分析默认要求用户保留的全部34条完成；阶段性分析须显式传`--allow-partial`，图中标PARTIAL。原42条与用户取消记录均保留，取消的部分运行不混入正式统计。分析与绘图读取本地raw文件。

四并发通过数值一致性验证；实际运行曾根据吞吐与显存余量切换到串行。CPU亲和性试验无收益，已恢复原设置。调度器另有经真实Windows共享锁验证的有界写入重试。训练源码和超参数始终不变，过程见log.md。elapsed_seconds为进程内perf_counter墙钟差，含等待/暂停间隔；N4000 seed0跨夜，不等于纯GPU计算时间，并发耗时也不能直接相加为项目总时长。

## 验证证据与审阅文件

- `metadata/validation/validation.json`：50 步官方更新对照、输入逐位一致、完整尾批、评估无副作用、严格阈值逻辑、跨进程断点恢复。
- `metadata/validation/benchmark.json`：当前 GPU 的 FP64 更新/评估实测，不借用 Part B 的速度。
- `configs/manifest.json`：当前34条保留条件；`manifest_original_42.json`：原计划；`metadata/user_cancellation.json`：取消依据。
- `runs/<run_id>/console.log`、`command.json`、`resolved_config.json`、`environment.json`、`metrics.csv`、`norms.npy`、`events.json`、`status.json`、`checkpoints/`：原始证据。
- `results/runs.csv`、`summary.csv`、`metrics.csv`、`qa.json`：独立重算的结果与完整性校验。
- `figures/figure3d.*`、`figure3e.*`、逐seed图、`figure3d_diagnostics.*`与参考诊断图：PNG/PDF均供审阅。
- `metadata/checkpoint_audit.json`、`process_outcomes.json`、`pdf_export_validation.json`：检查点、退出码来源及PDF验收证据。
- 工作区根目录 `D:\USC 作业\CSCI567\log.md`：操作日志、线上下载记录、问题与修复、实验判断。

结果与边界见REPORT.md。N200三个seed均未在预算内越过60%；N500/1000/2000/4000均达到，t60均值随N增加而下降。N8000仅单seed补充，不能当成三seed稳定结果。参考运行和这些曲线不能单独证明范数是唯一机制；方向干预留给后续工作。
