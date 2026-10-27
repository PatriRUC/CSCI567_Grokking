# Part C 本地复现实验工作日志

## 2026-10-26：启动与约束

- 用户授权：按 Part C 执行计划复现 Figure 3d-e；所有计算先在本地完成，完整成果交用户审阅；工作过程、问题与判断持续记录于本文件。
- 本轮不推送 GitHub、不使用远程实验服务器。论文与依赖可通过网络读取/下载，实验计算和文件保存在本机。
- 已读计划：`Project/PartC_Figure3de_Codex执行计划.md`。现有小组 checkout 只有 Part B；官方参考代码版本为 `3a0a57346230a7255e5390fcd565812d00f1a281`。
- 本地 GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 566.24；D 盘可用约 392 GB。CIM CPU/内存查询受环境限制，未获得结果；这不阻碍训练环境检查。
- 现有参考 checkout 均无未提交修改。尚无正式 Part C 代码、训练结果或可用环境确认。
- 下一步：在工作区建立隔离的运行环境与正式本地目录，核实官方历史中是否有 3d-e 的专用配置；实现并验证串行 FP64 训练，实测成本后执行冻结的矩阵。

日志规则：详细数值、命令 stdout/stderr 和配置保存在对应实验目录；本文件记录它们的位置、重要操作、结果、问题、修复和科学判断。不在日志中记录私密推理，只记录可审阅的决策依据。

## 本地工作区与协议冻结

- 正式 checkout：`Project/CSCI567_Grokking`；本地分支 `part-c-local-reproduction`，尚未推送。Part C 独立目录为 `Project check-in/实验部分/mnist_dynamics`。
- 本机现有 Python 3.8 的 torch 2.4.1 只有 CPU；建立工作区 `.venv-partc`（Python 3.12.14），正在安装官方 CUDA 12.4 wheel 的 torch 2.5.1 / torchvision 0.20.1。网络隔离造成首次下载 DNS 失败，按工具审批后重试。
- 已读取官方仓库完整 main 历史，MNIST 只有 ce1df03 初次上传，没有额外 3e 扫描脚本。冻结计划中的 3d 六 alpha、3e 八 N、各三 seed，共 42 runs；3e alpha=8 等缺失设置明确为项目选择，标记 qualitative。
- 训练主实现已写入 code/train_dynamics.py：FP64、无投影、可恢复 sampler、独立训练 RNG、全量评估、每步全局/分层范数、训练指标拟合检查点、原子落盘。尚未通过运行验证，不启动正式长跑。

## 2026-10-26T20:40:50-07:00：继续与线上操作登记

- 新增用户要求：本地处理不了时可使用官方 GitHub 仓库内容；所有线上操作记录于本文件。实验继续本地执行，完整材料完成后交审阅。
- 线上操作（补记此前已执行）：对 `https://github.com/KindXiaoming/Omnigrok.git` 执行 `git fetch --unshallow origin`，仅下载官方历史。首次 sandbox DNS 失败，获批重试成功；历史中无额外 MNIST dynamics 扫描文件。
- 线上操作：读取 `https://pytorch.org/get-started/previous-versions/`，核对 torch 2.5.1 / torchvision 0.20.1 的 CUDA 12.4 官方安装组合。
- 线上操作：从 `https://download.pytorch.org/whl/cu124` 下载 torch 2.5.1+cu124、torchvision 0.20.1+cu124 及依赖到 `.venv-partc`；从 Python 包索引安装 numpy 2.1.3、matplotlib 3.9.2。未上传项目文件。
- 线上操作：通过 torchvision MNIST downloader 获取公开数据，文件现已在 `mnist_dynamics/data/MNIST/raw`。下一步检查 torchvision 定义的镜像地址与数据 hash，并将来源归档。
- 中断恢复：此前工具进程句柄已失效；源码五个脚本和 MNIST 八个文件均存在。重新验证当前环境，不把失效句柄当成功证据。

## CUDA 与首轮数值验证

- 本机 CUDA 检查成功：torch 2.5.1+cu124、RTX 4060 Laptop。沙箱内导入耗时异常且 matplotlib 默认缓存目录不可写；沙箱外本机检查约 7 秒完成，后续 GPU 命令使用工具审批的本地执行，不使用远程算力。
- 数据来源镜像由 torchvision 列为 `http://yann.lecun.com/exdb/mnist/` 与 `https://ossci-datasets.s3.amazonaws.com/mnist/`；当前下载输出未完整保留，不能断言具体成功镜像。八个原始/压缩文件 SHA256 已归档于 `metadata/data_hashes.json`。
- 首轮 validate 在输入逐位一致性断言失败：GPU FP64 /255 与 ToTensor 的 CPU FP64 /255 存在舍入差异。修复为 CPU 先转 FP64 并除以 255，再传 GPU，保持官方数据处理数值。尚未启动任何正式运行。

## 本地验证通过，开始正式参考运行

- `code/validate_dynamics.py` 完成：输入转换逐位相等；50 步更新对照最大参数误差 0；连续 200 步 vs 100 步保存/新进程恢复至 200 步，模型、AdamW、sampler 和范数轨迹均完全相同；评估无副作用、503 样本尾批、严格 >60% 与删失逻辑均通过。证据：`metadata/validation/validation.json` 及 process0/1/2.log。
- 本地 500-step 测时：更新加范数约 0.002 秒/步；完整评估 N=1000 约 0.029 秒、N=30000 约 0.101 秒。每条 100k 估计 206-219 秒，不含全部启动/检查点开销；42 条顺序运行约 2.5-3 小时，正式参考实测后校正。
- 启动单个 reference：`python -u code/run_sweep.py --only fig3d_n1000_a8_s0`。已冻结 N=1000, alpha=8, wd=1e-4, seed=0, FP64, 100000 更新；逐步范数、训练拟合检查和检查点均启用。

## 参考运行中间观察与后续运行策略

- 参考运行在 step=3500 满足预设训练拟合标准并保存 fit.pt。至约 6 万步，训练保持高准确率，范数先下降后回升；不预设普通 AdamW 的范数必须单调下降，继续完整预算，不按测试集结果调参。
- `uv pip freeze` 首次使用沙箱默认 cache 出现权限失败；重试指定工作区 `tmp/uv-cache` 后成功，记录于 metadata/requirements-freeze.txt。此操作仅读取本地依赖，没有新的线上请求。
- 准备额外验证四个独立本地训练进程与串行 smoke 的参数/范数完全一致，用来评估是否可在不改变算法/配置的情况下提高设备利用率。只有参考运行完成及并发验证通过后才并发正式条件。

## 首条完整参考运行完成

- `fig3d_n1000_a8_s0` 完成 100000 更新，进程内耗时 224.9 秒。全量测试首次 >60% 的观测 step=36685，上一次未达到的观测为 36018；训练拟合 checkpoint step=3500。
- norms.npy 有 100001×4 个有限数值，逐层范数平方和与全局范数平方一致。正在进一步验证实际 fit checkpoint 的恢复，并从 raw 日志生成阶段图。
- 开始本机四进程 smoke 并发一致性检查，命令 `python -u code/validate_concurrency.py --workers 4`；此为执行效率验证，不改变任何正式训练配置。

## 并发验证与正式扫描

- 四个独立本地进程的 200-step smoke 全部与串行参考完全一致，模型/优化器最大误差 0、范数误差 0。记录：metadata/concurrency4/validation.json。并发会增加每条运行墙钟时间，但不改变数值；先观察首批完整条件的整体吞吐。
- 扫描调度器加入最多四进程、每条命令/输出/退出码、配置 hash 检查、完成项跳过和失败后停止新增任务。它自动把每次正式运行的开始与结束记录到工作区 log.md，原始 stdout/stderr 保留于各 run 的 console.log。
- 下一条正式命令：`python -u code/run_sweep.py --max-concurrent 4`，跳过已完成参考条件，完成冻结的剩余41条。本轮仍只在本地，未新增线上操作。

- 2026-10-26T20:47:57-07:00：开始本地运行 `fig3d_n1000_a1_s0`；命令见 `runs/fig3d_n1000_a1_s0/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T20:47:57-07:00：开始本地运行 `fig3e_n1000_a8_s0`；命令见 `runs/fig3e_n1000_a8_s0/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T20:47:57-07:00：开始本地运行 `fig3e_n8000_a8_s0`；命令见 `runs/fig3e_n8000_a8_s0/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T20:47:57-07:00：开始本地运行 `fig3d_n1000_a1_s1`；命令见 `runs/fig3d_n1000_a1_s1/command.json`，stdout/stderr 见 console.log。

## 恢复证据和阶段可视化

- 实际 reference fit.pt 从 step=3500 恢复到3700，与原运行这200步的范数最大误差0；拟合训练指标重新计算通过。证据：metadata/reference_checkpoint_replay/verification.json。
- `analyze_dynamics.py --allow-partial` 独立核对参考运行的配置、每步范数、完整样本计数、loss单位及t60；随后 `make_figures.py` 生成带 PARTIAL 标记的阶段图。人工查看 reference_diagnostics.png 与 figure3d.png，曲线和标签可读；最终将待42条完成后重算覆盖，未将单seed阶段结果当最终交付。
- 完整报告生成器已准备，只有 results/qa.json 显示42条完整通过时才会写 results/REPORT.md；大体积运行/checkpoint不纳入Git，小型代码/元数据/结果可供审阅。

## 并发吞吐调整（2026-10-26）

- 四并发正式运行约38000步时，每进程约377秒；参考串行100000步用224.9秒。并发数值一致性已通过，但本机四并发总吞吐略差，因此停止调度器PID34664，仅停止新增任务，保留四个训练子进程继续完成。
- 四个实际训练进程PID32896、17312、32260、27800在停止调度器后仍存在；待它们退出后自动启动 `python -u code/run_sweep.py --max-concurrent 1`，已完成项按配置hash跳过。训练代码与协议均未改变。本次无线上操作。

- 补充独立分析校验：逐条核对完整测试评估时点与冻结协议一致，复核previous_eval_step、末次评估和阈值事件；参考运行通过。3e图增加样本SD误差棒；审阅包新增离线SHA256核验脚本。准备A/B/D交接生成器，待42条验收后才生成最终交接文件。
- 已更新执行计划的实际环境、参考结果、阶段状态；原设计描述保留，实际接口以正式README为准。

- 首批尚在训练的原始CSV检查：3d alpha=1 的seed0/1首次观测t60分别为5/4步；3e seed0的N1000为29000步、N8000为2500步。这些是进行中运行的已观测事件，不能替代完整三seed结论。报告将同时统计训练拟合早于t60的数量及3e逐seed非单调点，避免只按图片预设结论。

- 交接材料脚本已覆盖A的150–250字摘要、B/C实际协议对照、D参考fit.pt的独立输出续训命令。README补齐环境重建命令。新增CPU checkpoint审计：最终阶段将核对每个保存模型的范数、FP64、训练索引、优化器步数和sampler状态；当前不影响正在运行的训练。

- 2026-10-26T21:04:32-07:00：开始本地运行 `fig3d_n1000_a1_s2`；命令见 `runs/fig3d_n1000_a1_s2/command.json`，stdout/stderr 见 console.log。
- 四并发保留进程已正常完成并通过独立完整性检查：fig3d_n1000_a1_s0；100000 steps；t60=5；fit=667；最终测试accuracy=0.7905；进程内耗时=987.3672946000006秒。
- 四并发保留进程已正常完成并通过独立完整性检查：fig3e_n1000_a8_s0；100000 steps；t60=29000；fit=3500；最终测试accuracy=0.8788；进程内耗时=987.9456252999953秒。
- 四并发保留进程已正常完成并通过独立完整性检查：fig3e_n8000_a8_s0；100000 steps；t60=2500；fit=26000；最终测试accuracy=0.8916；进程内耗时=981.2928618000005秒。
- 四并发保留进程已正常完成并通过独立完整性检查：fig3d_n1000_a1_s1；100000 steps；t60=4；fit=667；最终测试accuracy=0.7729；进程内耗时=986.9993507999752秒。

- 首批完整结果显示alpha=1两个seed的最终测试准确率为79.05%/77.29%，低于早期约90%的水平；早越过阈值不等于更高末步性能。分析输出补充观测峰值及末步相对峰值差，仅作描述，不用于选择模型/配置；最终报告将披露非单调轨迹。

- 2026-10-26T21:08:02-07:00：完成 `fig3d_n1000_a1_s2`，100000 steps，进程内计时 206.3 秒；t60=4，拟合 step=667。这是单次结果，最终统一汇总。

- 2026-10-26T21:08:02-07:00：开始本地运行 `fig3d_n1000_a2_s0`；命令见 `runs/fig3d_n1000_a2_s0/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T21:11:32-07:00：完成 `fig3d_n1000_a2_s0`，100000 steps，进程内计时 206.7 秒；t60=6，拟合 step=667。这是单次结果，最终统一汇总。

- 2026-10-26T21:11:32-07:00：开始本地运行 `fig3d_n1000_a2_s1`；命令见 `runs/fig3d_n1000_a2_s1/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T21:15:02-07:00：完成 `fig3d_n1000_a2_s1`，100000 steps，进程内计时 206.3 秒；t60=6，拟合 step=667。这是单次结果，最终统一汇总。

- 2026-10-26T21:15:02-07:00：开始本地运行 `fig3d_n1000_a2_s2`；命令见 `runs/fig3d_n1000_a2_s2/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T21:18:33-07:00：完成 `fig3d_n1000_a2_s2`，100000 steps，进程内计时 206.2 秒；t60=6，拟合 step=667。这是单次结果，最终统一汇总。

- 2026-10-26T21:18:33-07:00：开始本地运行 `fig3d_n1000_a4_s0`；命令见 `runs/fig3d_n1000_a4_s0/command.json`，stdout/stderr 见 console.log。

- 9条完整运行通过更新后的独立分析。3d alpha1：三seed t60均值4.333步、SD0.577，末步测试均值76.59%，观测峰值均值91.07%；alpha2：t60均值6步、SD0，末步87.9133%，峰值90.98%。峰值仅用于描述后期回落；未改变冻结配置与预算。

- 2026-10-26T21:22:41-07:00：完成 `fig3d_n1000_a4_s0`，100000 steps，进程内计时 244.6 秒；t60=667，拟合 step=1334。这是单次结果，最终统一汇总。

- 2026-10-26T21:22:41-07:00：开始本地运行 `fig3d_n1000_a4_s1`；命令见 `runs/fig3d_n1000_a4_s1/command.json`，stdout/stderr 见 console.log。

- 运行到alpha4 seed1时吞吐从约2.1ms/step降到约4.4ms/step。只读检查GPU 52°C、2565MHz、55%利用率，CPU总占用约28%。没有修改其他进程或系统电源设置。
- 对本任务实际调度器PID40540、训练进程PID22240做可恢复的CPU亲和性试验：原mask1048575（20个逻辑CPU），试用mask4095（前12个）。开始基准：step38000、elapsed176.699s。训练代码、参数和随机流均不变。第一次PowerShell命令存在foreach管道语法错误，解析失败未执行；修正后成功。待测后决定保留或恢复。

- 亲和性试验从38000步/176.699s到49000步/230.153s，11000步用53.454s（约4.86ms/step），未改善。已将两个任务进程恢复原mask1048575，后续子进程继承原设置。不会改变其他进程或系统设置；接受当前实际吞吐继续实验。此试验不改变GPU数值协议。

- 恢复亲和性后串行速度仍慢于初始基准（alpha4 seed1在step57000时进程内313.164s）。当前GPU此前仅55%利用率；第一批四并发的4条约988s完成，作为总吞吐参考。决定保持所有训练配置不变，让当前单条完成，再对剩余批次恢复已验证的四并发；仅停止自己的串行调度器PID40540，保留worker22240。CPU亲和性保持原值，不修改系统设置。

- 后续资源读数：AC供电、电池100%；GPU利用率98%、温度70°C、功率73W，显存已用5441/8188 MiB、空闲2517 MiB。与初始实验时资源状态不同，不能保证再加三进程安全。没有终止或修改其他应用。
- 给尚未启动的新调度器加显存启动检查：请求四并发时空闲显存不足5500 MiB则自动串行，避免用初始空闲状态推断当前容量；此项只控制启动数量，不改变训练数值。当前等待队列启动时会应用这一检查。

- 2026-10-26T21:33:56-07:00：请求四并发时GPU空闲显存为3184 MiB，低于保守启动余量5500 MiB；自动改为串行，保留相同配置。

- 2026-10-26T21:33:56-07:00：开始本地运行 `fig3d_n1000_a4_s2`；命令见 `runs/fig3d_n1000_a4_s2/command.json`，stdout/stderr 见 console.log。

- 保留的alpha4 seed1完成100000步，进程内665.658秒；t60=140、fit=1334。随后队列启动时空闲显存3184 MiB，启动检查自动选择串行，已跳过11条完成条件，继续alpha4 seed2。由于旧调度器停止，此条完成信息在此补记。

- 已完成11条的44个检查点提前通过CPU独立审计：模型FP64/有限性、训练索引、源码hash、optimizer step、sampler epoch/cursor、范数全部一致；最大范数差1.4210854715202004e-14。证据metadata/checkpoint_audit_progress.json，明确complete=false；最终仍需42条全部审计。

- 2026-10-26T21:43:10-07:00：完成 `fig3d_n1000_a4_s2`，100000 steps，进程内计时 548.1 秒；t60=667，拟合 step=1334。这是单次结果，最终统一汇总。

- 2026-10-26T21:43:10-07:00：开始本地运行 `fig3d_n1000_a6_s0`；命令见 `runs/fig3d_n1000_a6_s0/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T21:51:49-07:00：完成 `fig3d_n1000_a6_s0`，100000 steps，进程内计时 511.9 秒；t60=18009，拟合 step=2668。这是单次结果，最终统一汇总。

- 2026-10-26T21:51:49-07:00：开始本地运行 `fig3d_n1000_a6_s1`；命令见 `runs/fig3d_n1000_a6_s1/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T22:00:38-07:00：完成 `fig3d_n1000_a6_s1`，100000 steps，进程内计时 521.2 秒；t60=16008，拟合 step=2668。这是单次结果，最终统一汇总。

- 2026-10-26T22:00:38-07:00：开始本地运行 `fig3d_n1000_a6_s2`；命令见 `runs/fig3d_n1000_a6_s2/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T22:09:35-07:00：完成 `fig3d_n1000_a6_s2`，100000 steps，进程内计时 528.8 秒；t60=18009，拟合 step=2668。这是单次结果，最终统一汇总。

- 2026-10-26T22:09:35-07:00：开始本地运行 `fig3d_n1000_a8_s1`；命令见 `runs/fig3d_n1000_a8_s1/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T22:18:14-07:00：完成 `fig3d_n1000_a8_s1`，100000 steps，进程内计时 510.9 秒；t60=32683，拟合 step=4002。这是单次结果，最终统一汇总。

- 2026-10-26T22:18:14-07:00：开始本地运行 `fig3d_n1000_a8_s2`；命令见 `runs/fig3d_n1000_a8_s2/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T22:25:20-07:00：完成 `fig3d_n1000_a8_s2`，100000 steps，进程内计时 421.9 秒；t60=36685，拟合 step=4002。这是单次结果，最终统一汇总。

- 2026-10-26T22:25:20-07:00：开始本地运行 `fig3d_n1000_a10_s0`；命令见 `runs/fig3d_n1000_a10_s0/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T22:32:27-07:00：完成 `fig3d_n1000_a10_s0`，100000 steps，进程内计时 420.9 秒；t60=62698，拟合 step=4669。这是单次结果，最终统一汇总。

- 2026-10-26T22:32:27-07:00：开始本地运行 `fig3d_n1000_a10_s1`；命令见 `runs/fig3d_n1000_a10_s1/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T22:40:24-07:00：完成 `fig3d_n1000_a10_s1`，100000 steps，进程内计时 470.0 秒；t60=53360，拟合 step=4669。这是单次结果，最终统一汇总。

- 2026-10-26T22:40:24-07:00：开始本地运行 `fig3d_n1000_a10_s2`；命令见 `runs/fig3d_n1000_a10_s2/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T22:49:08-07:00：完成 `fig3d_n1000_a10_s2`，100000 steps，进程内计时 517.3 秒；t60=62031，拟合 step=4669。这是单次结果，最终统一汇总。

- 2026-10-26T22:49:08-07:00：开始本地运行 `fig3e_n200_a8_s0`；命令见 `runs/fig3e_n200_a8_s0/command.json`，stdout/stderr 见 console.log。

## Figure 3d 完整阶段验收

- 18/18条3d运行均完成100000步，连同前期2条3e共20/42条通过原始数据独立分析；没有失败。3e剩余扫描已自动开始N200 seed0。
- 三seed t60均值±样本SD（步）：alpha1为4.333±0.577；alpha2为6±0；alpha4为491.333±304.264；alpha6为17342±1155.278；alpha8为35351±2310.556；alpha10为59363±5209.437。全部18条在预算内达到阈值。
- alpha6/8/10三个seed均有train-fit早于t60；alpha1/2/4则t60早于严格train-fit标准。结论须结合两种不同阈值定义，不将其当相同难度任务。alpha1末步测试均值76.59%，低于其观测峰值均值91.07%。未根据结果删去任何seed。
- 更新报告均值精度为两位小数、范数图显式包含末步、manifest核验42个唯一ID；训练源码及冻结配置保持不变。正在生成阶段图核对完整3d布局，整体仍标PARTIAL，因为3e尚未完成。

- Figure3d主图、逐seed六面板和参考accuracy/MSE/norm诊断PNG已实际查看，轴、图例、末步与标记可读。整体PARTIAL标签保留到42条全部完成。alpha4虽提前越过60%，在train-fit后仍有进一步改善，因此结果说明增加完整曲线的补充解释，避免把t60当全部动力学。
- 20条完整运行的80个checkpoint通过CPU审计，最大范数差2.842170943040401e-14；训练源码hash始终f5c37991aabeb0e7880a0d3c962ac5703069ac45980b4c5ad787123fa404feaa。阶段证据metadata/checkpoint_audit_progress.json。
- 剩余工作：等待已运行队列完成3e；重做全42条独立分析和checkpoint审计；生成最终图、REPORT/HANDOFF/ABSTRACT；实际查看3e图与所有最终图；更新README/执行计划的完成状态；打包并离线验证审阅ZIP；完成后再交用户审阅。Goal继续保持active，尚不能标为complete。

## Goal 自动继续：Figure 3e 扫描

- 已重新轮询训练session11042，确认进程仍在运行；没有重复启动队列。当前20/42完成，N200 seed0约40000步，训练accuracy100%、测试约20.95%。这只是进行中观测，仍保留完整100000步预算。
- 本阶段没有新增线上操作。接下来继续等待现有本地队列，按原定42条范围完成全部实验及最终交付。

- 2026-10-26T22:58:33-07:00：完成 `fig3e_n200_a8_s0`，100000 steps，进程内计时 557.2 秒；t60=None，拟合 step=1000。这是单次结果，最终统一汇总。

- 2026-10-26T22:58:33-07:00：开始本地运行 `fig3e_n200_a8_s1`；命令见 `runs/fig3e_n200_a8_s1/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T23:08:18-07:00：完成 `fig3e_n200_a8_s1`，100000 steps，进程内计时 576.8 秒；t60=None，拟合 step=1000。这是单次结果，最终统一汇总。

- 2026-10-26T23:08:18-07:00：开始本地运行 `fig3e_n200_a8_s2`；命令见 `runs/fig3e_n200_a8_s2/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T23:17:59-07:00：完成 `fig3e_n200_a8_s2`，100000 steps，进程内计时 572.4 秒；t60=None，拟合 step=1000。这是单次结果，最终统一汇总。

- 2026-10-26T23:17:59-07:00：开始本地运行 `fig3e_n500_a8_s0`；命令见 `runs/fig3e_n500_a8_s0/command.json`，stdout/stderr 见 console.log。

- N200三seed均完成100000步并通过阶段独立分析，三条t60均为null（0/3 reached），最终测试accuracy分别50.45%、52.83%、50.14%。按预先固定规则保留右删失，不将100000当实际达到时间，也不报告普通t60均值。当前总进度23/42，N500扫描进行中。

- 检查仓库入口README：根目录仅有3a–c链接，check-in与实验部分入口仍写3d–e待接手。最终42条验收后需同步这三个入口，保留其他组员待整合任务的真实状态。
- 审阅打包器补入执行计划、三个仓库入口文档及工作日志的归档副本，并以SHA256索引覆盖这些上下文文件；离线核验脚本同时验证。最终打包仍以全42条QA和checkpoint审计通过为门槛，当前没有提前生成最终包。

- 2026-10-26T23:27:00-07:00：完成 `fig3e_n500_a8_s0`，100000 steps，进程内计时 532.8 秒；t60=39500，拟合 step=2000。这是单次结果，最终统一汇总。

- 2026-10-26T23:27:00-07:00：开始本地运行 `fig3e_n500_a8_s1`；命令见 `runs/fig3e_n500_a8_s1/command.json`，stdout/stderr 见 console.log。

- 按计划补齐figure3d_diagnostics.png/pdf：六个初始化条件在共同评估点上的三seed均值，包含train/test element-mean MSE、global与3层含偏置范数。原始记录已有，无新增训练。
- 已实际查看新增图，发现底部图例间距不足，调整图例到图框下方后重绘；最终全42条完成时会再次统一生成去除PARTIAL标签。结果说明新增拟合时与末步MSE对照，不将达到train-fit视为动力学停止或范数单调收缩。

- 2026-10-26T23:35:09-07:00：完成 `fig3e_n500_a8_s1`，100000 steps，进程内计时 480.7 秒；t60=37500，拟合 step=2000。这是单次结果，最终统一汇总。

- 2026-10-26T23:35:09-07:00：开始本地运行 `fig3e_n500_a8_s2`；命令见 `runs/fig3e_n500_a8_s2/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T23:38:31-07:00：完成 `fig3e_n500_a8_s2`，100000 steps，进程内计时 198.4 秒；t60=38500，拟合 step=2000。这是单次结果，最终统一汇总。

- 2026-10-26T23:38:31-07:00：开始本地运行 `fig3e_n1000_a8_s1`；命令见 `runs/fig3e_n1000_a8_s1/command.json`，stdout/stderr 见 console.log。

- N500三seed已完成并通过阶段分析：t60分别39500、37500、38500步，均值38500±样本SD1000；最终测试均值82.0133%。当前26/42完成，N1000的其余seed已接续。N200三条删失与N500三条有效达到均完整保留。
- 最终PDF验收使用PDF技能的本地Poppler流程，已确认bundled pdfinfo/pdftoppm可用；新增validate_pdf_exports.py，在全42条QA通过后渲染5张PDF，逐页检查，再记录结构/哈希证据。未安装新软件或发起线上操作。
- 补充诊断图调整后的PNG已再次实际查看，图例间距正常。正式训练代码与冻结配置没有改变。

- 2026-10-26T23:42:09-07:00：完成 `fig3e_n1000_a8_s1`，100000 steps，进程内计时 213.5 秒；t60=26500，拟合 step=3500。这是单次结果，最终统一汇总。

- 2026-10-26T23:42:09-07:00：开始本地运行 `fig3e_n1000_a8_s2`；命令见 `runs/fig3e_n1000_a8_s2/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T23:46:27-07:00：完成 `fig3e_n1000_a8_s2`，100000 steps，进程内计时 253.0 秒；t60=29500，拟合 step=3500。这是单次结果，最终统一汇总。

- 2026-10-26T23:46:27-07:00：开始本地运行 `fig3e_n2000_a8_s0`；命令见 `runs/fig3e_n2000_a8_s0/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T23:49:58-07:00：完成 `fig3e_n2000_a8_s0`，100000 steps，进程内计时 205.1 秒；t60=19500，拟合 step=15000。这是单次结果，最终统一汇总。

- 2026-10-26T23:49:58-07:00：开始本地运行 `fig3e_n2000_a8_s1`；命令见 `runs/fig3e_n2000_a8_s1/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T23:53:28-07:00：完成 `fig3e_n2000_a8_s1`，100000 steps，进程内计时 204.8 秒；t60=18500，拟合 step=15500。这是单次结果，最终统一汇总。

- 2026-10-26T23:53:28-07:00：开始本地运行 `fig3e_n2000_a8_s2`；命令见 `runs/fig3e_n2000_a8_s2/command.json`，stdout/stderr 见 console.log。

- 2026-10-26T23:56:58-07:00：完成 `fig3e_n2000_a8_s2`，100000 steps，进程内计时 205.1 秒；t60=19000，拟合 step=15000。这是单次结果，最终统一汇总。

- 2026-10-26T23:56:58-07:00：开始本地运行 `fig3e_n4000_a8_s0`；命令见 `runs/fig3e_n4000_a8_s0/command.json`，stdout/stderr 见 console.log。

- 结果说明补充每个N的训练拟合step及逐seed的t60−fit时间差，明确正负值和两种阈值难度不同。N2000三seed的t60为19500/18500/19000，fit为15000/15500/15000；数据量对t60和拟合后时间差的影响应分别解释，不能仅凭t60称作等待间隔。

- 2026-10-27T13:42:44-07:00：完成 `fig3e_n4000_a8_s0`，100000 steps，进程内计时 49541.7 秒；t60=4000，拟合 step=18500。这是单次结果，最终统一汇总。

- 2026-10-27T13:42:44-07:00：开始本地运行 `fig3e_n4000_a8_s1`；命令见 `runs/fig3e_n4000_a8_s1/command.json`，stdout/stderr 见 console.log。

- 监测cell126发生tool-host timeout，随后返回stale host generation；重新轮询原unified exec session11042成功，原队列仍live，已经完成32/42并运行N4000 seed1，没有重启训练或重复提交。观测超时只影响监测连接。
- 已完成32条通过更新后的独立原始数据分析；新快照区分32条数据核验与此前20条checkpoint审计，最终仍须全42条审计。正式训练源码和配置未改动，无新增线上操作。

- 2026-10-27T13:48:22-07:00：完成 `fig3e_n4000_a8_s1`，100000 steps，进程内计时 335.5 秒；t60=4000，拟合 step=32500。这是单次结果，最终统一汇总。

- 2026-10-27T13:48:22-07:00：开始本地运行 `fig3e_n4000_a8_s2`；命令见 `runs/fig3e_n4000_a8_s2/command.json`，stdout/stderr 见 console.log。

## 调度器状态写入故障与恢复

- 原session11042的调度器因PermissionError [WinError 5]退出码1：atomic_json替换metadata/sweep_status.json.tmp到sweep_status.json时被Windows共享占用阻止。保留的N4000 seed2子进程随后正常完成100000步；重新检查时无本任务Python进程存活，34/42条status为complete，因此恢复不会重复运行这34条。
- 仅在run_sweep.py的调度状态写入层增加有界重试（8次、最大退避0.5秒）；train_dynamics.py保持不变，训练算法/配置/RNG没有改变。真实Windows文件锁回归验证通过：3次重试后成功，0.357秒；证据metadata/scheduler_io_validation.json，训练源码hash仍为f5c37991aabeb0e7880a0d3c962ac5703069ac45980b4c5ad787123fa404feaa。
- 调度器同时增加每条后续子进程的process_exit.json，保留真实退出码；先前未单独留存的退出码不补造。将从已有成功日志或独立成果校验说明其证据来源。
- 下一条本地命令：python -u code/run_sweep.py --max-concurrent 1，按配置hash跳过34条完成项，继续剩余8条。没有新增线上操作。

- 2026-10-27T13:55:34-07:00：开始本地运行 `fig3e_n8000_a8_s1`；命令见 `runs/fig3e_n8000_a8_s1/command.json`，stdout/stderr 见 console.log。

## 用户取消后续队列

- 2026-10-27T13:56:04-07:00：用户在侧对话批注1明确取消后续N8000、N16000、N30000任务，只让正在进行的一条跑完。
- 检查时N4000 seed2已正常完成100000步；未发现匹配的Python调度器或训练进程。本次没有停止任何训练进程。
- 已从默认configs/manifest.json移除8条未完成任务：N8000 seeds1/2、N16000 seeds0/1/2、N30000 seeds0/1/2。原42条manifest完整备份为configs/manifest_original_42.json。
- 保留全部34条已完成结果，包括此前已完成的N8000 seed0；没有删除数据、模型或已有结果。取消明细见metadata/user_cancellation.json，队列已无pending项。
- 后续汇总需按用户缩减范围调整原42条硬编码验收门槛；N8000单seed只能作为补充结果。侧对话仅完成本次取消请求，未执行其他实验或最终汇总。

- 2026-10-27T13:58:13-07:00：取消后的复查发现旧调度器仍持有原队列，并启动了已取消的N8000 seed1。已停止调度器PID29812及该取消任务的wrapper PID11732、实际训练PID40768；未停止或修改N4000已完成结果。进程操作证据metadata/cancellation_process_actions.json。
- N8000 seed1的部分产物保留，状态明确标为cancelled_by_user，原状态另存；它不计入34条保留结果。其余7条仍取消。
- run_sweep.py增加用户取消名单检查：加载manifest和每次启动前均跳过取消ID，即使误选原42条备份manifest也不会重新启动这8条。仅变更调度控制，train_dynamics.py及训练超参数未改动。

- 检测到configs/manifest.json在原42条队列已加载后被意外截为前34条（文件mtime为2026-10-27 13:56:04），原因/作者未知。没有按34条重新定义完成范围。保留原文件快照metadata/manifest_observed34_before_restore.json，核对42个配置文件、所有已存在resolved_config及完成项config_hash后恢复原42条清单；证据metadata/manifest_restoration_audit.json。原队列未重启，仍按其已加载的42条范围继续。
- 首轮清单诊断脚本遗漏code模块路径，导入失败且未写文件；补充sys.path后完成上述保留与恢复。训练源码和具体配置文件均未改动。
- 打包与离线核验增强为交叉验证42个唯一manifest ID、42条结果及逐运行配置hash/状态/必需数据文件，避免仅凭qa.json的完成标记通过。

- 默认计划与原42条备份计划的dry-run均核验：有效队列34条且全部complete，取消8条不会启动，pending0。复查时原manifest保留42个条件，运行时按取消名单过滤；取消记录而非原计划条数决定后续启动范围。未执行真实训练。
- 最后进程核验：匹配本项目run_sweep/train_dynamics的Python进程数为0。取消请求已完成，当前没有继续训练的任务。

## 同步用户侧对话的最终范围

- 进一步读取metadata/user_cancellation.json与工作日志，确认34条清单并非意外损坏：用户在侧对话批注1明确要求取消后续N8000/16000/30000，只保留当时进行的N4000运行完成。此前“原因未知并恢复42条”的判断现已纠正；该恢复已撤回，manifest_restoration_audit.json标记superseded并记录原因。保留原始42条备份及全部已完成数据。
- 当前范围为34条：3d18条、3e主实验15条（N200/500/1000/2000/4000各3seed）、N8000 seed0一条补充。其余8条为用户取消；N8000 seed1在22000步的部分产物保留且排除主结果，不与右删失混同。
- 全34条独立数据分析与136个checkpoint的CPU审计通过，最大范数差2.842170943040401e-14。退出码来源如实区分27条调度成功日志还原与7条未留存记录，不补造历史退出码。
- 报告、摘要与交接说明已按此范围生成；摘要196字符。修复报告读取中文日志JSON时Windows默认GBK解码错误，显式使用UTF-8。科学训练代码hash与配置文件未改变。
- 最终5张PDF已本地生成并由Poppler渲染成功，接下来查看全部导出、更新入口文档并打包。此阶段未启动新训练，也无新增线上操作。

- 最终5张PNG和5张PDF渲染图均已实际查看，标题、图例、坐标、参考时刻与单seed补充标签无裁切/重叠；PDF源文件哈希与已查看版本记录于metadata/pdf_export_validation.json。
- 取消保护dry-run验证通过：默认清单34条、原42条备份经取消过滤后34条、显式选择取消ID后0条；没有启动训练。三个仓库入口已更新，Part B源码/数据无改动。
- 结果报告已逐表核对：N200三条右删失；N500/1000/2000/4000的t60均值分别38500/28333.33/19000/3500步；N8000单seed2500步单列补充。接下来构建本地ZIP并核对CRC、SHA256、修订范围及逐运行配置/状态一致性。

## 最终本地验收与归档

- 用户修订范围的34条保留运行、136个检查点、五组PNG/PDF、参考恢复、统计规则、配置/日志与A/B/D交接均已完成核对。摘要220字符，其中中文字符达到150以上。详细要求—证据审计见metadata/completion_data_audit.json。
- 审阅包初版离线验证通过：458个文件哈希，34条结果/配置/状态交叉一致，取消8条依据可追溯；checkpoint_index.json与cancelled_artifacts_index.json已生成，原始模型/数据被Git忽略且未夹带Part B改动。
- 计划最终验收清单已据实际证据完成，旧内部续接快照移除并由完成数据审计替代。下面重新归档最终文档状态，再验证最终ZIP。最终校验收据保存在ZIP旁，避免归档文件自引用哈希。

- 最终审阅ZIP复核通过：34条保留运行、8条用户取消、458个文件哈希一致；ZIP为91,278,981字节，SHA256=08ca29d446b3a902aeb5fd1e6a80114ac74322e9d4343970b58924336f9c892e。路径：Project/PartC_Figure3de_本地审阅包.zip。校验收据：Project/PartC_Figure3de_完成验收.json及ZIP旁.sha256文件。
- 已确认本任务无活动训练或调度进程；Git仅有三个入口README更新和新Part C目录，数据、范数轨迹及模型权重不进入Git。按用户要求保留本地供审阅，没有推送远端。
- 本次审阅范围的代码、配置、图表、日志、原始结果索引、参考检查点和A/B/D交接均已完成；原42条中的取消任务不再列为待运行工作。当前工作区log.md包含最终校验记录，ZIP内log.md为归档时快照。

## 结果解读、Check-in总结与GitHub同步（用户新授权）

- 用户要求对照论文原图解释Figure3d–e的复现现象、差异与可能原因，并将所有已完成项目材料同步到PatriRUC/CSCI567_Grokking。本次明确授权GitHub同步，替代此前本地审阅阶段不推送的限制；保留既有34条范围及8条用户取消记录，不启动新训练。
- 将读取本地论文与结果，核对arXiv官方论文页面/官方GitHub代码及GitHub归档文档，fetch团队仓库最新状态，再提交并同步代码、数值、图、说明、日志及可下载原始产物。所有线上动作继续记录在此文件。

- 线上核对已完成：git fetch origin --prune显示团队main仍为ed040b7；gh api验证当前协作者Why1Z-zyz（yanze zhou）拥有push权限，main未受保护，仓库尚无Release。仅为本仓库设置该已验证账号的name与GitHub noreply署名，未改全局Git配置。
- 已读取arXiv v2官方摘要页面及GitHub大文件/Release官方文档；大体积产物采用Release资产，代码/图表/结果/说明正常Git提交。现有34条原始产物共664,313,388字节，其中136个checkpoint为552,942,780字节。
- 本地论文第5页Figure3已渲染并逐图查看。系统pdftotext误选MiKTeX，因注册表/配置访问受限失败；bundled目录也无pdftotext，改用已安装pypdf读取全部16页成功，无新安装。论文第4–5页正文与附录A已核对。
- 新增原图对照解读与可粘贴Check-in总结。明确：官方notebook调用处评估完整测试集、采用FP64，不能把这两点误写成已确认差异；训练/评估RNG耦合和数据成员是候选来源，原因尚未隔离。核对的作者代码仍固定在3a0a573。
- 已把原图对照、可粘贴总结、数值报告、交接材料互相链接，并更新生成器以保留解释入口。新增完整原始包脚本（仅标准库），计划把34条运行全部文件和136个检查点归档并逐文件校验，不重新训练。普通Git不包含大体积二进制；同仓库Release提供完整原始ZIP及审阅ZIP。
- Git将保存执行计划与工作日志快照；归档日志以打包时为边界，仓库日志以发布核验前提交为边界，最终远端检查继续追加在本工作区log.md，避免自引用提交哈希。
- 完整原始包已构建并通过逐文件SHA256、ZIP读取CRC、34条配置/状态与136个检查点存在性校验：494个哈希条目，516,429,902字节，SHA256=9637f31385f1ebccb1ec614c8cf272535295d7752796c4fa8f0f965627037f3e。原始训练代码仍为f5c37991...，没有改动。
- 准备Git内容共123个新增轻量文件约3.77MB；最大单文件约1.74MB。凭据/私钥常见模式扫描无匹配；Part B无改动。保留原本地审阅ZIP及历史收据，另建本次发布审阅包，避免覆盖已验收历史文件。
- 本次发布审阅包离线校验通过：466个文件/上下文哈希、34条结果与配置/状态一致、8条取消依据可追溯；91,341,878字节，SHA256=56a9911c060d30c8c53e54e27f57665ac52e5478313b000176cb31220876ebc6。SHA256SUMS.txt与Release说明已生成。旧本地91,278,981字节审阅包及收据保留为历史。
- 下一步线上操作：再次fetch确认基线后，以普通非force push同步main；创建part-c-checkin-2026-10-27草稿Release，上传两个ZIP及校验文件，核对GitHub返回大小/摘要，再发布。用户已明确授权同步；不修改团队Part B文件。
- 本地提交894fb26暂未推送。Git提示Windows autocrlf会转换文本，发现这可能破坏新机器checkout后的训练源码/原始manifest字节哈希。因此新增Part C局部.gitattributes固定文本LF、PNG/PDF/NPY等二进制不转换，不影响Part B。发布收据包含ZIP哈希，排除在ZIP本身之外避免自引用；按最终版本重打审阅包，完整原始包数值/源码未变化，无需重建。
- 最终发布审阅包重打后再次通过467个文件/上下文哈希校验：91,343,302字节，SHA256=cef7bf083fe6b50069b6b47be3d81efa0bd3e33406a76ea0487a1b11c9fd6efb。SHA256SUMS.txt与Git发布准备记录已更新，旧56a9911c摘要为本轮中间构建。
- 推送前Git blob核对捕获原42条manifest的历史字节为CRLF，简单LF规则会改变其b85fc7db哈希；已为该冻结fixture单独设为-text保留原始字节，其余源码保持LF。此问题在本地被拦截，尚未上传错误版本。
