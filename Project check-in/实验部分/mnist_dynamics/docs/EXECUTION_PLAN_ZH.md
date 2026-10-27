# Part C：Figure 3d-e 完成计划（供 Codex 后续执行）

制定日期：2026-10-25，America/Los_Angeles。

**执行结果（2026-10-27）：按用户侧对话修订范围完成34条保留运行，进入最终本地审阅。** 下文原42条设计保留为历史依据；实际范围、接口及状态以正式README、取消记录和验收文件为准。

**最终执行范围（2026-10-27用户侧对话修订）：取消后续N8000/16000/30000任务，只让当时运行的一条完成。** 因此保留34条完整运行：3d为18条，3e主实验为N200/500/1000/2000/4000各3seed共15条，N8000 seed0一条仅作补充。其余8条取消，原42条计划保留为历史设计。34条原始数据与136个检查点已通过核验，报告和摘要/交接材料已生成；5张PNG/PDF已渲染检查。最终打包以此用户保留范围为准，不恢复已取消训练，也不声称完成原42条全部网格。

## 1. 任务目标与完成定义

完成任务分配中的 C：研究初始化尺度、训练样本量与实际泛化时间的关系。

1. Figure 3d：N=1000，比较不同初始化倍率的训练/测试准确率曲线，附 MSE、全局范数与各层范数。
2. Figure 3e：扫描 N，记录首次观测到测试准确率超过 60% 的 optimizer step；展示每次运行和预算内未达到的情况。
3. 至少一条完整参考运行：从初始化到预算结束，保存每步更新后的全局范数，并提供仅依赖训练指标选定的拟合检查点。
4. 拟合检查点能恢复模型、AdamW 状态、随机数及后续批次；通过恢复后继续训练与未中断运行的一致性检查。
5. 交付源码、配置、命令、原始日志、图、逐运行结果和简短说明，可交给 A 汇总，并支持 D 后续设计干预。

本阶段使用普通 AdamW 自由训练。不要加入 Part B 的每步范数投影，不实施切向放大/减小干预。复现成功与否均如实报告；不能只保留出现 grokking 的 seed。

## 2. 启动时先读这些状态

### 已核实材料

| 来源 | 已核实内容 |
| --- | --- |
| 本地任务分配 | `D:\USC 作业\CSCI567\Project\Project_Check-in_任务分配.md`，重点第 5、6、8、9 节 |
| 小组仓库 | `https://github.com/PatriRUC/CSCI567_Grokking`，HEAD=`ed040b7fa7ebbc95b928bb990829bed266ffcb6f` |
| 小组当前状态 | 固定版本已提交 Part B 三 seed 材料；README 明确 Part C 尚待继续，未包含 dynamics 代码/结果 |
| 官方仓库 | `https://github.com/KindXiaoming/Omnigrok`，HEAD=`3a0a57346230a7255e5390fcd565812d00f1a281` |
| 官方训练入口 | `mnist/grokking/mnist-grokking.ipynb`，首段说明是结果 “akin to Figure 3(d)” |
| 本地论文 | `D:\USC 作业\CSCI567\Project\Grokking_Ziming Liu.pdf`：Figure 3 在 PDF 第 5 页，附录 A 在第 13 页 |
| 本次环境 | 当前 bundled Python 有 NumPy，缺少 torch、torchvision、matplotlib；其他环境及 GPU 可用性未调查 |

计划阶段通过 `git ls-remote` 核对两个 HEAD，均与本地参考副本一致。当时GitHub历史页面读取失败；执行阶段已拉取官方main完整历史并检查其4个commit，仍未找到Figure 3e专用扫描配置。该结论限定于已固定的公开main历史，未声称覆盖其他未公开来源。

参考副本位置：

```text
D:\USC 作业\CSCI567\tmp\partb-plan\CSCI567_Grokking
D:\USC 作业\CSCI567\tmp\partb-plan\Omnigrok
```

它们用于查证，不自动视为正式长期 checkout。后续先检查现有正式仓库和组内可用环境；尽量复用。读取适用 AGENTS.md，检查 git status，不覆盖同学或用户未提交的工作。

### 执行状态（后续每阶段更新）

- [x] 阅读任务分配、Part B 计划、官方动力学 notebook。
- [x] 查看论文原图，核对 3d/3e 的 weight decay、预算和 60% 图轴。
- [x] 核对当前小组/官方仓库版本，确认目前没有已提交的 Part C 结果。
- [x] 确定正式 checkout、GPU 环境和实验输出位置。
- [x] 完成配置来源表并冻结正式配置。
- [x] 实现训练、日志、恢复、分析脚本。
- [x] 通过短程数值与恢复验证，测量成本。
- [x] 完成一条 100000-step 参考运行。
- [x] 完成用户修订范围的3d与3e主扫描，关键比较达到3 seeds；N8000单seed明确另列补充。
- [x] 生成并检查图表、结果说明和交接材料。

正式目录：`D:\USC 作业\CSCI567\Project\CSCI567_Grokking\Project check-in\实验部分\mnist_dynamics`；本地分支 `part-c-local-reproduction`。解释器为工作区 `.venv-partc/Scripts/python.exe`，PyTorch 2.5.1+cu124，RTX 4060 Laptop，FP64。42 条配置冻结在 configs/manifest.json。参考 `fig3d_n1000_a8_s0` 完成100000步，fit=3500、t60=36685、最终测试83.86%，训练耗时224.9秒；从实际fit检查点续训200步与原轨迹范数差为0。四并发通过数值验证，但首批吞吐低于串行，后续改为串行。原计划的CLI示例和文件名不是实际接口；以正式 README 为准。所有线上操作及执行判断见工作区 log.md。

## 3. 配置来源：已知与待定分别记录

| 参数 | Figure 3d | Figure 3e | 来源/处理 |
| --- | --- | --- | --- |
| 网络 | 784→200→200→10，隐藏层 ReLU，含 bias | 同左作为共同基线 | 附录 A 与官方 notebook |
| 损失 | one-hot、element-mean MSE | 同左 | 附录 A / notebook |
| 优化器 | AdamW，lr=0.001 | 同左作为共同基线 | 附录 A / notebook |
| batch | 200 | 200 作为共同基线 | 附录 A / notebook；保留末尾小 batch |
| weight decay | **1e-4** | **5e-3** | 分别来自原图标题；不能用 notebook 的 0.01 替代 |
| N | **1000** | 扫描，精确原始列表待查 | 3d 图标题；3e 横轴到 30000，不能据像素读出精确列表 |
| alpha | 比较多尺度，精确列表待查 | **待查** | 3d 色条覆盖约 1-10；notebook 默认 8 不等于 3e 配置 |
| steps | 计划 100000 | **100000 的观测预算** | 3d 图轴及 notebook；3e 图例明确预算 |
| dtype | FP64 | 先沿用 FP64，记录依据为 notebook | 官方 notebook 显式 `torch.float64`；论文未逐图说明 |
| train subset | 官方代码固定前 N 个 | 先按同一规则取嵌套子集 | 论文文字称随机子集，与 notebook 的前 N 有差别，报告这项差异 |
| seeds | 原图精确 seed/重复数待查 | 原图精确 seed/重复数待查 | 小组正式比较计划 seeds 0/1/2 |
| 评估频率 | notebook 使用稀疏时点，见下文 | 原始频率待查 | 3e 首次越阈时间对评估频率敏感 |

优先用官方专用图代码/补充材料确定缺项；当前 notebook 只是通用单次演示，不能证明所有原图设置。附录 C 的 alpha=9 对应 Figure 10 的 weight-decay 实验，也不能直接填入 3e。

### 缺项的处理规则

执行初期集中查找官方历史/补充材料与组内已有记录，先查约 30 分钟。若仍缺配置，不无限等待：使用下面预先写明的项目方案推进，标记 `reproduction_level=qualitative` 和 `source_status=project_choice`，说明与原图的对应和不确定性。若后来找到原始配置，另建 config/run，不改写已完成运行的 provenance。

没有原始 alpha、N 网格、重复数和评估口径，就不能将 3e 称为精确数值复现；不通过根据测试结果反复改变这些参数来“接近原图”。

## 4. 正式实现约定

### 模型、数据、优化器

- 使用三层 Linear 的 PyTorch 默认初始化；同一 seed 的基础模型先构造一次并保存/hash，再将全部 weight 和 bias 乘 alpha。记录缩放前后全局范数。
- FP64 必须覆盖模型、输入和 one-hot 标签。官方 notebook 在构造模型和调用 ToTensor 前设置默认 dtype；确认输入转换与之数值一致，避免先以 FP32 做 /255 再升为 FP64而引入未记录差异。
- 保持原始 MNIST 像素 /255，不额外标准化、增广或改变标签。保存样本索引、数据文件 hash、预处理定义。基线使用前 N，与 B 的成员规则一致，但不用 B 的 FP32 参数文件当作 FP64 基础初始化。
- AdamW 的 betas、eps、weight_decay、foreach/fused、dtype、TF32 与确定性选项全部显式记入实际配置。先用标准串行单模型，不为提速先改为多模型堆叠。
- 训练 loss 为对样本和十个输出都取均值的 MSE；另外保留 `mse_per_sample_sum_outputs=10*mse_element_mean`，避免与官方绘图/Part B 的报告单位混淆。

### 批次顺序与 RNG

**核实后的 cycle 行为**：notebook 的 cell 4 自定义了 `while True: for x in iterable: yield x`，随后将其用于 `DataLoader(..., shuffle=True)`；每轮会重新建立 loader 遍历并重新 shuffle。它不是 `itertools.cycle` 的首轮缓存行为。不要仅根据函数名推断。

用显式、可恢复的 sampler 保存 epoch、当前 permutation、cursor 和 generator state，包含末尾不足 200 的 batch。同一 seed、同一 N 的不同 alpha 使用相同批次序列；不同 N 不要求逐 batch 相同，因为样本集合和 epoch 长度不同。

官方全局 RNG 同时参与模型、训练 loader 与评估 shuffle，改变评估调用会影响后续 shuffle。项目主实现采用独立训练 generator 和固定评估顺序，避免日志频率改变训练路径；这是一项需披露的随机数实现调整，不声称逐位匹配作者轨迹。短程对照时，给原始模型与新模型喂同一显式 batch 序列以验证更新本身。

### Step 与评估边界

定义 `theta_t` 为完成 t 次 optimizer.step 后的参数；step=0 是缩放后的初始化。先更新得到 theta_t，再记录该步范数；若 t 属于评估集合，在同一 theta_t 上评估并按需保存检查点，之后才取下一个 batch。

原 notebook 在下一次更新前评估，日志中的 steps 也是已完成更新数；用上述定义对齐，不随意平移一位。额外加入最终 step=100000 评估，显式记录这是补充时点。

评估全 N 训练样本和完整 10000 测试样本，固定顺序、无随机抽样、保留尾批。官方函数用 `N//batch_size`，N 不整除 50 时可能忽略尾样本；项目主评估修正这一点并记录差异。使用 eval/no_grad，结束后回 train，评估不能修改模型/优化器或训练 RNG。

## 5. 预先规定的实验矩阵

下列是缺乏完整原始配置时的项目方案，不冒充作者精确网格。冻结后才启动正式扫描。

| 集合 | 条件 | 种子与预算 | 用途 |
| --- | --- | --- | --- |
| smoke | N=1000，alpha=1/8，wd=1e-4 | seed=0，各 200 steps | 验证实现、日志、范数、恢复；不用于现象结论 |
| reference | N=1000，alpha=8，wd=1e-4 | seed=0，100000 steps | 首条完整参考训练，也是 3d 中的一个条件 |
| 3d 最小比较 | N=1000，alpha=1/8，wd=1e-4 | seeds 0/1/2，各 100000 steps | 先回答初始化是否对应不同等待行为，共 6 runs |
| 3d 扩展 | alpha=1/2/4/6/8/10，其余同上 | seeds 0/1/2 | 含最小比较共 18 runs；新增 12 runs |
| 3e pilot | N=1000/8000，alpha=8（项目假定），wd=5e-3 | seed=0，各 100000 steps | 先验证 3e 路线与完整阈值记录 |
| 3e 正式 | N=200/500/1000/2000/4000/8000/16000/30000；alpha=8（项目假定）；wd=5e-3 | seeds 0/1/2，各 100000 steps | 共 24 runs；已冻结且配置一致的 pilot 可计入 |

完整项目方案共 42 条正式运行、4.2M 次更新，reference 和 pilot 都是子集，不能重复计数。资源不足时先完成 3d 最小比较和 3e 三个 N={1000,8000,30000}×3 seeds，共 15 条正式运行；状态注明为部分网格。

### 评估集合

- 3d 主曲线：复用 notebook 的时点规则。T=100000 时 `log_freq=ceil(T/150)=667`；评估 t=0..29、30/40/.../140、667 的正整数倍且小于 T，并加 T。不要用缩短预算后的 log_freq 产生正式对比。
- reference 的训练拟合检查：额外在每 100 step 全量评估训练集，只用于确定训练拟合检查点；不因此增加测试评估或改变 batch RNG。
- 3e：若找不到原始频率，固定 t={0,10,50,100,200}∪{500,1000,...,100000} 的全量评估。所有 N/seeds 用同一集合，配置中说明项目选择。
- 不用插值伪造更精确的越阈时刻；报告前一次未达到的评估 step 和第一次达到的 step。
- 不因达到 60% 就终止主要运行，统一跑到预算末步，以保留完整动力学并避免结果文件长度不一致。

## 6. 3e 的阈值与统计规则

原图纵轴写 `Test Accuracy > 60%`，项目主指标使用严格大于：

```text
t60 = min{ t in evaluation_steps : full_test_accuracy(theta_t) > 0.60 }
```

10000 个测试样本时，6000 个正确恰好等于 60%，尚不算跨越。若后续找到源代码用 >=，另外记录该口径并说明，不悄悄换规则。

每次运行保留 `event_observed`、`t60`（未达到则 null）、`previous_eval_step`、`last_observed_step`、`budget_steps`、`run_status`。达到后再下降也保留首次越阈；可另报末步准确率，不用持续性标准替换原指标。

- 100000 steps 内未达到：标记右删失 `threshold_not_reached`，图中在预算处用专门三角/箭头，不填写 t60=100000。
- 进程错误、手动中止、NaN、缺日志：属于失败或未完成，不等同于完成预算仍未越阈；不能计入有效删失样本。
- 所有 seed 达到时才报告该 N 的完整算术均值及样本 SD（ddof=1）。有删失时默认不画该 N 的普通总体均值连线，报告达到数/计划数、各 seed 结果和删失标记；若额外展示已达到者均值，明确为条件均值，不能解释成总体均值。
- step=0 已达到时保留真实 t60=0；对数纵轴另作注释/零时刻标记，不伪造为 1。
- 阈值时间是评估分辨率下的首次观测，不是精确连续跨越时间。不同评估频率的结果不混合求平均。

只有训练先拟合、测试后改善时才讨论延迟泛化。3d 可辅助记录 `t_fit` 与 `t60-t_fit`，但它们不是原图 3e 从初始化开始计数的 t60，不能混成同一个指标。

## 7. 参考轨迹和拟合检查点

### 范数记录

参考运行保存 step=0..100000 的 `global_norm`，其中每个 t>0 的值都在完成该步 AdamW 更新后计算。必须有 100001 个值，不能从稀疏曲线插值。

全局范数为所有可训练 weight/bias 的 L2 范数；同时保存三个 Linear 层的范数和可选 weight/bias 子范数。验证全局范数平方等于分层范数平方之和。后续 D 使用检查点 k 时，目标是对齐 `R[k+1]`、`R[k+2]` 等更新后范数，不能偏移一位。

范数逐步写入可恢复的数值文件，按固定间隔 flush；训练异常时保留已经写入的有效前缀。不要每步保存整个模型。

### 拟合时刻（项目定义，非论文原指标）

预定检查标准：首次在计划的训练评估时点同时满足 `full_train_accuracy>=0.99` 和 `train_mse_element_mean<=1e-3`。只使用训练集，保存其完整指标及 step。若预算内没有满足，明确 `fit_checkpoint_missing`，不从测试改善时刻反推检查点。

如需放宽标准，只能在训练指标驱动的预实验中决定并在正式运行前冻结；旧定义与新定义的运行不混合。保存临近的周期性检查点，避免唯一文件损坏。

### 检查点至少保存

```text
run_id, config_hash, source_commit, code_hash, step_completed
model.state_dict, optimizer.state_dict
Python / NumPy / torch CPU / all CUDA RNG states
independent sampler generator state, epoch, current permutation, cursor
dataset indices / hashes, parameter ordering and dtype
last committed metric/norm rows, fit criterion and fit status
```

检查点中的模型和优化器必须属于同一个 step；不能将早期模型配上末步 AdamW 动量。恢复时下一批数据、日志位置、范数写入位置都从保存边界继续。

### 有意义的验证（通过才长跑）

1. **源实现对照**：同初始化、同 FP64 输入/标签、相同显式批次、相同 AdamW 超参数，比较原始串行模型和新实现 20-50 次更新后的参数/损失。
2. **恢复一致性**：A 连续跑 200 steps；B 跑到 100、保存并在新进程恢复到 200。核对后 100 次 batch 索引、参数、优化器、指标和范数轨迹。固定同设备/版本，尽量逐位一致；若只能近似，记录最大绝对误差，建议初始验收阈值 1e-10，超出先排查而不是扩大容差放行。
3. **记录完整性**：范数序列无缺步/重步、指标有限、step 与检查点一致、恢复不重复写行。3e 用构造的指标序列检查严格 >、未达到、step=0 与失败状态，不需要 GPU。
4. **评估无副作用**：评估前后参数、优化器及训练 generator 不变。

不在 C 内运行正式方向干预。正常续训恢复检查通过后，将检查点与范数轨迹交给 D；未来的 c=1 投影重放属于干预实现验证的下一阶段。

## 8. 实施阶段和检查点

| 阶段 | Codex 要完成的动作 | 完成证据 / 下一步门槛 |
| --- | --- | --- |
| P0 目录与来源 | 检查正式 checkout、当前改动、AGENTS、已有 GPU/环境；固定代码版本，整理第 3 节缺项 | `provenance.json`、配置来源表；不得把参考副本已有 B 结果改成 C |
| P1 实现 | 建独立 mnist_dynamics；实现串行训练、全量评估、可恢复 sampler、逐步 norm、checkpoint、分析 | smoke 配置可完整运行；每项配置有 hash 和 run_id |
| P2 验证与测时 | 完成第 7 节验证；分别测训练、全量评估、范数统计与落盘成本 | `validation.json` 通过、`benchmark.json`；再允许 100k 长跑 |
| P3 单参考运行 | 完成 N=1000/alpha=8/wd=1e-4/seed0 的完整预算 | 曲线、100001 个范数、拟合检查点或明确缺失说明、最终 checkpoint |
| P4 最小比较 | 补齐 3d alpha=1/8 三 seeds，完成 3e pilot 和三个 N 的三 seeds | 有效运行与删失状态完整；先出阶段图，记录尚未扫描条件 |
| P5 扩展 | 资源允许再补齐 3d 六 alpha、3e 八 N 三 seeds | 18+24 条正式运行清单无遗漏/重复；失败项可追溯 |
| P6 分析交接 | 从 raw 文件重算指标、绘图、写说明、复核恢复附件 | 图与表一致，A 可汇总、D 可读取参考轨迹；验收清单更新 |

P0-P2 先解决实现可信性；P3 完成前不直接启动整个矩阵。GPU 单机并发数按实测确定，先串行；并发不能改变单次 run_id、seed 或日志路径。

### 算力与存储预算

不得用 Part B 的 FP32 堆叠训练 24-26 分钟/seed 去推断 Part C 的 FP64 100k-step 成本。当前 GPU 型号、可用时段和 FP64 吞吐未知，暂不承诺墙钟完成时间。

P2 按代表性 N=1000 和 N=30000 各测一段：

```text
estimated_run_seconds = 100000 * measured_update_and_norm_seconds
                     + n_evaluations * measured_full_eval_seconds
                     + measured_checkpoint_and_io_seconds
```

对短测、正式 reference、并发情况下分别记录误差；以 reference 的实测再估算剩余运行。计算最小 15 runs 与完整 42 runs 的预算，先汇报可完成范围再自动排队。保留一条完整参考运行的优先级高于铺开许多短跑。

100001 个 FP64 全局范数约 0.8 MB，逐层记录另计；模型/AdamW 检查点按实际大小乘保留数量估算。保留 init、拟合、周期性和 final，避免每步写全模型。

## 9. 建议文件结构和命令接口

新代码已放在小组仓库的 `Project check-in/实验部分/mnist_dynamics/`，与B平行。以下保留原设计结构；实际接口与可直接执行命令以正式README为准。

```text
mnist_dynamics/
  README.md
  configs/                    # smoke / reference / fig3d / fig3e，JSON
  code/train_dynamics.py      # 单 run；自由训练
  code/run_sweep.py           # dry-run / manifest / 顺序执行或有限并发
  code/validate_dynamics.py   # 更新对照、恢复、日志、阈值逻辑
  code/analyze_dynamics.py    # 从原始日志生成事件表与汇总
  code/make_figures.py        # CPU 绘图
  metadata/                  # 来源、版本、环境、验证、测时
  results/                   # 可共享的轻量最终数值与 summary
  figures/                   # PNG/PDF
  runs/<run_id>/             # 大型原始轨迹、模型、optimizer；默认不入 Git
```

实际每条run包含resolved_config.json、environment.json、metrics.csv、norms.npy、events.json、console.log、status.json、checkpoints/及命令记录。run_id编码条件与seed，config_hash单独存入状态/检查点并核验。历史退出码7条未单独留存，27条可由调度成功日志还原为0，缺失值如实记录而不补造；最终34条模型和指标均独立核验。

配置中除优化参数外，还保存 `reproduction_level`、每个待定项的来源状态、数据索引、评估 step 列表、范数/拟合规则、batch/RNG 方案。

计划 CLI：

```powershell
# 先进入正式 checkout 中的 mnist_dynamics；选定环境后记录解释器绝对路径。
python code/validate_dynamics.py --config configs/smoke.json --output metadata/validation
python code/run_sweep.py --config configs/fig3d.json --dry-run
python code/train_dynamics.py --config configs/reference.json --output runs/reference
python code/run_sweep.py --config configs/fig3d.json --max-concurrent 1
python code/run_sweep.py --config configs/fig3e.json --max-concurrent 1
python code/analyze_dynamics.py --runs runs --output results
python code/make_figures.py --input results --output figures
# 恢复接口应显式指定检查点并验证所属配置。
python code/train_dynamics.py --resume runs/reference/checkpoints/latest.pt
```

实际dry-run输出用户保留的条件与路径；测时见benchmark记录，正式sweep核验并跳过已完成项。历史退出码保留情况如上，恢复后的调度器增加process_exit.json；缺失历史值不补造。不要复用 Part B 的 collect/aggregate 脚本，它们硬编码 630/1890 条固定范数条件。

## 10. 图表、统计与结果说明

Figure 3d：optimizer step 对数横轴；train/test accuracy 统一 0-1 纵轴；用一致颜色/图例编码 alpha；保留逐 seed 曲线。可另给跨 seed 均值/样本 SD 图，但不在缺失时点上随意插值求平均。step=0 单独保留，不能直接放上 log 横轴。

Figure 3e：N 横轴、首次观测 t60 的 log 纵轴；每 seed 用点；有效预算内未达到用不同标记，并同时给 reached/total 表。原图显示 Mean，但目前未确认其删失处理，所以采用第 6 节的明确统计规则，不声称完全复刻原图均值算法。

补充图：训练/测试 element-mean MSE、全局/逐层范数轨迹、参考运行拟合时刻及越阈时刻。主图不要求范数单调，也不把末步准确率替代首次越阈时间。

说明至少回答：

1. 哪些 alpha 条件出现训练先拟合、测试后改善？哪些没有？
2. 增大 N 后 t60 是否改变，跨 seed 是否同向？哪些条件未达到或失败？
3. 训练误差接近零以后 MSE 和范数是否继续变化？
4. 与论文一致和不同的是什么；其中哪些可能来自配置、数值精度、子集、评估频率和有限预算？
5. 结论仅是当前设置下的动力学复现，不等同于范数唯一因果解释，也未检验切向更新备择假设。

A：交付 150-250 字可粘贴摘要、3d/3e 图和当前完成状态。B：共享架构、数据和术语，明确 batch/精度/weight decay/约束的差异。D：提供参考 run 的配置、拟合 checkpoint、从初始化起的每步范数及后续 batch 恢复方式。发送组外消息或联系同学不属于本轮动作；先准备好交接文件。

## 11. 最终验收

- [x] 原图设置、notebook 默认与项目自定设置清楚分开；未查明的来源没有被写成确定事实。
- [x] 3d 使用 wd=1e-4、3e 使用 wd=5e-3；主实验自由训练，没有投影。
- [x] 记录实际精度、所有优化器参数、训练索引、随机状态和每次运行配置。
- [x] 标准更新对照与跨进程恢复验证通过，有原始记录。
- [x] 至少一条完整参考轨迹；每步更新后 global norm 无缺步；拟合检查点真实可恢复，或准确列为缺项。
- [x] 关键比较有 3 seeds；若未达到，则只标记初步或部分完成，不提前声称全部完成。
- [x] t60 规则和评估频率固定；删失、错误、未完成明确区分，没有将上限当普通达到时间。
- [x] 逐 seed 指标、有效数量、均值/SD 和图能从 raw 日志重算，末步结果也保留。
- [x] 图的轴、图例、单位、log 轴零值处理正确；查看实际 PNG/PDF，不能只凭脚本退出成功。
- [x] README 含从环境到运行/恢复/重绘的真实可用命令，尚未实现的接口不冒充已有功能。
- [x] 小型材料与代码可提交；大文件有存储索引/hash；diff 不夹带 MNIST 数据和无关 B 改动。
- [x] A/B/D 的本地交接材料齐全，状态说明与实际文件一致。

任务分配允许 Check-in 只报告阶段进度。未完成的扫描、缺失的拟合检查点或未恢复的日志，必须列为剩余工作；计划、环境配置、smoke 成功、完整运行和现象复现使用不同状态。

## 12. 后续恢复时的首批动作

1. 读本文件和最新任务分配，检查正式 checkout 是否已有 C 的新代码/结果，避免重复劳动。
2. 确认已有 GPU/环境可用情况；先只读调查，不立即安装或跑长任务。
3. 写 provenance 与配置来源表，处理第 3 节未知项，冻结最小实验方案。
4. 创建/完善 mnist_dynamics，先完成 P1-P2；通过后只启动 reference。
5. 每阶段更新本文件的状态、实际路径、run_id、耗时和下一步。长跑定期给出有新信息的进度，结束后核验结果，不能停留在“已启动”。

## 13. 固定来源

- [小组仓库固定版本](https://github.com/PatriRUC/CSCI567_Grokking/tree/ed040b7fa7ebbc95b928bb990829bed266ffcb6f)。
- [官方动力学 notebook](https://github.com/KindXiaoming/Omnigrok/blob/3a0a57346230a7255e5390fcd565812d00f1a281/mnist/grokking/mnist-grokking.ipynb)：cells 4、7、9 对应 cycle/评估、参数、训练；SHA256=`e930648fda27ad67f6d31a7039deccf7a70e89ca659369da328124797aa67b0f`。
- [论文 arXiv v2](https://arxiv.org/abs/2210.01117v2)，原图与附录以本地 PDF 核对；图中的 wd 与阈值优先于 notebook 的通用默认值。
- [Part B 完成计划](./PartB_Figure3abc_完成计划.md)，用于共享接口和避免误用固定范数代码。

用户保留范围的实验、图表、数值说明与交接材料已完成。原42条计划的8条取消不是失败或阈值删失；N8000 seed1的22000步部分产物保留本地但排除统计。最终审阅包、验收证据与工作日志按修订后的范围交付。
