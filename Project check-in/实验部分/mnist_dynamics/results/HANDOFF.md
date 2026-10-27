# Part C 本地交接材料

按用户侧对话修订，保留34条有效运行并通过独立数据验收；原42条中的8条后续任务已取消。3d为18条，3e主实验15条，N8000 seed0另作补充。按用户新授权同步GitHub，完整产物见[交付说明](../PUBLICATION.md)。

## 给 A：可粘贴摘要

本阶段完成34条MNIST本地训练，每条十万次参数更新。3d比较6种初始化、各三个随机种子；三个种子中大初始化均较晚越过测试阈值。3e在N=200–4000范围各有三个随机种子，15条中12条观测到测试准确率超过60%，其余为右删失。N=8000仅保留seed0作补充，不报均值或样本标准差。使用FP64、AdamW自由训练；拟合检查点与逐步范数支持后续实验。所有指标从原始记录独立重算，保留预算内未越阈结果。结果为披露设置差异的定性复现。

主图：figures/figure3d.png、figure3e.png，均同时提供PDF；完整说明在results/REPORT.md。3e明确标出缩减N范围与单seed补充；用户取消与测试阈值未达到分开记录。

包含原图对照和差异的Check-in段落见[CHECKIN_SUMMARY_ZH.md](CHECKIN_SUMMARY_ZH.md)，详细分析见[INTERPRETATION_ZH.md](INTERPRETATION_ZH.md)。

## 给 B：共享基线与差异

共同使用MNIST前N个训练样本、784→200→200→10带偏置ReLU网络、one-hot element-mean MSE、AdamW lr=.001。B设置据相邻mnist_landscape/README.md核对。

| 协议 | B：3a–c | C：3d–e |
| --- | --- | --- |
| 参数范数 | 每步投影到初始化目标范数 | 自由变化，无投影 |
| 精度 | FP32 | FP64 |
| batch | 100 | 200 |
| weight decay | 0 | 3d为1e-4；3e为5e-3 |
| 每条件预算 | 10000更新 | 100000更新 |
| 观测量 | 固定范数下训练/测试误差景观 | 准确率时间曲线、首次观测t60 |

alpha为初始参数倍率，不是参数范数本身。C的MSE输出同时记录element mean与十倍的per-sample sum-output值，不应混用；t60严格要求测试准确率>60%。不同协议的最终误差不能当成同条件因果比较。

## 给 D：参考检查点与逐步范数

参考目录为runs/fig3d_n1000_a8_s0；fit.pt对应step=3500。交接需要checkpoints/fit.pt、resolved_config.json、train_indices.npy、norms.npy及code/train_dynamics.py。

norms.npy为100001×4的FP64数组，列顺序global、layer1、layer2、layer3；第t行对应完成t次更新后的含偏置参数L2范数。fit.pt含同一步模型、AdamW、Python/NumPy/CPU/CUDA RNG以及sampler的permutation/cursor/generator；恢复之后取下一个batch，不重新跳过或重新抽样已消耗批次。

为避免覆盖参考输出，先在新的空目录复制原norms.npy，再把该目录传给--output。例如从当前mnist_dynamics目录执行：

```powershell
$python = 'D:\USC 作业\CSCI567\.venv-partc\Scripts\python.exe'
$replay = 'runs/d_replay_review'
New-Item -ItemType Directory -Path $replay -ErrorAction Stop
Copy-Item -LiteralPath 'runs/fig3d_n1000_a8_s0/norms.npy' -Destination $replay
& $python code/train_dynamics.py --resume runs/fig3d_n1000_a8_s0/checkpoints/fit.pt --output $replay --stop-after 3700
```

现成验证记录在metadata/reference_checkpoint_replay/verification.json：实际fit checkpoint正常续训200步的范数误差为0。跨进程smoke另外验证模型与优化器状态完全一致。未来投影干预的c=1重放尚需D自行验证，不能用此正常续训验证替代。

## 审阅包与原始文件

审阅ZIP包含代码、配置、完整评估CSV、全部34条保留运行的逐步范数、命令/环境/控制台日志、图、报告和参考fit checkpoint。另有完整原始ZIP包含全部136个检查点，两者随同仓库Release提供；ZIP中的metadata/checkpoint_index.json提供路径、大小与SHA256。取消任务的部分产物保留本地并另有索引。MNIST和Python环境未打包。

ZIP解压后可以从已汇总CSV重新绘图。重新运行完整analyze_dynamics.py还需要本地全部checkpoint；审阅ZIP的完整性可用纯标准库命令 `python code/verify_bundle.py <审阅包.zip>` 核对。完整训练需按requirements-freeze.txt准备GPU环境及MNIST。尚未执行任何方向干预。
