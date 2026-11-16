# 实验部分

| 工作 | 当前状态 | 材料 |
|---|---|---|
| Figure 3a–c：MNIST 固定范数景观 | seeds 0、1、2 完成，共 1,890 条结果 | [代码与交接说明](mnist_landscape/README.md) |
| Figure 3d–e：MNIST grokking 训练动力学 | 按论文原图设置完成 3d（N=1000，alpha=1…10）与 3e（N=128…32768，60% 阈值），seeds 0、1、2 | [代码与交接说明](mnist_dynamics/README.md) |
| 本组备择假设的区分实验：相同范数轨迹下的方向干预 | seeds 0、1、2 完成（参考与 c=1、2、0.5，分叉后各 100,000 步）；60% 测试准确率的等待步数在 c=2 时为参考的 0.65 倍，c=0.5 时为 1.66 倍 | [代码与交接说明](mnist_direction/README.md) |

三 seed 结果摘要（[3a–c](mnist_landscape/results/SUMMARY.md)、[3d–e](mnist_dynamics/results/SUMMARY.md)、[方向干预](mnist_direction/results/SUMMARY.md)）可供 check-in 介绍参考。
