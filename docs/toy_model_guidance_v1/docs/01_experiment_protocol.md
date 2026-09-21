# 01 — Universal Experiment Protocol

每个 Toy Model 在写代码前先完成一份 Experiment Spec。

## A. Research Question

必须能用一句话表达，而且最好是可证伪的。

坏例子：

> 深度网络是不是更好？

好例子：

> 参数量近似相同、训练预算相同时，把容量分配到更多层，是否会改变 Spiral 决策边界的学习速度与几何复杂度？

## B. Mechanistic Hypothesis

不是预测一个数字，而是预测机制。

格式：

> 如果机制 M 成立，那么当我们改变 X 时，应观察到 Y；同时内部诊断 Z 应出现对应变化。

## C. Competing Hypotheses

至少写一个替代解释。

例如：

- H1：深度提高表达效率；
- H2：优势主要来自参数化导致的优化差异；
- H3：结果只是某个 seed 或学习率对深模型更友好。

## D. Variables

明确：

- Independent Variable；
- Controlled Variables；
- Measured Variables；
- Nuisance Variables。

特别区分：

- matched parameter count；
- matched FLOPs；
- matched training steps；
- matched wall-clock；

它们不是一回事。

## E. Prediction Before Running

实验前写：

- 方向性预测；
- 哪些图会改变；
- 哪些内部量会改变；
- 哪种结果会推翻当前解释。

禁止跑完以后再补预测。

## F. Minimum Measurements

所有实验至少保存：

- 完整 config；
- random seed；
- git commit（如可用）；
- per-step / per-checkpoint metrics；
- final metrics；
- selected checkpoints；
- figures 的原始数据；
- 一个 `results_note.md`。

## G. Multi-seed Rule

结构比较默认至少 3 个 seed。

如果现象不稳定：

- 增加 seed；
- 报均值与方差；
- 不选择最好看的 seed 代表总体结论。

## H. Plot Rule

比较图必须尽量做到：

- 相同 x/y 范围；
- 相同数据切分；
- 相同 checkpoint；
- 相同颜色语义；
- 清楚标注 log scale；
- seed 聚合显示方差；
- 图能从 CSV / JSON 重新生成。

## I. Interpretation Ladder

看到结果后按顺序解释：

1. What happened?
2. Is it reproducible?
3. Which variable changed?
4. Which mechanism is consistent with it?
5. What competing explanation remains?
6. What experiment distinguishes them?

## J. Stop Rule

如果一个实验没有改变你的信念，也没有暴露新问题，就不要继续无目的 sweep。

下一实验必须服务于：

- confirmation；
- falsification；
- boundary condition；
- mechanism isolation；

之一。
