# Stage 03 — Optimization & Signal Propagation

## Core 1 — Initialization Microscope

构造 10 / 30 / 100 层 MLP，不必先训练。

对不同初始化：

- too small；
- Xavier；
- He；
- too large。

输入固定随机 batch。

记录每层：

- activation mean/std/RMS；
- zero fraction；
- gradient RMS；
- weight spectral norm（小网络）；
- optional layer Jacobian norm。

先观察 forward/backward signal，再推导 variance propagation。

## Core 2 — Optimizer Microscope

在相同 function task 上比较：

- SGD；
- Momentum；
- Adam。

不是只比较 final loss。

记录：

- gradient norm；
- update norm；
- update / weight；
- distance from initialization；
- loss vs step；
- loss vs cumulative update distance。

问题：

> optimizer 改变的是方向、尺度、噪声，还是隐式偏置？

## Core 3 — Learning Rate Phase Diagram

小范围 grid：

- too small；
- useful；
- unstable。

画：

- final loss；
- max gradient；
- non-finite region；
- convergence speed。

目的不是找“最佳 LR”，而是认识优化的 phase behavior。

## Diagnostic Deep Dive — Deep Linear Network

使用 deep linear network，因为部分训练动力学可分析。

比较相同整体线性映射的：

- shallow linear；
- factorized deep linear。

观察参数化本身如何改变优化。

## Exit Criteria

能够从激活/梯度曲线解释：

- vanishing；
- exploding；
- initialization scale；
- optimizer update scale；
- 为什么“网络能表达”不意味着“训练得到”。
