# Stage 01 — Function Approximation & Generalization

## 核心问题

> 神经网络到底学到了一个怎样的函数？

这是整个课程的坐标系。

## Core 1 — Function Zoo

任务：

- sin；
- low + high frequency；
- polynomial；
- narrow peak；
- piecewise；
- discontinuity；
- noisy labels。

模型：

- small MLP；
- ReLU / tanh；
- shallow / deep。

先只研究行为，不急着讨论“大模型”。

### 必做可视化

- target + training samples + prediction；
- residual；
- interpolation / extrapolation；
- derivative；
- error vs distance from support。

### 思考

同样的 train MSE 是否可能对应完全不同的 learned function？

## Core 2 — Spectral Bias

目标：

构造：

`f(x)=sin(x)+a sin(kx)`

观察训练过程中不同频率误差的下降速度。

### 必做

每个 checkpoint 对 prediction 做 FFT / Fourier projection，画：

- target amplitude；
- prediction amplitude；
- residual amplitude；
- low-frequency error vs step；
- high-frequency error vs step。

### 关键问题

这是 representation limitation，还是 optimization dynamics？

尝试用 Fourier Features 改输入后再次测试。

## Core 3 — Interpolation vs Extrapolation

改变训练 support，而 evaluation support 固定。

观察：

- support 内误差；
- support 外误差；
- boundary distance error；
- tail affine fit；
- symmetry residual；
- periodicity residual。

不要只说“神经网络不会外推”，要描述它实际外推出了什么函数。

## Diagnostic Deep Dive — ReLU Piecewise Linear Geometry

对一维 ReLU MLP：

- 画 hidden pre-activation；
- 找每个 neuron 的 activation breakpoint；
- 对比网络深度和折点数量；
- 将折点与 target 高频区域对应。

这是理解“网络表示函数”的第一组内部诊断。

## Exit Criteria

你应该能解释：

- train fit 与 function recovery 的区别；
- interpolation 与 extrapolation 的区别；
- spectral bias 是什么现象；
- 为什么 Fourier feature 改变了问题的参数化；
- 如何判断“表达不了”和“优化不到”。

## Bridge to later stages

World Model 的 multi-step rollout，本质上也是函数在训练分布之外被反复调用。
Policy 的 covariate shift 也是一种输入 support 外推。
