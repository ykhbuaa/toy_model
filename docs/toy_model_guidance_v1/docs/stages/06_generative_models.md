# Stage 06 — Generative Models

目标不是把 VAE / GAN / Diffusion 全都实现一遍，而是理解“预测一个值”和“建模一个分布”的差别。

## Core 1 — Conditional Multimodality

构造：

`x -> y`

但同一个 x 对应两个合法 y mode。

先训练 MSE regression。

观察它预测 mode average。

这是后面 Diffusion Policy 最重要的前置直觉之一。

## Core 2 — 2D Density Modeling

使用 two moons / Gaussian mixture。

对一个小 diffusion / score model 可视化：

- data density；
- noisy density；
- score vector field；
- denoising vector field；
- sample trajectories；
- mode coverage。

必须画 vector field，而不是只展示最终 samples。

## Core 3 — Denoising Time

检查不同 noise level：

- easy coarse denoising；
- difficult fine detail；
- prediction norm；
- score norm；
- error by timestep。

理解 diffusion 不是“反复去噪”这句口号，而是不同噪声尺度上的 vector field learning。

## Optional — Flow Matching

在同一个 2D dataset 上实现 flow matching。

比较：

- trajectory shape；
- integration steps；
- learned vector field。

为 π0 一类 flow-based action model做概念准备。

## Exit Criteria

能够解释：

- conditional mean 为什么会在 multimodal action 中失败；
- score 是什么；
- denoising trajectory 是什么；
- diffusion / flow 在 toy 2D 空间中到底学了哪个向量场。
