# Stage 04 — Normalization, Residual & Representation

## Core 1 — Residual Connection

比较：

`y = F(x)`

和：

`y = x + F(x)`。

在深 MLP 上逐渐增加 depth。

记录：

- loss；
- per-layer grad norm；
- activation norm；
- residual branch norm / identity norm；
- input-output Jacobian singular values；
- seed stability。

核心问题：

> residual 主要改变 function class，还是 parameterization / information path？

## Core 2 — Normalization

先不要混着比较 BN / LN / RMSNorm 全家桶。

选择一个明确问题：

> LayerNorm 是否让不同层的尺度更稳定？

记录 pre-norm / post-norm：

- mean；
- RMS；
- gradient RMS；
- Jacobian stats；
- sensitivity to initialization / learning rate。

再研究 Pre-LN vs Post-LN Transformer 时复用同样测量。

## Core 3 — Representation Compression / Expansion

训练一个小分类任务，逐层看：

- covariance eigenspectrum；
- effective rank；
- probe accuracy；
- class separation；
- neuron sparsity。

然后加入 bottleneck，看哪些信息先丢失。

## Optional — Toy Models of Superposition

使用稀疏 feature toy model，观察：

- features > neurons；
- interference；
- feature geometry；
- sparsity 改变表示方式。

这个实验会成为后期理解大模型 representation 和 mechanistic interpretability 的桥。

## Exit Criteria

能够解释：

- residual 作为 identity path 的意义；
- normalization 改变什么统计量；
- representation dimension 与 neuron count 不等价；
- PCA / effective rank / probe 各自能说明什么、不能说明什么。
