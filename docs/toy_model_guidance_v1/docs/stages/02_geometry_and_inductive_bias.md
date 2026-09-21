# Stage 02 — Geometry & Inductive Bias

现有 Spiral 实验是这个阶段很好的起点，不需要废弃。

## Core 1 — Spiral Depth Comparison

继续保留：

- linear negative control；
- parameter-matched shallow / medium / deep MLP；
- multi-seed；
- decision boundary；
- logit margin；
- confusion matrix。

新增内部诊断：

- per-layer activation PCA；
- ReLU zero fraction；
- input gradient norm map；
- gradient norm by layer；
- boundary complexity proxy。

### 研究问题升级

不仅问：

> 谁 accuracy 更高？

而要问：

> 深度改变了决策边界的表示效率、学习速度，还是优化条件？

## Core 2 — Architecture Prior

构造具有明确 symmetry / locality 的 synthetic data。

例如二维 pattern：

- translation of local motif；
- permutation-sensitive vs permutation-invariant task。

比较：

- generic MLP；
- weight sharing / convolution-like architecture。

目标：直接看到 inductive bias 如何减少所需样本。

## Diagnostic — Representation Geometry

固定 probe set，保存每层 hidden state。

画：

- PCA；
- within-class / between-class distance；
- covariance eigenspectrum；
- effective rank；
- linear probe accuracy by layer。

避免结论：

> PCA 分得开，所以模型理解了类别。

PCA 是观察，不是证明。

## Exit Criteria

能够区分：

- architecture capacity；
- parameter efficiency；
- optimization；
- inductive bias；
- representation geometry。

能够说明“相同参数量”为什么不等于“公平地比较所有东西”。
