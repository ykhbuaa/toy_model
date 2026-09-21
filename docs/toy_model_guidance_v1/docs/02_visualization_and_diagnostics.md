# 02 — Visualization & Model Diagnostics

可视化不是装饰，而是测量仪器。

整个课程统一采用六级诊断阶梯。

## Level 0 — Task Metrics

回答“模型做得怎么样”。

典型量：

- regression MSE / MAE；
- cross entropy；
- accuracy；
- success rate；
- one-step prediction error；
- rollout return。

必要但绝不充分。

## Level 1 — Function / Decision Geometry

回答“模型学成了什么函数”。

### 一维函数

必须考虑：

- target / samples / prediction overlay；
- residual vs x；
- interpolation vs extrapolation 分区；
- error vs distance from training support；
- derivative `df/dx`；
- curvature `d²f/dx²`；
- target / prediction / residual 的 Fourier spectrum；
- checkpoint function movie / small multiples。

对于 ReLU MLP，可以进一步画：

- 每个 hidden neuron 的 pre-activation；
- activation breakpoint；
- 最终 piecewise-linear function 的折点密度。

### 二维分类

建议：

- dataset geometry；
- decision regions；
- top-two logit margin；
- true-class margin；
- predictive entropy；
- input gradient norm `||∂logit/∂x||`；
- checkpoint decision-boundary evolution。

## Level 2 — Training Dynamics

回答“模型是怎样学到这个函数的”。

统一记录：

- loss；
- train/test metric；
- global gradient norm；
- per-layer gradient norm；
- weight norm；
- update norm；
- update-to-weight ratio；
- distance from initialization；
- optional gradient cosine similarity across checkpoints。

关键图：

- metric vs step；
- per-layer norm heatmap；
- gradient norm vs layer；
- update ratio vs layer；
- seed envelope。

## Level 3 — Activation / Representation Geometry

回答“信息在层内变成了什么样”。

记录：

- activation mean/std；
- activation RMS；
- fraction zero（ReLU）；
- saturation fraction（tanh/sigmoid）；
- feature covariance；
- covariance eigenspectrum；
- effective rank；
- PCA projection；
- class centroids / within-class vs between-class distance；
- linear probe accuracy；
- CKA between layers or runs（后期）。

注意：PCA 图不能替代定量指标。

## Level 4 — Sensitivity / Jacobian / Spectrum

回答“映射在局部如何放大、压缩或扭曲”。

逐阶段引入：

- input-output Jacobian；
- Frobenius norm；
- singular values；
- condition number；
- layer Jacobian product；
- spectral norm of weights；
- Hessian / sharpness 只在有明确问题时使用。

不要因为“高级”而计算 Hessian。

## Level 5 — Causal Diagnostics

从 Attention 阶段开始。

使用：

- neuron / head ablation；
- attention head masking；
- activation patching；
- value-vector intervention；
- token removal / counterfactual input；
- latent replacement；
- action / observation perturbation。

目标是从：

> 这个特征和输出相关

升级到：

> 改变这个内部变量会系统改变模型行为。

# 各阶段必做诊断

| Stage | 必做 |
|---|---|
| Function | function + residual + spectral error + derivative |
| Geometry | boundary + margin + input-gradient + layer PCA |
| Optimization | activation/gradient/weight/update norms |
| Norm/Residual | layer statistics + Jacobian singular values |
| Representation | PCA/eigenspectrum/effective rank/linear probe |
| Attention | attention + QK logits + entropy + ablation |
| Generative | density/score field/denoising trajectory/mode coverage |
| World Model | one-step vs rollout error + latent trajectory + horizon curve |
| Policy | action distribution + rollout + covariate shift + multimodality |
| VLA | modality intervention + action-token/flow diagnostics + OOD grid |

# 统一输出建议

每个 run：

```text
run/
├── config.json
├── manifest.json
├── metrics.csv
├── traces/
│   ├── layer_stats.csv
│   └── checkpoint_metrics.csv
├── checkpoints/
├── figures/
└── results_note.md
```

`figures/` 是结果的视图，`metrics.csv` / `traces/*.csv` 才是数据源。

# 诊断实现原则

1. forward hook 只负责采样，不在 hook 内做复杂 plotting；
2. hook 必须在结束时 remove；
3. 默认 detach；
4. 不保存全部 activation tensor，优先在线统计 mean/std/norm；
5. 只有专门的 representation experiment 才保存样本级 hidden states；
6. Jacobian 计算限制在 toy-sized batch；
7. 对比实验中的图轴必须一致。
