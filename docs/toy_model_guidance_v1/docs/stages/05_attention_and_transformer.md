# Stage 05 — Attention & Transformer

不要直接训练语言模型。

## Core 1 — Content Addressing

任务：

```text
A -> 7
B -> 2
C -> 9
query: B
answer: 2
```

训练 single-head attention。

直接观察：

- Q；
- K；
- QK^T logits；
- softmax attention；
- V；
- attention output。

目标：把 Attention 理解成可学习的 content-addressed routing。

## Core 2 — Distractor & Context Length

加入：

- irrelevant key-value pairs；
- repeated keys；
- longer context；
- corrupted values。

观察：

- attention entropy；
- retrieval accuracy；
- QK margin；
- head output norm。

## Core 3 — Position

构造必须依赖位置的 copy / shift / previous-token task。

比较：

- no positional information；
- learned position；
- sinusoidal / RoPE（后面再加）。

核心问题：

> Attention 本身知道顺序吗？

## Core 4 — Multi-head Specialization

设计需要两种 relation 的任务。

观察不同 head：

- attention pattern；
- entropy；
- output norm；
- head ablation effect。

不要因为图看起来不同就声称“head 有语义”。

必须做 ablation。

## Diagnostic Deep Dive — Causal Intervention

- zero one head；
- patch head output；
- replace one token representation；
- counterfactual key/value。

从 correlation 进入 causality。

## Exit Criteria

能够手算一个极小 attention；
能够解释 QK 与 OV 两条路径；
能够从 retrieval toy task 解释 positional information；
能够使用 ablation 判断一个 head 是否对行为有因果作用。
