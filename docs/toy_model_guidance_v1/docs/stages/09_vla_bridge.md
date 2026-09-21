# Stage 09 — Bridge to VLA

这一阶段才开始把前面的 primitive 组合起来。

目标不是训练 7B VLA，而是构造一个 tiny multimodal control problem。

## Tiny VLA Environment

输入：

- synthetic image / object features；
- language instruction；
- robot state。

输出：

- discrete action token；或
- continuous action chunk。

例如：

场景里有 red / blue / square / circle。
语言：

> move to the red square

模型必须联合：

- vision；
- language；
- action。

## Core 1 — Representation Alignment

诊断：

- vision embedding；
- text embedding；
- fused representation；
- target-object probe；
- modality ablation。

问题：

> 模型依赖视觉还是语言？两者在哪里融合？

## Core 2 — Action Representation

比较：

- discrete token；
- continuous regression；
- mixture；
- flow/diffusion action。

把 RT-2 / OpenVLA / π0 的差异重新翻译成：

> action distribution 用什么参数化？

## Core 3 — Compositional Generalization

训练只见部分组合：

- red square；
- blue circle；

测试：

- red circle；
- blue square。

不要只测 IID success。

构造 OOD matrix。

## Core 4 — Causal Modality Tests

- swap language；
- mask image feature；
- patch object representation；
- perturb proprioception；
- change irrelevant visual distractor。

看 action 是否按机制变化。

## Core 5 — Closed Loop

必须最终进入 rollout。

VLA 离线 token accuracy 不能替代闭环成功率。

报告：

- action prediction metric；
- closed-loop success；
- recovery after perturbation；
- horizon sensitivity。

## Paper Bridge

阅读 RT-2 / OpenVLA / π0 时统一回答：

1. vision backbone 学到什么？
2. language 如何进入？
3. action 如何表示？
4. temporal context 多长？
5. 输出是 token、regression、diffusion 还是 flow？
6. pretraining data 与 robot data 如何组合？
7. offline metric 与 closed-loop metric 是什么？
8. 论文里的 improvement 可以拆成哪些 toy hypothesis？

## Exit Criteria

你应该已经能把新的 VLA 论文拆成：

- representation；
- fusion；
- action parameterization；
- temporal modeling；
- optimization；
- data mixture；
- closed-loop distribution shift。

到这里才适合系统进入真实 VLA 代码库。
