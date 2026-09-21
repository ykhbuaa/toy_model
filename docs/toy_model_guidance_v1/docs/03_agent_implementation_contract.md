# 03 — Coding Agent Implementation Contract

本文件是给 coding agent 的约束，不是建议。

## 1. Agent 的职责

Agent 负责：

- 实现数据生成；
- 模型；
- 训练；
- evaluation；
- diagnostics；
- visualization；
- tests；
- CLI；
- 结果保存。

Agent 不负责擅自修改研究问题。

## 2. 实现前必须输出

在写代码前先给出：

1. 要回答的 Research Question；
2. changed variable；
3. controlled variables；
4. 预期新增/修改文件；
5. 预期输出 metrics 和 figures；
6. 最小 smoke test。

如果这些和 Experiment Spec 不一致，先修正，不编码。

## 3. 教学代码要求

注释优先解释：

- tensor shape；
- 数学含义；
- 为什么这样设计；
- 哪个控制变量因此被固定；
- 数值稳定性原因。

不要写大量解释 Python 语法的注释。

## 4. 透明训练循环

前四阶段保持原生 PyTorch loop 可读。

暂不引入：

- PyTorch Lightning；
- HuggingFace Trainer；
- Hydra；
- callback framework；
- registry system。

当课程后期复杂度真的要求时再引入。

## 5. Reproducibility

必须：

- 显式 seed；
- dataset local RNG；
- 保存 config；
- train/test 分开 seed；
- checkpoint step 可配置；
- device 可配置；
- 输出目录不覆盖旧 run。

## 6. Diagnostics First-Class

不能只返回 loss。

按 Experiment Spec 返回所需：

- gradient norms；
- activation stats；
- weight/update norms；
- representation samples；
- Jacobian metrics；
- attention statistics；

但只实现当前实验真正需要的部分。

## 7. Rule of Two

某个 helper 如果只在一个实验使用，优先放实验目录。

当第二个实验也需要时，再下沉到 `src/toy_model`。

例外：

- seed；
- serialization；
- run manifest；
- 基础 diagnostics；

这些从一开始就可以公共化。

## 8. Tests

至少包括：

- shape；
- dtype；
- deterministic seed；
- known analytical case；
- error handling；
- metric sanity；
- diagnostics 不改变 model output。

例如 Jacobian 测试应使用线性层：

`y = Wx`

并检查自动微分 Jacobian 是否等于 W。

## 9. 禁止隐藏结论

Agent 可以汇总结果，但不得自动写：

> 模型 A 证明了深度一定更好。

结论必须区分：

- observation；
- interpretation；
- unresolved alternative explanation。

## 10. 每次交付格式

交付时列出：

- changed files；
- run commands；
- tests run；
- generated outputs；
- expected observations；
- known limitations；
- code-review questions for learner。
