# 00 — Course Charter

## 1. 最终目标

最终方向是 VLA 与 World Model，但前期不直接堆大模型。

我们要建立一组可以迁移的“观察语言”：

- Function：模型实现了怎样的映射？
- Representation：中间表示保存、丢失、展开了什么信息？
- Optimization：优化器实际沿什么方向移动？
- Inductive Bias：结构偏好哪些函数和解？
- Information Flow：信息如何跨层、跨 token、跨 modality 流动？
- Dynamics：一步预测怎样变成多步 rollout？
- Distribution：模型是在预测均值、条件分布、score，还是 trajectory distribution？
- Causality：某个内部特征只是相关，还是对输出有因果作用？

## 2. 课程不是“知识点列表”

每个主题必须至少包含四件事：

1. 一个极小的可控任务；
2. 一个机制假设；
3. 一个能看到模型内部的诊断；
4. 一个对应的经典或重要论文。

例如 Residual Connection 不以“记住缓解梯度消失”为结束，而要实际观察：

- 无 residual 时各层 gradient norm；
- 有 residual 时各层 gradient norm；
- block Jacobian；
- residual branch / identity branch 的相对大小；
- 深度增加以后训练曲线如何变化。

## 3. 核心实验哲学

### 3.1 一个实验只回答一个主要问题

禁止同时改变：

- 网络结构；
- 优化器；
- 数据规模；
- augmentation；
- 初始化；
- 训练预算。

如果必须同时改变，必须把它声明为“system comparison”，不能声称识别了单一机制。

### 3.2 区分四种失败

看到模型失败，必须问：

- Function class mismatch：模型表达不了？
- Optimization failure：表达得了但训练不到？
- Statistical failure：数据不足或分布错？
- Evaluation failure：指标根本没测到关心的能力？

### 3.3 先做行为，再做内部，再做干预

推荐诊断顺序：

1. 输出行为；
2. 函数/几何形状；
3. 激活与梯度；
4. 表示几何；
5. Jacobian / spectrum；
6. ablation / patching / causal intervention。

不要一开始就上复杂 mechanistic interpretability。

## 4. 快速推进策略

每阶段分成：

- Core：必须做，用于建立主干直觉；
- Diagnostic Deep Dive：选做，但至少每阶段做一个；
- Paper Bridge：把 toy phenomenon 对应到论文；
- Transfer Question：思考如何迁移到 World Model / VLA。

代码主要由 agent 实现，学习者的时间优先用于：

- 预测结果；
- review 代码；
- 设计 ablation；
- 看内部量；
- 写解释；
- 设计能推翻当前解释的下一实验。

## 5. 禁止事项

前四阶段尽量避免：

- 大数据集；
- 大模型；
- Lightning / Trainer 等隐藏训练循环的框架；
- 复杂配置系统；
- 自动超参搜索；
- 只给最终 accuracy 的实验；
- 没有 seed 重复的结构比较；
- 没有保存原始 metrics 的漂亮图。

我们的目标不是 benchmark，而是理解。
