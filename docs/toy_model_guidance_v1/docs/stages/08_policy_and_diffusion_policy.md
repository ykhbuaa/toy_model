# Stage 08 — Policy & Diffusion Policy

## Core 1 — Behavior Cloning

二维 reaching。

输入：

`state + goal`

输出：

`action`。

先构造 unimodal expert。

理解 supervised policy learning。

## Core 2 — Covariate Shift

只在 expert states 上训练。

部署 learner 后观察它进入训练集没有覆盖的 states。

画：

- expert state distribution；
- learner visited states；
- distance-to-training-support；
- failure location。

把它和 Stage 01 的 extrapolation、Stage 07 的 rollout shift 连接起来。

## Core 3 — Multimodal Expert

有障碍物时：

- left route；
- right route；

都合法。

MSE BC 可能平均成撞障碍物的动作。

画条件 action distribution，而不只是 trajectory。

## Core 4 — Diffusion Policy Toy

在二维动作或短 action chunk 上训练 conditional diffusion。

观察：

- noisy action samples；
- denoising trajectory；
- final action modes；
- rollout success。

对比：

- deterministic MSE；
- Gaussian policy；
- mixture density；
- diffusion。

## Core 5 — Receding Horizon / Action Chunk

研究：

- single action；
- action chunk；
- execute full chunk；
- receding horizon replan。

理解 Diffusion Policy 成功不只来自“用了 diffusion”。

## Exit Criteria

能够把 policy 视为条件 action distribution；
能够解释 covariate shift；
能够解释 multimodality；
能够说明 diffusion policy 解决了什么问题、没解决什么问题。
