# Stage 07 — World Models

把 World Model 拆成三个对象：

1. Representation；
2. Dynamics；
3. Rollout / Prediction。

## Core 1 — Known-State Dynamics

二维环境：

`state=(x,y,vx,vy)` 或简单 grid state。

学习：

`(s_t, a_t) -> s_{t+1}`。

必须分别报告：

- one-step error；
- h-step rollout error；
- error vs horizon。

核心观察：

> one-step error 很小不等于 rollout 稳定。

## Core 2 — Learned Latent State

输入改成 image observation。

学习：

`o_t -> z_t`

和：

`(z_t,a_t) -> z_{t+1}`。

诊断：

- latent PCA；
- latent vs true state correlation / probe；
- latent trajectory；
- reconstruction；
- transition error。

## Core 3 — Distribution Shift in Rollout

teacher-forced dynamics 与 free rollout 对比。

观察模型自己的预测如何改变下一步输入分布。

这和 Behavior Cloning 的 covariate shift 是同一个深层结构。

## Core 4 — Planning Through Model

在极小环境中使用 learned model 做短 horizon planning。

比较：

- oracle model planning；
- learned model planning；
- one-step good but planning bad 的 case。

## 必做图

- true vs predicted trajectory；
- error vs horizon；
- latent trajectory；
- state-space vector field；
- rollout divergence heatmap。

## Exit Criteria

能够解释：

- state / observation / latent 的区别；
- one-step 与 multi-step objective 的错位；
- compounding error；
- model bias 如何影响 planning；
- world model 为什么不仅是“预测下一帧”。
