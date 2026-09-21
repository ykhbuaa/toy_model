# Deep Learning Toy Model Research Curriculum

这不是一套“按知识点讲完深度学习”的课程，而是一套实验驱动的研究训练系统。

目标是让学习者逐渐形成以下能力：

1. 把神经网络理解成可学习的函数、映射、表示与动力系统；
2. 把论文中的改进拆成可验证的机制假设；
3. 用最小控制实验区分 representation / optimization / data / inductive bias；
4. 不只看最终 loss，而是能诊断激活、梯度、Jacobian、表示几何、频谱和因果干预；
5. 最终把这些直觉迁移到 Transformer、生成模型、World Model、Diffusion Policy 和 VLA。

## 推荐阅读顺序

先读通用文件：

- `docs/toy_model_guidance_v1/docs/00_course_charter.md`：整个项目的学习原则与路线。
- `docs/toy_model_guidance_v1/docs/01_experiment_protocol.md`：所有 Toy Model 必须遵守的实验协议。
- `docs/toy_model_guidance_v1/docs/02_visualization_and_diagnostics.md`：统一的可视化与模型诊断框架。
- `docs/toy_model_guidance_v1/docs/03_agent_implementation_contract.md`：交给 coding agent 的实现约束。
- `docs/toy_model_guidance_v1/docs/04_code_review_checklist.md`：学习者 review 代码时使用。
- `docs/toy_model_guidance_v1/docs/05_repo_migration_plan.md`：现有仓库如何最小代价升级。
- `docs/toy_model_guidance_v1/docs/PAPER_MAP.md`：论文与实验之间的对应关系。

再按顺序推进 `docs/toy_model_guidance_v1/docs/stages/`。

## 阶段

1. `docs/toy_model_guidance_v1/docs/stages/01_function_and_generalization.md`
2. `docs/toy_model_guidance_v1/docs/stages/02_geometry_and_inductive_bias.md`
3. `docs/toy_model_guidance_v1/docs/stages/03_optimization_and_signal_propagation.md`
4. `docs/toy_model_guidance_v1/docs/stages/04_normalization_residual_representation.md`
5. `docs/toy_model_guidance_v1/docs/stages/05_attention_and_transformer.md`
6. `docs/toy_model_guidance_v1/docs/stages/06_generative_models.md`
7. `docs/toy_model_guidance_v1/docs/stages/07_world_models.md`
8. `docs/toy_model_guidance_v1/docs/stages/08_policy_and_diffusion_policy.md`
9. `docs/toy_model_guidance_v1/docs/stages/09_vla_bridge.md`

不要以“读完”为通关标准，而以每个阶段末尾的 **Exit Criteria** 为准。

## 项目循环

Research Question
→ Mechanistic Hypothesis
→ Prediction Before Running
→ Controlled Experiment
→ Diagnostics
→ Observation
→ Competing Explanations
→ Falsification Experiment
→ Connection to Paper
→ Transfer to Larger Models

最重要的习惯：**实验前先预测。**

如果结果出来之后才开始想解释，很容易把任何现象都合理化。
