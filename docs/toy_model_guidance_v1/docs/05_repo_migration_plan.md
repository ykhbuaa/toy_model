# 05 — Repository Migration Plan

## 当前判断

现有仓库已经有正确的骨架：

```text
experiments/
src/toy_model/
tests/
pyproject.toml
```

并且 Spiral 实验已经体现了：

- numbered experiment；
- reusable package；
- evaluation；
- checkpoint；
- 多 seed；
- decision boundary / margin / confusion matrix；
- raw CSV / JSON。

因此不建议推倒重来。

当前最需要修正的是：

1. 根 `README.md` 应成为课程入口；
2. 根目录的单个 implementation guide 应迁移到 `docs/`；
3. visualization / diagnostics 需要成为正式能力；
4. 通用 IO / seed / run manifest 不应长期散落在单个 experiment utils 中；
5. 增加 curated `reports/`，保存真正值得复盘的实验结论。

## 推荐目标结构

```text
toy_model/
├── README.md
├── pyproject.toml
├── docs/
│   ├── README.md
│   ├── 00_course_charter.md
│   ├── 01_experiment_protocol.md
│   ├── 02_visualization_and_diagnostics.md
│   ├── 03_agent_implementation_contract.md
│   ├── 04_code_review_checklist.md
│   ├── 05_repo_migration_plan.md
│   ├── PAPER_MAP.md
│   ├── stages/
│   └── templates/
├── experiments/
│   ├── README.md
│   ├── <existing 01_*>
│   ├── 02_geometry/
│   ├── 03_optimization/
│   ├── 04_signal_flow/
│   ├── 05_representation/
│   ├── 06_attention/
│   ├── 07_generation/
│   ├── 08_world_model/
│   ├── 09_policy/
│   └── 10_vla/
├── src/toy_model/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── diagnostics/
│   ├── visualization/
│   └── utils/
├── reports/
├── runs/              # gitignored
└── tests/
```

注意：不要为了和这里完全一致而重命名已经工作的 `01_*`。保持历史稳定比目录完美更重要。

## diagnostics 的演进

不要一次创建十几个空模块。

### Migration A — 现在就做

```text
src/toy_model/diagnostics/
├── __init__.py
├── parameters.py      # weight norm, grad norm, update norm
└── activations.py     # small forward-hook recorder
```

### Migration B — 到相应课程再加

```text
jacobian.py
spectrum.py
representation.py
attention.py
dynamics.py
```

## visualization 的演进

从 `spiral_experiment_utils.py` 中优先抽出跨实验部分：

```text
src/toy_model/visualization/
├── training.py
├── function_1d.py
└── classification_2d.py
```

建议迁移：

- plot_training_curves
- make_classification_grid
- plot_decision_boundary
- plot_logit_margin
- plot_confusion_matrix

`plot_checkpoint_boundaries` 可以暂留 local，等第二个 2D 实验需要时再抽。

## utils 的演进

可从实验 helper 中迁移：

```text
src/toy_model/utils/
├── random.py          # set_seed
├── io.py              # save_json / save_rows_csv
└── runs.py            # create_run_directory / manifest
```

## runs 与 reports 分离

`runs/`：

- 自动生成；
- 大；
- 可删除；
- 默认 gitignore。

`reports/`：

- 人工挑选；
- 小；
- 版本控制；
- 每个核心实验留下结论、关键图和复现命令。

推荐：

```text
reports/
└── 02_geometry_spiral_depth/
    ├── README.md
    ├── summary.csv
    └── figures/
```

## 不建议做的重构

当前阶段不要：

- 通用 Trainer；
- Experiment Registry；
- Plugin system；
- Hydra；
- YAML 套 YAML；
- Dataset base class；
- 复杂 callback tree。

课程仓库最重要的是机制透明度，不是工业扩展性。

## 迁移顺序

1. 写根 README 与 docs index；
2. 把现有 Spiral guide 移到 docs/experiments 或 reports；
3. 增加 diagnostics/parameters.py；
4. 增加 diagnostics/activations.py；
5. 把 seed / IO 抽到 utils；
6. 再逐步迁移通用 plotting；
7. 给下一个实验使用新结构；
8. 不要求旧实验一次性全部重写。
