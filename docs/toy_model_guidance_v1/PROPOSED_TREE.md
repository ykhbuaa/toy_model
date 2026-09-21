# Proposed Directory Tree

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
│   │   ├── 01_function_and_generalization.md
│   │   ├── 02_geometry_and_inductive_bias.md
│   │   ├── 03_optimization_and_signal_propagation.md
│   │   ├── 04_normalization_residual_representation.md
│   │   ├── 05_attention_and_transformer.md
│   │   ├── 06_generative_models.md
│   │   ├── 07_world_models.md
│   │   ├── 08_policy_and_diffusion_policy.md
│   │   └── 09_vla_bridge.md
│   └── templates/
│       ├── EXPERIMENT_SPEC_TEMPLATE.md
│       └── RESULTS_REVIEW_TEMPLATE.md
├── experiments/
│   ├── README.md
│   ├── <keep existing first-stage folder>
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
│   │   ├── parameters.py
│   │   ├── activations.py
│   │   ├── jacobian.py          # add later
│   │   ├── spectrum.py          # add later
│   │   ├── representation.py    # add later
│   │   └── attention.py         # add later
│   ├── visualization/
│   │   ├── training.py
│   │   ├── function_1d.py
│   │   └── classification_2d.py
│   └── utils/
│       ├── random.py
│       ├── io.py
│       └── runs.py
├── reports/
├── runs/
└── tests/
```

The files marked “add later” should not be created as empty abstractions now.
