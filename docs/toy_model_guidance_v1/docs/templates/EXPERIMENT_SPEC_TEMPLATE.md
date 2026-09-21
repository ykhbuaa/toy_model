# Experiment Spec — <name>

## 1. Research Question

一句可证伪的问题。

## 2. Why It Matters

它和哪一个深度学习机制有关？
未来如何迁移到 Transformer / World Model / Policy / VLA？

## 3. Hypotheses

### H1
机制描述。

### H2
竞争解释。

## 4. Prediction Before Running

- Behavior:
- Training dynamics:
- Internal diagnostics:
- Falsifying observation:

## 5. Variables

### Changed
- ...

### Controlled
- ...

### Measured
- ...

### Nuisance
- ...

## 6. Conditions

| Condition | Architecture | Params | Optimizer | Steps | Seed |
|---|---:|---:|---|---:|---:|

## 7. Required Diagnostics

### Level 0
- ...

### Level 1
- ...

### Internal
- ...

## 8. Required Figures

1. ...
2. ...

## 9. Required Files

- config.json
- metrics.csv
- traces/
- checkpoints/
- figures/
- results_note.md

## 10. Tests

- deterministic data generation
- shape
- analytical diagnostic test
- smoke train

## 11. Interpretation Rules

哪些结果支持 H1？
哪些结果支持 H2？
什么结果两者都解释不了？
