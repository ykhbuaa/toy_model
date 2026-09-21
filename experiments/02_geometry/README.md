# Spiral Geometry Experiments

This experiment suite begins the two-dimensional geometry stage of the
Toy Model curriculum.

## Research question

At approximately matched parameter counts, does allocating MLP capacity
to greater depth make a curved Spiral decision boundary easier to learn?

The main comparison is:

| Model | Hidden dimensions | Trainable parameters |
|---|---:|---:|
| Linear | None | 9 |
| Shallow MLP | `[256]` | 1539 |
| Medium MLP | `[36, 36]` | 1551 |
| Deep MLP | `[26, 26, 26]` | 1563 |

## Experiment specification

Before running, the prediction is that deeper models will learn a usable
curved boundary within the same step budget, while the linear negative
control will remain structurally mismatched. If depth changes the learned
geometry, the effect should also appear in at least one internal measure:
layer activation separation, parameter gradient scale, input sensitivity,
or boundary transition density. A result that changes only accuracy while
all diagnostics remain indistinguishable leaves optimization and seed
effects unresolved.

The linear model is a negative control. The three nonlinear models have
parameter counts within roughly two percent of one another.

## Data generator

For class `c`, a latent progress variable controls radius and angle:

```text
u ~ Uniform(0, 1)
r = radius_min + (radius_max - radius_min) * u
theta = 2*pi*c/num_classes + 2*pi*turns*u + angular_noise
x = [r*cos(theta), r*sin(theta)]
```

A positive `radius_min` prevents all classes from becoming intrinsically
ambiguous at the origin.

## Commands

Inspect independently sampled training and test sets:

```bash
python experiments/02_geometry/inspect_spiral.py
```

Train one MLP and save boundary snapshots:

```bash
python experiments/02_geometry/train_spiral.py
```

Run the controlled depth comparison over three seeds:

```bash
python experiments/02_geometry/compare_spiral_depths.py
```

Run all tests:

```bash
pytest tests -v
```

The depth comparison also saves, for every model and seed:

- `activation_stats.csv`: activation mean, standard deviation, RMS and
  ReLU zero fraction on the fixed test probe;
- `pca_metrics.csv` and `layer_pca.png`: two-dimensional hidden-layer PCA
  with within-class and between-class distances;
- `parameter_gradient_norms.csv`: final parameter and full-batch gradient
  norms;
- `input_gradient_grid.npz` and `input_gradient_norm.png`: predicted-logit
  sensitivity over the same evaluation grid;
- `metrics.json`: the scalar boundary-complexity proxy and all diagnostics;
- `results_note.md`: the pre-run prediction and seed-aggregated observations,
  without turning them into an automatic mechanistic conclusion.

The boundary-complexity value is the fraction of neighboring grid cells
whose predicted class changes. It is a resolution-dependent geometry
proxy, not a measurement of the exact mathematical boundary length.

## Important interpretation

A deeper model obtaining better accuracy does not by itself prove that
depth always improves optimization. The result must be read together
with training loss and seed variance:

- low training accuracy indicates an optimization or representation issue;
- high training but low test accuracy indicates a generalization issue;
- large seed variance indicates optimization sensitivity;
- a failing linear baseline demonstrates structural mismatch with the
  curved decision boundary.
