# Deep Learning Toy Models

实验驱动地理解深度学习，从函数拟合一路走到 World Model 与 VLA。

## Philosophy

不是以“实现更多模型”为目标，而是通过小规模、变量可控、现象可视化的实验建立：

- function intuition；
- optimization intuition；
- representation diagnostics；
- inductive bias；
- information flow；
- generative / dynamical modeling；
- robot policy intuition。

## Start Here

见 `docs/README.md`。

## Repository Layers

- `experiments/`：一个实验回答一个研究问题；
- `src/toy_model/`：跨实验复用的最小公共代码；
- `tests/`：保护数学与实验契约；
- `runs/`：自动生成结果，不长期版本控制；
- `reports/`：人工挑选的关键实验结论；
- `docs/`：课程、实验协议和阶段路线。

## Current Principle

先预测，再运行；
先行为，再内部；
先相关，再因果；
先 toy，再大模型。
