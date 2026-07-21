# Spiral 二维分类实验：实现说明

本次实现把课程从一维函数拟合推进到二维分类几何，研究问题是：

> 在参数量近似相同的条件下，把容量分配给更深的 MLP，是否更容易学习 Spiral 数据的弯曲决策边界？

## 一、实现范围

新增了三类可复用模块：

1. `toy_model.data.sample_spiral`
   - 生成平衡的多分类 Spiral 数据；
   - 独立控制类别数、每类样本量、旋转圈数、半径范围和角度噪声；
   - 使用函数局部随机数生成器，不污染全局 PyTorch 随机状态；
   - 默认设置正的 `radius_min`，避免不同类别在原点退化为不可辨认的数据。

2. `toy_model.training.fit_classification`
   - 使用 full-batch Adam 和 `CrossEntropyLoss`；
   - 模型直接输出 logits，不手动应用 Softmax；
   - 记录逐步 loss、accuracy 和指定 checkpoint；
   - 沿用现有回归模块中的 device 解析和 state-dict 快照逻辑。

3. `toy_model.evaluation`
   - cross entropy；
   - overall accuracy；
   - per-class accuracy；
   - confusion matrix；
   - top-two logit margin；
   - signed true-class margin；
   - 正确类别平均概率。

没有加入通用 Trainer、Registry、Hydra、Callback 或 DataLoader 抽象。

## 二、控制变量设计

对比模型：

| Model | Hidden dimensions | Parameters |
|---|---:|---:|
| Linear | none | 9 |
| Shallow MLP | `[256]` | 1539 |
| Medium MLP | `[36, 36]` | 1551 |
| Deep MLP | `[26, 26, 26]` | 1563 |

Linear 是结构失配的负对照。三个非线性网络的参数量差异约在 2% 以内。

固定变量包括：

- 相同 Spiral 数据生成机制；
- 相同训练与测试样本数量；
- 所有非线性模型均使用 ReLU；
- 相同 Adam、学习率、训练步数和 full-batch 设置；
- 相同随机种子集合；
- 相同评价网格。

主要改变变量是：

- 网络深度；
- 与深度配套的宽度分配。

## 三、为什么模型不加 Softmax

`torch.nn.CrossEntropyLoss` 接收未归一化 logits，并在内部执行数值稳定的
`log_softmax + negative log likelihood`。因此模型最后一层应保持线性：

```python
logits = model(x)
loss = torch.nn.CrossEntropyLoss()(logits, labels)
```

只有在可视化概率或报告置信度时，才显式计算：

```python
probabilities = torch.softmax(logits, dim=-1)
```

## 四、运行方法

将 ZIP 内容解压到仓库根目录后：

```bash
conda activate uad
cd toy_model
pip install -e . --no-deps
pytest tests -v
```

先检查数据：

```bash
python experiments/02_geometry/inspect_spiral.py
```

训练一个模型：

```bash
python experiments/02_geometry/train_spiral.py
```

参数量匹配的三种深度对比：

```bash
python experiments/02_geometry/compare_spiral_depths.py
```

缩短训练进行快速检查：

```bash
python experiments/02_geometry/compare_spiral_depths.py \
    --steps 500 \
    --seeds 0 \
    --grid-points 120
```

## 五、输出内容

单模型实验保存：

- `config.json`
- `metrics.json`
- `training_history.csv`
- `model.pt`
- 多个 step checkpoint
- training/test dataset
- training loss
- training accuracy
- final decision boundary
- checkpoint decision boundaries
- top-two logit margin
- confusion matrix

多模型实验保存：

- 每个 model × seed 的指标；
- `metrics.csv`
- `training_history.csv`
- `summary.csv`
- 实验问题、假设、控制变量和汇总结果；
- 每个模型与 seed 的 decision boundary 和 margin map；
- 跨 seed accuracy 与 cross-entropy 汇总图。

所有图表文字均为英文。

## 六、如何解释结果

- Linear 训练与测试都差：主要是函数类与弯曲边界结构失配。
- MLP 训练也差：可能是优化预算不足，也可能是当前容量不足。
- MLP 训练好、测试差：主要是几何泛化或数据噪声问题。
- 深模型均值更高但方差更大：表达效率可能更强，但优化更敏感。
- 更深模型 margin 更大：不仅分类正确，而且在测试样本上离决策边界更远。
- 不应仅根据 accuracy 判断边界是否合理；需要联合观察边界图、margin 和 seed 方差。

## 七、本次验证

在隔离重建环境中完成：

- Python 静态编译通过；
- 15 个新增单元测试通过；
- 数据可视化脚本通过；
- 单模型训练、checkpoint、绘图和保存流程通过；
- 多模型比较、CSV 汇总和图表流程通过。

使用默认 Spiral 数据、三个 seed、每个模型 2000 个 full-batch steps 的无绘图验证结果：

| Model | Mean test accuracy | Std |
|---|---:|---:|
| Linear | 0.4463 | 0.0101 |
| Shallow | 0.9896 | 0.0061 |
| Medium | 0.9970 | 0.0032 |
| Deep | 0.9981 | 0.0013 |

这些数字只用于确认任务和实现行为合理，不应替代你在本地 CUDA 环境中正式运行后的实验结果。
