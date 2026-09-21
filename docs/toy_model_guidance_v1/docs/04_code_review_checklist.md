# 04 — Learner Code Review Checklist

你不需要逐行审查所有 Python 细节。重点审查“这个代码是否真的在回答研究问题”。

## A. Research Validity

- [ ] 研究问题是否只有一个主变量？
- [ ] 实际代码改变的变量是否和 spec 一致？
- [ ] 是否存在未声明的超参变化？
- [ ] 参数量匹配是否真的计算过？
- [ ] 如果比较 compute，是否真的匹配 compute，而非只匹配 step？
- [ ] train/test 是否独立采样？
- [ ] 是否有数据泄漏？

## B. Reproducibility

- [ ] Python / NumPy / PyTorch seed 是否明确？
- [ ] dataset generator 是否污染全局 RNG？
- [ ] 是否保存 config？
- [ ] 是否保存 raw metrics？
- [ ] 是否保留 checkpoint？
- [ ] 多 seed 是否成对使用同样的数据规则？

## C. Training Correctness

- [ ] classification 是否输出 logits 而不是提前 softmax？
- [ ] optimizer.zero_grad 是否正确？
- [ ] model.train/eval 是否正确切换？
- [ ] loss 的 reduction 是否符合预期？
- [ ] dtype/device 是否一致？
- [ ] 是否检查 NaN/Inf？
- [ ] evaluation 是否处于 no_grad？

## D. Diagnostics Correctness

- [ ] forward hooks 是否会 remove？
- [ ] diagnostics 是否意外保留计算图？
- [ ] gradient norm 是 optimizer.step 前测的吗？
- [ ] update norm 是同一组参数 before/after 的差吗？
- [ ] activation statistics 是否区分不同 layer？
- [ ] Jacobian 的 input/output shape 是否明确？
- [ ] representation analysis 是否用固定 probe set？

## E. Visualization Honesty

- [ ] 对比图是否共享坐标轴？
- [ ] 是否故意截断 y 轴制造视觉差异？
- [ ] log scale 是否标出？
- [ ] 是否只展示最好看的 seed？
- [ ] 色彩语义是否一致？
- [ ] 图的原始数据能否重新生成？
- [ ] interpolation / extrapolation 区间是否视觉区分？

## F. Interpretation

Review 完代码后先回答：

1. 我认为这个实验最终最可能出现什么结果？
2. 如果结果相反，我会改变哪条理解？
3. 哪个图最能区分两个竞争机制？
4. 哪个 metric 只是结果指标，哪个是真正的 mechanism diagnostic？

这四个问题比“代码风格是否优雅”重要。
