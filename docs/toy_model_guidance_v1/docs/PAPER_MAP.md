# Paper Map — 只在实验产生问题后读

原则：论文不是先修列表，而是“实验现象的解释材料”。

## Function / Generalization

- Cybenko (1989), Approximation by superpositions of a sigmoidal function
- Rahaman et al. (2019), On the Spectral Bias of Neural Networks
- Jacot et al. (2018), Neural Tangent Kernel（选读）
- Zhang et al. (2017), Understanding deep learning requires rethinking generalization
- Nakkiran et al. (2020), Deep Double Descent（选读）

## Optimization / Signal Propagation

- Glorot & Bengio (2010), Understanding the difficulty of training deep feedforward neural networks
- He et al. (2015), Delving Deep into Rectifiers
- Saxe et al. (2014), Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
- Schoenholz et al. (2017), Deep Information Propagation（选读）

## Normalization / Residual

- Ioffe & Szegedy (2015), Batch Normalization
- Ba, Kiros & Hinton (2016), Layer Normalization
- He et al. (2016), Deep Residual Learning for Image Recognition
- Xiong et al. (2020), On Layer Normalization in the Transformer Architecture（后期）

## Representation

- Elhage et al. (2022), Toy Models of Superposition
- Kornblith et al. (2019), Similarity of Neural Network Representations Revisited
- Alain & Bengio (2017), Understanding intermediate layers using linear classifier probes

## Attention / Transformer

- Vaswani et al. (2017), Attention Is All You Need
- Olsson et al. (2022), In-context Learning and Induction Heads
- Elhage et al. (2021), A Mathematical Framework for Transformer Circuits（选读）

## Generative Models

- Kingma & Welling (2013), Auto-Encoding Variational Bayes
- Goodfellow et al. (2014), Generative Adversarial Nets（概念对比）
- Ho et al. (2020), Denoising Diffusion Probabilistic Models
- Song et al. (2021), Score-Based Generative Modeling through Stochastic Differential Equations
- Lipman et al. (2023), Flow Matching for Generative Modeling

## World Models

- Ha & Schmidhuber (2018), World Models
- Hafner et al. (2019/2020), Dreamer
- Hafner et al. (2023; later Nature version), DreamerV3 / Mastering Diverse Domains through World Models

## Robot Policy

- Behavior Cloning / DAgger：先理解 covariate shift
- Chi et al. (2023/2024), Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

## VLA

这些论文用于建立架构谱系，不追求逐篇复现：

- RT-2 (2023)
- OpenVLA (2024)
- π0 / flow-based VLA (2024)

阅读 VLA 时始终把问题拆回：

- perception representation；
- language conditioning；
- action representation；
- temporal horizon；
- distribution modeling；
- closed-loop robustness；
- pretraining transfer。
