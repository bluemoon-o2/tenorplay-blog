---
title: "Talk Linear Attention"
date: 2026-03-10T12:00:00+08:00
lastmod: 2026-03-10T12:00:00+08:00
author: "TensorPlay Team"
tags: ["LLM", "Linear Attention"]
categories: ["技术博客"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## 引言
自2017年Vaswani等人提出Transformer架构以来，基于Softmax Attention的注意力机制成为序列建模的核心组件，支撑了大语言模型的快速发展。但Softmax Attention存在**二次计算复杂度**的固有问题，随着序列长度$L$的增加，$O(L^2d)$的计算量和内存开销呈平方级增长，成为长序列建模和大模型高效推理、训练的核心瓶颈。

为解决这一问题，Katharopoulos等人在2020年提出**线性注意力（Linear Attention）**，通过移除Softmax Attention中的softmax归一化项，将注意力的计算复杂度降至线性级别，实现了推理阶段的常数内存占用。但原始线性注意力存在训练效率低、性能不及Softmax Attention的问题，后续研究者通过分块并行、硬件高效实现、门控机制、记忆更新策略等方式不断优化，诞生了Flash Linear Attention、门控线性注意力（Gated Linear Attention, GLA）、并行化DeltaNet等技术，让线性注意力成为兼具效率和性能的Transformer替代方案。

本文将从Softmax Attention的痛点出发，逐层剖析线性注意力的核心原理、训练困境、硬件优化方案、性能提升手段，以及针对联想召回任务的改进方向，最终梳理线性注意力的泛化趋势和未来研究方向。

## 1 注意力的痛点：从Softmax Attention说起
Transformer的核心是**多头自注意力（Multi-Head Self-Attention, MHSA）**，其本质是通过查询（Query, Q）、键（Key, K）、值（Value, V）的交互实现序列中元素的关联建模。对于维度为$L \times d$的输入序列$X$（$L$为序列长度，$d$为隐藏层维度），Softmax Attention的核心计算过程为：
$$
O = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} \odot M\right) V
$$
其中$Q=XW_Q, K=XW_K, V=XW_V$为可学习参数映射后的矩阵，$M$为因果掩码（避免未来信息泄露），$\odot$为按元素相乘。

### 1.1 Softmax Attention的复杂度分析
Softmax Attention的计算和内存开销主要体现在三个阶段，整体复杂度为$O(L^2d + Ld^2)$：
1. **Q/K/V映射**：复杂度$O(Ld^2)$，为线性级；
2. **注意力矩阵计算**：$QK^\top$生成$L \times L$的注意力矩阵，结合softmax归一化，复杂度$O(L^2d)$，为平方级；
3. **注意力加权V**：注意力矩阵与V相乘，复杂度仍为$O(L^2d)$。

### 1.2 训练与推理的模式差异
Softmax Attention的**并行训练**和**递归推理**模式存在显著差异，这也是其效率瓶颈的重要体现：
- **训练阶段（并行形式）**：可一次性计算整个序列的注意力，矩阵乘法密集，能充分利用GPU的张量核心，但平方级的计算量随序列长度增长快速膨胀；
- **推理阶段（递归形式）**：生成式任务中需逐token计算，需维护**KV-Cache**存储历史的K/V向量，内存开销为$O(Ld)$，随序列长度线性增长，且每一步仍需计算当前query与所有历史key的相似度，计算量为$O(Ld)$。

简言之，Softmax Attention实现了高精度的序列建模，但平方级复杂度使其在长序列场景下的训练和推理效率极低，这为线性注意力的提出奠定了基础。

## 2 线性注意力：Softmax Attention的轻量化替代
线性注意力的核心思路是**移除Softmax Attention中的softmax归一化项**，将注意力的核心计算简化为QK的直接内积与V的加权，从根本上改变注意力的计算范式。

### 2.1 线性注意力的核心公式
原始线性注意力的计算式为：
$$
O = \left(QK^\top \odot M\right) V
$$
对比Softmax Attention，仅移除了softmax归一化和尺度因子$\frac{1}{\sqrt{d}}$，但这一改动让注意力的计算特性发生了本质变化。

对于生成式推理的逐token计算场景（时刻$t$），线性注意力的输出可重写为：
$$
o_t = \sum_{j=1}^t (q_t^\top k_j) v_j
$$
通过**矩阵乘法结合律**，可将上式进一步变形为：
$$
o_t = q_t^\top \underbrace{\left(\sum_{j=1}^t k_j v_j^\top\right)}_{S_t \in \mathbb{R}^{d \times d}}
$$
其中$S_t$为**矩阵值的隐藏状态**，且满足递归更新关系：
$$
S_t = S_{t-1} + k_t v_t^\top
$$
最终推理输出简化为：
$$
o_t = q_t^\top S_t
$$

### 2.2 线性注意力的核心优势：常数内存推理
从上述递归公式可以看出，线性注意力的推理阶段仅需维护一个固定维度$d \times d$的隐藏状态$S_t$，而非随序列长度增长的KV-Cache，**内存开销降至常数$O(d^2)$**，这是其相较于Softmax Attention的核心优势。

线性注意力的训练与推理复杂度对比如下：
- 训练阶段（并行形式）：仍为$O(L^2d)$，与Softmax Attention一致；
- 推理阶段（递归形式）：计算量$O(Ld^2)$，内存$O(d^2)$，均为线性/常数级。

## 3 线性注意力的训练困境与分块并行方案
虽然线性注意力实现了推理阶段的效率飞跃，但**原始线性注意力的训练效率极低**，无法直接落地，其核心问题在于递归形式的训练特性与GPU硬件的适配性差，而分块并行（Chunkwise Parallel）是解决这一问题的关键方案。

### 3.1 线性注意力的训练核心问题
若直接采用递归形式训练线性注意力，会面临三大问题：
1. **严格串行计算**：缺乏序列级并行性，无法利用GPU的并行计算能力；
2. **无矩阵乘法操作**：所有计算为按元素加/乘或归约，无法利用GPU的张量核心（Tensor Core），计算效率低；
3. **高IO成本**：需要物化每个时间步的隐藏状态，隐藏层维度$d$较大时，数据在显存（HBM）和片上内存（SRAM）之间的交互成本极高。

尽管理论上可通过**并行扫描**将训练计算复杂度降至$O(Ld^2)$，但实际工程中完全不可行。

### 3.2 分块并行形式（Chunkwise Parallel Form）
Hua等人（2022）、Sun等人（2023）提出的**分块并行**是线性注意力训练的核心工程方案，其核心是将长度为$L$的序列划分为若干个长度为$C$的块（Chunk），在**块内并行计算**、**块间递归传递状态**，实现“并行”与“递归”的折中。

分块并行的核心步骤分为三步：
1. **块内状态计算**：对每个块独立计算局部状态$K_{[i]}^\top V_{[i]}$，块内完全并行；
2. **块间状态传递**：将前一个块的全局状态$S_{[i]}$传递给下一个块$S_{[i+1]}$，仅块间存在串行依赖；
3. **输出计算**：每个块的输出由**前序块的全局状态贡献**和**当前块的局部注意力贡献**两部分组成，即$O_{[i+1]} = Q_{[i+1]} S_{[i]} + \text{LocalAttention}(Q_{[i+1]}, K_{[i+1]}, V_{[i+1]})$。

### 3.3 分块参数$C$的选择
分块参数$C$决定了线性注意力训练的效率和性能平衡，其特性为：
- $C=L$：退化为**完全并行形式**，训练效率高但计算复杂度仍为$O(L^2d)$；
- $C=1$：退化为**完全递归形式**，计算复杂度$O(Ld^2)$但训练效率极低；
- 实际工程中$C$取64/128/256（16的倍数），以适配GPU张量核心，同时将递归步骤从$L$降至$L/C$，大幅提升训练效率。

分块并行让线性注意力实现了**训练长度的线性扩展**，是其从理论走向工程实现的关键。

## 4 硬件高效的线性注意力实现：Flash Linear Attention
原始PyTorch实现的线性注意力在速度上甚至不及优化后的Softmax Attention（如FlashAttention-2），核心原因是**IO成本未做优化**。受FlashAttention启发，研究者提出**Flash Linear Attention**，通过**IO感知的硬件优化**和**核融合**，大幅降低线性注意力的训练IO成本，提升运行速度。

### 4.1 Flash Linear Attention的两种实现版本
Flash Linear Attention针对不同训练场景设计了**非物化版本**和**物化版本**，分别适配短序列和长序列/大尺度训练：
#### 4.1.1 非物化版本（Nonmaterialization Version）
- 核心思路：将隐藏状态$S_t$全程保存在GPU片上内存（SRAM）中，避免HBM与SRAM之间的频繁数据交互；
- 优势：IO成本最小化，仅需从HBM加载一次Q/K/V，适合**短序列训练**（IO成本占主导）；
- 劣势：缺乏块间的序列并行性，需要大批次大小（Batch Size）才能充分利用GPU的流多处理器（SM），否则SM利用率低。

#### 4.1.2 物化版本（Materialization Version）
- 核心思路：将训练分为**串行状态计算**和**并行输出计算**两个阶段，通过核融合将块内状态计算和块间状态传递融合为单个核函数，同时块间输出计算完全并行；
- 优势：支持分块并行，SM利用率高，适合**长序列/大尺度训练**；
- 劣势：IO成本略高（K/V需加载两次，$S_t$需存储和加载一次），可通过**反向传播重计算**降低内存占用。

### 4.2 硬件优化的效果
Flash Linear Attention通过IO感知的设计和核融合，在运行速度上实现了质的飞跃，相较于纯PyTorch实现的线性注意力，其在不同序列长度下的速度均大幅提升，甚至可与优化后的FlashAttention-2媲美，成为线性注意力的标准工程实现（开源地址：https://github.com/sustcsonglin/flash-linear-attention）。

## 5 性能提升：门控线性注意力（Gated Linear Attention, GLA）
线性注意力虽解决了效率问题，但**性能较Softmax Attention存在显著差距**（如困惑度PPL更高、LM Eval指标更低），核心原因是原始线性注意力的隐藏状态$S_t$为简单的累加更新，缺乏**数据依赖的记忆遗忘机制**。为此，Yang等人在ICML 2024提出**门控线性注意力（GLA）**，通过引入**数据依赖的乘法门控**，让线性注意力的性能接近甚至超过Softmax Attention和主流的状态空间模型（SSM）如Mamba。

### 5.1 GLA的核心门控机制
原始线性注意力的隐藏状态更新为简单累加：
$$
S_t = S_{t-1} + k_t v_t^\top
$$
GLA在其中引入**门控矩阵$G_t$**，实现数据依赖的记忆遗忘，更新公式为：
$$
S_t = G_t \odot S_{t-1} + k_t v_t^\top
$$
其中门控矩阵$G_t$为**标量门控的单位矩阵扩展**，即：
$$
G_t = \alpha_t \cdot \mathbf{1}, \quad \alpha_t = \sigma\left(x_t W_{\alpha_1} W_{\alpha_2}\right)^{\frac{1}{\tau}}
$$
$\sigma$为Sigmoid激活函数，$\tau$为温度系数，$\alpha_t \in (0,1)$决定了对历史记忆的保留/遗忘比例，且由当前输入$x_t$动态决定。

### 5.2 GLA的分块并行形式
GLA保留了线性注意力的分块并行特性，并针对门控的**累积衰减**设计了**衰减感知的分块并行形式**，通过定义衰减系数$\Lambda, \Gamma, \gamma$，实现块间门控状态的高效传递，保证并行训练时的衰减一致性。

其核心是通过累积衰减$b_t = \prod_{j=1}^t \alpha_j$，将块内的门控衰减归一化，最终实现GLA的并行计算式：
$$
O = \left( (Q \odot B) \left( \frac{K}{B} \right)^\top \odot M \right) V
$$
其中$B$为累积衰减系数的矩阵形式，该公式既保证了门控的有效性，又能利用GPU张量核心进行矩阵乘法，兼顾性能和效率。

### 5.3 GLA的性能表现
在1.3B参数量、100B tokens训练的实验中，GLA的性能全面超越传统线性注意力（如RetNet）和主流SSM（如Mamba），接近甚至超过Softmax Attention（Transformer++），核心指标如下：

| 模型               | PPL  | LM Eval | Retrieval |
|--------------------|------|---------|-----------|
| Transformer++      | 16.9 | 50.9    | 41.8      |
| RetNet（线性注意力）| 18.6 | 48.9    | 30.6      |
| Mamba（SSM）| 17.1 | 50.0    | 27.6      |
| GLA（门控线性注意力）| 17.2 | 51.1    | 37.7      |

此外，GLA在**长序列泛化**和**召回导向任务**上的表现远优于传统线性注意力和SSM，成为线性注意力的性能标杆。

### 5.4 GLA与状态空间模型（SSM）的关系
GLA的核心贡献之一是建立了**线性注意力与SSM的数学关联**，证明了**门控线性注意力是可扩展的状态空间模型**，且主流SSM（Mamba、Mamba-2、HGRN-2、RWKV-6等）均为门控线性注意力的特殊子集：
- 所有SSM的递归更新均可表示为门控线性注意力的形式$S_t = G_t \odot S_{t-1} + k_t v_t^\top$；
- 可扩展的SSM要求门控$G_t$为$\alpha_t \beta_t^\top$的形式，以将递归转换为矩阵乘法形式，适配GPU张量核心。

这一关联为线性注意力和SSM的融合研究提供了理论基础。

## 6 联想召回的改进：DeltaNet与并行化
尽管GLA大幅提升了线性注意力的性能，但**线性注意力和SSM在联想召回任务（Associative Recall）上仍存在显著缺陷**。联想召回任务要求模型根据查询键检索对应的数值（如输入$A4B3C6 \to$查询$A?C?$输出$4,6$），是检验模型长序列记忆和关联能力的核心任务，而线性注意力/SSM在该任务上的准确率远低于预期。

### 6.1 DeltaNet的核心思路
DeltaNet的核心是引入**记忆检索-更新机制**，替代线性注意力的简单累加更新，实现对键值对的精准记忆和召回，其核心步骤为：
1. **记忆检索**：根据当前key从历史隐藏状态中检索旧的value：$v_t^{old} = S_{t-1} k_t$；
2. **记忆融合**：将旧value与当前value通过门控融合：$v_t^{new} = \beta_t v_t + (1-\beta_t) v_t^{old}$，其中$\beta_t = \sigma(W_\beta x_t) \in (0,1)$；
3. **记忆更新**：从隐藏状态中移除旧value，写入新value：$S_t = S_{t-1} - v_t^{old} k_t^\top + v_t^{new} k_t^\top$；
4. **输出计算**：$o_t = S_t q_t$。

DeltaNet通过**显式的记忆增删**实现了精准的键值对关联，在联想召回任务上的准确率大幅超越GLA、Mamba、RetNet等模型。

### 6.2 DeltaNet的并行化困境与解决
原始DeltaNet的核心问题是**伪值向量$u_t = v_t^{new} - v_t^{old}$依赖历史隐藏状态$S_{t-1}$**，无法直接并行训练，而研究者通过**重参数化**和**WY表示**解决了这一问题：
1. **重参数化**：将DeltaNet的更新公式重写为矩阵乘法形式：$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$，其中$I$为单位矩阵；
2. **WY表示**：利用Householder矩阵的乘积表示，将递归的矩阵乘积转换为可分块并行的向量累加，实现$S_t$的分块并行计算。

### 6.3 并行化DeltaNet的性能与混合优化
并行化后的DeltaNet在保持联想召回优势的同时，实现了高效的GPU训练，在1.3B模型上的PPL与Softmax Attention持平，LM Eval指标更优。为进一步提升召回性能，研究者提出**混合DeltaNet**，结合滑动窗口注意力/全局注意力与DeltaNet：
1. **混合1**：隔层使用滑动窗口注意力和DeltaNet；
2. **混合2**：在第2层和中间层引入全局注意力，其余层为DeltaNet。

混合DeltaNet的性能超越了纯Softmax Attention，成为目前线性注意力在综合任务上的最优方案，核心指标如下：

| 模型               | PPL  | LM Eval | Retrieval |
|--------------------|------|---------|-----------|
| Transformer++      | 16.9 | 50.9    | 41.8      |
| GLA                | 17.2 | 51.1    | 37.7      |
| DeltaNet           | 16.9 | 51.6    | 34.7      |
| DeltaNet+全局注意力（2层） | 16.6 | 51.8    | 47.9      |

## 7 线性注意力的泛化与未来方向
在GLA和DeltaNet的基础上，研究者进一步提出**广义线性Transformer**，将线性注意力的门控更新从**按元素乘积**扩展为**结构化矩阵乘法**，同时梳理了线性注意力未来的核心研究方向。

### 7.1 广义线性Transformer
原始线性注意力和GLA的更新为**按元素乘积**（$S_t = S_{t-1} \odot G_t + k_t v_t^\top$），复杂度为$O(d^2)$，但无法建模通道间的交互；而直接将其扩展为全矩阵乘法（$S_t = S_{t-1} G_t + k_t v_t^\top$）虽能建模通道交互，但复杂度升至$O(d^3)$，无法落地。

为此，研究者提出**结构化矩阵乘法**的广义更新公式：
$$
S_t = S_{t-1}(I - a_t b_t^\top) + v_t k_t^\top
$$
其中$I - a_t b_t^\top$为**单位矩阵+低秩矩阵**，既实现了通道间的交互，又将计算复杂度控制在$O(kd^2)$（$k$为低秩维度），而DeltaNet正是该公式的特殊情况（$a_t = b_t = \sqrt{\beta_t} k_t$）。

### 7.2 开放与未来工作
线性注意力的研究仍处于快速发展阶段，核心开放问题和未来方向包括：
1. **更通用的关联算子**：探索$S_t = S_{t-1} \cdot M_t + k_t v_t^\top$中更通用的矩阵算子$\cdot$，兼顾效率和表达能力；
2. **长序列建模的进一步优化**：结合稀疏注意力、滑动窗口注意力与线性注意力，实现超长篇序列（如100K+）的高效建模；
3. **硬件与算法的协同设计**：针对不同硬件（GPU/TPU/NPU）设计定制化的线性注意力实现，进一步挖掘硬件效率；
4. **多模态线性注意力**：将线性注意力扩展至视觉、语音等多模态任务，解决多模态长序列的建模瓶颈。

## 8 总结
线性注意力作为Softmax Attention的轻量化替代方案，通过移除softmax归一化实现了**推理阶段的常数内存占用**，解决了Transformer长序列建模的核心瓶颈。从原始线性注意力的训练困境，到分块并行的工程突破，再到Flash Linear Attention的硬件高效实现、GLA的性能提升、DeltaNet的联想召回优化，线性注意力已从理论方案发展为**兼具效率和性能**的序列建模技术，且与状态空间模型建立了紧密的数学关联。

目前，混合DeltaNet已在综合任务上超越传统Softmax Attention，而广义线性Transformer为线性注意力的进一步泛化提供了理论框架。随着硬件与算法的协同设计、多模态扩展等研究的推进，线性注意力将成为大语言模型长序列训练、高效推理的核心技术，支撑更大规模、更长序列的大模型落地。

线性注意力的发展证明，**高效性与性能并非不可兼得**，通过对注意力机制的数学重构、工程优化和机制创新，能够在降低计算和内存开销的同时，甚至超越原始架构的性能，这也是大模型轻量化、实用化的核心研究方向。

## 引用

> **引用**：转载或引用本文内容时，请注明原作者和来源。

**Cited as:**

> TensorPlay Team. (March 2026). 谈线性注意力.
https://blog.tensorplay.cn/zh/posts/linear-attention

Or

```bibtex
@article{tensorplay2026-linear-attention,
  title   = "Talk Linear Attention",
  author  = "TensorPlay Team",
  journal = "blog.tensorplay.cn",
  year    = "2026",
  month   = "March",
  url     = "https://blog.tensorplay.cn/zh/posts/talk-linear-attention"
}
```