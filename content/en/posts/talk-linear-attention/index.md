---
title: "Talk Linear Attention"
date: 2026-03-10T12:00:00+08:00
author: "TensorPlay Team"
tags: ["LLM", "Linear Attention"]
categories: ["Technical Blog"]
ShowReadingTime: true
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## Introduction
Since Vaswani et al. proposed the Transformer architecture in 2017, the attention mechanism based on Softmax Attention has become the core component of sequence modeling, supporting the rapid development of Large Language Models (LLMs). However, Softmax Attention inherently suffers from **quadratic computational complexity**: as the sequence length $L$ increases, the computational and memory overhead of $O(L^2d)$ grows quadratically, which has become a core bottleneck for long-sequence modeling and efficient inference and training of LLMs.

To address this issue, Katharopoulos et al. proposed **Linear Attention** in 2020. By removing the softmax normalization term in Softmax Attention, it reduces the computational complexity of attention to a linear level and achieves constant memory usage in the inference phase. Nevertheless, the original Linear Attention has drawbacks such as low training efficiency and inferior performance compared to Softmax Attention. Subsequent researchers have continuously optimized it through chunkwise parallelism, hardware-efficient implementation, gating mechanisms, and memory update strategies, leading to the emergence of technologies like Flash Linear Attention, Gated Linear Attention (GLA), and Parallelized DeltaNet. These advancements have made Linear Attention a competitive alternative to Transformers that balances efficiency and performance.

This article starts from the pain points of Softmax Attention, and analyzes the core principles, training dilemmas, hardware optimization schemes, performance improvement methods of Linear Attention, as well as the improvement directions for associative recall tasks. Finally, it sorts out the generalization trends and future research directions of Linear Attention.

## 1 The Pain Point of Attention: Starting from Softmax Attention
The core of the Transformer is **Multi-Head Self-Attention (MHSA)**, which essentially models the correlations between elements in a sequence through the interaction of Queries (Q), Keys (K), and Values (V). For an input sequence $X \in \mathbb{R}^{L \times d}$ (where $L$ is the sequence length and $d$ is the hidden state dimension), the core computation process of Softmax Attention is:
$$
O = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} \odot M\right) V
$$
where $Q=XW_Q, K=XW_K, V=XW_V$ are matrices mapped by learnable parameters, $M$ is the causal mask (to avoid future information leakage), and $\odot$ denotes element-wise multiplication.

### 1.1 Complexity Analysis of Softmax Attention
The computational and memory overhead of Softmax Attention is mainly reflected in three stages, with an overall complexity of $O(L^2d + Ld^2)$:
1. **Q/K/V Projection**: Complexity of $O(Ld^2)$, linear in sequence length;
2. **Attention Matrix Computation**: $QK^\top$ generates an $L \times L$ attention matrix, and combined with softmax normalization, the complexity is $O(L^2d)$, quadratic in sequence length;
3. **Attention-Weighted V**: Multiplication of the attention matrix with $V$, with a complexity of $O(L^2d)$.

### 1.2 Mode Differences Between Training and Inference
Softmax Attention has significant differences between its **parallel training** and **recurrent inference** modes, which is also an important manifestation of its efficiency bottleneck:
- **Training Phase (Parallel Form)**: The attention of the entire sequence can be computed at once, which is dense in matrix multiplications and can fully utilize the tensor cores of GPUs. However, the quadratic computational load expands rapidly with the increase of sequence length;
- **Inference Phase (Recurrent Form)**: Token-by-token computation is required for generative tasks, which needs to maintain a **KV-Cache** to store historical K/V vectors. The memory overhead is $O(Ld)$, growing linearly with the sequence length, and the similarity between the current query and all historical keys still needs to be computed at each step, with a computational load of $O(Ld)$.

In short, Softmax Attention enables high-precision sequence modeling, but its quadratic complexity results in extremely low training and inference efficiency in long-sequence scenarios, laying the foundation for the proposal of Linear Attention.

## 2 Linear Attention: A Lightweight Alternative to Softmax Attention
The core idea of Linear Attention is to **remove the softmax normalization term in Softmax Attention**, simplifying the core computation of attention to the direct inner product of QK and the weighting of V, which fundamentally changes the computational paradigm of attention.

### 2.1 Core Formulas of Linear Attention
The computation formula of the original Linear Attention is:
$$
O = \left(QK^\top \odot M\right) V
$$
Compared with Softmax Attention, only the softmax normalization and the scaling factor $\frac{1}{\sqrt{d}}$ are removed, but this change fundamentally alters the computational characteristics of attention.

For the token-by-token computation scenario in generative inference (at time step $t$), the output of Linear Attention can be rewritten as:
$$
o_t = \sum_{j=1}^t (q_t^\top k_j) v_j
$$
Through the **associative law of matrix multiplication**, the above formula can be further transformed into:
$$
o_t = q_t^\top \underbrace{\left(\sum_{j=1}^t k_j v_j^\top\right)}_{S_t \in \mathbb{R}^{d \times d}}
$$
where $S_t$ is a **matrix-valued hidden state** that satisfies the recurrent update relation:
$$
S_t = S_{t-1} + k_t v_t^\top
$$
Finally, the inference output is simplified to:
$$
o_t = q_t^\top S_t
$$

### 2.2 Core Advantage of Linear Attention: Constant-Memory Inference
It can be seen from the above recurrent formula that the inference phase of Linear Attention only needs to maintain a fixed-dimension hidden state $S_t \in \mathbb{R}^{d \times d}$, instead of a KV-Cache that grows with the sequence length. The **memory overhead is reduced to a constant $O(d^2)$**, which is the core advantage of Linear Attention over Softmax Attention.

The comparison of training and inference complexity of Linear Attention is as follows:
- Training Phase (Parallel Form): Still $O(L^2d)$, consistent with Softmax Attention;
- Inference Phase (Recurrent Form): Computational load of $O(Ld^2)$ and memory of $O(d^2)$, both linear/constant in sequence length.

## 3 Training Dilemmas of Linear Attention and Chunkwise Parallel Solution
Although Linear Attention achieves a leap in efficiency in the inference phase, the **original Linear Attention has extremely low training efficiency** and cannot be directly deployed. Its core problem lies in the poor adaptability of the recurrent training characteristics to GPU hardware, and **Chunkwise Parallel** is the key solution to this problem.

### 3.1 Core Training Problems of Linear Attention
Directly using the recurrent form to train Linear Attention faces three major problems:
1. **Strict Sequential Computation**: Lack of sequence-level parallelism, unable to utilize the parallel computing power of GPUs;
2. **No Matrix Multiplication Operations**: All computations are element-wise addition/multiplication or reduction, which cannot leverage GPU tensor cores, leading to low computational efficiency;
3. **High I/O Cost**: Requires materialization of the hidden state at each time step. When the hidden dimension $d$ is large, the interaction cost of data between High Bandwidth Memory (HBM) and Static Random-Access Memory (SRAM) is extremely high.

Theoretically, parallel scan can reduce the training computational complexity to $O(Ld^2)$, but it is completely impractical in actual engineering.

### 3.2 Chunkwise Parallel Form
Proposed by Hua et al. (2022) and Sun et al. (2023), **Chunkwise Parallel** is the core engineering solution for Linear Attention training. Its core is to divide a sequence of length $L$ into several chunks of length $C$, perform **intra-chunk parallel computation** and **inter-chunk recurrent state passing**, achieving a trade-off between parallelism and recurrence.

The core steps of chunkwise parallelism are divided into three parts:
1. **Intra-chunk State Computation**: Compute the local state $K_{[i]}^\top V_{[i]}$ independently for each chunk, with full parallelism within the chunk;
2. **Inter-chunk State Passing**: Pass the global state $S_{[i]}$ of the previous chunk to the next chunk $S_{[i+1]}$, with only serial dependencies between chunks;
3. **Output Computation**: The output of each chunk consists of two parts: the **contribution from the global state of previous chunks** and the **contribution from the local attention of the current chunk**, i.e., $O_{[i+1]} = Q_{[i+1]} S_{[i]} + \text{LocalAttention}(Q_{[i+1]}, K_{[i+1]}, V_{[i+1]})$.

### 3.3 Selection of Chunk Parameter $C$
The chunk parameter $C$ determines the balance between training efficiency and performance of Linear Attention, with the following characteristics:
- $C=L$: Degrades to the **fully parallel form**, with high training efficiency but the computational complexity remains $O(L^2d)$;
- $C=1$: Degrades to the **fully recurrent form**, with a computational complexity of $O(Ld^2)$ but extremely low training efficiency;
- In practical engineering, $C$ is set to 64/128/256 (multiples of 16) to adapt to GPU tensor cores, and the number of recurrent steps is reduced from $L$ to $L/C$, which greatly improves training efficiency.

Chunkwise parallelism enables **linear scaling of training length** for Linear Attention, which is the key to its transition from a theoretical scheme to engineering implementation.

## 4 Hardware-Efficient Implementation of Linear Attention: Flash Linear Attention
The original PyTorch implementation of Linear Attention is even slower than the optimized Softmax Attention (e.g., FlashAttention-2), the core reason being **unoptimized I/O cost**. Inspired by FlashAttention, researchers proposed **Flash Linear Attention**, which greatly reduces the training I/O cost of Linear Attention and improves the running speed through **I/O-aware hardware optimization** and **kernel fusion**.

Flash Linear Attention has been open-sourced in the repository [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), a Triton-based library that supports multiple platforms (NVIDIA, AMD, Intel) and integrates state-of-the-art linear attention models, becoming the de facto standard implementation of Linear Attention.

### 4.1 Two Implementation Versions of Flash Linear Attention
Flash Linear Attention designs **Nonmaterialization Version** and **Materialization Version** for different training scenarios, adapting to short-sequence and long-sequence/large-scale training respectively:
#### 4.1.1 Nonmaterialization Version
- Core Idea: Keep the hidden state $S_t$ in GPU SRAM throughout the training process to avoid frequent data interaction between HBM and SRAM;
- Advantages: Minimized I/O cost, only need to load Q/K/V from HBM once, suitable for **short-sequence training** (where I/O cost dominates);
- Disadvantages: Lack of sequence-level parallelism between chunks, requires a large batch size to keep GPU Streaming Multiprocessors (SMs) busy, otherwise the SM utilization is low.

#### 4.1.2 Materialization Version
- Core Idea: Divide training into two stages: **sequential state computation** and **parallel output computation**. Fuse intra-chunk state computation and inter-chunk state passing into a single kernel through kernel fusion, and the inter-chunk output computation is fully parallel;
- Advantages: Supports chunkwise parallelism with high SM utilization, suitable for **long-sequence/large-scale training**;
- Disadvantages: Slightly higher I/O cost (K/V need to be loaded twice, $S_t$ needs to be stored and loaded once), and memory usage can be reduced through **recomputation in the backward pass**.

### 4.2 Effect of Hardware Optimization
Through I/O-aware design and kernel fusion, Flash Linear Attention achieves a qualitative leap in running speed. Compared with the pure PyTorch implementation of Linear Attention, its speed is greatly improved across different sequence lengths, and it can even match the optimized FlashAttention-2, becoming the standard engineering implementation of Linear Attention.

## 5 Performance Improvement: Gated Linear Attention (GLA)
Although Linear Attention solves the efficiency problem, its **performance is significantly inferior to Softmax Attention** (e.g., higher Perplexity (PPL) and lower LM Eval scores). The core reason is that the hidden state $S_t$ of the original Linear Attention is a simple cumulative update, lacking a **data-dependent memory forgetting mechanism**. To this end, Yang et al. proposed **Gated Linear Attention (GLA)** at ICML 2024. By introducing a **data-dependent multiplicative gate**, GLA makes the performance of Linear Attention close to or even surpass that of Softmax Attention and mainstream State-Space Models (SSMs) such as Mamba.

### 5.1 Core Gating Mechanism of GLA
The hidden state update of the original Linear Attention is a simple accumulation:
$$
S_t = S_{t-1} + k_t v_t^\top
$$
GLA introduces a **gating matrix $G_t$** into it to achieve data-dependent memory forgetting, with the update formula:
$$
S_t = G_t \odot S_{t-1} + k_t v_t^\top
$$
where the gating matrix $G_t$ is an **extension of the identity matrix with scalar gating**, i.e.:
$$
G_t = \alpha_t \cdot \mathbf{1}, \quad \alpha_t = \sigma\left(x_t W_{\alpha_1} W_{\alpha_2}\right)^{\frac{1}{\tau}}
$$
$\sigma$ is the Sigmoid activation function, $\tau$ is the temperature coefficient, and $\alpha_t \in (0,1)$ determines the retention/forgetting ratio of historical memory and is dynamically determined by the current input $x_t$.

### 5.2 Chunkwise Parallel Form of GLA
GLA retains the chunkwise parallel characteristics of Linear Attention, and designs a **decay-aware chunkwise parallel form** for the **cumulative decay** of the gate. By defining decay coefficients $\Lambda, \Gamma, \gamma$, it achieves efficient passing of inter-chunk gating states and ensures the consistency of decay during parallel training.

Its core is to normalize the intra-chunk gating decay through cumulative decay $b_t = \prod_{j=1}^t \alpha_j$, and finally realize the parallel computation formula of GLA:
$$
O = \left( (Q \odot B) \left( \frac{K}{B} \right)^\top \odot M \right) V
$$
where $B$ is the matrix form of cumulative decay coefficients. This formula not only ensures the effectiveness of gating but also can utilize GPU tensor cores for matrix multiplication, balancing performance and efficiency.

### 5.3 Performance of GLA
In experiments with 1.3B parameters and training on 100B tokens, GLA's performance comprehensively surpasses traditional Linear Attention (e.g., RetNet) and mainstream SSMs (e.g., Mamba), and is close to or even surpasses Softmax Attention (Transformer++). The core indicators are as follows:

| Model               | PPL  | LM Eval | Retrieval |
|--------------------|------|---------|-----------|
| Transformer++      | 16.9 | 50.9    | 41.8      |
| RetNet (Linear Attention) | 18.6 | 48.9    | 30.6      |
| Mamba (SSM)        | 17.1 | 50.0    | 27.6      |
| GLA (Gated Linear Attention) | 17.2 | 51.1    | 37.7      |

In addition, GLA performs far better than traditional Linear Attention and SSMs on **long-sequence generalization** and **recall-oriented tasks**, becoming the performance benchmark of Linear Attention.

### 5.4 The Relationship Between GLA and State-Space Models (SSMs)
One of the core contributions of GLA is to establish a **mathematical connection between Linear Attention and SSMs**, proving that **Gated Linear Attention is a scalable state-space model**, and mainstream SSMs (Mamba, Mamba-2, HGRN-2, RWKV-6, etc.) are all special subsets of Gated Linear Attention:
- The recurrent update of all SSMs can be expressed in the form of Gated Linear Attention: $S_t = G_t \odot S_{t-1} + k_t v_t^\top$;
- Scalable SSMs require the gate $G_t$ to be in the form of $\alpha_t \beta_t^\top$ to convert recurrence into matrix multiplication form, adapting to GPU tensor cores.

This connection provides a theoretical foundation for the fusion research of Linear Attention and SSMs.

## 6 Improvement for Associative Recall: DeltaNet and Parallelization
Although GLA greatly improves the performance of Linear Attention, **Linear Attention and SSMs still have significant defects in associative recall tasks**. Associative recall tasks require the model to retrieve the corresponding values based on query keys (e.g., input $A4B3C6 \to$ query $A?C?$ output $4,6$), which is a core task to test the model's long-sequence memory and association capabilities. However, the accuracy of Linear Attention/SSMs on this task is far lower than expected.

### 6.1 Core Idea of DeltaNet
The core of DeltaNet is to introduce a **memory retrieval-update mechanism** to replace the simple cumulative update of Linear Attention, achieving accurate memory and recall of key-value pairs. Its core steps are:
1. **Memory Retrieval**: Retrieve the old value from the historical hidden state according to the current key: $v_t^{old} = S_{t-1} k_t$;
2. **Memory Fusion**: Fuse the old value with the current value through a gate: $v_t^{new} = \beta_t v_t + (1-\beta_t) v_t^{old}$, where $\beta_t = \sigma(W_\beta x_t) \in (0,1)$;
3. **Memory Update**: Remove the old value from the hidden state and write the new value: $S_t = S_{t-1} - v_t^{old} k_t^\top + v_t^{new} k_t^\top$;
4. **Output Computation**: $o_t = S_t q_t$.

Through **explicit memory addition and deletion**, DeltaNet achieves accurate key-value pair association, and its accuracy on associative recall tasks is significantly higher than that of GLA, Mamba, RetNet and other models.

### 6.2 Parallelization Dilemma and Solution of DeltaNet
The core problem of the original DeltaNet is that the **pseudo-value vector $u_t = v_t^{new} - v_t^{old}$ depends on the historical hidden state $S_{t-1}$**, which cannot be directly trained in parallel. Researchers solved this problem through **reparameterization** and **WY representation**:
1. **Reparameterization**: Rewrite the update formula of DeltaNet into a matrix multiplication form: $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$, where $I$ is the identity matrix;
2. **WY Representation**: Utilize the product representation of Householder matrices to convert the recurrent matrix product into chunkwise parallel vector accumulation, realizing the chunkwise parallel computation of $S_t$.

### 6.3 Performance and Hybrid Optimization of Parallelized DeltaNet
The parallelized DeltaNet maintains the advantage of associative recall and achieves efficient GPU training. Its PPL on the 1.3B model is consistent with Softmax Attention, and its LM Eval index is better. To further improve recall performance, researchers proposed **Hybrid DeltaNet**, which combines sliding window attention/global attention with DeltaNet:
1. **Hybrid 1**: Use sliding window attention and DeltaNet alternately for each layer;
2. **Hybrid 2**: Introduce global attention in the 2nd layer and the middle layer, and use DeltaNet for the remaining layers.

Hybrid DeltaNet's performance surpasses pure Softmax Attention, becoming the optimal solution of Linear Attention on comprehensive tasks so far. The core indicators are as follows:

| Model               | PPL  | LM Eval | Retrieval |
|--------------------|------|---------|-----------|
| Transformer++      | 16.9 | 50.9    | 41.8      |
| GLA                | 17.2 | 51.1    | 37.7      |
| DeltaNet           | 16.9 | 51.6    | 34.7      |
| DeltaNet + Global Attention (2 layers) | 16.6 | 51.8    | 47.9      |

## 7 Generalization and Future Directions of Linear Attention
On the basis of GLA and DeltaNet, researchers further proposed the **Generalized Linear Transformer**, which extends the gate update of Linear Attention from **element-wise product** to **structured matrix multiplication**, and sorts out the core research directions of Linear Attention in the future.

### 7.1 Generalized Linear Transformer
The update of the original Linear Attention and GLA is **element-wise product** ($S_t = S_{t-1} \odot G_t + k_t v_t^\top$) with a complexity of $O(d^2)$, but it cannot model cross-channel interactions. Directly extending it to full matrix multiplication ($S_t = S_{t-1} G_t + k_t v_t^\top$) can model cross-channel interactions, but the complexity rises to $O(d^3)$, which is not deployable.

To this end, researchers proposed a generalized update formula with **structured matrix multiplication**:
$$
S_t = S_{t-1}(I - a_t b_t^\top) + v_t k_t^\top
$$
where $I - a_t b_t^\top$ is an **identity matrix plus a low-rank matrix**, which not only realizes cross-channel interactions but also controls the computational complexity at $O(kd^2)$ ($k$ is the low-rank dimension). DeltaNet is a special case of this formula ($a_t = b_t = \sqrt{\beta_t} k_t$).

### 7.2 Open and Future Work
The research on Linear Attention is still in a stage of rapid development, and the core open problems and future directions include:
1. **More General Associative Operators**: Explore more general matrix operators $\cdot$ in $S_t = S_{t-1} \cdot M_t + k_t v_t^\top$ that balance efficiency and expressive power;
2. **Further Optimization for Long-Sequence Modeling**: Combine sparse attention, sliding window attention with Linear Attention to achieve efficient modeling of ultra-long sequences (e.g., 100K+);
3. **Co-Design of Hardware and Algorithms**: Design customized Linear Attention implementations for different hardware (GPU/TPU/NPU) to further tap into hardware efficiency;
4. **Multimodal Linear Attention**: Extend Linear Attention to multimodal tasks such as computer vision and speech processing to solve the modeling bottleneck of multimodal long sequences;
5. **Unified Framework for Linear Attention and SSMs**: Based on the mathematical connection between Linear Attention and SSMs, build a unified modeling framework to integrate the advantages of both.

## 8 Conclusion
As a lightweight alternative to Softmax Attention, Linear Attention achieves **constant memory usage in the inference phase** by removing the softmax normalization, solving the core bottleneck of Transformer in long-sequence modeling. From the training dilemmas of the original Linear Attention, to the engineering breakthrough of chunkwise parallelism, and then to the hardware-efficient implementation of Flash Linear Attention, the performance improvement of GLA, and the associative recall optimization of DeltaNet, Linear Attention has evolved from a theoretical scheme to a sequence modeling technology that **balances efficiency and performance**, and has established a close mathematical connection with state-space models.

At present, Hybrid DeltaNet has surpassed traditional Softmax Attention on comprehensive tasks, and the Generalized Linear Transformer provides a theoretical framework for the further generalization of Linear Attention. With the advancement of research on hardware-algorithm co-design and multimodal extension, Linear Attention will become the core technology for long-sequence training and efficient inference of LLMs, supporting the deployment of larger-scale and longer-sequence LLMs.

The development of Linear Attention proves that **efficiency and performance are not mutually exclusive**. Through mathematical reconstruction, engineering optimization and mechanism innovation of the attention mechanism, it is possible to reduce computational and memory overhead while even surpassing the performance of the original architecture. This is also the core research direction for the lightweight and practical application of LLMs.

## Citation

> **Citation**: Please cite the original author and source when reprinting or quoting the content of this article.

**Cited as:**

> TensorPlay Team. (March 2026). Talk Linear Attention.
https://blog.tensorplay.cn/en/posts/talk-linear-attention

Or

```bibtex
@article{tensorplay2026-talk-linear-attention,
  title   = "Talk Linear Attention",
  author  = "TensorPlay Team",
  journal = "blog.tensorplay.cn",
  year    = "2026",
  month   = "March",
  url     = "https://blog.tensorplay.cn/en/posts/talk-linear-attention"
}
```