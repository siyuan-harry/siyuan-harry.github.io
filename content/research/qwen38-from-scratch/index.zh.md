
---
title: "从零手搓 Qwen3.8：一次架构演进的逐件拆解"
date: 2026-11-13T00:00:00+08:00
draft: false
summary: "从 Qwen3.0 到 Qwen3.8，Transformer 的几乎每个部件都被工程化地重写了。这篇记录自己手搓 Gated DeltaNet、Gated Attention、Partial RoPE、Zero-centered RMSNorm 和稀疏 MoE 的全过程，以及和 HuggingFace 实现的逐项数值对齐。"
showToc: true
tocOpen: false
tags: ["LLM", "Qwen", "架构", "MoE", "手搓"]
categories: ["研究"]
---

之前今年年初时，我们从零手写了一个 Qwen3 (阿里-千问3) dense 模型 (followed stanford cs336 assignment1)。参考该链接: 
https://github.com/siyuan-harry/llm-from-scratch. 

当时我的感受是，从 2019 年的 GPT-2 到 2024 llama-2 到 2025 Qwen3，Transformer 的架构变化比较有限，主要是三处零件替换:

- 激活函数和 FFN：GELU 的 MLP 换成了 SwiGLU；
- 位置编码：可学习的绝对位置编码换成了 RoPE；
- 归一化：Post-Norm 的 LayerNorm 换成了 Pre-Norm 的 RMSNorm。

所以 Qwen3 dense 是一个很「干净」的 LLM。我B站的视频也直说 LLM 其实非常简单。

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/20261113123231794.png)

而这期的 Qwen3.8，**画风完全不一样**。如果说 Qwen3 是换零件，那 Qwen3.8 就是把整台机器拆开、几乎每个部件都做了工程化的「魔改」：注意力换成了 Linear Attention 和 Gated Attention 的混合体，FFN 变成了几百个专家的稀疏 MoE，RoPE 只转 1/4 的维度，RMSNorm 的参数改成以 0 为中心，还额外挂了一个MTP 模块来预测下下个 token 等等……

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/%E4%BB%8E%E9%9B%B6%E6%89%8B%E6%90%93%20Qwen3.8%20%E6%9E%B6%E6%9E%84%E6%BC%94%E8%BF%9B.png)

这些改动每一个都只带来一点边际收益：省下一点推理显存、稳定训练、让 decode 变快一点。但叠在一起，就是今天最新开源模型的工程架构，水位已经涨到了这里。

今天这期手搓 Qwen3.8，我们就逐个理解这些新模块：**它和 Qwen3.0 有什么不同、它是怎么演进来的、为什么合理、还有哪些提升空间、代码 (甚至GPU Kernel) 怎么写**，以及我踩的坑。

规矩和此前一样：不用 `nn.Linear`、`nn.LayerNorm`、`nn.Conv1d`、`nn.Transformer` 这些高层模块，全部尽可能从张量运算手写。

**那为什么要这样从零手搓呢？** 因为我发现，想要理解一个模型架构的时候，只看论文的文本、数学公式，或者架构图，受制于篇幅，各有各的省略之处。但只有用代码搓出来，并且真的能跑通训练，这个架构才算是真的完整呈现在我的眼前。能通过读代码非常直观地看到变量是怎么传递的，每个 tensor 的形状在每一个计算中是怎么变化的、它承载的信息是怎么转换的，这样才能明白每一个计算的细节，从而理解整个架构真正的含义。而跑通真实训练，能感受到不同组件对计算效率的影响，这就可以自然地玩一些 Infra 优化。

本期手搓 qwen3.8 的代码已开源: https://github.com/siyuan-harry/qwen38-from-scratch ，欢迎大家点点 stars，也希望能帮助到各位同行的研究工作。我们接下来开始做简单的架构拆解。

## 1. 全景：Qwen3.0 dense vs Qwen3.8（约 1.5 分钟）

| 部件 | 上期 Qwen3.0（我们的实现） | Qwen3.8 |
|---|---|---|
| Token mixer | 每层都是 softmax 全注意力 | 3 层 Gated DeltaNet 线性注意力 + 1 层 Gated Attention 全注意力循环 |
| 注意力头 | MHA，head_dim = d_model / n_head | **GQA**，head_dim 与 d_model 解耦 |
| 注意力输出 | 直接进 o_proj | sigmoid gate --> o_proj |
| Q/K | 无归一化 | QK-Norm |
| 位置编码 | RoPE，全维度，相邻两维配对，θ=1e4 | Partial RoPE，只转 25% 维度，rotate_half 配对，θ=1e7 |
| Norm | RMSNorm，gain 初始化为 1 | Zero-centered RMSNorm，w 初始化为 0，并施加 weight decay |
| FFN | dense SwiGLU | dense SwiGLU 或 稀疏 MoE + 共享专家 |
| 训练目标 | next-token | next-token + MTP（多 token 预测） + MoE 负载均衡损失 |
| 初始化 | Xavier 风格 | normal(0, 0.02)，embedding 与 lm_head 不共享 |

这张表基本说明了 qwen3 到 qwen3.8 在架构上的演进。也在 Github 仓库里了。在这个 overview 部分，我们也可以额外提两嘴：

Qwen3.8 开源中有两种区别：一个是推理时全量参数激活的 dense 模型，另一个是 MoE 模型。两者的注意力部分完全一样，区别只在 FFN。我们的实现两种都支持，可以通过一个命令行参数 `--num_experts` 切换。

另外，Qwen3.8 的 text decoder 架构是从 Qwen3-Next（2025 年 9 月）到 Qwen3.5、3.6 一路沿用下来的，所以官方 HuggingFace 上它的 `model_type` 还写着 `qwen3_5`。这也说明了 qwen3.8 架构比较经典，历经多代能够有效地 scale up。当然最新的 Qwen3.8-Flash-Next 是全新架构，我们先保持观望状态。

## 2. 混合注意力：Gated DeltaNet

### 2.1 为什么要动注意力？

> 下面是 GPT 老师给画的图，审美不太行，但信息呈现还算ok. 凑合看吧。。

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/02_kv_cache_vs_state.png)

Softmax Attention 有两个老问题：第一个，训练时计算量随序列长度平方增长；第二个，推理时每生成一个 token 都要把之前所有的 K、V 存下来，也就是 KV cache，显存随长度线性增长。Qwen3.8 支持 262K 的上下文，如果每一层都是标准注意力，KV cache 会非常吓人。

Qwen3.8 的做法是：**每 4 层里只有 1 层保留标准注意力，另外 3 层换成线性注意力 Gated DeltaNet**。线性注意力层只维护一个固定大小的状态矩阵 S，不管序列多长，推理显存都是 O(1)。

### 2.2 演进史：从线性注意力到 Gated DeltaNet

那么线性注意力是怎么得出的呢？这条线可以用一个“状态矩阵 S”的更新公式串起来：

1. **线性注意力 (2020, Katharopoulos et al.,*Transformers are RNNs*)** 把标准 Attention 里面的 softmax 去掉，把 QK 相乘拆开，让KV先做外积。这是个非常有洞察力、非常有趣的数学变换，第一次理解这件事让我惊为天人，因为它通过 Anderson 1972 / Kohonen 1972 的 Linear Associator 的思想，把 Transformers 和它10年前的前辈 “RNN” 联系在了一起。并且发现它们居然能互补。总之，做了这个变换后，我们发现 Attention 在数学上就直接等于 RNN, 每处理1个新的 token，都只是对一个固定大小的状态 S 做信息写入：

$$\textbf S_t = \textbf S_{t-1}+\boldsymbol k_t\boldsymbol v_t^\top, \quad \boldsymbol o_t = \textbf S_t^\top \boldsymbol q_t$$

但这么做的问题是「只加不减」，状态会被越写越乱 (interference)，记忆力很差。

2. **加遗忘门 (RetNet, Mamba2, GLA, etc.，2023~2024)** 

$$\textbf S_t = \alpha_t(\textbf S_{t-1})+\boldsymbol k_t\boldsymbol v_t^\top$$ 

每一步按比例衰减旧记忆，相当于学会了“遗忘”。但衰减因素 alpha 是一刀切的，没法精确修改某一条记忆。

3. **Delta rule (2021 年 Schlag et al. 的 *Fast Weight Programmers* 提出 DeltaNet; 2024 年 Yang et al. 解决了并行训练)** 

$$\textbf S_t = \textbf S_{t-1}+\boldsymbol \beta_t \boldsymbol k_t (\boldsymbol v_t - \mathbf S_{t-1}^\top \boldsymbol k_t)^\top$$ 

先用 k 去状态 S 里读出旧值，再按误差去纠正。这就是经典的 delta rule，相当于对状态矩阵 S 做一步在线梯度下降，能精确覆盖某个 key 对应的记忆。

4. **Gated DeltaNet (2024 年底提出. ICLR 2025. Yang, Kautz, Hatamizadeh)** 把 2 和 3 合起来，也就是给 DeltaNet 里的 $\mathbf{S}_{t-1}$ 外面也乘一个衰减因子 alpha t: 

$$\mathbf{S}_t=\boldsymbol \alpha_t(\mathbf{S}_{t-1})+\boldsymbol \beta_t(\boldsymbol v_t-\alpha_t \mathbf{S}_{t-1} \boldsymbol k_t) \boldsymbol k_t^\top,\quad \boldsymbol o_t=\mathbf{S}_t \boldsymbol q_t.$$

这样，既能用 alpha 来对此前的状态做整体遗忘，又能通过 beta 精确改写其中某个 channel 的记忆。论文标题就叫 *Improving Mamba2 with Delta Rule*。

5. **混合架构 (MiniMax-01 的 7:1、Qwen3-Next / Qwen3.5 / Qwen3.8 的 3:1, Kimi Linear / K3 也是 3:1).** 纯线性注意力在“大海捞针”这类精确检索任务上还是不如 softmax 注意力，所以保留少量全注意力层兜底。这样，大部分层其实只需要局部 (以及压缩后的全局) 信息，只有少数层需要精确回看整个历史。3:1 是 Qwen 团队实验出来的效率和效果的折中。

不过它目前也不是最优解。后面大家还是有别的探索，例如：

- **门控粒度**：Gated DeltaNet 的衰减 α 是每个头一个标量。Kimi Linear (2025 年 10 月) 提出的 KDA 把它再化到每个通道一个衰减，表达能力更强。也算是一个工程魔改了。
- **混合比例和位置**：3:1 不一定是最优，哪些层放全注意力也值得搜索。也有团队 (比如 MiniMax 的 M2) 在实际大规模部署后又回到了全注意力，说明线性注意力在复杂推理和检索上的短板还没有完全解决。
- **硬件友好的 kernel**：线性注意力的理论优势要靠高度优化的 kernel (比如 FLA 库的 Triton 实现) 才能兑现。以及我算是也搓了一个适配 Apple GPU 的 kernel. 见下文。

### 2.3 代码实现思路

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/04_gdn_forward.png)

一个 Gated DeltaNet 层的前向分五步。这块老生常谈，先做 qkv = in_proj_qkv(x) -> 因果 depthwise conv1d (kernel=4) -> SiLU -> 拆分为 q, k, v (qk形状一致，v比qk的头数多). 然后计算用于输出GatedRMSNorm 的 z, 以及组合新旧记忆的乘数 beta ($\beta^{(i)}=\sigma(\boldsymbol W_\beta \boldsymbol x^{(i)})\in(0,1)$), 和衰减参数 g 也就是经典 GDN 公式里的 $\alpha_t$. 最后给 qk head 复制到和 value head 一样多 (qk head equal num in GDN), 目的是让同一份 qk 寻址方式能检索到多份 value 的信息.  然后进入 Delta Rule，出来后做 norm (用上刚才算的 z) 和输出投影。

类似于 softplus 数值稳定怎么写，衰减 g 用 Mamba 的写法，这些细节此处不展开了。看代码即可。

**关键: Delta Rule 的实现**

最关键的是 delta rule 本身。逐 token 的递推写法很简单，我也在代码里写了一个 `recurrent_gated_delta_rule` 函数，实现的就是下面这个递推式：

$$\mathbf{S}_t=\alpha_t(\mathbf{S}_{t-1})+\beta_t(v_t-\alpha_t \mathbf{S}_{t-1}k_t)k_t^\top,\quad o_t=\mathbf{S}_tq_t.$$

但它是一个长度为 T 的串行循环，训练太慢几乎不可用。所以这个想法从2021年被提出后就被搁置，直到 Yang et al., 2024 年在 "Parallelizing Linear Transformers" 这篇文章发现，如果对这个 S_t 递推公式做一个数学上的等价变换，能把 DeltaNet 的训练并行化，从而让 DeltaNet 完成从不可用到可用的跨越:

$$\mathbf S_{t} = \alpha_t (\mathbf I-\beta_t\boldsymbol k_t \boldsymbol k_t^\top) \mathbf S_{t-1} + \beta_t\boldsymbol k_t \boldsymbol v_t^\top, \quad \alpha \in [0,1]$$

注意到，新递推公式中 rank-1 的更新结构 $(\mathbf I-\beta_t\boldsymbol k_t \boldsymbol k_t^\top)$ 等价于广义的 Householder transformation. 这个结构非常关键，它有很好的代数性质，使得一串这样的矩阵连乘可以用 WY representation 压缩表示，支持硬件高效的 chunk-wise 分块并行化 (每次都感慨这个想法太聪明了)。所以此后对于 GDN 的计算，通常都是把序列切成 chunks (分块，比如每块 64 个 token)，每个 chunk 内用 WY / UT 变换的技巧并行计算，而前后两个 chunk 之间只传递一个状态矩阵。和串行比较，示意图如下：

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/05_chunk_parallel.png)

上面的是串行，算完上一个 token 才能算下一个。下面是切分 chunk 并行的实现，如果 chunk size 为 64 的话，计算路径缩短 64 倍。而且每一步能充分利用电脑 GPU 的稠密矩阵计算能力。(举个实际例子，比如用户发进来一条128个 token 的问题，那么原来要计算 128 遍矩阵向量乘法，现在只需要两遍矩阵乘加一遍结果传递).

### 2.4 踩了大坑，努力爬坑

为什么要细讲 GDN 的计算？因为我在自己电脑的硬件上 (Apple M4 Pro) 实现这一步 chunk 内并行化计算时遇到了不小的麻烦。虽然到后来证明是个乌龙事件，但还是在解决问题的过程中学到了很多。

具体来讲，这一步需要解一个单位下三角线性方程组 $(\mathbf I+\mathbf A) \mathbf X = \mathbf R$，可以用求逆的方法得到 $\mathbf T = (\mathbf I+\mathbf A)^{-1}$，再解出 $\mathbf X = \mathbf T @ \mathbf R$。问题就在求逆这一步上。官方代码用的是 `torch.linalg.solve_triangular` ，但我实测它在 Mac 的 MPS 上非常慢，一次前向要 130 多毫秒。调查后发现，torch 2.9 的这个 function 写得非常糟糕。在默认情况下，它会为 batch 里的每个矩阵单独提交一次 GPU kernel。而这个算子的 batch 的大小等于训练 batch (默认B=32) 乘以 value head number (H=8) 乘以 每条的 chunk 数 (默认 N=4), 乘起来有 1024 个小三角矩阵要求逆，做 profilling 发现每次提交一个 kernel 在我电脑上大概花 0.066ms, 但完成这个 kernel 的计算可能只花几微秒.. 这非常不经济，相当于时间几乎全花在调度开销上，根本没发挥出 GPU 的并行计算优势。为了解决这个问题，我只好自己写了一个**只用矩阵乘法的分块求逆**，把同一轮里 1024 个矩阵的合并操作合成一次 batched matmul，大概能提速十几倍，并且结果精确。实测与官方对比效果如下：

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/benchmark_inverse.png)

可以发现，随着 batch 从 1 增长到 1024，PyTorch 官方的这个实现耗时翻了680倍，基本是线性的时间复杂度增长。而我们的这个算子基本是对数时间复杂度。也就是随着 batch 增大，它的优势越来越大。

**实现思路**。我的思路很直觉化。就是利用分块求逆恒等式 (下式), 把同一轮里 1024 个矩阵操作合成一次 batched matmul，尽可能发挥 GPU 的大矩阵并行计算优势。

$$\begin{bmatrix}\mathbf L_{11}&\mathbf 0\\ \mathbf L_{21}&\mathbf L_{22}\end{bmatrix}^{-1}=\begin{bmatrix}\mathbf X_{11}&\mathbf 0\\ -\mathbf X_{22}\mathbf L_{21}\mathbf X_{11}&\mathbf X_{22}\end{bmatrix}.$$

这个公式告诉我们，要求一个单位下三角矩阵 (设其大小为 $2m \times 2m$) 的逆，需要三样东西：

1. $\mathbf X_{11}=\mathbf L_{11}^{-1}$
2. $\mathbf X_{22}=\mathbf L_{22}^{-1}$
3. $\mathbf L_{21}$，直接从原矩阵里截取即可，不需要计算。

其中，第 1 和第 2 样是 $m\times m$ 对角块的逆。而单位下三角矩阵的对角块仍是单位下三角，所以 $\mathbf L_{11}$ 和 $\mathbf L_{22}$ 本身也是单位下三角矩阵，于是同一个公式可以接着套在它们身上。于是这个恒等式可以自底向上递归。

- 从 64 个 $1\times1$ 的逆（都是 1）出发，每轮把相邻两个 $b\times b$ 的逆合并成一个 $2b\times 2b$ 的逆，$1\to2\to4\to\cdots\to64$ 共 $\log_2 64=6$ 轮 (这就是对数时间复杂度的来源)。每一轮内部的所有 chunk pair、以及 1024 个 C x C 大小的矩阵彼此都无关，因此每轮就可以直接视为一次 batched 矩阵乘 (batch 从 32768 逐轮降到 1024)。
- 这是精确的代数计算，**没有近似操作**，所以 fp32 下最大误差 $3\times10^{-6}$。上面自己机器的截图里也可以看到其精确性。

这就是 `invert_unit_lower_triangular()` 函数，感兴趣的同学可以去源码里看细节。

**哪里乌龙了？**

满心欢喜地以为自己搞定了 pytorch 的一个算子优化，准备开开心心提 PR 的时候发现最新的 torch 2.15 已经把这个算子修了，和我思路不一样，但效果似乎更好一些。不过问题不大，还是学到了很多有趣的知识。在此分享出来。

## 3. Gated Attention

除了线性注意力外，剩下 1/4 的全注意力层也被魔改了。核心改动很简单：注意力的输出先乘一个 sigmoid gate，再进 output projection.

为什么要这么做？因为 2023 年 StreamingLLM 那篇论文发现，LLM 普遍会把大量注意力分数倾倒倒在第一个 token 上，叫做 attention sink (如下图)

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/07_gated_attention.png)

2024 年也有工作发现模型内部会出现个别数值巨大的激活 (massive activation)。原因之一是 softmax 必须归一化到 1，当一个头“不想看任何东西”时，只能找个垃圾桶 token 倒掉。

Qwen 团队 2025 年的论文 *Gated Attention for Large Language Models* 系统比较了在注意力里加 Gate 的各种位置，发现在 SDPA 输出之后、逐元素地加一个依赖于输入的 sigmoid gate 效果最好。这个门引入了非线性和稀疏性，让一个头可以直接把输出关掉，于是 attention sink 和 massive activation 基本消失，训练更稳定，长上下文外推也更好。这篇论文后来拿了 NeurIPS 2025 的 best paper。

不过这也有提升空间。 Gate 是逐元素的，参数和计算多一倍的 q_proj 输出；而逐头 (headwise) 的 gate 更便宜，但效果略差，这是一个可以继续权衡的点。

这块代码比较简单，代码片段：

```python
# q_proj 输出维度翻倍：每个头是 [query(head_dim), gate(head_dim)]
q, gate = self.q_proj(x).view(B, T, self.n_head, 2 * self.head_dim).split(self.head_dim, dim=-1)

q = self.q_norm(q).transpose(1, 2)                   # QK-Norm
k = self.k_norm(self.k_proj(x).view(...)).transpose(1, 2)
q, k = PartialRoPE.apply(q, cos, sin), PartialRoPE.apply(k, cos, sin)
k, v = repeat_kv(k, self.n_rep, dim=1), repeat_kv(v, self.n_rep, dim=1)   # GQA

attn = scaled_dot_product_attention(q, k, v, mask)   # 还是上期手写的那个
attn = attn * torch.sigmoid(gate)                    # 新增的 output gate
return self.o_proj(attn)
```

## 4. Partial RoPE

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/08_partial_rope.png)

上期我们讲 RoPE，是把 q、k 的每一对维度都按位置旋转。Qwen3.8 只旋转 head_dim 的前 25%，剩下 75% 不带任何位置信息。

partial rotary 其实不是新东西，2021 年 GPT-J、2022 年 GPT-NeoX-20B 就是只旋转 25% 的维度。DeepSeek 的 MLA 也把带 RoPE 的部分和不带 RoPE 的部分拆开。思路是一部分维度专门负责位置，其余维度专心负责语义匹配，不受旋转干扰。这对长上下文外推可能也更友好，因为没有被旋转的维度不会遇到没见过的角度。

这块的代码如下，也比较简单。

```python
rotary_dim = int(head_dim * 0.25)
x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
x_rot = x_rot * cos + rotate_half(x_rot) * sin
return torch.cat([x_rot, x_pass], dim=-1)
```

细心的读者可能发现，官方配置里写的是 mRoPE (多模态 RoPE)，会把旋转维度分成「时间」、「高」、「宽」三段，给图像和视频用。但对纯文本输入，这三段的位置 id 完全相同，数学上就退化成普通 1D RoPE，所以这里我只实现了 1D 版本。

## 5. Zero-centered RMSNorm + weight decay

这是一个看起来最不起眼，但很有「工程魔改」味道的改动。没什么理论，就是这么做会让训练更顺。

背后的故事很有意思。Qwen 团队在训 Qwen3-Next 时发现，用了 QK-Norm 之后，部分 norm 层的权重会异常增大，影响训练稳定性。

**直觉上的解法**是给 norm 的权重也加 weight decay。但普通 RMSNorm 的 gain 初始化是 1，weight decay 会把它往 **0** 拉，这显然是个错误的先验：0 意味着这个通道被关掉，而 1 才是中性的尺度。这就相当于对偏离的惩罚不对称了 (奖励往0的偏离)。所以实践里通常不对 norm 的 gain 做 decay，结果就是没有任何东西约束它的增长。

那怎么办呢？有没有办法既给 gain 做 weight decay, 又保持计算上的正确呢？

解决方案出奇地简单。就直接把 w 初始化为 0, 然后在计算的时候给它加 1 不就好了吗？`out = x/rms(x) · (1 + w)`，你看，现在 weight decay 把 w 拉向 0，等价于把 gain 拉回 **1**，正好是我们想要的“别长太大”。这个 (1 + w) 的写法 Gemma 早已实现，Qwen3-Next 第一次把它和 weight decay 组合起来。

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/09_zero_centered_rmsnorm.png)

代码很简单。真正的改动在训练脚本的参数分组里：

```python
# model.py
x = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
x = x * (1.0 + self.weight)      # weight 初始化为 0

# train_model.py：哪些参数不做 weight decay
NO_WEIGHT_DECAY_SUFFIXES = (
    "A_log",              # Gated DeltaNet 的衰减率
    "dt_bias",            # Gated DeltaNet 的时间步 bias
    "gated_norm.weight",  # GDN 里面的 GatedRMSNorm - 权重以 1 为中心
)
```

注意 Gated DeltaNet 输出端的那个门控 RMSNorm，官方并没有做成 zero-centered，所以它要排除在 weight decay 之外。这种细节只有读官方代码才看得出来。

## 6. 略讲 Sparse MoE 与 MTP

接下来，这几代 Qwen 架构里比较有趣的部分还剩 Sparse MoE，以及 MTP。我们就略讲一下这两块。

首先，Qwen3.6-35B-A3B 的 FFN 是一个稀疏 MoE：256 个小专家，每个 token 生成时只选分数最高的 8 个，再加 1 个所有 token 都会经过的共享专家。总参数 35B，但每个 token 只激活 3B，这样的话，推理成本基本只有 3B 模型的大小，省了10倍算力。

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/10_moe_routing.png)

> 上图我觉得 GPT 做的一般，没有把 “256个专家里挑8个” 这个概念直观地展示出来。

这里我实现的是官方 transformers 里的标准 aux loss（系数 0.001）。Qwen 团队 2025 年有一篇论文专门讨论，aux loss 应该在**全局 batch** 上统计，而不是在每个 micro-batch 上统计，否则会逼着专家在局部数据上也均匀，反而损害专精。 然后真正的 MoE 训练离不开专家并行和 grouped GEMM 这类专用 kernel，我这里的逐专家循环只是“能跑”，实际上超慢。这个优化不动了，不过也是因为 MoE 已经有很不错的 kernel 了。

Qwen3.8 训练时还挂了一个 MTP模块，也就是同时预测下一个 token 和 下下个 token。这样能在训练时提升样本效率，每个样本的信息量被榨得更干。

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/11_mtp.png)

## 7. 怎么确认写对了？

这是我让 AI 帮忙在机器上跑了个验证，对比 Huggingface 实现的效果差别：

```text
Qwen reproduction | real numerical verification
PyTorch 2.9.0 | Transformers 4.57.1 | CPU float32 | seed 20261112
Tiny random weights | 4 layers (3 DeltaNet + 1 attention)

[PASS] Block inverse vs triangular solve     max_abs=1.431e-06
[PASS] Block inverse gradients (lower part)  max_abs=1.192e-06
[PASS] Delta outputs / T=1,63,64,65,129      max_abs=1.788e-07
[PASS] Delta gradients / q,k,v,g,beta        max_abs=2.049e-07
[PASS] Dense logits vs HF Qwen3Next          max_abs=4.061e-07
[PASS] MoE logits vs HF Qwen3Next            max_abs=5.439e-07
[PASS] Dense + MTP + AdamW resume            max_abs=0.000e+00
[PASS] MoE + MTP + AdamW resume              max_abs=0.000e+00

8/8 checks passed | 1.36s
```

主要检查这三个部分：

1. **检查计算本身**。把自己写的分块求逆 kernel 和 torch 原版三角求解比较，再把分块 delta rule 和逐 token 参考实现比较。这次输出和梯度都对齐在很小的浮点误差内。
2. **检查推理**。用相同的小模型配置，把我的随机初始化权重转换到 Hugging Face 的 Qwen3-Next. 两个相同参数的模型收到同一份输入，比较所有位置的 logits。这次最大绝对误差分别约为 4.06e-7 和 5.44e-7。这说明被测配置下的前向计算与官方实现对齐，是没问题的。
3. **检查训练**。Dense 和 MoE 都带上 MTP，做真实的反向传播和 AdamW 更新，再恢复 model & optimizer state ，比较下一步更新后的参数。本次续训和连续训练的参数完全一致。

除了这次检查外，还在自己电脑上跑了 5000 步的训练， loss 曲线看上去还不错，仓库里可以看到我的 Mac MPS 训练日志。

## 8. 没有实现的部分

最后诚实地列出没做的东西：

| 没做的 | 原因 |
|---|---|
| **视觉编码器 / 多模态 mRoPE** | Qwen3.8 是原生多模态模型，但我们只在 TinyStories 纯文本上预训练；纯文本下 mRoPE 等价于 1D RoPE。 |
| **推理缓存**（全注意力的 KV cache；Gated DeltaNet 的卷积状态和递推状态） | `generate` 和上期一样，每生成一个 token 都重算整段序列。加上不难：递推版本的 delta rule 已经写好了，缺的是缓存管理。 |
| **MTP 投机解码** | MTP 目前只用来提供训练信号； |
| **融合 kernel** (FLA 的 Triton delta rule kernel、causal-conv1d 的 CUDA kernel) | Mac 的 MPS 也跑不了 Triton/CUDA。作为折中，我用矩阵乘法分块求逆替代了三角求解。 |
| **高效 MoE** (专家并行、grouped GEMM、全局 batch load balancing) | 这些属于分布式训练基础设施。 | |
| **混合精度、262K 长上下文、后训练（SFT/RL/思考模式）** | 超出“复现架构 + 小规模预训练”的范围。 |

## End

回头看大语言模型架构的变迁：2019年，GPT-2 定义了一个 decoder-only llm 的骨架；2025年, Qwen3 替换了几个关键零件，变成一个干净、通用的现代 LLM；而 2026 年最新的 Qwen3.8 是在这个骨架上，把推理成本、训练稳定性、长上下文这几个目标拆成几十个小问题，一个个用工程手段去推动到极致。没有哪一个是革命性的，但合在一起，就是今天开源模型的真实样貌。

代码全部开源在仓库里： https://github.com/siyuan-harry/qwen38-from-scratch 。训练日志之类的也在里面。欢迎点赞点 stars.

## References

- Qwen3.8 官方配置：`Qwen/Qwen3.8-27B`、`Qwen/Qwen3.8-35B-A3B`（HuggingFace）
- 官方实现：transformers `models/qwen3_5/`、`models/qwen3_5_moe/`；vLLM `qwen3_5_mtp.py`
- Katharopoulos et al., 2020, *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*
- Schlag, Irie, Schmidhuber, 2021, *Linear Transformers Are Secretly Fast Weight Programmers*
- Yang et al., 2024, *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*
- Yang, Kautz, Hatamizadeh, 2025, *Gated Delta Networks: Improving Mamba2 with Delta Rule*
- Qiu et al. (Qwen), 2025, *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*
- Xiao et al., 2023, *Efficient Streaming Language Models with Attention Sinks*
- DeepSeek-AI, 2024, *DeepSeekMoE*；DeepSeek-AI, 2024, *DeepSeek-V3 Technical Report*
- Qiu et al. (Qwen), 2025, *Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models*
- Gloeckle et al., 2024, *Better & Faster Large Language Models via Multi-token Prediction*
- Qwen Team, 2025, *Qwen3 Technical Report*；Qwen3-Next 官方博客
