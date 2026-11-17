---
title: "智能增长没有永动机：为什么 AI 的未来依然取决于人类建设的基础设施？"
date: 2026-11-17T12:00:00+08:00
draft: false
summary: "从合成数据、递归自我改进到世界模型，讨论信息、验证器与因果性对 AI 能力增长的约束。"
showToc: true
tocOpen: false
tags: ["AI", "合成数据", "RSI", "世界模型", "信息论"]
categories: ["研究"]
---

> **AI; DR**: 本文正文内容每一个字均是自己手敲，并经历多遍删改。AI 帮我查证了一些参考文献，这些文献或加固或更改了我的一些观点。文末可见参考文献列表。除了该文献列表是 AI 整理以外，本文其它部分不含 AI 生成内容.

下面为本文简介。

> **TL; DR**: 从大规模合成数据的多样性瓶颈，到递归自我改进 (RSI) 的智能爆炸，再到世界模型能否真正理解物理规律，看似不同的前沿问题，背后却共享着同一个理论约束。
> 
> 这也帮助我们洞察当下 AI 发展的特点：数学、编程领域进步神速，未必是因为模型获得了普适的超级智能，而是这些领域拥有相对廉价、可靠的验证器。未来 AI 能力的拓展，很大程度上取决于人类能否为更多真实场景搭建这样的基础设施。
> 
> 本文尝试从经典的信息论与算法理论出发，重新审视当下最热门的 AI 技术路线。**如果你正在从事 AI 算法或应用开发，希望这篇文章能提供判断力：哪些瓶颈值得投入资源攻克，哪些路线从一开始就不值得去探索。**

---

<center>*</center>

1775 年，巴黎科学院宣布，此后不再受理永动机方案。那时热力学定律还没有建立。评审们只是见过太多失败，"此路不通"本身已经成了一条知识。半个世纪后，卡诺 Carnot (1824) 证明热机效率存在**上界**，这条知识才成为理论。

「理论」来源于经验，而又超越经验。一经提出，就令后人醒悟，不再于错误的道路上白费力气。本文并非空泛讨论哲学，而是要讲理论思考如何帮助自己应对日常工作中的实际问题、乃至初步形成自己关于技术路线的审美与判断力。

回国后这一年，我在上海模速空间一家 AI 安全公司里负责数据工程、评测、模型算法、Agentic 框架开发相关工作。最近突然意识到手头有越来越多工作「似曾相识」：一个强化学习的新项目，也许和自己一年前做的红队数据生成项目的瓶颈源自同一套理论。而这些理论原型往往已有几十年的历史。

诚然，即使不关心这些理论原型，一个算法团队也完全可以把手头工作做到平均水平、能交付并满足客户验收标准。但如果能从理论视角关注算法工作的性能上界，也许就有机会帮团队规避掉白费力气的路线，从而做到出类拔萃。

毕竟人生苦短，若只是完成琐碎工作，多少缺乏了些趣味。

本文结合工作中遇到的几个具体问题，浅谈一些关于其理论原型的思考。并延展一点对眼下 AI 领域学界前沿问题中技术路线走向的个人判断。

这些想法目前总体还是直觉性的，尚不严谨。供有兴趣的同行批评讨论。

目录:

```
1 引子：五十万条数据，一万条新意
2 智能爆炸的逻辑缺口
3 智能增长从何而来？
4 数学天才的短板
5 为智能搭建脚手架
6 世界模型的一厢情愿: 生成不等于理解
7 关键在于因果性 (Causality)
	7.1 正道
8 结语: 不必重新发明永动机
```

## 1 引子：五十万条数据，一万条新意

作为 AI 安全团队，我们有时需要生成有害的图片和文本，训练安全护栏去识别涉政、涉恐、技术犯罪、企业信息泄密等风险 (类别参照国家标准 GB/T 45654-2025)。容易发现，当每个风险类别只需几十上百条数据时，用单一 abliterated LLM 配合固定的 prompt 模板就能得到质量不错的语料。

但规模一旦扩大，事情就变得诡异起来。假设我们需要几十万条训练数据，却仍然使用单一模型和单一 Prompt 模板，可能生成了 50 万条语料，但经过语义余弦相似度去重后，最终只剩下 1 万条左右真正有差异 (有信息量) 的样本。数量上我们拥有 50 万条数据，实际上却只有几十分之一。这种数据训练出的护栏模型会有严重的过拟合与分布偏狭问题 (模型看似见过大量不同的表达，实际上反复学习的是少数几种语义模式)。

正是： **数据的繁荣，信息的贫瘠。**

对此问题，容易想到的解决方法一般是多搞几个 prompt 模板、分层多步生成，以增加输入多样性的方式撬动模型输出语料的多样化，更全面地 exploit 单一模型的生成空间。在这个基础上再叠加「多模型生成」，是不是就可以把“1万条有效样本”变成“几十万条”了呢？可惜现实总不如理论美好。Artificial Hivemind (Jiang et al., 2025) 系统记录了多个模型在输出模式上的同质化，多模型的乘数因此大打折扣。

难点不止数量。模型生成的内容与真实数据之间还存在更致命的系统性偏差。对此，人们想到可以用真实样本做 few-shot 指引，但这依然取决于它带来的增益能否覆盖采集与标注成本。类似的困境在工业流程控制里也有：历史数据大多来自稳态工况，分布偏狭，而仿真合成的数据又有 sim-to-real 的鸿沟。因此好的数据永远昂贵。

**存在理论上界。** 从根本上而言，训练一个模型就是对世界信息的一次有损压缩 (Delétang et al., 2024)。模型是一个「收敛了的世界」。可想而知，沿着真实世界 --> 训练数据 --> 模型 --> 生成 这条链的循环无可避免会走向死寂，因为关于世界的信息每过一环只会减少，不会增加 (数据处理不等式 by Cover & Thomas, 2006)。

**你可以把这理解为熵的不可逆性。** LLM 预训练里直接的证据来自 Shumailov et al. (2024) 在 Nature 上的论文，即递归地用模型自己的输出训练自己会导致模型崩塌至完全不可用。而 Gerstgrasser et al. (2024) 发现只要每轮训练保留外部真实数据，不要全部都用自己生成的数据，就能避免模型坍塌。

学界对后训练的研究也殊途同归 - 如果没有新的外部信息输入，效果依然越训越糟糕: RLHF 显著降低输出多样性 (Kirk et al., 2024)；RL 训练后的模型在 pass@1 上更强，但 k 大时 pass@k 反而不如 baseline (Yue et al., 2025)。

多模型叠加的乘数打折也不意外 - 对此的合理猜测是，不同模型压缩的是大同小异的互联网，以人类偏好为锚点的后训练又把它们推向单调的 good answers.

这些证据已经足够我们深入考虑另一个学术界的热门议题：模型的**递归自我改进** (Recursive Self-Improvement, **RSI**)，并有逻辑地辨认其中的真假。

## 2 智能爆炸的逻辑缺口

模型能否自主地优化出数据 pipeline 与架构，训练出比自己更强的模型？如果能，那么模型A训练出比自己更聪明的模型B、模型B进一步搞出自己的升级版，最终会形成无上限的智能爆炸。Good (1965) 提出了这个想法，Schmidhuber 的 Gödel machine (2003) 尝试将它形式化。

半个世纪过去，今年这个话题以 RSI 为名，引发学术社区热烈讨论。OpenAI 和 Anthropic 宣称其前代模型已经很大程度参与到最新模型的训练中。但所谓 RSI 引起的智能爆炸似乎还未出现。如 2026 年 11 月 10 日 Forecasting Research Institute 的最新综述所强调，“公开数据显示 AI 能力的进展**并未明显加速**”。

为什么？

## 3 智能增长从何而来？

本文至此强调的原则是「模型的智能提升根植于高质量数据 (信息)」。关键在于是否有**外界新信息流入**。这些新信息只有两种合法来源：

1. 新数据 (分布外样本)
2. 验证器判据 (外部反馈)

两者之外，没有第三种。因为信息不会凭空产生。这规定着 RSI 乃至一切 AI 训练的理论上界。

<center>*</center>

我们分别考察 (1) 新数据 和 (2) 验证器判据 在当今 (2026年11月) 的现状，然后再基于现状和理论进行推演，以此对 RSI 「何种意义上为真」做讨论。

先说数据。高质量文本持续稀缺的局面不会改变。早在 2024 年 12 月，Ilya 已公开表明公域互联网上的文本数据快要被 LLM 用尽，并直言数据会越来越成为智能增长的瓶颈。所以大家见到2025年 LLM 的发展重心从预训练渐渐转向了后训练。

如前文所述，通过合成数据增加数据总量的方法也难以奏效。相比起训练时见到的数据，模型生成的只是一个更加单调 (平庸) 的世界。所以高质量文本数据的价值水涨船高 eg. 今年消息，字节跳动会为一道数学难题支付出题者1000元。

再说验证器 (verifier). 做 verifier 显然是当今大势所趋，也符合从预训练 (数据瓶颈) 到后训练 (奖励信号瓶颈) 的技术变迁逻辑。验证器提供的就是奖励信号 - 模型在一个新场景有了奖励信号，知道怎么做是对的、怎么是错的，才有机会在这个场景里不断优化能力。例如数学场景的验证器就是计算结果数值正确/错误，或者证明过程的逻辑是否能通过 Lean 编译验证。这条路真的让前沿模型爆发了极强的数学能力。

那么什么是 RSI 的验证器？比较权威的定义来自 Anthropic (2026): **AI** **系统能够完全自主地设计和开发自己的后继系统**。 - 这太模糊，我们把它拆解的具体一些：模型能自主修改自己的 harness、训练 pipeline、强化学习算法、验证器、模型架构，甚至自主抓取编排新的训练数据，以让自己一代代变强。

谨慎的读者可能已经发现逻辑漏洞。问题就出在「**变强**」的标准是什么？现在模型还在用刷榜的方式证明自己变强，但如果真的 AI 能递归开发自己，那么总有一天所有人类榜单都会被 AI 自主迭代的智能凿穿。然后 AI 无可避免地陷入自己给自己定义验证器、自己给自己挑数据的阶段。研究早已表明，没有外部反馈的模型自我纠错并不可靠 (Huang 等, 2024)，因为它没有新信息流入。这就撞上了本文一直强调的理论上界。所以 RSI 从逻辑上并非直接引起智能爆炸。

RSI 本身的定义可能也在逻辑上令人困惑: 一个通过压缩获取的智能模型“递归地”自己训练自己... anyway, 这不是一个值得探讨的问题。

此处我无意继续讨论智能爆炸的可能性。更根本的问题是：**它是否有必要存在？**

<center>*</center>

智能要强到何种地步，才算 “超级智能爆炸” ？毕竟当今最前沿的 AI 模型在3小时内已经可以突破菲尔兹奖级别的难题，说明它在智力层面已经远远超过了人类最强数学家。难道这还不够吗？还想要提升到什么程度去呢？

或者说，这么强的数学能力，就已经足够了吗？

## 4 数学天才的短板

> 瑜不掩瑕。

2026年，以 GPT-6 Astra 为代表的前沿大模型在数学领域展示出了颠覆性的智能水平。达到国际数学奥林匹克金牌水平已经是基操。更进一步，它们开始独立推翻数十年的数学猜想 (三维 Jacobian 猜想、持续约 80 年的 Erdős 单位距离猜想)。到了今年10月，OpenAI 的内部模型已经能在短短几小时里硬解人类顶尖数学家穷极一生难以解决的数学难题，甚至做出 “可能是160年以来最伟大的数学成果” (证明弱黎曼猜想、Navier-Stokes 存在性...)

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/20261117101514760.png)

人们如此痴迷于在代码和数学领域提升模型能力，除了这些领域的提升容易量化以外，也包含着希望它们能帮助 AI 变得更聪明的美好愿景。例如 Huan et al. (2025) 发现数学能力的提升可以 (有条件地) 泛化到其它领域。如果这个愿景为真，说明 AI 各方面的智能水平现今都已经超越人类。**是吗？**

显然不是这样。能拿奥林匹克数竞金牌的模型，做不好一个逻辑严谨的建筑平面施工图。按道理后者在智力上的挑战远不如前者。

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/20261117101501919.png)

除了建筑设计外，还有更常见的问题 - 只要你和 AI 协作写过稍复杂的代码，就能察觉前沿模型习惯「防御型编程」，代码里兜底修饰过多，导致恶性 bug 隐蔽不报错、性能漏洞难以追溯。看来 AI 即使能解决顶尖数学难题，也没有足够聪明的编程习惯。

**即使在数学领域，前沿模型依然并非全能。**

 AI 在数学领域成绩斐然的重要因素在于形式化验证。在解决数学难题的过程中它们可以不断生成候选证明，并通过 Lean 等证明助手检查其逻辑有效性，因而外部信息被不断输入给 LLM，让它能始终在一个基于真实反馈纠错的状态中持续进步，而非在自我学习的死循环里越来越窄化、平庸。

但形式化验证反馈的问题也是它只管 AI 解答的逻辑正确性，其它一概不管。所以 AI 是给出了逻辑，但它的写作质量非常差劲，导致人类数学家需要几个月的时间去理解和验证 AI 的证明 (已经很多顶尖数学家抱怨过这个问题)。

我们停下来想一想：如果 AI 和形式验证器声称一个数学猜想已经被解决，但人类不能理解它的解决方式，那么这个问题对人类而言和「未解决」难道不是一样么？

## 5 为智能搭建脚手架

总体而言，当下 AI 的智能水平符合 Karpathy 提出的“锯齿状智能 (Jagged Intelligence)”。 AI 在一些有形式化验证反馈的领域具备智能尖峰，但在大部分反馈不足的领域表现平平。甚至山峰和峡谷紧挨在一起：在数学推理能力上因为有 Lean 形式验证给出的密集反馈，AI 进步极快、能力极强。但因为没有「文本写作质量」的量化标准，导致 AI 在讲清楚自己的证明过程这块表现一塌糊涂。

制作新场景的形式验证器，等同于给 AI (LLM) 制造通往各种新场景的脚手架，填平其智能洼地。所以 Evaluation (评估AI性能) 的重要性在今天被再次前提。这将是一场跨行业的盛宴。

无论是 Google DeepMind 的 AlphaProof, 还是 SWE-Gym (ICML 2025), 都证实验证器 (verifier) 的路径确实有效，但它所带来模型能力的提升基于大量人力搭建的昂贵基础设施 (数学 Lean、编程 SWE-Bench-Pro 等)。和 RSI 的原始定义已经关系不大。

个人判断：无论 RSI 导致智能爆炸是否为真，当下都确实需要大量专业人士帮助 AI 搭建 infra (基建) 以接入各行各业新数据形态 - 我们不能指望一个从来没做过视频制作与剪辑的 LLM 在这个本就判断标准模糊的新领域实现“自己衡量自己的进步”、“自我增长”。Anthropic 已经跑通了基于 NLP 能力的视频生成。未来会有更多人在此尝到甜头。

## 6 世界模型的一厢情愿: 生成不等于理解

「复现世界」是「智能爆炸」之外，人们的另一个狂野愿景。从算法信息论的角度，这件事与生成一串真正的随机数难度相当：人类从未做到过。

冯·诺依曼 1951 年有句著名的调侃：

> 试图用算术方法产生随机数的人，都已身陷罪恶。
> 
> Anyone who considers arithmetical methods of producing random digits is, of course, in a state of sin.

真正的随机数在算法领域并不存在。 $\pi$ 的数字在统计检验下基本就是个均匀分布的随机序列，但它的前 n 位只需一段很短的程序就能算出。所以「随机性」有两层含义：统计意义上的分布均匀，以及算法意义上的不可压缩。后者正是 Kolmogorov 复杂度：一个对象 $a$ 的复杂度 $K(a)$，是能输出它的最短程序的长度。

它有一个朴素而有力的推论：

$$
K(f(x)) \le K(x) + K(f) + O(1)
$$

这意味着，确定性的计算无法凭空创造超出输入 $x$ 与程序 $f$ 本身的算法信息。本文开篇引例中「生成 50 万条数据却只剩 1 万条有效」，正是这个意思。

2024年初，OpenAI Sora 挂着世界模型的名字横空出世，后续渐渐没了声响。视频生成模型可以把已有的信息展开成千变万化的画面，却不代表它发现了支配真实世界的物理规律。真正的科学发现恰恰是其反方向：从无数复杂现象中提炼出一个更简短、能够解释和预测它们的规律 (用更少的比特描述它)。

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/20261117101444927.png)

Solomonoff (1964) 的通用归纳理论给出了一个诱人的理论极限: 对于可计算环境，通用预测器最终能够收敛到真实的预测分布。但问题在于，这个通用预测器本身不可计算 (Leike & Hutter, 2015)。

## 7 关键在于因果性 (Causality)

学习物理规律，其实就是让模型认识到世界的**因果性**。世界模型不仅要回答「接下来会发生什么」，还必须回答「如果我做了某件事，接下来会发生什么」。这在因果推断中是两个不同的问题：从观测数据学到的 $P(s_{t+1}\mid s_t,a_t$)，几乎不等于主动干预下的 $P(s_{t+1}\mid s_t,do(a_t))$。两套完全不同的物理机制可能生成看起来一模一样的视频，却在施加干预后产生截然不同的结果。这正是李飞飞 (2026) 对世界模型做出的 Renderer & Simulator 区分，或者 LeCun (2026) 区分的普通预测性世界模型与条件性 (因果) 世界模型。两人都知道，后者的难度远大于前者。

> 一个对“未来”充分的统计量,不一定对“干预后的未来”充分。

因为观察数据缺乏反事实信息，物理机制也不直接可见，模型就无法有效区分偶然与必然。这不是模型的问题，你我的大脑都一样。

因此，无论是 Sora 这样的像素生成模型，还是 JEPA 这样的潜空间预测模型，都不能仅凭预测或重建效果好，就证明其掌握了真实的因果机制。要可靠地预测未见过的干预，最终仍然需要足以识别机制的交互数据或可信的因果假设 - 这也是工业生产中仍然依赖基于数学的机理模型，而非接入黑箱 probablistic model 的原因。**一个看起来真实的世界，和一个能够被真实行动所检验的世界，是两回事。**

这就回到了贯穿全文的核心问题：**世界模型的验证器 (verifier) 在哪里？** 数学有 Lean，编程有测试和实际运行，模型都能够通过相对廉价的外部反馈不断纠错，最终实现大师级表现。但世界模型要验证自己预测的物理规律是否正确，最终只能求助于真实世界 - 而真实世界恰是它想要模拟的对象。如果用模型生成的世界来验证模型自身的正确性，就陷入了循环论证；如果每次都到现实中进行实验，又要面对昂贵的交互成本、无法穷尽的干预空间，以及难以控制的真实环境。这就是世界模型所面临的真实困境。

### 7.1 正道

但这并不意味着世界模型没有价值。恰恰相反，它真正的价值就是前文收敛的主线: **让 AI 获得在更多场景中理解、预测和干预真实世界的能力**。 自动驾驶等 Physical AI 场景，可以通过有限的真实交互训练局部的因果动力学模型，再利用现实反馈不断修正预测。Dreamer (2025 Nature) 等研究已经证明了这条路线在特定环境中的有效性。这样，世界不再只是 AI 被动学习的数据来源，还 (至少局部地) 成为了一个能够不断产生新信息的验证器。看来世界模型的未来，在于让 AI 真正进入世界。

## 8 结语: 不必重新发明永动机

写到这里，似乎是用几条老旧的理论，对 RSI、合成数据、世界模型这些当今前沿的研究方向做了一番批判。目的在于慎重地考虑被追逐的概念的真真假假，将理论不可行的排除，并挑选出真正在实践基础上茁壮成长的技术路线。

这是在对抗幻觉。

1775 年，巴黎科学院拒绝再受理永动机方案。此后的两百多年里，人类的工程能力突飞猛进，但再没有必要重新发明一次永动机。

当然，本文的推论并非定论，重点在于引发读者的思考。理论始终会被新的实践证据更新，我今日所写也将被推翻。对此，我乐见其成。

因为科学从不保证正确，它依靠被推翻而存续。

皓首穷经的理论阅读与写作现已结束。让我与真实的世界去继续接触。

<p align="right"> by Siyuan (Harry) </p>

## 参考文献 References


- Carnot, S. (1824), _Réflexions sur la puissance motrice du feu et sur les machines propres à développer cette puissance_, Paris: Bachelier.
- Académie Royale des Sciences (1775), *Histoire de l'Académie Royale des Sciences*, 该年卷（实际刊于 1778）决议不再审查永动机方案
- Cover & Thomas (2006), *Elements of Information Theory*, 2nd ed., Wiley（数据处理不等式）
- Cunningham et al. (2026), [The Economics of Recursive Self-Improvement](https://arxiv.org/abs/2609.15802?utm_source=chatgpt.com), arXiv:2609.15802.
- Li & Vitányi (2008), *An Introduction to Kolmogorov Complexity and Its Applications*, 3rd ed., Springer；Kolmogorov (1965), Three approaches to the quantitative definition of information, *Problems of Information Transmission* 1(1):4–7
- LeCun & Manyika (2026), [Learning Abstractions: A Conversation with Yann LeCun](https://www.amacad.org/publication/daedalus/learning-abstractions-conversation-yann-lecun?utm_source=chatgpt.com), _Dædalus_, Winter/Spring 2026.
- Li, F.-F. (2026), [A Functional Taxonomy of World Models: Renderers, Simulators, Planners, and the Loop That Connects Them](https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models?utm_source=chatgpt.com), June 3, 2026.
- Delétang et al. (2024), Language Modeling Is Compression, ICLR 2024（Chinchilla 70B 压缩 ImageNet patch 至 43.4%、LibriSpeech 至 16.4%）
- Kirk et al. (2024), Understanding the Effects of RLHF on LLM Generalisation and Diversity, ICLR 2024
- Yue et al. (2025), Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?, NeurIPS 2025 (Oral, Best Paper Runner-up)；arXiv:2504.13837
- Huan et al. (2026), [Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning](https://proceedings.mlr.press/v306/huan26a.html?utm_source=chatgpt.com), ICML 2026, PMLR 306: 46232–46268; arXiv:2507.00432.
- Hubert et al. (2026), [Olympiad-Level Formal Mathematical Reasoning with Reinforcement Learning](https://doi.org/10.1038/s41586-025-09833-y?utm_source=chatgpt.com), _Nature_ 651: 607–613. (AlphaProof)
- Anthropic (2026), [When AI Builds Itself](https://www.anthropic.com/institute/recursive-self-improvement?utm_source=chatgpt.com), The Anthropic Institute.
- Jiang et al. (2025), Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond), NeurIPS 2025 Datasets & Benchmarks (Best Paper)；arXiv:2510.22954
- von Neumann (1951), Various techniques used in connection with random digits, *Monte Carlo Method*, NBS Applied Mathematics Series 12:36–38
- Solomonoff (1964a), [A Formal Theory of Inductive Inference, Part I](https://doi.org/10.1016/S0019-9958\(64\)90223-2?utm_source=chatgpt.com), _Information and Control_ 7(1): 1–22.
- Leike & Hutter (2015), [On the Computability of AIXI](https://arxiv.org/abs/1510.05572?utm_source=chatgpt.com), UAI 2015: 239–248.
- Yao (1982), Theory and application of trapdoor functions, FOCS；Blum & Micali (1984), How to generate cryptographically strong sequences of pseudo-random bits, *SIAM J. Computing* 13(4)
- Shannon (1948), A Mathematical Theory of Communication, *Bell System Technical Journal* 27:379–423, 623–656
- Ukkonen (1992), Approximate string-matching with q-grams and maximal matches, *Theoretical Computer Science* 92(1):191–211；DOI 10.1016/0304-3975(92)90143-4
- Pan et al. (2025), [Training Software Engineering Agents and Verifiers with SWE-Gym](https://proceedings.mlr.press/v267/pan25g.html?utm_source=chatgpt.com), ICML 2025, PMLR 267: 47717–47737.
- Deng et al. (2025), [SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?](https://arxiv.org/abs/2509.16941?utm_source=chatgpt.com), arXiv:2509.16941.
- de Moura & Ullrich (2021), _The Lean 4 Theorem Prover and Programming Language_, CADE-28, LNCS 12699: 625–635.
- Broder (1997), On the resemblance and containment of documents, *Proc. Compression and Complexity of Sequences (SEQUENCES '97)*, IEEE, 21–29；Indyk & Motwani (1998), Approximate nearest neighbors: towards removing the curse of dimensionality, STOC
- Sutskever, I. (2024), [Sequence to Sequence Learning with Neural Networks — Test of Time Award Lecture](https://ceointerviews.ai/interview/733336/?utm_source=chatgpt.com), NeurIPS 2024, December 13, 2024.
- Thompson (1933), On the likelihood that one unknown probability exceeds another, *Biometrika* 25:285–294；Robbins (1952), Some aspects of the sequential design of experiments, *Bull. AMS* 58:527–535
- Kučinskas & Rosenberg (2026), [How Close Is Recursive Self-Improvement? A Review of the Evidence](https://forecastingresearch.org/research/recursive-self-improvement-v1?utm_source=chatgpt.com), Forecasting Research Institute, November 10, 2026.
- Auer, Cesa-Bianchi & Fischer (2002), Finite-time analysis of the multiarmed bandit problem, *Machine Learning* 47:235–256 (UCB1)；Auer et al. (2002), The nonstochastic multiarmed bandit problem, *SIAM J. Computing* 32(1):48–77 (EXP3)
- Kocsis & Szepesvári (2006), Bandit based Monte-Carlo planning, ECML 2006 (UCT)
- Perez et al. (2022), Red Teaming Language Models with Language Models, Findings of EMNLP 2022, 3419–3448
- Chao et al. (2023), Jailbreaking Black Box Large Language Models in Twenty Queries (PAIR), arXiv:2310.08419；SaTML 2025
- Mehrotra et al. (2023), Tree of Attacks: Jailbreaking Black-Box LLMs Automatically (TAP), arXiv:2312.02119；NeurIPS 2024
- Romera-Paredes et al. (2023), Mathematical discoveries from program search with large language models (FunSearch), *Nature* 625(7995):468–475
- Romera-Paredes et al. (2024), [Mathematical Discoveries from Program Search with Large Language Models](https://doi.org/10.1038/s41586-023-06924-6?utm_source=chatgpt.com), _Nature_ 625(7995): 468–475.
- Novikov et al. (2025), AlphaEvolve: A coding agent for scientific and algorithmic discovery, arXiv:2506.13131（另有 Google DeepMind 博客，2025-05-14）
- Brown et al. (2024), Large Language Monkeys: Scaling Inference Compute with Repeated Sampling, arXiv:2407.21787
- Mouret & Clune (2015), Illuminating search spaces by mapping elites (MAP-Elites), arXiv:1504.04909
- Samvelyan et al. (2024), Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts, NeurIPS 2024；arXiv:2402.16822
- Hong et al. (2024), Curiosity-driven Red-teaming for Large Language Models, ICLR 2024
- Good (1965), Speculations Concerning the First Ultraintelligent Machine, *Advances in Computers* 6:31–88
- Schmidhuber (2003), Gödel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements, arXiv:cs/0309048
- Shumailov et al. (2024), AI models collapse when trained on recursively generated data, *Nature* 631(8022):755–759
- Dohmatob et al. (2024), A Tale of Tails: Model Collapse as a Change of Scaling Laws, ICML 2024；arXiv:2402.07043
- Gerstgrasser et al. (2024), Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data, COLM 2024；arXiv:2404.01413
- Huang et al. (2024), Large Language Models Cannot Self-Correct Reasoning Yet, ICLR 2024
- OpenAI (2026b), [On the Navier–Stokes Millennium Prize Problem](https://openai.com/index/navier-stokes-solution/?utm_source=chatgpt.com).
- OpenAI (2026c), [The Quasi-Riemann Hypothesis](https://github.com/openai/math/tree/main/preprints/The-Quasi-Riemann-Hypothesis-October-5-2026?utm_source=chatgpt.com), Preprint, October 5, 2026.
- OpenAI (2024), [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/?utm_source=chatgpt.com), Technical Report, February 15, 2024.
- Pearl, J. (2009), [_Causality: Models, Reasoning, and Inference_](https://doi.org/10.1017/CBO9780511803161?utm_source=chatgpt.com), 2nd ed., Cambridge University Press.
- Hafner et al. (2025), [Mastering Diverse Control Tasks through World Models](https://doi.org/10.1038/s41586-025-08744-2?utm_source=chatgpt.com), _Nature_ 640: 647–653. (DreamerV3)
- Cui et al. (2025), The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models, arXiv:2505.22617
- Schmidhuber (2009), Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes, *Anticipatory Behavior in Adaptive Learning Systems*, LNAI 5499:48–76；arXiv:0812.4360
