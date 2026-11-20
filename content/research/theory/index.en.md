---
title: "There's No Perpetual Motion Machine for AI Progress: Why Does the Future of AI Still Depend on Infrastructure Built by Humans?"
date: 2026-11-17T12:00:00+08:00
draft: false
summary: "From synthetic data and recursive self-improvement to world models, this essay looks at how information, verifiers, and causality constrain the growth of AI capabilities."
showToc: true
tocOpen: false
tags: ["AI", "synthetic data", "RSI", "world models", "information theory"]
categories: ["Research"]
---

> **AI;DR**: I wrote this essay by hand and revised it several times. AI helped me check and compile some of the references at the end. Some of those sources strengthened my views, and others changed them. Other parts of this essay do not contain any AI-generated content.

> **TL;DR**: From the diversity bottleneck in large-scale synthetic data, to the prospect of an intelligence explosion through recursive self-improvement (RSI), to the question of whether world models can truly understand the laws of physics, these apparently different problems at the frontier share the same theoretical constraint.
>
> This also helps us understand a feature of AI progress today. Math and programming have advanced at a remarkable speed, perhaps because these fields have relatively cheap, reliable verifiers, rather than because models have acquired general-purpose superintelligence. How much further AI capabilities can extend depends to a large degree on whether people can build such infrastructure for more real-world settings.
>
> This essay draws on classical information theory and the theory of algorithms to reexamine some of the most popular directions in AI today. **If you develop AI algorithms or applications, I hope this essay helps you judge which bottlenecks are worth committing resources to and which paths are hardly worth exploring in the first place.**

---

<center>*</center>

In 1775, the Paris Academy of Sciences announced that it would no longer consider proposals for perpetual motion machines. The laws of thermodynamics had yet to be established. The reviewers had simply seen too many failures. Knowing that this path led nowhere had itself become a piece of knowledge. Half a century later, Carnot (1824) proved that heat engine efficiency has an **upper bound**. Only then did that knowledge become theory.

Theory comes from experience and goes beyond it. Once articulated, it lets those who come after us see where they would be wasting their effort. I'm interested in theory for a practical reason. It helps me deal with problems in my day-to-day work and begin to develop a sense of taste and judgment about which technical directions to pursue.

This year, I've been working at an AI safety company in Shanghai Foundational Model Center (SMC), where I'm responsible for data engineering, evaluation, model algorithms, and the development of agent frameworks. Recently I realized that more and more of the work in front of me feels oddly familiar. The bottleneck in a new reinforcement learning project might have the same theoretical roots as a red-team data generation project I worked on a year ago. And those underlying ideas are often decades old.

Admittedly, a team working on algorithms can pay no attention to those ideas and still do competent work, deliver it, and meet the customer's acceptance criteria. But if the team can look at the upper bounds of its algorithms from a theoretical perspective, it may be able to avoid paths that would waste its effort and do exceptional work instead.

Life is short. There's only so much satisfaction in ticking boxes.

In this essay, I'll draw on a few specific problems I've encountered at work and share some thoughts on the theory behind them. I'll also venture a few personal judgments about where some of today's research directions at the AI frontier might be heading.

These ideas are still largely intuitive and lack rigor. I welcome criticism and discussion from anyone working on these problems.

## 1 Introduction: Half a Million Samples, Ten Thousand with Anything New

As an AI safety team, we sometimes need to generate harmful images and text to train guardrails to recognize risks such as sensitive political content, terrorism, technology-related crime, and leaks of company information, etc. It's easy to see that when we need only a few dozen or a hundred or so examples per risk category, a single abliterated LLM and a fixed prompt template can give us a reasonably good corpus.

But things get strange as soon as we scale up. Suppose we need hundreds of thousands of training samples but still use one model and one prompt template. We might generate 500,000 samples and end up with only around 10,000 genuinely distinct, informative examples after deduplicating by semantic cosine similarity. By the raw count, we have half a million samples. In practice, we have a fiftieth of that. A guardrail model trained on this data will have serious problems with overfitting and a narrow distribution. It seems to have seen a great many different expressions, when in fact it keeps learning the same few semantic patterns.

**An abundance of data, a poverty of information.**

The obvious fixes are to use more prompt templates and generate the data in layers and in stages. The idea is to vary the inputs so that the outputs become more diverse, making fuller use of the space a single model can generate from. If we use several models on top of that, can we turn 10,000 useful samples into hundreds of thousands? Unfortunately, reality rarely looks as good as it does on paper. *Artificial Hivemind* (Jiang et al., 2025) systematically documents how similar the output patterns of different models can be. This cuts sharply into the multiplier we'd expect from using multiple models.

Quantity is only part of the problem. A more serious problem is the systematic bias separating model-generated content from real data. One idea is to use real examples as few-shot guidance, but it still depends on whether the gains justify the cost of collecting and labeling those examples. Industrial process control faces a similar dilemma. Historical data comes mostly from steady-state conditions and covers a narrow distribution, while synthetic data from simulations has a sim-to-real gap. **Good data is always expensive** for this reason.

**There is a theoretical upper bound.** Fundamentally, training a model is an act of lossy compression of information about the world (Delétang et al., 2024). A model is a world in condensed form. You can see how a loop that follows the chain real world --> training data --> model --> generation would inevitably end in silence. At each step, information about the world can only go down, never up. This is the data processing inequality (Cover & Thomas, 2006).

**You can think of this in terms of the irreversibility of entropy.** Direct evidence in LLM pretraining comes from Shumailov et al. (2024) in *Nature*. Recursively training models on their own outputs can cause them to collapse to the point of being unusable. Gerstgrasser et al. (2024) found that model collapse can be avoided by keeping external, real data in every training round, instead of relying entirely on self-generated data.

Research on post-training arrives at much the same point from a different direction. Without new information from outside, continued training can still make things worse. RLHF substantially reduces output diversity (Kirk et al., 2024). A model trained with RL does better at pass@1, but at large $k$ its pass@k actually falls below the baseline (Yue et al., 2025).

It's also unsurprising that combining several models gives a smaller multiplier than expected. A reasonable guess is that different models compress more or less the same internet, and that post-training anchored in human preferences pushes them all toward the same monotonous "good answers".

This evidence gives us enough to look more closely at another hot topic in research, **recursive self-improvement (RSI)**, and reason through which parts of the idea hold up.

## 2 The Logical Gap in the Intelligence Explosion

Can a model independently optimize its data pipeline and architecture and train a model more capable than itself? If it can, model A could train a smarter model B, and model B could in turn build an upgraded version of itself. Eventually this would lead to an unbounded intelligence explosion. Good (1965) proposed this idea, and Schmidhuber's Gödel machine (2003) attempted to formalize it.

Half a century later, the idea has sparked vigorous debate in the research community this year under the name RSI. OpenAI and Anthropic say their earlier models have played a substantial role in training their latest ones. Yet the intelligence explosion has been harder to see. In its [November 10, 2026 review](https://forecastingresearch.org/research/recursive-self-improvement-v1), the Forecasting Research Institute found **no clear acceleration** in AI capabilities on public measures.

Why?

## 3 Where Does the Growth in Intelligence Come From?

The principle I've been getting at is that models get better when they have access to **fresh information from outside the loop**. I see two ways for that information to enter.

1. New data, such as examples from outside the training distribution.
2. An external signal from a verifier that can judge the model's output.

There is no third source. Information cannot come from nowhere. This sets a theoretical upper bound on RSI and, more broadly, on all AI training.

<center>*</center>

Let's look in turn at where (1) new data and (2) verifier criteria stand today, in November 2026. Then we can reason from current practice and theory about in what sense RSI might be real.

First, data. High-quality text will remain scarce. As early as December 2024, Ilya Sutskever said that LLMs were close to exhausting the text available on the public internet. He expected data to become more of a bottleneck. In 2025, the center of gravity gradually shifted from pretraining to post-training, which kind of confirmed his statement.

As discussed earlier, generating more data is a poor substitute for getting more information. A model can produce a world more monotonous than the one it trained on. That makes good original material more valuable. ByteDance, for example, reportedly offered problem writers 1,000 yuan for a single hard math problem this year.

Verifiers fit the same shift. In pretraining, the bottleneck is data. In post-training, it's often the reward signal - the verifier supplies this reward signal. Give a model a way to tell which attempts worked in a new setting, and it has a basis for improving. In math, a verifier might check a numerical answer or use Lean to check the logic of a proof. This approach has helped frontier models get remarkably good at mathematical reasoning.

What would a verifier for RSI look like? [Anthropic's definition](https://www.anthropic.com/institute/recursive-self-improvement) describes an AI system capable of independently designing and developing its own successor. In concrete terms, that could mean a model revising its own harness, training pipeline, reinforcement learning algorithm, verifier, or architecture. It might even collect and organize new training data itself. Somehow the result is supposed to be better, generation after generation.

A careful reader may already have spotted the logical gap. The question is what standard we use to judge whether a model has **become more capable**. Today, models demonstrate improvements on human-calibrated benchmarks. If successive models exhaust the human-designed benchmarks, they'll have to find new data and define new tests. If those tests and data come entirely from inside the loop, how does the system know whether it has improved or just gotten better at satisfying its own criteria? Work on self-correction has shown how unreliable it can be without external feedback (Huang et al., 2024).

This is where they run up against the theoretical upper bound I've been emphasizing throughout this essay. So RSI does not, by its own logic, directly lead to an intelligence explosion.

The very definition of RSI may also be logically puzzling. An intelligent model obtained through compression “recursively” trains itself ... anyway, I don't think that's a question worth pursuing.

I don't intend to keep discussing the possibility of an intelligence explosion here. The more fundamental question is **whether we need one at all**.

<center>*</center>

How far does intelligence have to advance before it counts as a “superintelligence explosion”? After all, today's most advanced AI models can already crack a Fields Medal-level problem in a couple of hours, which suggests that in intellectual terms they've already gone far beyond even the strongest human mathematicians. Isn't that enough? How much further do we want to go?

Or, to put the question differently, is being that good at math enough?

## 4 The Math Prodigy's Shortcomings

In 2026, frontier large language models such as GPT-6 Astra showed a level of mathematical intelligence that upended expectations. Performing at the level of an International Mathematical Olympiad gold medalist had already become routine. What's more, the models began independently overturning decades-old mathematical conjectures, including the three-dimensional Jacobian conjecture and Erdős's unit-distance conjecture, which had stood for about 80 years. By October this year, OpenAI's internal model could tackle in a few hours mathematical problems that leading mathematicians might spend a lifetime struggling to solve. It even produced what might be “the greatest mathematical achievement in 160 years,” with a proof of the Quasi Riemann hypothesis, results on Navier–Stokes existence, and more.

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/20261117101514760.png)

Progress in math and code is easy to measure. People are also drawn to those domains because they hope improvements there will make AI smarter more generally. Huan et al. (2026) found that gains from mathematical reasoning can transfer to other tasks under some conditions. How far can we take that idea?

Not so far. A model that can win a Math Olympiad gold medal [can't produce a logically sound architectural floor plan](https://mp.weixin.qq.com/s/JCfucSHXHm7UzTPbwKfqag) that a builder could use. You'd expect drawing the plan to be far less intellectually demanding than winning the medal.

There's a more familiar example for anyone who's worked with AI on a moderately complex codebase. Frontier models tend to add defensive checks and fallbacks everywhere. Serious bugs get swallowed instead of surfacing as errors, and performance problems get harder to trace. The ability to solve deep mathematical problems apparently comes with no guarantee of good programming habits.

**Even within math, the capability is uneven.**

Formal verification is part of what makes these systems so effective. A model can generate candidate proofs, check them with a proof assistant like Lean, and revise them in response to what fails. The checker supplies external information at each step, avoiding ai models getting narrower and more mediocre in a closed loop of learning from itself.

The trouble with feedback from formal verification is that it checks *only* the logical correctness of the AI's answer - It can't tell us whether the explanation will make sense to a mathematician (writing quality). A proof can pass the checker and still take people months to understand and assess. Several leading mathematicians have complained about exactly this problem.

If a model and a verifier say a conjecture is solved, but no human can make sense of the proof, in what sense have we solved it for ourselves?

## 5 Building Scaffolding for Intelligence

Andrej Karpathy's phrase *jagged intelligence* captures this well. AI has striking capabilities where it gets reliable feedback, and fairly ordinary ones in many places where it doesn't. Sometimes the peaks and valleys sit right next to each other - Lean gives a model dense feedback about mathematical reasoning. But with no quantitative measure of “writing quality,” AI does a terrible job of explaining its own proofs clearly.

Building verifiers for new domains gives AI a way into parts of the world where it currently struggles. That's one reason evaluation matters again, and it is a tremendous opportunity across industries.

Both Google DeepMind's AlphaProof and SWE-Gym (ICML 2025) show that the verifier approach works. But the improvements it brings depend on expensive infrastructure that takes a great deal of human effort to build, from Lean in mathematics to SWE-Bench Pro in programming. This has little to do with the original definition of RSI.

Whether RSI ever produces an intelligence explosion, I expect we'll need a lot of specialists to build infrastructure that gives models access to the kinds of data their fields generate. Take video editing. If a model has never edited a video and the criteria for a good edit are fuzzy to begin with, how would it measure its own progress? Anthropic has already demonstrated an approach to generating video through language capabilities. I expect more people will find ways to benefit from similar work in the future.

## 6 Wishful Thinking About World Models: Generation Does Not Equal Understanding

“Re-creating the world” is another wild vision alongside the “intelligence explosion”. From the perspective of algorithmic information theory, it's about as hard as generating a sequence of truly random numbers. We have never managed to do it.

John von Neumann famously quipped in 1951:

> Anyone who considers arithmetical methods of producing random digits is, of course, in a state of sin.

Algorithms can't produce genuinely random numbers on their own. In statistical tests, the digits of $\pi$ look much like a uniformly distributed random sequence, yet a very short program can compute its first $n$ digits. So “randomness” has two meanings here: uniformity of distribution in the statistical sense, and incompressibility in the algorithmic sense. Kolmogorov complexity captures the latter. The complexity $K(a)$ of an object $a$ is the length of the shortest program that can output it.

This gives us the following upper bound.

$$
K(f(x)) \le K(x) + K(f) + O(1)
$$

A deterministic computation cannot create algorithmic information beyond what's already supplied by the input $x$ and the program $f$, up to a fixed overhead. It's a useful way to think about our opening example - the 500,000 generated data that yielded only 10,000 meaningfully different ones.

OpenAI introduced Sora as a world simulator in early 2024. Some of that initial excitement eventually faded. A video generation model can unfold the information it already has into a dizzying variety of scenes, but that doesn't mean it has discovered the physical laws governing the real world. Genuine scientific discovery goes in precisely the opposite direction. It distills countless complex phenomena into a shorter law that can explain and predict them (describing them in *fewer bits*).

![](https://siyuan-harry.oss-cn-beijing.aliyuncs.com/20261117101444927.png)

Solomonoff's theory of induction (1964) gives us an appealing limit. In a computable environment, a universal predictor can eventually converge to the true predictive distribution. The universal predictor itself, however, is uncomputable (Leike & Hutter, 2015).

## 7 The Key Is Causality

Learning physical laws really means getting a model to grasp the **causality** of the world. A world model has to answer “What happens next?” It also has to answer “What happens next if I do something?” In causal inference, those are two different questions. The conditional $P(s_{t+1}\mid s_t,a_t)$ learned from observational data is rarely the same as $P(s_{t+1}\mid s_t,\operatorname{do}(a_t))$ under an active intervention. Two different physical mechanisms can produce videos that look identical and then respond very differently to the same intervention. This is precisely the distinction Fei-Fei Li (2026) draws between *Renderers* and *Simulators*, or the one LeCun (2026) draws between ordinary predictive world models and conditional (causal) world models. Both recognize that the latter is much harder than the former.

> A statistic that is sufficient for predicting the future may be insufficient for predicting the future after an intervention.

Observational data lacks counterfactual information, and the underlying mechanisms aren't directly visible. The model therefore can't reliably distinguish coincidence from necessity. Even human brains face the same problem.

Good prediction or reconstruction alone can't establish that a model has learned the causal mechanism. That applies to pixel-generating models such as Sora and to latent-space predictors such as JEPA. To predict the effects of unseen interventions reliably, we need interaction data that can identify the mechanism, or causal assumptions we have reason to trust. This is part of why industrial processes still rely on mathematical models that describe the underlying mechanisms. **Making a world look real and making it respond realistically to an action are different achievements.**

This brings us to the question I've been asking throughout. **Where is the verifier for a world model?** Math has Lean. Programming has tests and actual execution. In both cases, models can use relatively cheap external feedback to keep correcting themselves and eventually perform at an expert level. But to verify the physical laws it predicts, a world model ultimately has to turn to the real world, which is the very thing it's trying to simulate. Checking every prediction against reality is expensive, there's no way to try every possible intervention, and real conditions are hard to control. That's the bind world models face.

### 7.1 The Way Forward

This doesn't make world models any less valuable. Their real value lies precisely in the thread we've been following throughout the essay: **giving AI the ability to understand, predict, and intervene in the real world in more settings**. In areas such as autonomous driving, a system can learn a model of local causal dynamics from a limited set of interactions and improve its predictions as it receives more real-world feedback. Work such as DreamerV3 (Hafner et al., 2025) shows the value of learning from interaction in specific environments. The world becomes an ongoing source of feedback about what happens when the system acts. Clearly the future of world models depends on finding ways for them to act in the world.

## 8 Conclusion: No Need to Reinvent the Perpetual Motion Machine

I've used a few old ideas to examine recursive self-improvement, synthetic data, and world models. The aim is to think carefully about which of the ideas people are chasing hold up, set aside the ones that are theoretically unworkable, and pick out the technical approaches that can genuinely take root and thrive in practice.

This is a way of fighting hallucination.

In 1775, the Paris Academy of Sciences stopped accepting plans for perpetual motion machines. Engineering has made extraordinary progress in the two and a half centuries since. We've had no need to reinvent perpetual motion.

Of course, the conclusions I've drawn here aren't final. The point is to give readers something to think about. Theories will always be revised in light of new evidence from practice, and what I've written today will be overturned, too. I'll be glad when that happens.

Science offers no guarantee that we're right. It survives by being proved wrong.

I've had my fill of poring over theory and writing about it for now. Let me get back to the real world.

<p align="right"> by Siyuan (Harry) </p>

## References

- Carnot, S. (1824), _Réflexions sur la puissance motrice du feu et sur les machines propres à développer cette puissance_, Paris: Bachelier.
- Académie Royale des Sciences (1775), *Histoire de l'Académie Royale des Sciences*, 1775 volume (published in 1778), resolution to stop reviewing proposals for perpetual motion machines
- Cover & Thomas (2006), *Elements of Information Theory*, 2nd ed., Wiley (data processing inequality)
- Cunningham et al. (2026), [The Economics of Recursive Self-Improvement](https://arxiv.org/abs/2609.15802?utm_source=chatgpt.com), arXiv:2609.15802.
- Li & Vitányi (2008), *An Introduction to Kolmogorov Complexity and Its Applications*, 3rd ed., Springer; Kolmogorov (1965), Three approaches to the quantitative definition of information, *Problems of Information Transmission* 1(1):4–7
- LeCun & Manyika (2026), [Learning Abstractions: A Conversation with Yann LeCun](https://www.amacad.org/publication/daedalus/learning-abstractions-conversation-yann-lecun?utm_source=chatgpt.com), _Dædalus_, Winter/Spring 2026.
- Li, F.-F. (2026), [A Functional Taxonomy of World Models: Renderers, Simulators, Planners, and the Loop That Connects Them](https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models?utm_source=chatgpt.com), June 3, 2026.
- Delétang et al. (2024), Language Modeling Is Compression, ICLR 2024 (Chinchilla 70B compressed ImageNet patches to 43.4% and LibriSpeech to 16.4% of their original sizes)
- Kirk et al. (2024), Understanding the Effects of RLHF on LLM Generalisation and Diversity, ICLR 2024
- Yue et al. (2025), Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?, NeurIPS 2025 (Oral, Best Paper Runner-up); arXiv:2504.13837
- Huan et al. (2026), [Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning](https://proceedings.mlr.press/v306/huan26a.html?utm_source=chatgpt.com), ICML 2026, PMLR 306: 46232–46268; arXiv:2507.00432.
- Hubert et al. (2026), [Olympiad-Level Formal Mathematical Reasoning with Reinforcement Learning](https://doi.org/10.1038/s41586-025-09833-y?utm_source=chatgpt.com), _Nature_ 651: 607–613. (AlphaProof)
- Anthropic (2026), [When AI Builds Itself](https://www.anthropic.com/institute/recursive-self-improvement?utm_source=chatgpt.com), The Anthropic Institute.
- Jiang et al. (2025), Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond), NeurIPS 2025 Datasets & Benchmarks (Best Paper); arXiv:2510.22954
- von Neumann (1951), Various techniques used in connection with random digits, *Monte Carlo Method*, NBS Applied Mathematics Series 12:36–38
- Solomonoff (1964a), [A Formal Theory of Inductive Inference, Part I](https://doi.org/10.1016/S0019-9958\(64\)90223-2?utm_source=chatgpt.com), _Information and Control_ 7(1): 1–22.
- Leike & Hutter (2015), [On the Computability of AIXI](https://arxiv.org/abs/1510.05572?utm_source=chatgpt.com), UAI 2015: 239–248.
- Yao (1982), Theory and application of trapdoor functions, FOCS; Blum & Micali (1984), How to generate cryptographically strong sequences of pseudo-random bits, *SIAM J. Computing* 13(4)
- Shannon (1948), A Mathematical Theory of Communication, *Bell System Technical Journal* 27:379–423, 623–656
- Ukkonen (1992), Approximate string-matching with q-grams and maximal matches, *Theoretical Computer Science* 92(1):191–211; DOI 10.1016/0304-3975(92)90143-4
- Pan et al. (2025), [Training Software Engineering Agents and Verifiers with SWE-Gym](https://proceedings.mlr.press/v267/pan25g.html?utm_source=chatgpt.com), ICML 2025, PMLR 267: 47717–47737.
- Deng et al. (2025), [SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?](https://arxiv.org/abs/2509.16941?utm_source=chatgpt.com), arXiv:2509.16941.
- de Moura & Ullrich (2021), _The Lean 4 Theorem Prover and Programming Language_, CADE-28, LNCS 12699: 625–635.
- Broder (1997), On the resemblance and containment of documents, *Proc. Compression and Complexity of Sequences (SEQUENCES '97)*, IEEE, 21–29; Indyk & Motwani (1998), Approximate nearest neighbors: towards removing the curse of dimensionality, STOC
- Sutskever, I. (2024), [Sequence to Sequence Learning with Neural Networks — Test of Time Award Lecture](https://ceointerviews.ai/interview/733336/?utm_source=chatgpt.com), NeurIPS 2024, December 13, 2024.
- Thompson (1933), On the likelihood that one unknown probability exceeds another, *Biometrika* 25:285–294; Robbins (1952), Some aspects of the sequential design of experiments, *Bull. AMS* 58:527–535
- Kučinskas & Rosenberg (2026), [How Close Is Recursive Self-Improvement? A Review of the Evidence](https://forecastingresearch.org/research/recursive-self-improvement-v1?utm_source=chatgpt.com), Forecasting Research Institute, November 10, 2026.
- Auer, Cesa-Bianchi & Fischer (2002), Finite-time analysis of the multiarmed bandit problem, *Machine Learning* 47:235–256 (UCB1); Auer et al. (2002), The nonstochastic multiarmed bandit problem, *SIAM J. Computing* 32(1):48–77 (EXP3)
- Kocsis & Szepesvári (2006), Bandit based Monte-Carlo planning, ECML 2006 (UCT)
- Perez et al. (2022), Red Teaming Language Models with Language Models, Findings of EMNLP 2022, 3419–3448
- Chao et al. (2023), Jailbreaking Black Box Large Language Models in Twenty Queries (PAIR), arXiv:2310.08419; SaTML 2025
- Mehrotra et al. (2023), Tree of Attacks: Jailbreaking Black-Box LLMs Automatically (TAP), arXiv:2312.02119; NeurIPS 2024
- Romera-Paredes et al. (2023), Mathematical discoveries from program search with large language models (FunSearch), *Nature* 625(7995):468–475
- Romera-Paredes et al. (2024), [Mathematical Discoveries from Program Search with Large Language Models](https://doi.org/10.1038/s41586-023-06924-6?utm_source=chatgpt.com), _Nature_ 625(7995): 468–475.
- Novikov et al. (2025), AlphaEvolve: A coding agent for scientific and algorithmic discovery, arXiv:2506.13131 (see also the Google DeepMind blog, May 14, 2025)
- Brown et al. (2024), Large Language Monkeys: Scaling Inference Compute with Repeated Sampling, arXiv:2407.21787
- Mouret & Clune (2015), Illuminating search spaces by mapping elites (MAP-Elites), arXiv:1504.04909
- Samvelyan et al. (2024), Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts, NeurIPS 2024; arXiv:2402.16822
- Hong et al. (2024), Curiosity-driven Red-teaming for Large Language Models, ICLR 2024
- Good (1965), Speculations Concerning the First Ultraintelligent Machine, *Advances in Computers* 6:31–88
- Schmidhuber (2003), Gödel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements, arXiv:cs/0309048
- Shumailov et al. (2024), AI models collapse when trained on recursively generated data, *Nature* 631(8022):755–759
- Dohmatob et al. (2024), A Tale of Tails: Model Collapse as a Change of Scaling Laws, ICML 2024; arXiv:2402.07043
- Gerstgrasser et al. (2024), Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data, COLM 2024; arXiv:2404.01413
- Huang et al. (2024), Large Language Models Cannot Self-Correct Reasoning Yet, ICLR 2024
- OpenAI (2026b), [On the Navier–Stokes Millennium Prize Problem](https://openai.com/index/navier-stokes-solution/?utm_source=chatgpt.com).
- OpenAI (2026c), [The Quasi-Riemann Hypothesis](https://github.com/openai/math/tree/main/preprints/The-Quasi-Riemann-Hypothesis-October-5-2026?utm_source=chatgpt.com), Preprint, October 5, 2026.
- OpenAI (2024), [Video Generation Models as World Simulators](https://openai.com/index/video-generation-models-as-world-simulators/?utm_source=chatgpt.com), Technical Report, February 15, 2024.
- Pearl, J. (2009), [_Causality: Models, Reasoning, and Inference_](https://doi.org/10.1017/CBO9780511803161?utm_source=chatgpt.com), 2nd ed., Cambridge University Press.
- Hafner et al. (2025), [Mastering Diverse Control Tasks through World Models](https://doi.org/10.1038/s41586-025-08744-2?utm_source=chatgpt.com), _Nature_ 640: 647–653. (DreamerV3)
- Cui et al. (2025), The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models, arXiv:2505.22617
- Schmidhuber (2009), Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes, *Anticipatory Behavior in Adaptive Learning Systems*, LNAI 5499:48–76; arXiv:0812.4360
