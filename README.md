# Elicit Machine Learning Reading List

## Purpose

The purpose of this curriculum is to help new [Elicit](https://elicit.com/) employees learn background in machine learning, with a focus on language models. I’ve tried to strike a balance between papers that are relevant for deploying ML in production and techniques that matter for longer-term scalability.

If you don’t work at Elicit yet - we’re [hiring ML and software engineers](https://elicit.com/careers).

## How to read

Recommended reading order:

1. Read “Tier 1” for all topics
2. Read “Tier 2” for all topics
3. Etc

✨ Added after 2024/4/1

## Table of contents

- [Fundamentals](#fundamentals)
  * [Introduction to machine learning](#introduction-to-machine-learning)
  * [Transformers](#transformers)
  * [Key foundation model architectures](#key-foundation-model-architectures)
  * [Training and finetuning](#training-and-finetuning)
- [ML in practice](#ml-in-practice)
  * [Production deployment](#production-deployment)

## Fundamentals

### Introduction to machine learning

**Tier 1**

- [A short introduction to machine learning](https://www.alignmentforum.org/posts/qE73pqxAZmeACsAdF/a-short-introduction-to-machine-learning)
- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk&t=0s)
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)

**Tier 2**

- ✨ [An intuitive understanding of backpropagation](https://cs231n.github.io/optimization-2/)
- [What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4)
- [An introduction to deep reinforcement learning](https://thomassimonini.medium.com/an-introduction-to-deep-reinforcement-learning-17a565999c0c)

**Tier 3**

- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5)

### Transformers

**Tier 1**

- ✨ [But what is a GPT? Visual intro to transformers](https://www.youtube.com/watch?v=wjZofJX0v4M)
- ✨ [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- ✨ [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)

**Tier 2**

- ✨ [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- ✨ [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**Tier 3**

- [A Practical Survey on Faster and Lighter Transformers](https://arxiv.org/abs/2103.14636)
- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848)
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Compositional Capabilities of Autoregressive Transformers: A Study on Synthetic, Interpretable Tasks](https://arxiv.org/abs/2311.12997)
- [Memorizing Transformers](https://arxiv.org/abs/2203.08913)
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)

</details>

### Key foundation model architectures

**Tier 1**

- [Language Models are Unsupervised Multitask Learners](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe) (GPT-2)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)

**Tier 2**

- ✨ [LLaMA: Open and Efficient Foundation Language Models](http://arxiv.org/abs/2302.13971) (LLaMA)
- ✨ [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) ([video](https://www.youtube.com/watch?v=EvQ3ncuriCM)) (S4)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) (T5)
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (OpenAI Codex)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (OpenAI Instruct)

**Tier 3**

- ✨ [Mistral 7B](http://arxiv.org/abs/2310.06825) (Mistral)
- ✨ [Mixtral of Experts](http://arxiv.org/abs/2401.04088) (Mixtral)
- ✨ [Gemini: A Family of Highly Capable Multimodal Models](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf) (Gemini)
- ✨ [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752v1) (Mamba)
- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) (Flan)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Consistency Models](http://arxiv.org/abs/2303.01469)
- ✨ [Model Card and Evaluations for Claude Models](https://www-cdn.anthropic.com/bd2a28d2535bfb0494cc8e2a3bf135d2e7523226/Model-Card-Claude-2.pdf) (Claude 2)
- ✨ [OLMo: Accelerating the Science of Language Models](http://arxiv.org/abs/2402.00838)
- ✨ [PaLM 2 Technical Report](https://arxiv.org/abs/2305.10403) (Palm 2)
- ✨ [Textbooks Are All You Need II: phi-1.5 technical report](http://arxiv.org/abs/2309.05463) (phi 1.5)
- ✨ [Visual Instruction Tuning](http://arxiv.org/abs/2304.08485) (LLaVA)
- [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861)
- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (Google Instruct)
- [Galactica: A Large Language Model for Science](https://arxiv.org/abs/2211.09085)
- [LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) (Google Dialog)
- [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2112.11446) (Meta GPT-3)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (PaLM)
- [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732) (Google Codex)
- [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) (Gopher)
- [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858) (Minerva)
- [UL2: Unifying Language Learning Paradigms](http://aima.cs.berkeley.edu/) (UL2)

</details>

### Training and finetuning

**Tier 2**

- ✨ [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [Learning to summarise with human feedback](https://arxiv.org/abs/2009.01325)
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

**Tier 3**

- ✨ [Pretraining Language Models with Human Preferences](http://arxiv.org/abs/2302.08582)
- ✨ [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](http://arxiv.org/abs/2312.09390)
- [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638v1)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Unsupervised Neural Machine Translation with Generative Language Models Only](https://arxiv.org/abs/2110.05448)

<details><summary><strong>Tier 4+</strong></summary>

- ✨ [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](http://arxiv.org/abs/2312.06585)
- ✨ [Improving Code Generation by Training with Natural Language Feedback](http://arxiv.org/abs/2303.16749)
- ✨ [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668v1)
- ✨ [LIMA: Less Is More for Alignment](http://arxiv.org/abs/2305.11206)
- ✨ [Learning to Compress Prompts with Gist Tokens](http://arxiv.org/abs/2304.08467)
- ✨ [Lost in the Middle: How Language Models Use Long Contexts](http://arxiv.org/abs/2307.03172)
- ✨ [QLoRA: Efficient Finetuning of Quantized LLMs](http://arxiv.org/abs/2305.14314)
- ✨ [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](http://arxiv.org/abs/2403.09629)
- ✨ [Reinforced Self-Training (ReST) for Language Modeling](http://arxiv.org/abs/2308.08998)
- ✨ [Solving olympiad geometry without human demonstrations](https://www.nature.com/articles/s41586-023-06747-5)
- ✨ [Tell, don't show: Declarative facts influence how LLMs generalize](http://arxiv.org/abs/2312.07779)
- ✨ [Textbooks Are All You Need](http://arxiv.org/abs/2306.11644)
- ✨ [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](http://arxiv.org/abs/2305.07759)
- ✨ [Training Language Models with Language Feedback at Scale](http://arxiv.org/abs/2303.16755)
- ✨ [Turing Complete Transformers: Two Transformers Are More Powerful Than One](https://openreview.net/forum?id=MGWsPGogLH)
- [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626)
- [Data Distributional Properties Drive Emergent In-Context Learning in Transformers](https://arxiv.org/abs/2205.05055)
- [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)
- [ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2107.02137)
- [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)
- [ExT5: Towards Extreme Multi-Task Scaling for Transfer Learning](https://arxiv.org/abs/2111.10952)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning](https://arxiv.org/abs/2106.02584)
- [True Few-Shot Learning with Prompts -- A Real-World Perspective](https://arxiv.org/abs/2111.13440)

</details>

## ML in practice

### Production deployment

**Tier 1**

- [Machine Learning in Python: Main developments and technology trends in data science, machine learning, and AI](https://arxiv.org/abs/2002.04803v2)
- [Machine Learning: The High Interest Credit Card of Technical Debt](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

**Tier 2**

- ✨ [Designing Data-Intensive Applications](https://dataintensive.net/)
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
