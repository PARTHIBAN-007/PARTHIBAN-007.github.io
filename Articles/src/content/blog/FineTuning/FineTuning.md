---
title: FineTuning
description: FineTuning Mechanism
date: 2026-11-08
category: FineTuning
---

Fine-tuning is the process of taking a pre-trained Large Language Model—which already understands general language and patterns from a massive dataset—and performing additional training on a smaller, specialized dataset to adapt it for a specific task or domain. Think of it like a doctor who has completed general medical school (pre-training) and then undergoes a residency in cardiology (fine-tuning) to gain specialized expertise while retaining their foundational knowledge.

# **Stages of LLM Training**

**Pre-Training** → **Supervised Fine-Tuning (SFT)** → **RLHF**
![](Pasted%20image%2020261108201839.png)

- **Pre-Training:** The objective is to train the LLM architecture to understand and recognize patterns in massive datasets (Internet data) to enable text generation.
    - The model learns on raw, unstructured text.
    - In the pre-training phase, the model learns how words relate to each other and the basic nuances of language.
    - While pre-training provides a broad base, the model actually gains a massive amount of "Domain Knowledge" (Medicine, Coding, History) during this phase simply because that data exists on the internet. However, it doesn't know how to **use** that knowledge to follow instructions or act as an assistant.
- **Supervised Fine-Tuning (SFT):** The LLM is trained on high-quality, task-oriented examples consisting of input-output pairs. This guides the model to apply its knowledge effectively and provide structured, helpful responses.
- **RLHF (Reinforcement Learning from Human Feedback):** This stage aligns the LLM with human preferences. It teaches the model to respond in a way humans find useful, safe, and accurate, leveraging the knowledge acquired during Pre-Training and SFT. RLHF specifically aligns the **style and behavior** of the model's responses.
![](Pasted%20image%2020261108201859.png)

# Supervised Fine Tuning:

In SFT, the dataset consists of high-quality data points in an **input-output (instruction-response)** format. This aligns the behavior of the LLM to respond in a structured manner.

- **Instruction Datasets:** These contain specific tasks so the pre-trained LLM can be adapted for domain-specific goals or general tasks like question answering and coding.
- **Comparison to Traditional ML:** To fine-tune a model, we follow principles similar to **Transfer Learning**. In traditional CV (Computer Vision), we might freeze the backbone and add new layers.
- **The LLM Difference:** In standard "Full Fine-Tuning" for LLMs, the weights are **not** frozen; all parameters are updated to minimize loss. However, since LLMs have billions of parameters, updating all of them requires massive compute resources.

To solve this compute bottleneck, **Parameter-Efficient Fine-Tuning (PEFT)** algorithms were introduced.

> **Note:** **LoRA** and **QLoRA** are not traditional full fine-tuning methods. They are PEFT techniques designed to address memory bottlenecks. They optimize resources by freezing the base model and updating only a small number of additional parameters (LoRA) or using quantization (QLoRA) to run on limited hardware.

**Key Algorithms and Methods**

**PEFT Methods:**

- **Adapters:** Sequential bottleneck layers.
- **LoRA:** Low-Rank Adaptation of weights.
- **QLoRA:** Quantized LoRA for 4-bit efficiency.

**Reinforcement Learning (RL) Algorithms:**

- **TRPO:** Trust Region Policy Optimization.
- **PPO:** Proximal Policy Optimization (The standard for RLHF).
- **GRPO:** Group Relative Policy Optimization (A newer, more efficient RL logic used in models like DeepSeek-R1).

# Adapter:

Research Paper: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751)

Full fine-tuning requires massive computational resources, similar to the original Pre-Training phase. Additionally, full fine-tuning can lead to **"catastrophic forgetting,"** where the model completely changes its behavior or loses general knowledge while trying to learn a specific domain. To incorporate domain knowledge or align responses, we often only need to modify a small fraction of the model's parameters.

**The Adapter Solution**  
To solve this, researchers introduced **Adapters**. Instead of updating the entire network, they inserted small, trainable modules (adapter blocks) into the existing core architecture.

1. The original weights of the LLM are **frozen**.
2. Only the weights within the added adapter blocks are updated.  
    If you are familiar with **Traditional Machine Learning** or Computer Vision, this is very similar to **Transfer Learning** used in architectures like ImageNet or ResNet. In those cases, when compute is a bottleneck, the backbone is frozen, and only the final "head" layers are replaced or trained to align the model with a new task.

**How Adapters Work (The Bottleneck)**  
Adapters use a "bottleneck" structure to remain efficient:

1. **Down-Projection:** The original $d$-dimensional features are projected into a much smaller dimension, $m$.
2. **Non-linearity:** An activation function (like ReLU) is applied.
3. **Up-Projection:** The features are projected back to the original $d$ dimensions.  
    **The Math of Parameters:**  
    The total number of parameters added per layer (including biases) is $2md + d + m$. By ensuring that $m \ll d$, we limit the added parameters to a tiny fraction of the original model—typically between **0.5% and 8%**. This bottleneck dimension ($m$) allows developers to trade off performance for parameter efficiency.  
    **Limitations**  
    While efficient in terms of storage, Adapters have a major drawback: **Inference Latency**. Because these blocks are added sequentially (one after another) within the core architecture, the model must perform extra computations for every layer during every forward pass, slowing down the response time.

> **Key Insight:** Researchers found that pre-trained language models have a low **"intrinsic dimension."** This means they can learn tasks efficiently even when restricted to a much smaller parameter subspace, which is the foundational theory that led from Adapters to LoRA.
![](Pasted%20image%2020261108201916.png)

# LoRA(Low Rank Adaptation):

Research Paper: [LORA: LOW-RANK ADAPTATION OF LARGE LAN GUAGE MODELS](https://arxiv.org/pdf/2106.09685)

Before diving into the LoRA architecture, let's refresh our understanding of matrices and their **Rank**.  
**What is the Rank of a Matrix?**  
The **Rank** is the number of independent rows or columns in a matrix. A row or column is independent if it cannot be created by simply multiplying or adding the others.  
Consider **Matrix A**:

$$  
A = \begin{bmatrix} 10 & 20 & 30 \\ 40 & 80 & 120 \\ 70 & 140 & 210 \end{bmatrix}

$$

Here, the **Rank of Matrix A = 1**.  
Why? Because Columns 2 and 3 are just multiples of Column 1: where $\text{Column 2} = \text{Column 1} \times 2$ ,$\text{Column 3} = \text{Column 1} \times 3$  
Since there is only one truly unique column, the rank is 1.  
Now, consider **Matrix B**:

$$  
B = \begin{bmatrix} 5 & 10 & 39 \\ 23 & 49 & 2 \\ 3 & 48 & 80 \end{bmatrix}

$$

The **Rank of Matrix B = 3**.  
In this case, no row or column can be derived from the others. Every single value adds unique information to the structure, making it "Full Rank."  
**Why does Rank matter here?**  
A $3 \times 3$ matrix consists of 9 elements. When a matrix has internal dependency (low rank), it means there is redundancy. Because of this redundancy, we can represent a large matrix using two smaller matrices. When these two are multiplied, they reconstruct the original dimensions with the exact same elements.  
**Low-Rank Factorization of Matrix A**  
Since Matrix A has a rank of 1, we can break it down into a $3 \times 1$ column vector and a $1 \times 3$ row vector:

$$  
A = \begin{bmatrix} 1 \\ 4 \\ 7 \end{bmatrix} \times \begin{bmatrix} 10 & 20 & 30 \end{bmatrix}

$$

By using this format, we only need to store **6 elements** (3 from each smaller matrix) instead of the original **9**. In the world of LoRA, this is how we update massive models with very little data.

**From Adapters to LoRA**  
Earlier "Adapter" methods added extra layers **sequentially** within the model architecture. This introduced two main problems:

1. **Increased Latency:** Data had to pass through extra layers during inference.
2. **Structural Changes:** The core architecture had to be modified.

## **LoRA's Solution:**

Instead of adding sequential layers, LoRA adds a **parallel** path consisting of two low-rank matrices ($A$ and $B$).

1. We **freeze** the original weights ($W_0$).
2. We train only the low-rank matrices $A$ and $B$.
3. The total weight update is $\Delta W = B \times A$.
4. During inference, we can merge these weights: $W_{total} = W_0 + BA$. This results in **zero added latency**.
![](Pasted%20image%2020261108201933.png)
where:

$$  
W' = W + \Delta W  
$$

$$  
W' = W_0 + \Delta W = W_0 + BA  
$$

$$  
B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \le \min(d,k)  
$$

$$  
r≤min⁡(d,k)  
$$

During training, W is **frozen** and does not receive gradient updates.
![](Pasted%20image%2020261108202009.png)

$$  
W' = W + \Delta W = W + \frac{\alpha}{r}(A \times B)  
$$

**$W$**: The **Frozen weights**.

**$A$ & $B$**: The **Trainable Adapters**.

**$r$ (Rank)**: The **Size** of the adapters. Smaller $r$ = less memory; larger $r$ = more learning capacity.

**$\alpha$ (Alpha)**: The **Strength**. It controls how much the adapters change the original model's behavior.

**$\frac{\alpha}{r}$**: The **Scaling Factor**. It keeps the training stable, regardless of what rank ($r$) you choose.

To ensure the model starts with the exact same output as the original frozen model, we use a **Random Gaussian initialization for A** and **Zero initialization for B**. This ensures that at the start of training, $\Delta W (B \times A)$ is zero.

> It is important to clarify that **LoRA does not actually split or decompose the existing pre-trained weight matrix** of the LLM into two smaller matrices. This is a common misconception where people confuse LoRA with standard Low-Rank Approximation or SVD (Singular Value Decomposition).  
> The concepts of **rank** and **decomposition** are used here primarily to illustrate an intuition: that a weight update can be represented in a lower-dimensional space and then projected back into the original dimensions while still capturing the essential "knowledge" or task-specific adjustments of a much larger matrix.  
> In this context, the **rank ($r$)** represents the degree of factorization:  
> • **Lower Rank:** Fewer trainable parameters, leading to less influence on the original model but higher efficiency.  
> • **Higher Rank:** More trainable parameters, allowing for greater influence and more complex updates.  
> • **Full Rank:** Setting the rank equal to the original model's dimensions would essentially result in **full fine-tuning**, as the update matrix would have the same capacity as the original weight matrix.

# QLoRA

Research Paper: [QLORA: Efﬁcient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)

QLoRA is a memory-optimized variant of LoRA (Low-Rank Adaptation) designed to significantly reduce the VRAM requirements for finetuning Large Language Models.

#### Quantization

Before explaining QLoRA, I would like to explain Quantization in brief.

In neural networks the parameters of the model are mostly represented in float. By default, PyTorch stores weights in float32.

IEEE 754 is the Technical Standard for Floating Point Arithmetic. It defines how computers represent and calculate fractions.

> Before this standard, every computer manufacturer (IBM, DEC, Intel) had their own way of doing math, meaning a program could give different results simply by being run on a different machine.

![](Pasted%20image%2020261108202022.png)

Reference: [Float example - IEEE 754 - Wikipedia](https://en.wikipedia.org/wiki/IEEE_754#/media/File:Float_example.svg)

8 bits form 1 byte. float32 represents 32 bits, which consumes 4 bytes of memory per parameter. A 32-billion parameter model consumes 128 GB of VRAM, which exceeds the 80 GB capacity of a single NVIDIA H100.

$$  
32,000,000,000 \times 4 = 128,000,000,000 \text{ bytes}  
$$

To fix this issue researchers came up with an idea of instead of loading models in float32 why not load the weights in lower bits like FP16, FP8, which will dramatically consume **less** space.

|**Format**|**Bytes per Param**|**Total Size (32B Model)**|
|---|---|---|
|**float32**|4|128 GB|
|**float16 / bfloat16**|2|64 GB|

Note: Model weights are not always in a single standard format. In many modern architectures, **Mixed Precision** is used: some layers or operations are kept in higher precision (like Float32) to maintain stability, while others are downscaled to lower precision (like Float16 or Int8) to save memory.

Representing weights in FP16 or FP8 consumes significantly less space, but does it make the model perform worse?

In practice, it rarely makes the model "worse" in a way that humans can easily detect. While there is a slight mathematical loss in precision (a tiny bit of performance degradation), the model’s core intelligence and reasoning usually remain intact.

![](Pasted%20image%2020261108202039.png)
Reference :[Slava_web-designer - Pixabay](https://pixabay.com/users/slava_web-designer-39623293/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=8668140)

We all know that images are represented by pixels with RGB values. Different RGB compositions result in different colors.

**Original Image:** It is colorful, bright, and has smooth gradients.

**Quantized Image:** It may lack the enriched color depth of the original, but it still clearly delivers the same content.  
To fit massive models onto consumer-grade hardware, this tiny trade-off in "color" (precision) is a necessary and highly effective sacrifice for massive memory savings.

> **Analogy:**  
> if someone asks your height, will you say 175.482 cm or 175 cm?  
> Most people prefer 175 cm because the extra decimals aren't necessary for the context. In the same way, the performance trade-off in quantization is feasible for the memory savings it provides.

#### Quantization Datatypes and its range

|**Format**|**Bits**|**Exponent Bits**|**Min Positive (Normal)**|**Max Finite Value**|
|---|---|---|---|---|
|**Float64**|64|11|$2.2 \times 10^{-308}$|$1.8 \times 10^{308}$|
|**Float32**|32|8|$1.18 \times 10^{-38}$|$3.4 \times 10^{38}$|
|**BFloat16**|16|8|$1.18 \times 10^{-38}$|$3.4 \times 10^{38}$|
|**Float16**|16|5|$6.1 \times 10^{-5}$|$65,504$|
|**FP8 (E5M2)**|8|5|$6.1 \times 10^{-5}$|$57,344$|
|**FP8 (E4M3)**|8|4|$0.0156$|$448$|

- **BFloat16 (Brain Float):** Google designed this specifically for AI. They took a Float32 and simply chopped off the end of the precision. Because it keeps all 8 exponent bits, it can handle the same massive numbers as FP32, which prevents models from "crashing" (overflowing) during training.
- **Float16 (Half):** This was the old standard. It has more precision (better decimals) than BF16, but its range is tiny. It can't even represent a number larger than **65,504**. In deep learning, gradients often shoot past 65,000, causing FP16 models to fail where BF16 succeeds.

If you want to deep dive into Quantization techniques and Math, I highly encourage you to check out [**Maarten Grootendorst](https://substack.com/@maartengrootendorst)’s blog on Quantization.**

Here is the Link: [A Visual Guide to Quantization - by Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

#### 4-bit Normal Float (NF4):

Neural Networks weights typically follow a Gaussian distribution. Instead of spacing out the 16 possible values of a 4-bit ($2^4$ = 16) number evenly. NF spaces them out according to the curve of a normal distribution. It captures the weight distribution with high precision.

In other words, Consider the weights of the Model in a distribution instead of distributing the weights as a discrete value it tries to represent the values to represent the underlying Gaussian representation

#### Double Quantization:

To map the 16-bit weights of a model to 4 bits, quantization constants are needed as a scaling factor which helps to map it properly. In a Large Language Model the quantization constants itself take up a huge space of memory. To resolve that QLoRA performs a quantization operation on the Quantization constants to shrink them from 32 bit to 8 bit which saves a significant amount of memory which is referred to as Double Quantization (Quantization operation on a Quantization Constant)

#### Paged Optimizers

Paged Optimizers is a memory management trick. QLoRA uses the NVIDIA Unified Memory to move optimizer states to CPU RAM if the GPU runs out of space, preventing Out of Memory (OOM) errors.

> LoRA saves memory by storing the optimizer states and gradients only for the small adapters; the massive base model remains frozen and does not require gradients or optimizer states. This makes the number of trainable parameters significantly lower compared to full finetuning.
> 
> However, the base model must still be loaded into memory in 16-bit or 32-bit precision. A 65-billion parameter model in 16-bit still requires 130 GB of VRAM.
> 
> QLoRA shrinks the frozen 16-bit base model weights down to 4-bit using quantization techniques and quantized datatypes (NF4).
> 
> - **LoRA** reduces the number of trainable parameters.
> - **QLoRA** reduces the memory footprint of the frozen parameters.

# References (Resources):

1. Parameter-Efficient Transfer Learning for NLP - [https://arxiv.org/pdf/1902.00751](https://arxiv.org/pdf/1902.00751)
2. LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS - [https://arxiv.org/pdf/2106.09685](https://arxiv.org/pdf/2106.09685)
3. QLORA:Efficient Finetuning of Quantized LLMs - [https://arxiv.org/pdf/2305.14314](https://arxiv.org/pdf/2305.14314)
4. [https://open.substack.com/pub/maartengrootendorst/p/a-visual-guide-to-quantization?r=4mboei&utm_campaign=post&utm_medium=web](https://open.substack.com/pub/maartengrootendorst/p/a-visual-guide-to-quantization?r=4mboei&utm_campaign=post&utm_medium=web)
5. [https://youtu.be/Bq9zqTJDsjg?si=qHDpTNt7R0IHpu8W](https://youtu.be/Bq9zqTJDsjg?si=qHDpTNt7R0IHpu8W)
6. [https://youtu.be/nhRYyXGkjSU?si=Xpe_vhxDtxXqk7GO](https://youtu.be/nhRYyXGkjSU?si=Xpe_vhxDtxXqk7GO)
7. [https://theneuralmaze.substack.com/t/finetuning-sessions](https://theneuralmaze.substack.com/t/finetuning-sessions)
8. [https://youtu.be/OkEGJ5G3foU?si=6pHMDvf2oNIpWhGK](https://youtu.be/OkEGJ5G3foU?si=6pHMDvf2oNIpWhGK)