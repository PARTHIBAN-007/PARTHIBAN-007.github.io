---
title: NVIDIA GPUs
description: Understanding high-performance matrix multiplication kernels
date: 2026-11-05
category: GPU
---

## GPU Generations

NVIDIA introduced the first Tensor Core architecture in 2017 with the Volta series. It also follows an approach of naming GPU generations after scientists and releasing new generations approximately every two years. The following picture shows the release timeline and the generations.

Currently, we are in the sixth generation of Tensor Core NVIDIA GPUs.

![](Pasted%20image%2020261105214337.png)

Let’s look at each GPU generation.

#### First Generation (Volta) – 2017

The Volta series was the first Tensor Core GPU architecture released by NVIDIA in 2017. Tesla V100 was the GPU that delivered the throughput of 32 CPUs in 2017.

|GPU|Single Precison(**32 bits**)|**GPU Memory**|**GPU Memory Bandwidth**|
|---|---|---|---|
|Tesla V100 Pcle|14 TFLOPS|32/16 GB HBM2|900GB/sec|

#### Second Generation (Turing) – 2018

Turing is the second generation of Tensor Core architecture, released in the subsequent year (2018). T4 is a Turing architecture GPU. Earlier LLMs like GPT and BERT were trained on the Volta and Turing series of GPUs.

|GPU|Single Precison(**32 bits**)|Mixed Precision (FP16/FP32)|**GPU Memory**|**GPU Memory Bandwidth**|
|---|---|---|---|---|
|Tesla V100 Pcle|8.1 TFLOPS|65 TFLOPS|16 GB|320 GB/s|

#### Third Generation (Ampere) – 2020

The Ampere generation includes different GPUs, such as the A2, A10, A16, A40, and A100.

The following table compares their specifications.

|GPU|Single Precision (FP32)|Mixed Precision (FP16/FP32)|GPU Memory|GPU Memory Bandwidth|
|---|---|---|---|---|
|A2|4.5 TFLOPS|36 TFLOPS|16 GB GDDR6|200 GB/s|
|A10|31.2 TFLOPS|125 TFLOPS|24 GB GDDR6|600 GB/s|
|A16|19.5 TFLOPS*|156 TFLOPS*|64 GB GDDR6 (4 × 16 GB)|800 GB/s|
|A40|37.4 TFLOPS|149.7 TFLOPS|48 GB GDDR6|696 GB/s|
|A100|19.5 TFLOPS|312 TFLOPS|40/80 GB HBM2(e)|1,555/2,039 GB/s|

#### Fourth Generation (Hopper, Lovelace) – 2022

In 2022, NVIDIA released two different GPU architectures in the same year for two different purposes.

Hopper is the most well-known and commonly used GPU architecture in the industry. It is also one of the most widely used and available GPU architectures among the other generations.

H100 is designed for high compute efficiency during training, whereas H200 is designed with inference quality in mind. H200 is the first series of inference-focused GPUs, while previous-generation GPUs were more focused on higher compute performance.

LLMs’ ARM (Auto-Regressive Model) architecture suffers during inference (Prefill vs. Decode). The H200 series is primarily focused on handling the Decode phase of LLM inference.

|GPU|**FP16 Tensor Core**|**GPU Memory**|**GPU Memory Bandwidth**|
|---|---|---|---|
|**H100 SXM**|1,979 TFLOPS|80GB|3.35TB/s|
|H200 **SXM**|1,979 TFLOPS|141GB|4.8TB/s|

H100 and H200 provide the same level of compute performance, but improvements in memory and memory bandwidth (mainly bandwidth) make H200 the best choice for inference. Reports ([H200 GPU | NVIDIA](https://www.nvidia.com/en-in/data-center/h200/)) suggest that H200 provides 2× better inference performance compared to the H100 series in LLMs like Llama 2.

H200’s 141 GB of memory is the first NVIDIA GPU to offer higher memory capacity compared to all previous generations.

---

Ada Lovelace GPUs (L4, L40) are designed for running small models and embedding models. They are designed to be affordable.

|GPU|**FP16 Tensor Core(Sparse)**|**GPU Memory**|**GPU Memory Bandwidth**|
|---|---|---|---|
|L4|242 TFLOPS|24GB|300GB/s|
|H200 **SXM**|362.1 TFLOPS|48GB GDDR6 with ECC|864GB/s|

The specifications of L4 and L40 suggest that they are not primarily designed for AI and HPC workloads in data centers. Instead, they are primarily intended for hosting smaller models.

#### Fifth Generation (Blackwell) – 2024

Blackwell is the flagship architecture behind the compute powering all the major AI advancements we see today.

|GPU|**FP16 Tensor Core (Sparse)**|**GPU Memory**|**GPU Memory Bandwidth**|
|---|---|---|---|
|B100|1.75 PFLOPS|192 GB HBM3e|4.10 TB/s|
|**B200**|2.25 PFLOPS|180 GB HBM3e|7.7 TB/s|
|**B300**|2.25 PFLOPS|**270 GB HBM3e**|7.7 TB/s|

B200 is designed for heavy workloads in AI and HPC, whereas B300 is similar to H200, focusing on inference with higher GPU memory capacity rather than higher GPU memory bandwidth.

B100 is still a mystery. NVIDIA announced the B100, but its specifications and other details are not officially available online, and it is not available for sale. B200 and B300 are available as clusters, unlike the Hopper series. However, they can be accessed through online GPU platforms.

#### Sixth Generation (Rubin) – 2026

Rubin is the sixth generation of NVIDIA GPUs and the most recent launch by NVIDIA. Rubin is said to deliver 10× the throughput compared to the previous generation (Blackwell).

|GPU|**FP16 Tensor Core (Sparse)**|**GPU Memory**|**GPU Memory Bandwidth**|
|---|---|---|---|
|Rubin|4 PFLOPS|288 GB HBM4|22 TB/s|

Currently, Rubin is in production, but it is not available for individual purchase or through cloud platforms such as RunPod or Modal.

#### Seventh Generation (Feynman) – 2028

The seventh generation of NVIDIA Tensor Core GPUs is expected to be released in 2028. It is currently named Feynman, referencing the contributions of Richard Feynman to physics.

## Data Center GPU Generations vs Tensor GPU Generation

Wherever I have mentioned the generation, I have provided a note specifying Tensor Core. There is an important distinction here. As I mentioned earlier, NVIDIA introduced Tensor Cores in 2017, but it doesn’t mean that NVIDIA introduced Data Center GPUs in 2017. As AI and HPC workloads rely heavily on Tensor Cores, they get the spotlight, and people assume that the generation started in 2017. However, NVIDIA clearly distinguishes between Data Center GPU Generations and Tensor Core Generations.

Let’s look at the following image to understand the Data Center and Tensor Core GPU Generations.

![](Pasted%20image%2020261105214355.png)

The bottom numbers represent Data Center GPU Generations, while the upper number sequence, starting from Volta, represents Tensor Core GPU Generations.

NVIDIA names its Data Center GPUs after scientists. Here are the actual names of the scientists associated with each GPU Generation.

|GPU Generation|Scientist Name|
|---|---|
|Tesla|Nikola Tesla|
|Fermi|Enrico Fermi|
|Kepler|Johannes Kepler|
|Maxwell|James Clerk Maxwell|
|Pascal|Blaise Pascal|
|Volta|Alessandro Volta|
|Turing|Alan Turing|
|Ampere|André-Marie Ampère|
|Hopper|Grace Hopper|
|Ada Lovelace|Ada Lovelace|
|Blackwell|David Blackwell|
|Blackwell Ultra|David Blackwell|
|Rubin|Vera Rubin|
|Feynman|Richard Feynman|

## NVIDIA CPU

People know NVIDIA as a GPU company, but NVIDIA also produces CPUs. However, these CPUs are not primarily designed for personal computing purposes like Intel or AMD CPUs. GPUs need CPUs to handle tasks such as launching GPU workloads, loading models, managing applications, processing data pipelines, and orchestrating computations, while GPUs perform the heavy parallel mathematical operations.

NVIDIA traditionally relied on CPUs from other companies, but it has developed its own ARM-based CPUs for accelerated computing and AI data centers. These CPUs are designed to handle CPU-intensive workloads such as tool calls, sandboxed code execution, evaluations, data pipelines, analytics, and orchestration.

NVIDIA Grace and Vera are its major data-center CPU platforms.

For example, GH200 represents a Grace CPU and an H200 GPU. Likewise, GB200 represents a Grace CPU and B200 GPUs.

## Multi GPU Instance

GPUs are known for parallelism, but even advanced flagship GPUs cannot run Frontier Models efficiently on a single GPU due to their massive computational and memory requirements. Multiple GPUs are connected together to improve performance.

A fundamental unit of a multi-GPU system is a node, where multiple GPUs (typically 8 GPUs in a standard NVIDIA DGX H100 system) are connected together to achieve higher performance. A cluster contains multiple interconnected nodes.Multiple GPUs within a node are connected through NVLink and NVSwitch.

#### Nvlink:

NVIDIA NVLink is a high-speed GPU interconnect that offers significantly higher bandwidth for multi-GPU systems compared to traditional PCIe-based solutions.

Traditionally, NVLink was introduced to provide high-speed communication between GPUs and CPUs. Later, it evolved to support high-bandwidth GPU-to-GPU communication, enabling efficient data sharing and synchronization across multiple GPUs.

|**Generation**|**GPU Architecture**|**Year**|**Bandwidth**|
|---|---|---|---|
|NVLink 1.0|Pascal (P100)|2016|160 GB/s|
|NVLink 2.0|Volta (V100)|2017|300 GB/s|
|NVLink 3.0|Ampere ([A100](https://www.thundercompute.com/blog/nvidia-a100-pricing))|2020|600 GB/s|
|NVLink 4.0|Hopper (H200)|2022|900 GB/s|
|NVLink 5.0|Blackwell (GB200)|2024|1.8 TB/s|
|NVLink 6.0|Vera Rubin|2026+|~3.6 TB/s|

![](Pasted%20image%2020261105214406.png)

#### NvSwitch:

NVSwitch is an advancement in NVLink technology that solves the multi-GPU interconnection problem in GPU communications. A GPU can be connected through multiple NVLinks, depending on the GPU generation, but connecting every GPU directly to every other GPU creates a complex and expensive interconnection environment.

Consider the example where each GPU is connected to other GPUs using NVLink. GPU1 can communicate with GPU5, GPU6, GPU7, and GPU8, but it may not have a direct connection to GPU2, GPU3, or GPU4. To establish direct connectivity with those GPUs, additional NVLink connections are required.

Already, it looks like a neural network with numerous connections. To solve this connectivity problem, NVSwitch was introduced.

![](Pasted%20image%2020261105214412.png)

NVSwitch solves this by serving as an interconnection fabric between all the connected GPUs, where each GPU can communicate with other GPUs in the network through NVSwitch.

NVSwitch provides high-bandwidth GPU-to-GPU communication, enabling multiple GPUs to communicate simultaneously. NVSwitch provides data transfer at the same speed as NVLink between two GPUs, when using the same generation of NVLink and NVSwitch.

![](Pasted%20image%2020261105214418.png)

#### Infini Band

AI and HPC workloads don't fit within a single node, where multiple nodes can be connected together using InfiniBand.

InfiniBand is a high-speed networking technology that allows multiple GPU nodes to be connected together. It provides high bandwidth and low latency for communication between nodes.

InfiniBand typically provides higher communication latency compared to NVLink, as NVLink is designed for high-speed communication within a GPU node, whereas InfiniBand is designed for communication across multiple nodes.

![](Pasted%20image%2020261105214424.png)

NVIDIA has some new competitors in the GPU and AI accelerator space following the AI boom. AMD, Google TPUs, Intel GPUs, and Groq LPUs are some of the major competitors in this space. In 2025, NVIDIA entered into a non-exclusive technology licensing agreement with Groq, while companies like OpenAI are also looking to design their own custom chips.

Despite the growing competition, NVIDIA continues to lead the market by a significant margin. Let's see how the competition evolves in the future.