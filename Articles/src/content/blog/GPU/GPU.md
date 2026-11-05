---
title: "Inside NVIDIA GPUs"
description: "Understanding GPU kernels"
date: 2026-11-05
category: GPU
---

# GPU
A GPU (Graphics Processing Unit) is a specialized parallel processing hardware designed to perform large amounts of computations simultaneously. It consists of multiple Streaming Multiprocessors (SMs) that execute thousands of threads in parallel, making it suitable for graphics rendering, AI/ML, and scientific computing.

#### GPU vs CPU

Before getting into GPU architecture, understanding the difference between a CPU and a GPU is important. It explains why we need a GPU architecture if we already have a CPU.

CPUs have a smaller number of cores, where each core is designed to handle complex tasks efficiently. When it comes to GPUs, they have a higher number of cores, where each core is optimized for parallel processing rather than handling complex tasks independently.

If a GPU's individual core is less capable of handling complex tasks than an individual CPU core, then how on earth are GPUs considered powerful? It is because of their parallelism architecture, where GPUs use multiple cores to distribute the work across them instead of relying on a single core to handle the entire workload.

If GPUs are powerful, why not use GPUs for all tasks? Instead of engineering a single powerful core, why not use multiple cores to do the work?

GPUs excel at tasks where the processes are independent of each other. Consider the Fibonacci series, where the value of the nth number depends on its previous two Fibonacci numbers. GPUs don't perform as efficiently on tasks that require strict sequential processing. CPUs excel at sequential tasks where each core is powerful. GPUs excel at parallel tasks where multiple cores process independent workloads simultaneously.

CPUs and GPUs are designed for two different purposes, with different architectures to achieve their respective purposes.

Ex: Consider an example of filling a 10×10 matrix, each element with a value of 1.

The CPU can fill each address sequentially, which takes more time compared to parallel processing. The GPU distributes the workload across multiple threads, where each thread fills one or more elements. The GPU wins at these kinds of tasks because multiple elements can be processed simultaneously.

Note: H100 is the most famous and Preferred GPU in the Industry right now. So H100 Architecture is commonly mentioned in some part of the article for reference. Where H100 Architecture can be referred to know about other architectures. Check the later section of the article to know About H100.

## GPU Architecture

Let’s look at the following image

![](Pasted%20image%2020261105214051.png)
Src: [https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/h100-og.jpg](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/h100-og.jpg)

![](Pasted%20image%2020261105214528.png)
#### GPC

GPC (Graphics Processing Cluster) is a hardware component of a GPU. Consider the H100; it has 8 GPCs. Within each GPC, there are TPCs, SMs, and Raster Engines.

The GPU's L2 cache is partitioned into multiple partitions, which are shared across GPCs. The relationship between GPCs and L2 cache partitions depends on the GPU architecture.

The Raster Engine converts geometric primitives into fragments (potential pixels). It performs rasterization, which determines which pixels are covered by the rendered geometry. Texture mapping and other operations are performed during subsequent stages of rendering.

Let's look in detail at what each GPC contains.

![](Pasted%20image%2020261105214145.png)

Some GPC’s in A100 has 9 TPC’s Where some has 8 TPC’s

#### TPC

A TPC (Texture Processing Cluster) in a GPC contains 2SMs in the H100 architecture.

It also contains texture processing hardware, which performs texture sampling, filtering, and related operations used in graphics rendering.

![](Pasted%20image%2020261105214137.png)

### Streaming Multiprocessor (SM)

The Streaming Multiprocessor is the fundamental compute unit and one of the most important components of a GPU. An SM is somewhat equivalent to a CPU core in terms of being a processing unit, but it is much more complex and contains multiple execution units.

An SM contains Tensor Cores, CUDA Cores, Warp Schedulers, a Register File, and L1 Cache / Shared Memory.

I mentioned that an SM is equivalent to a CPU core, but there are other compute units in a GPU that have the name "Core," such as CUDA Cores and Tensor Cores, which perform Floating-Point Operations and Matrix Multiplications, respectively.

In the NVIDIA H100, each SM is partitioned into four processing sub-partitions (SMSPs). Each sub-partition consists of a Warp Scheduler and associated execution resources such as CUDA Cores, Tensor Cores, LSU, and SFU.

The SM also contains shared resources such as L1 Cache / Shared Memory and the Register File, which are organized across the SM's processing sub-partitions.

Let's look at each component one by one.
![](Pasted%20image%2020261105214156.png)

#### CUDA Core

CUDA Cores are the processing units that execute standard floating-point operations, including multiply-add operations such as FMA (Fused Multiply-Add).

Ex: C = A * B + C
#### Tensor Core

Tensor Core is an important component that performs matrix multiplication and matrix-related operations. Tensor Cores were introduced in 2017 with the Volta architecture, which led to modern AI and HPC compute.
Tensor Cores are optimized for GEMM (General Matrix Multiply).

C = AB + αC
Where A, B, and C are matrices and α is a scalar.
Tensor Cores are optimized for performing GEMM operations.
CUDA Cores are used for performing FMA operations.

#### Special Function Units(SFU)
Special Function Units are the components that perform special mathematical(Transcendental) functions such as exponential, logarithm, trigonometric functions (sin, cos), and their inverse functions.

#### Load and Store Units(LSU)
Load and Store Units dispatch requests to execute load and store instructions, allowing threads to read data from and write data to memory.

#### Threads
A Thread is a logical execution entity that uses the GPU to execute its instructions. The Thread structure in a GPU is similar to the execution context in CPU architecture.

A Thread typically contains:
- **Instruction** – Operation to perform.
- **Thread ID** – Identification of the Thread.
- **Registers** - Thread's private variables
- **Execution State** – Current state of the Thread, whether the Thread is active, waiting, or executing.
- **Program Counter** – Holds the address of the next instruction to execute.

An example of a Thread instruction is adding A[i] and B[i] and storing the result in C[i].

#### Warp & Warp Scheduler
A Warp is typically referred to as a group of 32 Threads in NVIDIA GPUs. Why Grouping? It works on the principle of SIMT (Single Instruction, Multiple Threads), where a single instruction is executed across multiple Threads, which is the fundamental principle of parallelism in GPUs.

The Warp Scheduler is a scheduler that selects ready Warps and issues instructions for execution using SIMT.

Let's take the example of a Thread instruction: C[i] = A[i] + B[i].

The CUDA kernel contains the instruction, and multiple Threads execute the same instruction on different data elements. The Warp Scheduler selects a ready Warp and issues the instruction, where the 32 Threads in the Warp execute the instruction across their respective lanes using compute units such as CUDA cores.

### GPU Memory
Memory is needed to store and hold data and instructions required to perform parallel operations.

GPUs are designed from the perspective of high compute throughput. However, data cannot always be supplied to the compute units in a way that allows them to achieve their maximum efficiency. let’s look at how GPU Memory is defined.

Memory has become a hot topic following the surge in consumer memory chip prices.

Like CPUs, GPU memory is designed in a hierarchical structure to balance access latency and capacity.

![](Pasted%20image%2020261105214320.png)

#### SRAM

SRAM is referred to as ON-CHIP Memory because it is located within the GPU chip. SRAM is generally classified into Registers,L0 cache, L1 cache, and L2 cache. SRAM is one of the fastest accessible memories for compute units.

**Registers:**

Registers are the fastest accessible storage locations for compute threads. They are allocated to individual threads and are located within the SM. Each thread uses its registers to store data required for its computations.

**L0 Cache**

L0 cache is generally referred to as an instruction cache in certain GPU architectures. It holds instructions that the compute units need to execute the specified operations. Its implementation and scope depend on the GPU architecture.

**L1 Cache**

L1 cache is located within an SM. It provides low-latency access to data for the threads executing within the SM. Multiple warps within the same SM can access the L1 cache and perform read operations.

**L2 Cache**

L2 cache is a shared cache located within the GPU die. It is shared across multiple GPCs and SMs. Depending on the GPU architecture, L2 cache may be partitioned, with different GPCs accessing different partitions.

To put it in numbers:

4 GPC × 9 TPC × 2 SM = 72 SMs.

In this example, 72 SMs are distributed across 4 GPCs. The L2 cache partitions are shared according to the GPU's cache architecture.

Now you can draw a mental picture that L2 cache is a shared memory located within a GPU die and shared across GPCs, L1 cache is located within an SM and shared across the compute threads within that SM, and L0 cache is a lower-level cache located within a warp in a SM. Registers are allocated to individual threads.

All of them are located within the GPU chip, which is why they are referred to as ON-CHIP Memory. SRAM stands for Static RAM.

#### DRAM (VRAM)

DRAM (Dynamic RAM) is also commonly referred to as VRAM (Video RAM) when used as GPU memory. Traditionally, GPUs are utilized for video graphics such as games, 3D modelling, and graphics, so the term Video RAM fits well.

Compared to ON-CHIP Memory, it is referred to as OFF-CHIP Memory. Why OFF-CHIP Memory? VRAM is usually located outside the GPU die.

It has a higher capacity compared to ON-CHIP Memory, but with higher access latency.

**People often mention DRAM as HBM?**

DRAM already has many names. Is HBM one of those names? No.

HBM stands for High Bandwidth Memory. It is a type of DRAM technology that uses vertically stacked memory dies to provide higher bandwidth.

If you look deeper into the architecture, HBM memory stacks are placed close to the GPU die and connected through a high-bandwidth interface, typically using an interposer. This provides higher bandwidth compared to traditional memory architectures.

The underlying memory technology is referred to as HBM, not VRAM. VRAM is a general term for GPU memory, whereas HBM is a specific type of memory technology.

let’s summarize wiith the speed and capacity:

**Speed (Access Time):**

VRAM < L2 Cache < L1 Cache < L0 Cache < Registers

Faster access means the compute unit can retrieve data in less time.

**Capacity:**

VRAM > L2 Cache > L1 Cache > L0 Cache > Registers

Higher the memory capacity, slower the access.

If you want to describe it in other words:

**Latency:**

VRAM > L2 Cache > L1 Cache > L0 Cache > Registers

Higher latency means the compute unit takes more time to access the data.

Higher the memory capacity, higher the latency.

We have covered the compute and memory aspects of GPUs. Let’s look at some terms that are most commonly used in the context of GPUs.

### CUDA

Most people know CUDA (Compute Unified Device Architecture) as NVIDIA's proprietary parallel computing platform and programming model used to utilize GPUs effectively for compute workloads.

Some may confuse CUDA with CUDA cores from the previous section. Both are two different things with different purposes. CUDA cores are hardware processing units that perform arithmetic operations, whereas CUDA is a software platform used to program and utilize the GPU effectively for compute workloads.

### FLOP (Floating Point Operation)

FLOP is a metric used to measure the computational workload of a GPU. One FLOP is defined as one floating-point arithmetic operation.

One multiplication and one addition operation are considered 2 FLOPs.

Ex: $d = a * b + c$

For a matrix multiplication, consider matrix M (a × b) and matrix N (b × c). The total number of FLOPs required for the matrix multiplication is calculated as:

$FLOPs =$ $2 × a × b × c$

Ex: $A (10 × 20), B (20 × 30)$

$FLOPs = 2 × 10 × 20 × 30 = 12,000\space FLOPs$

There are some other notations that make it hard to remember the actual purpose of each term.

FLOP - Floating Point Operation

FLOP/s - Floating Point Operations per Second

FLOPs (small letter s) - Plural form of FLOP.

FLOPS (all uppercase) - Commonly used to represent Floating Point Operations per Second, a throughput metric. However, FLOP/s is the more precise notation.

### Kernel

If you're aware of how an Operating System works, then you might be thinking of a different meaning here. However, the meaning of Kernel in GPU is different. In GPU programming, a kernel refers to a function.

It is a function that specifies an operation to be performed, such as adding Matrix A and Matrix B. It is a different naming convention.

# NVIDIA GPU Ecosystem

NVIDIA is one of the most popular GPU manufacturers, and its chips are among the most widely used in the market.

NVIDIA introduced the GPU in 1999 and is currently a major player in the GPU market.

NVIDIA has four major categories of GPUs for different purposes:

**Personal Computing (Gaming / Video Editing):** GeForce RTX / GTX

**Data Center GPUs:** Hopper / Blackwell Series

**Professional Visualization:** NVIDIA RTX Professional GPUs

**Embedded and Edge Computing:** NVIDIA Jetson

Let’s look at Data Center GPUs and its generations

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

References:
[https://www.baseten.co/inference-engineering/](https://www.baseten.co/inference-engineering/)
[https://youtu.be/Nn5XlSwVlOw?si=ZDW8cg9XQUYtCw-u](https://youtu.be/Nn5XlSwVlOw?si=ZDW8cg9XQUYtCw-u)
[Inside NVIDIA GPUs: Anatomy of high performance matmul kernels - Aleksa Gordić](https://www.aleksagordic.com/blog/matmul)
[GPU Glossary](https://modal.com/gpu-glossary)
[A Beginner's Guide to NVIDIA GPUs in 2025: Architecture & Components](https://www.cudocompute.com/blog/a-beginners-guide-to-nvidia-gpus)
[NVIDIA GPUs for AI: Architecture and Scaling Explained](https://www.intoai.pub/p/what-every-ai-engineer-must-know-about-nvidia-gpus)
[https://youtu.be/dEZP1qUNTOk?si=o88pvA2CAHeOljaY](https://youtu.be/dEZP1qUNTOk?si=o88pvA2CAHeOljaY)
[Deep Dive Into Nvidia’s “Hopper” GPU Architecture](https://www.nextplatform.com/compute/2022/04/01/deep-dive-into-nvidias-hopper-gpu-architecture/1657433)
[NVIDIA GPU Architecture Explained: A Guide for AI (2026) | GPU Glossary](https://www.aquanode.io/gpu-glossary/gpu-architecture)
[NVLink vs PCIe: Do You Actually Need NVLink? (2026)](https://www.thundercompute.com/blog/what-is-nvlink)