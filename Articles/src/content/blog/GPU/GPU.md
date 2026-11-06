---
title: Inside NVIDIA GPUs
description: Understanding GPU kernels
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