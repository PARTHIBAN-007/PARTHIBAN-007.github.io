---
title: MLP(FFNN and MoE)
description: Attention Mechanism, Mixture of Experts
date: 2026-11-08
category: LLM Architectures
---
# MLP

In every Transformer-based LLM architecture, there are two core components: the **Attention Mechanism** and the **Multi-Layer Perceptron (MLP) block**. The Attention Mechanism processes token representations and produces context-aware representations by determining how much each token should attend to other tokens in the sequence. The MLP block then applies non-linear transformations to each token representation independently, enabling the model to learn more complex patterns, abstractions, and semantic relationships

## Feed Forward Neural Network

Feed Forward Neural Network is the MLP Block in core transformer architectures which is an expansion contraction block.

![](Pasted%20image%2020261108203009.png)

FFNN is an Expansion and contraction layer. context vectors from the attention mechanism is passed to the FFNN which then projects the context vector in a higher dimension vector space and try to understand complex patterns, understand semantic relationship then it abstracts it down the information and passes it as output.

let the dimension of the context vector is ${d_\text{model}}$ then the dimension of the expansion layer is $4 \times {d_\text{model}}$. In this example ${d_\text{model}}$ is 512 and dimension of Expansion layer is ${4 \times 512 = 2048}$ and then it projects it back to the ${d_\text{model}}$ which is 512.  
Now you can relate why I mentioned FFNN as expansion and contraction block.

## Mixture of Experts (MoE)

In the transformer architectures FFNN is used to understand complex patterns and semantic relationship but researchers found that language is not limited to one domain it is more diverse. language contains domains like grammar, technology, education, biology, finance, economics, history, Mathematics and much more. A single feed forward Neural Network can’t handle all these domains so researchers thought why only one FFNN? why not multiple FFNN? each should be well versed in one domain. Consider Expert 1 is for preforming tasks related to finance, Expert 2 is for Grammar. Likewise, each Expert block becomes good at one particular task which makes the LLM to be great at multiple domains. Also, one of the core features of MLP is that it activates only the parameters of a particular expert and the other experts’ weights are not activated which makes the LLM to have less no of parameters activated at inference.

Mixture of Experts is the reason for why modern LLMs are named like **Qwen/Qwen3.5-397B-A17B** where the total number of parameters of this model is 397 billion but only 17 billion Parameters are activated at inference time.

Mixture of Experts is an expert router mechanism which was introduced in 1991 as a supervised learning procedure for systems composed of many separate networks [https://ieeexplore.ieee.org/abstract/document/6797059](https://ieeexplore.ieee.org/abstract/document/6797059)

Mixture of Experts (MoE) is called a Sparse Model because, for a given token, only a subset of experts is activated and participate in the computation. The remaining experts are not used for that token. In contrast, a standard MLP with a single Feed-Forward Neural Network (FFNN) is called a Dense Model because every token passes through the same FFNN and all its parameters are involved in the computation.

let’s look at the following diagram to get better understanding of Mixture of Experts

![](Pasted%20image%2020261108203018.png)

Context vector from the attention mechanism is passed to the Mixture of Experts Block. Router mechanism decides which token should pass to which expert. Each Token is routed to the expert independently. One Token may be passed to multiple experts and their abstracted information from the experts is combined. why is one token passed to multiple experts? same token might be relevant in multiple domains.

![](Pasted%20image%2020261108203026.png)

Each Expert is nothing, but a FFNN which is specialized in one domain. Let’s Consider Expert 1 is for grammar, Expert 2 is for agriculture, Expert 3 is for Finance.

![](Pasted%20image%2020261108203032.png)

As we know Mixture of Experts is a router expert mechanism and each expert is specialized in a particular domain and an expert is nothing, but a FFNN. let’s look at the implementation of Mixture of Experts

![](Pasted%20image%2020261108203039.png)

Context vector: $sequence \space length \times d_{\text{model}}$ which is $(8 \times 6)$  
Router Matrix: $d_{\text{model}} \times { No \space of \space Experts}$ which is $(6 \times 4)$   
No of Experts: 4    
Top K: 2 (2 Experts are selected out of 4 Experts for each token)  

Context vector is multiplied with router matrix which gives the router logits for each token to all experts $sequence \space length \times { No \space of \space Experts}$ then the shape of the router logits is $(8 \times 4)$. From the Router logits top k values selected for each token. if top k is 2 then all tokens should route 2 experts out of 4. let’s consider token 1 which should route to expert 2 and expert 4 whereas token 2 should route to expert 1 and expert 4. The remaining values in Top K Experts logits are masked by setting the values to $-\infty$ so that $e^{-\infty}$ becomes 0 when SoftMax is applied to convert values into probabilities.

After Masking, SoftMax converts values into routing probabilities which gives how much important the token should provide to a particular expert. Let’s consider token 1 which have Top K Expert Routes with [0,0.7,0,0.3] then it means that the output of Expert 2 should contribute 70% and Expert 4 should contribute 30% while combining the outputs of Experts.

Tokens will be routed to experts in this order:
Token 1: Expert 2, Expert 4  
Token 2: Expert 1, Expert 4  
Token 3: Expert 2, Expert 3  
Token 4: Expert 1, Expert 3  
Token 5: Expert 2, Expert 4  
Token 6: Expert 1, Expert 2  
Token 7: Expert 3, Expert 4  
Token 8: Expert 1, Expert 3  

### Shared Experts

Shared Experts are experts where all the context vectors are pass through these expert layers. shared experts are experts which learns general knowledge, common patterns and good at multiple domains. Output of Shared Experts are combined with routed experts’ output to produce the final output.

![](Pasted%20image%2020261108215119.png)

Shared Experts are completely independent of router. whatever are the router logits, all context vectors will pass through the shared expert. A MLP Block may contain multiple shared networks.

Tokens will be routed to experts in this order:
Token 1: Shared Expert, Expert 2, Expert 4  
Token 2: Shared Expert, Expert 1, Expert 4  
Token 3: Shared Expert, Expert 2, Expert 3  
Token 4: Shared Expert, Expert 1, Expert 3  
Token 5: Shared Expert, Expert 2, Expert 4  
Token 6: Shared Expert, Expert 1, Expert 2  
Token 7: Shared Expert, Expert 3, Expert 4  
Token 8: Shared Expert, Expert 1, Expert 3  