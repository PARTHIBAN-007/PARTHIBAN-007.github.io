---
title: Sarvam 30B
description: Technical Deep Dive on Sarvam 30B LLM Architecture.
date: 2026-11-21
category: LLM Architectures
---
Sarvam is building sovereign AI for India. In March 2026, Sarvam released the architectures and open weights of its Sarvam 30B and 105B models. Let's explore the Sarvam 30B architecture to understand how Sarvam designed each component and how its design differs from other LLM architectures.

I assume you have a good understanding of LLM architectures and PyTorch basics, as I'm also going to explain the implementation details through code. If you're not familiar with the internals of LLM architectures, I highly encourage you to check out [LLM Architectures](https://parthiban-007.github.io/blog/#category-llm-architectures), where I've explained the core components of an LLM, including the Attention Mechanism, Mixture of Experts, Positional Encoding, and Normalization.

This blog is based on the implementation details available in [Hugging Face – Sarvam 30B](https://huggingface.co/sarvamai/sarvam-30b/blob/main/modeling_sarvam_moe.py). I've simplified and modified the original code to make the concepts easier to understand.

Feel free to skip the code sections if you're not interested in understanding the implementation details. I know there's only a niche audience interested in code-level implementations, but I highly encourage you to go through the code and its explanations to gain a deeper understanding of the architecture.

# Sarvam 30B LLM Architecture
Sarvam 30B Architecture uses Grouped Query Attention as its attention mechanism to handle the KV cache, and RoPE as a positional encoding mechanism to inject relative positional information into the attention layers rather than using absolute positional encodings. It uses a Mixture of Experts in the MLP block, with each expert designed for a specific purpose rather than acting as a general FFNN—and, importantly, Sarvam's Mixture of Experts uses a grouping concept in expert selection.

Each component of the Sarvam 30B architecture shows many advancements over the 2017 Transformer architecture and the 2018 GPT-1 architecture. In recent days, these have become advanced and common architectures used in open-source LLMs such as Gemma 3 and GPT-OSS. Let’s see each component in detail.
![](Pasted%20image%2020261117084518.png)

## Grouped Query Attention
During inference, autoregressive generation repeats key and value calculations for previous tokens in MHA, so **KV caching** was introduced to store the Keys and Values of each head. However, caching every head's Keys and Values requires huge memory storage. Attention mechanisms like MQA, GQA, and MLA were introduced to balance higher performance with lower storage.

MHA uses separate Keys and Values for each head, which increases the KV cache size. GQA reduces the number of unique Keys and Values compared to MHA by sharing the same Key and Value heads across a group of Query heads. Having multiple groups preserves model performance, while sharing Keys and Values within each group lowers storage. Keeping distinct Queries across all heads—irrespective of groups—further helps maintain strong performance, ultimately shrinking the KV cache without sacrificing much quality.  

Let's look at the following image to understand GQA: heads within the same group share a single Key head and Value head (and their corresponding $W_k, W_v$ projections), while different groups use different Key and Value heads, and Queries remain unique across every head.
![](Pasted%20image%2020261121104205.png)

> X $(batch \space size, sequence \space length , d_{model})$ -> X$(num \space tokens, d_{model})$
> Let's ignore the batch dimension of $X$ for now for easier illustration, where the actual computation techniques along with dimension broadcasting are explained clearly in the code section.

X $(num \space tokens, d_{model})$ passes through the projection weights$(w_q,w_k,w_v)$ across heads and groups to produce Queries, keys and values. The dimension of each head's $(w_q,w_k,w_v)$  is $(d_{model}, head \space dim)$, where the $head \space dim = \frac{d_{model}}{No \space of \space heads}$ and the dimension of resulting query, key and value of each heads are $(num \space tokens , head \space dim)$

Grouped Query Attention differs from MHA mainly in how the heads (and shared KV groups) are set up; after that, it follows MHA's steps to compute attention scores, attention weights, and the context vector.
![](Pasted%20image%2020261121102945.png)
Query represents a set of questions a token asks to learn about other tokens. To answer the Query, Keys provide simple identifying knowledge about each token. Based on the relevance between the question (Query) and the answer (Key), a proportional amount of each token's Value is combined to provide a context-aware representation.

Attention scores, the relevance between tokens are computed by multiplying the Queries $(num \space tokens , head \space dim)$ with the transpose of keys $(head \space dim, num \space tokens)$. Attention scores$(num \space tokens, num \space tokens)$ represent the dot product of each token to other tokens in the sequence. In the attention scores token 10 can attend to all other tokens in the sequence till n but intuitively the token 10 should have attend to tokens before it not after it. ARM LLM's process tokens in left to right like we read we don't look at future tokens. so for each token it can only attend to tokens which are before it and not after it by using attention masks which has -inifity values for future token i+1 ,... n for token i. Attention scores are  by $\frac{1}{\sqrt{head \space dim}}$ of each head to handle the variance of the weights. Attention scores values are not bounded by any range which makes it hard to represent the relation softmax is applied to convert the weights into probabilities which explains how does each token is important shares relation with the other token.


Attention scores represents the relevance between tokens which are computed by multiplying the Queries $(\text{num\_tokens}, \text{head\_dim})$ with the transpose of the Keys $(\text{head\_dim}, \text{num\_tokens})$. The resulting attention score matrix $(\text{num\_tokens}, \text{num\_tokens})$ represents the dot product of each token with every other token in the sequence, and these scores are scaled by $\frac{1}{\sqrt{\text{head\_dim}}}$ to keep the variance of the dot products stable.

In this matrix, token 10 can still attend to all tokens in the sequence up to $n$, whereas it should only attend to itself and the tokens before it, not after it. Autoregressive (ARM) LLMs process tokens from left to right just like how we read, without looking ahead at future words. Therefore, each token $i$ is restricted to attending only to tokens at positions $\le i$ by applying a causal attention mask, which sets future positions ($i+1, \dots, n$) to $-\infty$.

Attention scores are unbounded and hard to interpret directly, **softmax** is applied to convert them into attention weights $(\text{num\_tokens}, \text{num\_tokens})$. Softmax expresses each token's attention in terms of probabilities, ensuring that all attention weights for a token sum up to $1$ and clearly show how strongly it relates to the other tokens.
![](Pasted%20image%2020261121103134.png)

Now the attention weights $(\text{num\_tokens}, \text{num\_tokens})$ represent the relevance between tokens, they are multiplied by the Value matrix $(\text{num\_tokens}, \text{head\_dim})$ to capture a proportional amount of information from each token, generating each head's context vector $(\text{num\_tokens}, \text{head\_dim})$. The context vectors from all heads are then concatenated together, combining them back into a single tensor $(\text{num\_tokens}, d_{\text{model}})$.

Multiple heads are used to extract different perspectives, but those perspectives must be shared with each other to form a complete understanding. Just like players on the same team not only have to play their own game well, but also need to read the situation and communicate with teammates to contribute to the team, the different head perspectives must interact. Concatenation alone does not share these perspectives—it merely places them side by side in a single tensor. To blend and share the insights across all heads, the concatenated context vector passes through an output projection matrix $W_o$ $(d_{\text{model}}, d_{\text{model}})$, producing the final context vector $(\text{num\_tokens}, d_{\text{model}})$.

Grouped Query Attention (GQA) has become a reliable standard in modern LLMs—especially across 7B, 14B, and 30B scale models. For instance, Sarvam uses GQA for its 30B model while moving to Multi-Head Latent Attention (MLA) for its larger Sarvam 105B model.
### Code
Let's see the implementation details of GQA in PyTorch. 

```python
def repeat_kv(x,num_kv_groups):
    batch_size, num_heads , seq_len , head_dim = x.shape
    x = x[:,:,None,:,:].expand(-1,-1,num_kv_groups,-1,-1)
    return x.reshape(batch_size,num_heads * num_kv_groups,seq_len,head_dim)

class SarvamGroupedQueryAttention(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.key_value_heads = config.num_key_value_heads
        assert config.d_model % self.num_heads == 0
        self.head_dim = config.d_model // self.num_heads
        self.num_kv_groups = self.num_heads // self.key_value_heads
        if self.config.qk_norm:
            self.query_rms_norm = RMSNorm(config.d_model,config.rms_norm_eps)
            self.key_rms_norm = RMSNorm(config.d_model,config.rms_norm_eps)
        self.rope = SarvamRotaryPositionalEmbedding(config)
        self.query_key_value = nn.Linear(
            config.d_model, (self.num_heads + 2 * self.key_value_heads) * self.head_dim , bias = config.qkv_bias)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.dense = nn.Linear(config.d_model,config.d_model,bias= False)

  
    def forward(self,x,position_embeddings,attn_mask):
        batch_size, seq_len, d_model = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.view(batch_size,seq_len,self.num_heads + 2 * self.key_value_heads,self.head_dim)
        query,key,value = qkv.split([self.num_heads,self.key_value_heads,self.key_value_heads],dim = 2)
        query = query.transpose(1,2).contiguous() # (batch Size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len,head_dim)
        key = key.transpose(1,2).contiguous()
        value = value.transpose(1,2).contiguous()
        if self.config.qk_norm:
            query = self.query_rms_norm(query)
            key = self.key_rms_norm(key)
        sin, cos = position_embeddings
        query,key = apply_rope_embedding(query,key,sin,cos)
        key = repeat_kv(key,self.num_kv_groups)
        value = repeat_kv(value,self.num_kv_groups)
        attn_scores = query @ key.transpose(-1,-2)  / math.sqrt(self.head_dim)
        attn_scores = attn_scores + attn_mask
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context_vector = (attn_weights @ value).transpose(1,2).contiguous() # (batch Size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads,head_dim)
        context_vector = context_vector.view(batch_size,seq_len,-1) # (batch_size, seq_len, num_heads,head_dim) -> (batch_size, seq_len, d_model)
        attn_output = self.dense(context_vector)
        return attn_output
```

> Have a pen and paper to effectively understand the implementation of GQA.

SarvamGroupedQueryAttention class is defined for the GQA where it initializes 
`num_attention_heads` - No of Heads in the Attention Mechanism  
`num_key_value_heads` - No of Key Value Heads  / No of Groups  
`head_dim` - Dimension of each head and it is computed by $\frac{d_{model}}{No \space of \space Heads}$  
`num_kv_groups` - No of Heads in a group and it is computed by $\frac{No\space of\space Heads}{No\space of\space key\space Value\space Heads}$  
`qk_norm` - RMS Norm for Query and Key Vectors  
`rope` - Rotary Positional Encoding which is applied in the attention mechanism rather than adding in the input tokens like Sinusoidal Encoding.  

let's use some math to understand the configuration of GQA.

```python
num_attention_heads: int = 64
num_key_value_heads: int = 4
```
It means that there are 64 heads in the attention block and 4 groups. so each group contains 64/4 = 16 heads.  
No of Unique $(w_q,w_k,w_v)$ in the attention block are $(64,4,4)$ .   
All 16 heads in the same group will have the same key and value.

w_q, w_k, w_v are initialized as a single layer with the configuration of $(d_{model}, (num \space attention\space heads + 2 * key\space value\space heads ) * head\space dim )$,
num_attention_heads refers to the no of queries , 2 * key_value_heads refers to sum of no of uniue key heads and no of unique value heads and multipled by head_dim. The First num_attention_heads * head_dim is the w_q and the next section of no_of_key_value_groups is the w_k and the remaining is the w_v

$w_o$ is initialized as dense layer of $(d_{model},d_{model})$ and dropout is used to make sure all the weights of the model are contributing to the model's performance instead of putting all the workload on some parameters.

As $w_q,w_k,w_v$ are initialized as a single layer, the x is passed to that layer to get the qkv$(num \space tokens,(num \space attention\space heads + 2 * key\space value\space heads ) * head\space dim) )$,
which is then reshaped to split the heads as $(batch\space size, sequence \space length ,num \space attention\space heads + 2 * key\space value\space heads , head\space dim )$.  

The qkv are arranged as a single dimension. It can split in the 2nd index(head index) into get individual query, key and value based on the head configuration. The first num_attention_heads for query and the next kv_heads are for key and the remaining is for value. 

The heads are considered within a sequence. let's change the dimension to represent the sequence from the head's perspective to perform attention computation  on individual heads.  
Let's take an example,    
Sequence 0 - Head 0 , Head 1, Head 2, Head 3  
Sequence  1  - Head 0 , Head 1, Head 2, Head 3  
it has to change as,  
Head 0 - Sequence 0, Sequence 1  
Head 1 - Sequence 0, Sequence 1  
Head 2 - Sequence 0, Sequence 1  
Head 3 - Sequence 0, Sequence 1  

Now the query, key and value transpose their 1st and 2nd index to make the heads contains sequence.

RopE is gonna applied on Query and key to inject relative position information. Before applying rope, the Query and key vectors are normalized using RMSNorm and then the Positional Encoding is applied which injects the relative positions information of vectors through rotations. 

> Purpose and Implementation of RoPE is clearly explained  in the following section. So For now, assume that it injects Relative distance between tokens.

Currently all the heads in a group share the same key and value. instead of sharing a single key and value for a group. Let's duplicate the key's and values across group's heads so that each head has it's own key and value just like it has the query. repeat_kv() is used to expand the keys and values num_groups * num_kv_heads_per_group separately.

Now attention score is calculated by multiplying the query and key transpose and then scaled by $\frac{1}{\sqrt{head \space dim}}$ and the attn_mask is applied to make sure that the token doesn't have access to future token's represenation. attn_mask is not computed in the attention block which is passed by the Causal LM block. For now, assume that the attn mask is a $(num \space tokens, num \space tokens)$ vector which contains $-\inf$ to not allow attention mechanism to see future tokens.
softmax converts the weights into probabilities and the -inf values to 0. so the future tokens are still masked. Dropout is applied on the attention weights to make sure all weights contribute effectively to the model's performance.

Context vector is computed by multiplying the attention weights and value. Now, we have performed all the operations with respect to head wise, so get back to the original dimension. Let's undo the changes performed for head wise computations. Currently we have arranged the sequence on head, let's get back to sequence by taking transpose between 1st and 2nd index.
(batch Size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads,head_dim).  
contiguous() operation is performed to arrange the memory address in a contiguous order where opertions like Transpose distort the Tensor. In the next step we'are going to apply the view which works on contiguous memory address. So contiguous() operation is important. If the Tensor is disorted reshape can be used to change dimensions.

now let's get back to original shape of the x,
(batch_size, seq_len, num_heads,head_dim) -> (batch_size, seq_len, d_model).  
Now all undo operrations are perfomed. Now we have the context vector which contains the perspectives of all the heads. Now , pass the context vector to the dense layer$(W_o)$ to share the perspectives of the all the heads. The Grouped Query Attention is computed efficiently.
## RoPE

RoPE (Rotary Positional Encoding) injects relative position rather than absolute positional encodings, whereas Sinusoidal Positional Encodings inject absolute positional encodings.

Let's take an example:  
**Ex:** "There is a red car." vs. "A red car"
- `"A" + "pos_2"`, `"Red" + "pos_3"`, `"car" + "pos_4"`
- `"A" + "pos_0"`, `"Red" + "pos_1"`, `"car" + "pos_2"`

Even though the relative distance between `"Red"` and `"car"` is the same in both sentences, their absolute positional embeddings are not the same. To fix this, relative position is encoded.

Rotary Positional Encoding only changes the direction of the vector and does not change its magnitude.

Euler's rotation matrix for an angle $\theta$:

$$ 
R(\theta) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
$$

Where $\theta_i$ is:

$$
\theta_i = \frac{1}{10000^{\frac{2i}{d_{\text{head}}}}}
$$

In this formula, `base` is $10000$, and $d_{\text{head}}$ (`head_dim`) is used instead of $d_{\text{model}}$ because RoPE is applied across heads on queries and keys rather than on token embeddings, where $d_{\text{model}}$ is used.

In RoPE, two indices of the vector are taken and rotated using Euler's formula, and the pair $(x, y)$ is multiplied with $R(\theta)$ to get $x'$ and $y'$.

>**Note:** $x$ and $y$ are not two separate vectors; they are a pair of values from the same vector. The same rotation applies to every pair in the vector.

Expanding the rotation gives:
$$
x' = x \cos\theta - y \sin\theta
$$

$$
y' = x \sin\theta + y \cos\theta
$$

The individual $x$ and $y$ pair is represented in a different structure when expanded into component equations, whereas in vector form it has both $x$ and $y$ together in one vector ($\mathbf{x}$), written as:

$$
\mathbf{x}' = \mathbf{x} \odot \cos\theta + \text{rotate}(\mathbf{x}) \odot \sin\theta
$$

Here, $\mathbf{x}$ is multiplied with $\cos\theta$, and the rotated counterpart with the sign flip is multiplied with $\sin\theta$. Both the $\cos$ and $\sin$ computations are added together to inject relative positional information.

While classical RoPE pairs adjacent elements along the dimension, this implementation moves from adjacent pairs to splitting the vector into two halves (`x1` and `x2` along the last dimension) and pairing corresponding positions across the halves. Splitting into halves aligns cleanly with contiguous tensor slicing and vectorized operations in modern transformer implementations.

For `sin` and `cos`, the position-frequency angles are computed for half the head dimension and concatenated across the two halves so both halves receive the matching `cos` and `sin` values for rotation.

> Sarvam's codebase and documentation include support for partial RoPE (where only a portion of the head dimension is rotated), but Sarvam 30B does not utilize partial RoPE—so it is ignored here for simplicity.

### Code
```python
class SarvamRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.base = config.base
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        positions = torch.arange(config.max_seq_len)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("positions", positions, persistent=False)

    def forward(self,x):
        _ , seq_len , _ = x.shape
        freqs = self.positions[:seq_len,None].float() * self.inv_freq[None,:]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return sin, cos

def rotate_half(x):
    x1 = x[...,:x.shape[-1]//2]
    x2 = x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

  
def apply_rope_embedding(q,k,sin,cos):
    cos = cos[None,None,:,:] #(seq_len,head_dim) -> #(1,1,seq_len,head_dim)
    sin = sin[None,None,:,:]
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed , k_embed
```

`head_dim` sets the size of each attention head, and `base` sets the rotation speed scale (`10000`), saved on the module along with `inv_freq` (the rotation rates for each pair) and `positions` (the token positions up to `max_seq_len`). Both `inv_freq` and `positions` are stored as non-persistent buffers (`self.register_buffer(..., persistent=False)`) so they stay with the module without being saved as model parameters.

`freqs` combines the position indices with `inv_freq` to get the rotation angle for each position, and `emb` joins two copies of `freqs` together so both halves have matching angles. `cos` and `sin` convert those angles into cosine and sine values in the input tensor's data type and return `sin, cos`.

`rotate_half` splits the last dimension into two halves and swaps them with a minus sign on the second half so the rotation formula can be applied cleanly. `apply_rope_embedding` reshapes `cos` and `sin` to match `q` and `k`, then combines them with `q`, `k`, and `rotate_half` to add positional information.

## RMS Norm
Normalization is performed to stabilize activations (Hidden States) that might suffer from vanishing or exploding gradients. RMSNorm is used as the normalization technique in Sarvam 30B, offering a compute- and memory-efficient approach over LayerNorm.

$$
\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}
$$  

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot w
$$


$w$ — Learnable weights  
$\epsilon$ — Epsilon  

In the RMS calculation, the mean of the squares of $x$ is summed with $\epsilon$ (to prevent division-by-zero errors), and the square root is taken in the denominator. The input $x$ is divided by this value and multiplied by the learnable weights $w$. RMSNorm scales $x$ consistently, leading to stable and faster training.

```python
class RMSNorm(nn.Module):
    def __init__(self,d_model, epsilon = 1e-20):
        super().__init__()
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self,x):
        mean_square = x.pow(2).mean(dim= - 1, keepdim =True)
        rms = torch.sqrt(mean_square + self.epsilon)
        return (x / rms) * self.weights
```

The RMSNorm class initializes epsilon and learnable parameters (weights) of size $d_{\text{model}}$ filled with ones, since RMSNorm is calculated along the last dimension ($d_{\text{model}}$) of the input $x$. While torch.ones() creates a plain tensor, PyTorch does not automatically register it as part of the model. Wrapping it in nn.Parameter registers the tensor inside model.parameters() so the optimizer updates it during backpropagation (requires_grad=True by default), state_dict() saves it during checkpointing, and calling model.to("cuda") moves it to the GPU along with the rest of the model.

The mean of the squares is calculated along the last dimension ($d_{\text{model}}$), which normally collapses that dimension. Setting keepdim=True preserves the original dimensions, allowing PyTorch to broadcast for future calculations.

torch.sqrt takes the square root of the mean square plus epsilon, where epsilon acts as a tiny safeguard against dividing by zero if all values in a vector happen to be zero, and the values in $x$ are scaled by the RMS value and the weights.

## Mixture of Experts
Traditional LLMs use a single Feed Forward Neural Network(Expansion Contraction) block, which projects the context vector in a higher dimension (commonly 4x) and down projects it back to the original dimension. It helps the model learn richer representations in a higher-dimensional space and carry that information back to the original dimension.

Frontier labs inherited the router-expert concepts from 1990s research papers and built the Mixture of Experts. Instead of a single FFNN, why not have multiple FFNNs, where each FFNN specializes in distinct patterns or sub-contexts and handles the context well. In Mixture of Experts, where each individual FFNN is called an Expert, and the component that knows which vector should go to which experts is referred to as the Router.
![](Pasted%20image%2020261118190859.png)

Researchers decided to combine experts with general knowledge and experts with specialized knowledge, which led to the **shared expert** design. With shared experts, certain experts always process every token regardless of the router's selection, and the outputs of the shared experts and routed experts are added together so both general and specialized contributions form the final representation.
![](Pasted%20image%2020261118191036.png)

Mixture of Experts has been used by LLMs such as Mistral 8x7B and DeepSeek-V3. DeepSeek-V3 popularized Mixture of Experts in 2025, and since then many frontier labs have adopted MoE as a default component in their LLM architectures with various modifications.

Sarvam has done a great job in the routing mechanism by introducing a grouping mechanism. Instead of selecting top-$k$ experts from all available experts, Sarvam groups the experts first. Within each group, the sum of the top-$n$ experts ranks the groups, and the top-$k$ groups are selected for the next step of routing. Now the router can choose only among the selected groups. Let's say you want to pick a diamond card from a deck of 52 cards. Grouping's top-$k$ helps you pick a card from only the 26 red cards, which gives a higher chance. Grouping's top-$k$ helps pick experts that fit this context better.

![](Pasted%20image%2020261118190947.png)
After selecting the groups, all experts in the selected groups are available to the router. Every expert in those selected groups is considered as an individual expert and gets the chance to be picked by the router.
![](Pasted%20image%2020261118191011.png)

Experts are feed-forward neural networks, as you can see in the picture below.
![](Pasted%20image%2020261118191610.png)
Let's look closely at the expert's FFNN setup and see what is inside the expert.
![](Pasted%20image%2020261118191633.png)

A context vector of shape $(\text{sequence length} \times d_{\text{model}})$ passes through two linear layers (`gate` and `up_proj`) that each project it into a higher dimension (`intermediate_size`). One branch (`gate`) goes through a `SiLU` activation while the other (`up_proj`) has no activation. The `SiLU` output and the `up_proj` output are multiplied together, combining non-linear gating with the projected features into a tensor of shape $(\text{sequence length} \times \text{intermediate\_size})$. Finally, `down_proj` maps that higher-dimensional result back to `d_model`, giving an output of shape $(\text{sequence length} \times d_{\text{model}})$.
### Code
```python
# MLP - FFNN
class Expert(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model,config.intermediate_size,bias = False)
        self.up_proj = nn.Linear(config.d_model,config.intermediate_size,bias = False)
        self.down_proj = nn.Linear(config.intermediate_size,config.d_model,bias=False)
        self.act = nn.SiLU()

    def forward(self,x):
        return self.down_proj(self.act(self.gate(x)) * self.up_proj(x))
```
Expert Class inherits the torch module and defines the layers gate $(d_{model} * intermediate \space size)$
, Up_Proj $(d_{model} * intermediate \space size)$ and down_proj $(intermediate \space size * d_{model})$ and the activation function SILU. All the Linear layers are intialized with no Bias.
let's look at the execution, the Context Vector X $(batch \space size, sequence \space length , d_{model})$ passes through the gate projection which produces an output of $(batch \space size, sequence \space length , intermediate \space size)$ 
and the activation function SILU is applied on it and activation maintains the dimension. The Same Context Vector x passes through the Up_proj which then produces an output of shape $(batch \space size, sequence \space length , intermediate \space size)$ . Now the results of up_proj and gate has same dimension and the multiplication of their respective elements preserves the dimension and remains same. Then it passes through the down_proj which brings the context vector into the initial dimension of X$(batch \space size, sequence \space length , d_{model})$.

```python
class SarvamRouter(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.config = TransformerConfig()
        self.num_experts = config.num_experts
        self.n_groups = config.n_groups
        self.topk_group = config.topk_group
        self.topk = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(config.num_experts,config.d_model))
        self.expert_bias = nn.Parameter(torch.zeros((config.num_experts)),requires_grad = False)

    def reset_parameters(self):
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
  
    def group_limited_topk(self,scores):
        num_tokens , _  = scores.shape
        #(num_tokens, num_experts) -> (num_tokens, n_groups, num_group_experts)
        group_scores = scores.view(num_tokens, self.n_groups,-1).topk(2,dim=-1)[0].sum(dim=-1)
        topk_group_idx = torch.topk(group_scores,k = self.topk_group,dim=-1)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1,topk_group_idx,1)
        score_mask = group_mask.unsqueeze(-1).expand(num_tokens,self.n_groups,self.num_experts//self.n_groups).reshape(num_tokens,self.num_experts)
        masked_scores = score_mask.masked_fill(~score_mask.bool(), float('-inf'))
        probs , top_idx = torch.topk(masked_scores , k = self.topk, dim=-1)
        return probs, top_idx

    def forward(self,x):
        x = x.view(-1,x.shape[-1]) #(batch_size, seq_len,d_model) -> (batch_size * seq_len , d_model)
        logits = F.linear(x.to(torch.float32), self.weight.to(torch.float32))
        scores = torch.sigmoid(logits.float()).type_as(logits)
        routing_scores = scores + self.expert_bias
        probs,  topk_idx = self.group_limited_topk(routing_scores)
        scores = torch.gather(routing_scores,dim=-1, index = topk_idx).type_as(logits)
        topk_weight = scores / (scores.sum(dim=-1,keepdim = True) + 1e-20) if self.topk > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight , logits
```

let's understand the initializations in the Router Expert, then we can dive deep into each component and how it is used in the routing mechanism,
num_experts - No of Experts in the MLP Block
n_groups - No of groups
topk_group - Top K Expert in Each Group
topk - No of Experts per token
routed_scaling_factor - Scaling Factor
weight - Routing Mechanism Linear Layer. Weights are initialized with kaiming_uniform distribution. Shape of this layer is $(d_{model}, num \space experts)$ so that the output of this layer gives value for all the experts.
expert_bias - Bias for Each Expert

```python
num_experts: int = 128
num_experts_per_token: int = 6
n_groups: int = 1
topk_group: int = 1
routed_scaling_factor: float = 2.5
num_shared_experts: int = 1
intermediate_size: int = 8192
```

> The Configuration of Sarvam 30B doesn't utilizes the grouping mechanism(n_group = 1) but let's assume that it utilized grouping mechanism to route tokens to understand the architecure design.

The num_experts are grouped into n_groups. among the n_groups there will be num_group_experts = num_experts/n_groups.

The context vector x is reshaped to $(num \space tokens, d_{model})$ and passes through the Linear Layer and give routing scores for for each num_experts in $(num \space tokens, num \space experts)$ which is then applied with sigmoid to give a value in the range of [0,1].  and then after passing through the layer, the bias is added for each expert.

Now comes the interesting grouping mechanism, the routing scores for each experts grouped into n_groups through the view function
(num_tokens, num_experts) -> (num_tokens, n_groups, num_group_experts)
Let's look at the following where the experts are grouped.
Expert 1 , Expert 2, Expert 3, Expert 4,,,...Expert n  
**Grouping:**  
Group 1  - Expert 1, Expert 2, Expert 3, Expert 4   
Group 2  - Expert 5, Expert 6 , Expert 7, Expert 8  
...................   
Group x - .........., Expert n-1 , Expert n  

Topk[0] - Values of Top K  
Topk[1] - Indices of Top K   

In Each Group Top K Experts are choosen and their sum is added together.
In Group 1, Expert 2 and 3 are choosen as top k and their routing scores are added.
group scores $(num \space tokens, n \space groups)$ contains the sum of top k expert's routing score of each group for all tokens.

Top K groups are selected from the group scores which are computed on each group's topk expert's sum. The Purpose of this grouping mechanism is to select a group of experts which are more suited for the context vector rather than having all the experts. 
To Perform that we know have the top k groups so now we have to neglect the experts of the other groups, To do that, group_mask is created where it is initialized as zeros with the shape of group scores which is $(num \space tokens, n \space groups)$. Scatter function in pytorch fills the value in the src tensor on the specified values. topk_group_idx contains the indices of top_k groups filling 1 in the place on those indices make the group_mask makes it contains 1 on topk groups and 0 on other groups.

To convert back it into experts back the unsqueeze function add a dimension at specificied dimension(-1) which makes it as $(num \space tokens, n \space groups,1)$.  
Expand Function expands the tensor as many times as specified. (num_tokens,self.n_groups,self.num_experts//self.n_groups) makes to expands in the last dimension where the 0 and 1st index are same, 2nd index is expanded as specified so the experts in the groups contains the same values as a group. Experts are still grouped , let's reshape to get back to individual experts$(num \space tokens, num\space experts)$.
masked_fill function fills with the boolean value of score mask which sets the non Top K group's expert to negative infinity.
Now , Top K Experts are chosen for routing from the masked scores and the indices and values of top k experts are returned.

Among the routing scores, gather function chooses the scores based on the topk_idx(Scores of Top K Experts). Scores are normalized and then scaled with router_scaling_factor.
The Router class returns the topk_idx, topk_weights and logits

```python
class SarvamMixtureofExperts(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.config = config
        self.SarvamRouter = SarvamRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.shared_experts = nn.ModuleList([Expert(config) for _ in range(config.num_shared_experts)])

    def forward(self,x):
        batch, seq_len, d_model = x.shape
        topk_idx, topk_weight ,  logits = self.SarvamRouter(x)
        shared_out = sum(expert(x) for expert in self.shared_experts)
        flat_x = x.view(-1,x.shape[-1])
        routed_out = torch.zeros_like(flat_x)
        for i,expert in enumerate(self.experts):
            mask = (topk_idx==i)
            row_idx , which_k = torch.where(mask)
            if row_idx.numel()==0:
                continue
            expert_in = flat_x[row_idx]
            expert_out = expert(expert_in)
            w = topk_weight[row_idx,i].unsqueeze(-1)
            routed_out.index_add_(0,row_idx,w * expert_out)
        routed_out = routed_out.view(batch, seq_len, d_model)
        return shared_out + routed_out
```

Now Let's wire the Expert and the routing mechanism to get the Mixture of Experts Block.

SarvamRouter - SarvamRouter class is initialized to perform the routing operation.
experts - Expert Class is used to initialized the num_experts 
shared_experts - Shared Experts are initialized with the same Expert Class

shared experts are the general experts where all the context vector x passes through the shared experts irrespective of the routing mechanism. Each Shared Expert's output is added together and stored in shared_out.

Let's look at the response of routing experts,
the context vector x passes through the router and gets the topk_idx, topk_weight and logits.
context vector x is flattened to $(num \space tokens, d_{model})$ to perform operations smoothly with router response where is designed regardless of batch size.

From the theorotical perspective you might assume that each token chooses it's expert and go through the selected experts and the outputs are summed up, but in the implementation, each expert is selected and the tokens, which are allocated for the expert are processed which makes the implementation somewhat complex. let's look at that.

For a Expert I, topk_idx$(num \space tokens, topk \space experts)$ contains the allocated experts for each tokens, It Checks whether any token is allocated for the expert i which return True if a token is allocated to the expert or False if not. then the idx of the token is collected using where(). row_idx.numel() contains the number of elements in the tensor. It it is zero then it means that there is no token for the expert i and continue to the next expert. if it contains the tokens for the expert i then the token is selected from x  as expert_in and passes through the expert and get the output as expert_out. The expert_out contains response for the tokens of expert i. Now it is scaled with topk_weight for the specific token and expert. Unsqueeze Operation is performed to match with the expert_out dimension.

index_add_() adds the weight * expert_out for the specific token in the routed_out. routed_out is reshaped back to $(batch\space size, sequence\space length , d_{model})$ and then the shared_out and routed_out are added together.

Mixture of Experts is one of the concept which is easier in theory and somewhat complex to implement.


## Transformer Block
Let's assemble the attention mechanism, Mixture of Experts, RoPE, and normalizations to form the Transformer block.
```python
class SarvamTransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = SarvamGroupedQueryAttention(config=config)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.moe = SarvamMixtureofExperts(config=config)

    def forward(self, x, position_embedding, attn_mask):
        residual = x
        x = self.input_layernorm(x)
        x = self.attn(x, position_embedding, attn_mask)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.moe(x)
        x = x + residual
        return x
```

> I appreciate Sarvam's clear and deliberate naming conventions in its codebase, though here I am following a generalized version of naming so the structure stays familiar and easy to generalize across other LLM Architectures.

`input_layernorm` and `post_attention_layernorm` are initialized as separate `RMSNorm` layers so each has its own learnable weights and independent control rather than sharing weights. Residual connections, introduced in ResNet in 2015, help stabilize gradient flow across deep networks as representations pass through many stacked blocks.

The input `x` is stored in a variable for the residual connection, normalized through `input_layernorm`, and passed into the attention block (GQA) along with position information for RoPE and `attn_mask` for causal masking. The resulting output from `x` is added to the residual and saved for the next residual connection, then normalized through `post_attention_layernorm`, passed through the Mixture of Experts, and combined with the residual values before moving to the next `TransformerBlock`.
## SarvamCausalLM

```python
class SarvamCausalLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = SarvamRotaryPositionalEmbedding(config)
        self.blocks = nn.ModuleList(
            [SarvamTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def causal_mask(self, seq_len: int, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return torch.zeros(seq_len, seq_len, device=device).masked_fill(mask, float("-inf"))

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        x = self.embeddings(input_ids)
        position_embedding = self.rope(x)
        attn_mask = self.causal_mask(seq_len, x.device)

        for block in self.blocks:
            x = block(x, position_embedding, attn_mask)
  
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
```

SarvamCausalLM get the input tokens ids and then the embeddings

`self.embeddings` converts `input_ids` into dense vectors so the model can process them. `self.rope` provides position information, and `causal_mask` sets future positions to negative infinity so attention only attends to earlier tokens and cannot look ahead at future ones.

The representation(x), `position_embedding`, and `attn_mask` pass through multiple `config.n_layers` blocks. Then `self.norm` normalizes the representation to keep values stable before projection, and `self.lm_head` maps it to vocabulary `logits` so the autoregressive model can score candidate tokens and predict the next token across a series of prediction steps as the sequence is generated.

For Complete Code, [AI-Architectures-Foundry](https://github.com/PARTHIBAN-007/AI-Architectures-Foundry/blob/main/Sarvam/sarvam_30B.py).  
For Official Code and Documentation, [Sarvam's Hugging Face](https://huggingface.co/sarvamai/sarvam-30b/tree/main).

Sarvam have done a great work on building the soverign LLM for India and I wish sarvam to be the frontier lab which represents india in the World Stage and hoping for more Open Source(Weight) Models from Sarvam.

References:
1. [Sarvam Hugging Face](https://huggingface.co/sarvamai/sarvam-30b/tree/main)
2. [Vizuara Build LLM From Scratch](https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&si=aFUKl5Qr3Jvel7SH)
3. [Vizuara Deepseek From Scratch](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms)
4. [Maarten grootendorst Mixture of Experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
5. [Zachary Huang RoPE](https://youtu.be/hCzJo4ui1P8?si=h8OO7ns_zs_JN8FF)







