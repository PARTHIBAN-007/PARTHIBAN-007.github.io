---
title: Sarvam 30B
description: Technical Deep Dive on Sarvam 30B LLM Architecture.
date: 2026-11-21
category: LLM Architectures
---
Sarvam is building sovereign AI for India. In March 2026, Sarvam released the architectures and open weights of its Sarvam 30B and 105B models. Let's explore the Sarvam 30B architecture to understand how Sarvam designed each component and how its design differs from other LLM architectures.

I assume you have a good understanding of LLM architectures and PyTorch basics, as I'm also going to explain the implementation details through code. If you're not familiar with the internals of LLM architectures, I highly encourage you to check out [LLM Architectures](https://parthiban-007.github.io/blog/#category-llm-architectures)}, where I've explained the core components of an LLM, including the Attention Mechanism, Mixture of Experts, Positional Encoding, and Normalization.

This blog is based on the implementation details available in [Hugging Face – Sarvam 30B](https://huggingface.co/sarvamai/sarvam-30b/blob/main/modeling_sarvam_moe.py). I've simplified and modified the original code to make the concepts easier to understand.

Feel free to skip the code sections if you're not interested in understanding the implementation details. I know there's only a niche audience interested in code-level implementations, but I highly encourage you to go through the code and its explanations to gain a deeper understanding of the architecture.

# Sarvam 30B LLM Architecture
Sarvam 30B Architecture uses Grouped Query Attention as it's attention mechanism to handle the KV cache and RoPE as positional encoding mechanism to inject the relative positional encoding in the Attention Mechanisms rather than using Absolute Positional Encodings. Mixture of Experts in the MLP Block with each expert designed for specific purpose than a general FFNN and importantly Sarvam Mixture of Experts uses Grouping Concept in Expert Selection.  
Sarvam 30B Architecture's each component has many advancement from the 2017 Transformer Architecture and the 2018 GPT 1 Architecture. In the recent days these are the advanced and the most common architectures used in the Open Source LLMs such as Gemma 3, GPT Oss.
let's see each component in detail.
![](Pasted%20image%2020261117084518.png)

## Grouped Query Attention
Traditional Multi head Attention have a compute bottleneck which was handled by the KV cache, but KV Cache comes up with a storage tradeoff. To have a balance between MHA Performance and with lesser KV Cache Storage , mechanisms like MQA, GQA and MLA are introduced. Sarvam utilizes Grouped Query Attention to balance the Performance and Efficient Storage. Grouped Query Attention has been a more reliable architecture in the modern LLM's especially for the 7B,14B, 30B Models.Sarvam 105B LLM uses MLA.

Let's Focus on Grouped Query Attention.
KV cache found the repeated calculation steps(In Inference) in  MHA and comes up with an idea to cache the Keys and Value of each heads. Caching Each heads' key and Value increases the storage. Grouped Query Attention is proposed to have a group where each head in a group shares the same key and value so it introduces a balance of performance and efficient storage. Where MHA uses different keys and values for each heads which offers higher performance, MQA uses same keys and values for all heads which offers efficient storage.  
let's look at the following to understand the GQA that heads in a same group has a same key($w_k$) and same value($w_v$) and different keys and values with the other group heads. Query's are different across heads irrespective of groups
![](Pasted%20image%2020261121104205.png)

> X $(batch \space size, sequence \space length , d_{model})$ -> X$(num \space tokens, d_{model})$
> Let's ignore the batch dimension of x for now for easier illustration, where the actual computation techniques along with dimension broadcasting are explained clearly in the code section. 

X $(num \space tokens, d_{model})$ passes through the $(w_q,w_k,w_v)$  different heads of different groups which then produces Queries, keys and values. The dimension of each head's $(w_q,w_k,w_v)$  is $(d_{model}, head \space dim)$ where the $head \space dim$ is defined by $(\frac{d_{model}}{No \space of \space heads})$ and the dimension of query, key and value of each heads are $(num \space tokens , head \space dim)$

Grouped Query Attention differs from MHA only from the heads initialization and after that it follows the MHA for the computation of Attention Scores, Attention weights and context vector. 
![](Pasted%20image%2020261121102945.png)
Queries $(num \space tokens , head \space dim)$ are multiplied with the transpose of keys $(head \space dim, num \space tokens)$ which produces an attention scores$(num \space tokens, num \space tokens)$ which is then masked using attention mask and scaled by $\frac{1}{\sqrt{head \space dim}}$ of each head and softmax is applied to convert the weights into probabilities which forms the attention weights
![](Pasted%20image%2020261121103134.png)

Attention weights $(num \space tokens, num \space tokens)$ are multiplied with the value $(num \space tokens , head \space dim)$ to produce context vector $(num \space tokens, head \space dim)$. Context vectors of different heads are concatenated together which brings back the multiple heads to a single head of $(num \space tokens, d_{model})$. 

Multiple Heads are used to extract different perspectives but those perspectives has to be shared with each other to get a better understanding. Players in a same team not only has to play their game better but also needs to understand the situation, communicate with the other players to contribute to the team. So perspectives has to be shared.
Concatenation doesn't share the perspectives it only put the perspectives in a single place. To share perspectives of each heads the context vector passes through a $W_o$ $(d_{model},d_{model})$ and produces the final context vector$(num\space tokens, d_{model})$.

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
        query,key = apply_rope_embedding(query,key,cos,sin)
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
num_attention_heads - No of Heads in the Attention Mechanism  
num_key_value_heads - No of Key Value Heads  / No of Groups
head_dim - Dimension of each head and it is computed by $\frac{d_{model}}{No \space of \space Heads}$  
num_kv_groups - No of Heads in a group and it is computed by $\frac{No\space of\space Heads}{No\space of\space key\space Value\space Heads}$  
qk_norm - RMS Norm for Query and Key Vectors  
rope - Rotary Positional Encoding which is applied in the attention mechanism rather than adding in the input tokens like Sinusoidal Encoding.  

let's use some math to understand the configuration of GQA.

```python
num_attention_heads: int = 64
num_key_value_heads: int = 4
```
It means that there are 64 heads in the attention block and 4 groups. so each group contains 64/4 = 16 heads.  
No of Unique $(w_q,w_k,w_v)$ are $(64,4,4)$ .   
Each Groups's 16 heads will have the same key and value.

w_q, w_k, w_v are initialized as a single layer with the configuration of $(d_{model}, (num \space attention\space heads + 2 * key\space value\space heads ) * head\space dim )$
num_attention_heads refers to the no of queries , 2 * key_value_heads refers to sum of no of key heads and no of value heads and multipled by head_dim. The First no of queries * head_dim is the w_q and the next section of no_of_key_value_groups is the w_k and the remaining is the w_v

$w_o$ is initialized as dense layer of $(d_{model},d_{model})$ and dropout is used to make sure all the weights of the model are contributing to the model's performance instead of putting all the workload on some parameters.

As $w_q,w_k,w_v$ are initialized as a single layer, the x is passed to that layer to get the qkv$(num \space tokens,(num \space attention\space heads + 2 * key\space value\space heads ) * head\space dim) )$,
which is then reshaped for splitting the heads as $(batch\space size, sequence \space length ,num \space attention\space heads + 2 * key\space value\space heads , head\space dim )$.  

The qkv are arranged as a single layer it can split in the 2nd index(head index) into get individual query, key and value based on the head configuration. The first num_attention_heads for query and the next kv_heads are for key and the remaining is for value. 

The heads are considered within a sequence. let's change the dimension to represent the sequence from the head's perspective to perform attention computation  on individual heads.  
Let's take an example,    
Sequence 0 - Head 0 , Head 1, Head 2, Head 3  
Sequence  1  - Head 0 , Head 1, Head 2, Head 3  
it has to changed as,  
Head 0 - Sequence 0, Sequence 1  
Head 1 - Sequence 0, Sequence 1  
Head 2 - Sequence 0, Sequence 1  
Head 3 - Sequence 0, Sequence 1  

Now the query, key and value transpose their 1st and 2nd index to make the heads contains sequence.

RopE is gonna applied on Query and key to inject relative distance information. Before applying rope, the Query and key vectors are normalized using RMSNorm and then the Positional Encoding is applied which makes sure to inject the relative positions of vectors through rotations. 

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
RoPE(Rotary Positional Encoding) injects relative position rather than the absolute positional encodings. Sinusoidal Positional Encodings injects the absolute positional encodings.
let's take a example.  
Ex: There is a red car.  
A Red car  
A + "pos_2" , Red + "Pos_3", car + "Pos_4"  
A + "Pos_0" , "Red" + "Pos_1", car + "Pos_2".  

Even though the relative distance between red and car is same between both the sentence but their positional embeddings are not same. To Fix this, relative position is encoded.

Rotary Positional encoding only changes the direction but doesn't change the magnitude.
Euler's Rotation:
$$  
R(θ) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}  
$$
Where $\theta_i$,
$$

\theta_i = \frac{1}{10000^{\frac{2i}{d_{\text{model}}}}}

$$

In RoPE, two indices of the vector are token and rotated using euler's formula and the vector x , y is multiplied with $R(\theta)$  to get x' and y'.

Note: x and y are not two different vectors it is the consecutive set of indices of a same vector. It is one of the pair of the vector. Likewise all the pairs of the vector are taken and rotation is calculated using euler's rotation formula 

![](Pasted%20image%2020261108202248.png)

x' = x cos$\theta$ - y sin$\theta$   
y' = ysin$\theta$  + ycos$\theta$ 

it can be represented as (x y)cos$\theta$  + (-y x)sin$\theta$ 
x is directly multiplied with cos$\theta$ 
x is cut into half and rotate the halves and add a negative sign and then multiply with sin$\theta$ 
Both cos and sin computation are added together to inject relative positional encoding
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

  
def apply_rope_embedding(q,k,cos,sin):
    cos = cos[None,None,:,:] #(seq_len,d_model) -> #(1,seq_len,1,d_model)
    sin = sin[None,None,:,:]
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed , k_embed
```

head_dim - Instead of $d_{model}$ RoPE uses head_dim as base because it is applied to each head's query and key.
base - base of the theta's power.(10000)
inv_freq - computes the $\frac{1}{\theta ^ {\frac{2i}{head \space dim}}}$ . torch.arange gives the value of 0 to head_dim with 2 steps. It ignores the sequence because the sin is applied on even position and cos on odd position for the set of 2 indices. For a pair one value is enough.
positions is the context length of the LLM defined by max_seq_len and it is stored in buffer.
positions and inv_freq are multiplied together to form the frequencies. which is the concatenated for both even and odd indices. sin and cos of freqs are generated for the position.

apply_rotary_embedding applies the rotation of queries and key matrix using the sin and cosine values. sin and cos values are elongated for pytorch broadcasting. rotate_half function get the vector and split it into two half based on the last dimension(head_dim) and project the second at the first place and the first half at the second half place by concatenation and add negative sign to represent the formula.

Now the x is multiplied with cos$\theta$ and the rotated half is multipled with sin$\theta$. Now Both Query and key's have relative positional information which is considered in the attention mechanism to give the relative importance.

## RMS Norm

Normalization is performed to stabilize the weights which might be suffer from Vanishing or Exploding Gradients. RMS Norm is used as normalization in Sarvam 30B. It is compute and memory efficient Normalization technique.

$$ RMS(x) = \frac {1} {\sqrt{\frac{1}{n} \sum_{i=1}^{n} x^2 + \epsilon}}$$


$$RMSNorm(x) = x * RMS(x) * w $$

$w$ - Learnable weights  
$\epsilon$ - Epsilon

In the RMS Calculation the mean of the squares of x is summed up with epsilon(To Handle Division by Zero Error) is Root Squared in the denominator which is then multiplied with the input x and learnable weights x. RMS Scales the x and leads to stable and faster training.
### Code
```python
class RMSNorm(nn.Module):
    def __init__(self,d_model, epsilon = 1e-20):
        super().__init__()
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self,x):
        variance = x.pow(2).mean(dim= - 1, keepdims =True)
        rms = torch.rsqrt(variance + self.epsilon)
        return x * rms * self.weights
```

RMSNorm Class initializes epsilon and learnable parameters(weights) of $d_{model}$ shape with ones as RMS Norm is calculated along the last dimension$(d_{model}$)  of the Vector x. 
torch.ones() creates a tensor but that tensor is not included as parameters of the model when the model.to("cuda") is applied all the parameters of the model goes to GPU. Tensor still says in the same CPU, nn.Parameter helps the tensors to goes to the GPU.

Variance(mean of the squares) is calculated along the last dimension(d_model) which typically results in a $(batch \space size , sequence \space length)$. Keepdims = True helps to return a dimension of $(batch \space size , sequence \space length , d_{model})$ where the calculated variance is repeated across the last dimension. Dimension is preserved to match the further operations with x.

rsqrt - Reciprocal Square Root, which is the inverse of the square root of a number. Here the calculated variance is in the inverse of a square root where epsilon is a tiny value which is used to handle the division by zero(sqrt of variance might be zero) error.

## Mixture of Experts
Traditional LLM's uses a single Feed Forward Neural Network(Expansion Contraction) block which projects the context vector in a higher dimension(4x) and down project it back to the original dimension. It helps to learn in a higher dimension and carry forward the information back to the original dimension. 

Frontier labs inherited the Router Expert Concepts from 1995 research paper and build the Mixture of Experts. Instead of a single FFNN why not have a Multiple FFNN each FFNN is specialized in a domain and handles the context well. It is the Mixture of Experts where the individual FFNN is called as Expert and the one which knows what context should go to which experts is referred as Router.
![](Pasted%20image%2020261118190859.png)

Researchers decided to use a combination of a expert with general knowledge and a expert with specialized knowledge which leads to shared expert. In Shared Expert, some experts always process the task regardless of the router's selection and the results of Top K Router Experts and shared expert's are combined and proceed to the next layer.
![](Pasted%20image%2020261118191036.png)

Mixture of Experts has been used by LLM's like Mistral 8 x 7b , Deepseek v3.
Deepseek v3 has popularized the Mixture of Experts in 2025 and after that many Frontier Labs have MoE as a default component in their LLM Architectures with some variations.

Sarvam has did a fantastic job in the routing mechanism by introducing Grouping Mechanism. Instead of selecting top k experts from the available experts. Sarvam groups the experts first and among the groups the top n experts's sum is used to rank the groups. Top K groups are selected for next process of routing. Now the router can choose only among the selected groups. let's say you want to pick a diamond card from the deck of 52 cards. Grouping's Top K helps you to pick a card from only red cards of 26 which gives a higher probability. Grouping's Top K helps to pick a router which is more qualified for this context.
The following images illustrates the Grouping Mechanism.
![](Pasted%20image%2020261118190947.png)
After selecting the groups all the experts in the selected group are common expert available for the router. There is no bias of choosing a expert from a group which has higher ranking based on the Top K Experts Sum. All Experts are considered as individual expert which doesn't have any group influence. Experts in the top group might not be selected by the router at this stage.

![](Pasted%20image%2020261118191011.png)

Experts are nothing but a Feed Forward Neural Network as you can see in the below picture.
![](Pasted%20image%2020261118191610.png)
Let's look deep into the Expert's FFNN Configuration. what's inside the Expert.
![](Pasted%20image%2020261118191633.png)

Context Vector $(sequence \space length * d_{model})$ which passes through two different Linear layers each process them to project in higher dimension $(d_{model} * intermediate \space size)$ where one layer has activation SiLU and the other one doesn't have any activation. the output of the two linear layers are added together it Provides a combination of non linearity(activation) and linearity which produces an output of $(sequence \space length * intermediate \space size)$ which is then down projected with a linear layer $(intermediate \space size * d_{model})$ which extracts the information projected in the higher dimension and represent in the same input dimension and the resultant output as ${sequence \space length * d_{model}}$ .
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

Let's assemble the Attention Mechanism, Mixture of Experts, RoPE and Normalizations to form the Transformer Block. 
```python
class SarvamTransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = SarvamGroupedQueryAttention(config=config)
        self.post_attention_layernorm = RMSNorm(config.d_model,config.rms_norm_eps)
        self.moe = SarvamMixtureofExperts(config=config)

    def forward(self,x,pos,attn):
        residual = x
        x = self.input_layernorm(x)
        x = self.attn(x,pos,attn)
        x = x + residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.moe(x)
        x = x + residual
        return x
```

> All the ARM(AutoRegressive Model) architecture LLM's uses transformer's decoder block as their main block. But i personally feel to name it as decoder block or decoder layer but all the major Frontier Labs uses Transformer Block as the name. But when i see sarvam they have given more importance to variable naming such as using this Block as DecodeLayer and mentioning x as hidden_states, causal lm and many more. I respect Sarvam's naming conventions but TransformerBlock has been more general so i stick to the general term.

RMSNorm is initialized separately so that the input_layernorm and post_attention_layernorm has different weights and have more control instead of normalizing with a same weight.

Residual Connection is introduced in the ResNet architecture in 2015 to handle the vanishing gradient. As Transformer Block has many layers of processing, residual connection is needed to handle the vanishing gradient.  

The Input x is stored in a variable for residual connection and passes through the input_layernorm to normalize it and passes through the attention block(GQA) along with the position information which is used in RoPE and attn_mask for causal masking. Context Vector from x is added with residual connection and then it is stored for next layer of residual connection and goes to the post_attention_layernorm to normalize the attention values and passes through the Mixture of Experts and then adds the residual values and goes to the next layer of TransformerBlock.

## SarvamCausalLM

```python
class SarvamCausalLM(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size,config.d_model)
        self.rope = SarvamRotaryPositionalEmbedding(config)
        self.blocks = nn.ModuleList(
                    [SarvamTransformerBlock(config) for _ in range(19)]
                )
        self.norm = RMSNorm(config.d_model,config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model,config.vocab_size,bias = False)

    def causal_mask(self,S: int,device):
        mask = torch.triu(torch.ones(S,S,device = device,dtype = torch.bool),diagonal=1)
        return torch.zeros(S,S,device=device).masked_fill(mask,float('-inf'))

    def forward(self,x):
        B,S,D = x.shape
        x = self.embed(x)
        position_embedding = self.rope(x)  
        mask = self.causal_mask(S,x.device)
        for block in self.blocks:
            x = block(x,position_embedding,mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
```

SarvamCausalLM get the input tokens ids and then the embeddings

rope will give the positional embedding(sin and cos) for the position and causal mask is a 2D tensor which is filled with negative inf on future positions which makes sure that the Attention Mechanism can only look at the previous tokens not the future tokens.
The Input X along with positional_embedding and causal_mask is passed to the transformer block which process the x on multiple transformer blocks. After going through the transformer block it is then normalized using RMSNorm and the lm_head which projects the context vector to give the weights for each vector in the vocabulary and then predicts the next vector. The ARM(Auto Regressive Model) continues to predict the next tokens.

For Complete Code, [AI-Architectures-Foundry](https://github.com/PARTHIBAN-007/AI-Architectures-Foundry/blob/main/Sarvam/sarvam_30B.py).  
For Official Code and Documentation, [Sarvam's Hugging Face](https://huggingface.co/sarvamai/sarvam-30b/tree/main).

Sarvam have done a great work on building the soverign LLM for India and I wish sarvam to be the frontier lab which represents india in the World Stage and hoping for more Open Source(Weight) Models from Sarvam.

References:
1. [Sarvam Hugging Face](https://huggingface.co/sarvamai/sarvam-30b/tree/main)
2. [Vizuara Build LLM From Scratch](https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&si=aFUKl5Qr3Jvel7SH)
3. [Vizuara Deepseek From Scratch](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms)
4. [Maarten grootendorst Mixture of Experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
5. [Zachary Huang RoPE](https://youtu.be/hCzJo4ui1P8?si=h8OO7ns_zs_JN8FF)







