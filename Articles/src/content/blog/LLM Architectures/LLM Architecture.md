---
title: LLM Architectures
description: Attention Mechanism, Mixture of Experts
date: 2026-11-08
category: LLM Architectures
---

Hello everyone,

In this article, I will explain the core architecture of LLMs, and the architecture used in GPT-OSS. An LLM consists of several key components, and this article covers Positional Encoding, the Attention Mechanism, Normalization, and MLPs.

### GPT-OSS Components

- **Positional Encoding** — RoPE (Rotary Positional Embedding)
- **Attention Mechanism** — Grouped Query Attention (GQA)
- **Normalization** — RMSNorm
- **MLP** — Mixture of Experts (MoE)
![](Pasted%20image%2020261108202210.png)

I have structured this article to progress from traditional implementations of these components to the more advanced techniques used in GPT-OSS. The goal is to build a solid understanding of the fundamentals before exploring how these concepts are optimized and applied in modern LLM architectures.

# Positional Encoding

In the Transformer architecture, tokens are processed in parallel, and the self-attention mechanism does not inherently encode positional information. When it comes to language, position information defines the grammar and meaning of the sentence. Same sentence with different position arrangement will completely change the meaning of the sentence.

For example:

Sentence 1: “Dog bites Man”

Sentence 2: “Man Bites Dog”

Here both the sentences contain the same words [” Dog”,” bites”,” man”] but position of the words determine the meaning of the sentence. Traditional architectures such as RNNs and LSTMs process tokens sequentially, allowing positional information to be implicitly captured through the recurrent flow of information.

So, researchers decided to encode positional information with the input so that the transformer architecture can understand the position information and understand the meaning of the sentence.

## Sinusoidal Positional Encoding

Transformer architectures use sinusoidal positional encoding to incorporate positional information into token representations. Since the self-attention mechanism itself is permutation-invariant, positional information must be explicitly provided to the model. Sinusoidal positional encoding is deterministic and contains no trainable parameters. Sinusoidal embeddings have the same embedding dimensions as tokens, and the positional embeddings are added to the token embeddings and then passed to the Transformer block. There is no change in dimensions.

Ex: Sentence “Dog chased another dog”. The text is tokenized using a tokenizer which converts the text into tokens. Each token is mapped to a token ID, which is used to look up its corresponding embedding vector from the embedding matrix. Likewise, positional encodings for positions 0, 1, 2, and 3 are formulated using the following deterministic formulas and then added to the input token embeddings. Sinusoidal embeddings have a deterministic formula, so whatever the text is, position 0 gets the same positional encoding, position 10 always gets the same positional encoding, which helps the Transformer capture positional information.
![](Pasted%20image%2020261108202221.png)
In practice, an LLM's context length is determined by the maximum sequence length used during training and the positional encoding mechanism. For learned positional embeddings, the positional embedding table defines the maximum supported position. Sinusoidal positional encodings can be computed for arbitrary positions, but models often generalize poorly beyond the context lengths seen during training.

$$

PE_{\text({pos,2i})} = \sin({\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}})  
$$

$$  
PE_{\text({pos,2i+1})} = \cos({\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}})  
$$

where,

pos - position of the token (indexing starts from 0)

i - index of the embeddings (Even Index - Sin, Odd Index -Cosine)

$d_{\text{model}}$ - Dimension of Embedding
![](Pasted%20image%2020261108202230.png)
Let’s get into the math of the formula, researchers defined the sinusoidal positional encoding formula in such a way that it was deterministic and easier for attention mechanism to extract positional information. Sine and cosine functions are deterministic and continuous periodic functions, making them suitable for representing positions in a smooth and structured manner which will help the attention mechanism for extracting positional information.  
As positional encoding is added with token embeddings there has to be pattern for the position information which was defined based on the position and the indices of the position embedding. The value 10000 is a scaling constant chosen to spread frequencies across embedding dimensions. As a result, each position is represented by a unique combination of frequencies, allowing the model to distinguish positions across both short and long ranges. Lower embedding dimensions vary rapidly with position and capture fine-grained local position information, while higher dimensions vary more slowly and capture broader positional patterns across longer distances.

Using both sine and cosine provides complementary signals that help distinguish positions and capture relative positional relationships. An important property of sinusoidal encoding is that the encoding of position p+k can be represented as a linear function of the encoding at position p. This arises from trigonometric identities of sine and cosine and allows the attention mechanism to learn relative positional relationships between tokens.

## RoPE (Rotary Positional Encoding)

Sinusoidal positional encoding encodes positional information along with the input embeddings, but it also modifies the original input embeddings.

Ex: Sentence1: The dog chased the red car. Sentence 2: The red car.

Sentence 1: “the” + pos_3, “red” + pos_4, “car” + pos_5

Sentence 2: “the” + pos_0, “red” + pos_1, “car” + pos_2

“The red car” has the same meaning in both sentences, but the embeddings are different because of the way of addition of position embedding. Relative distances between the tokens influence the meaning of the sentence which was not captured by absolute positional encoding techniques like sinusoidal positional encoding.

Rotary Positional Encoding was introduced to change the angle of token embeddings rather than adding positional embeddings to the token embeddings. It doesn’t add the position embeddings along with the token embeddings. Instead, it injects the position information in the attention mechanism (where the position matters) by rotating the Q and K vectors (which are used in attention score calculation)

Euler’s Formula:

$$  
e^{i\theta} = \cos(\theta) + i\sin(\theta)  
$$

Euler’s Rotation Matrix:

$$  
R(θ) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}  
$$

let’s consider an example $v = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$ and $\theta = 90^\degree$

$$  
then, R(90^\degree) = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}  
$$

$$  
R(90^\circ)v = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ -2 \end{bmatrix} = \begin{bmatrix} 2\\ 1 \end{bmatrix}  
$$

The rotation matrix performs the rotation. RoPE only changes the angle, not the magnitude. Let’s check the magnitude of the vector. Magnitude v (1, -2) = $\sqrt{1^2 + (-2)^2} = \sqrt{5}$ and v (2, 1) = $\sqrt{2^2 + 1^2} = \sqrt{5}$. The vector has the same magnitude before and after rotation.

$$  
Attention Score = f(Q,K)  
$$

$$  
Attention Score = f(Q,K,m-n)  
$$

RoPE makes the attention score depend on the relative distance (m−n) between tokens.

$$  
R_nR_m^T=R_{n−m}  
$$

RoPE injects the relative position information between the tokens in the Attention mechanism. In the attention mechanism, Query and Key vectors are rotated according to their positions:

$\text{Attention Score} = Q^TK$

$Q_m = R_mQ$

$K_n = R_nK$

The attention score becomes: $Q_m^TK_n$

Using the property of rotation matrices:

$$  
(R_m^TQ^T)(R_nK)  
$$

$$  
Q^TR_m^TR_nK  
$$

$$  
Q^T R_{n-m} K  
$$

The final attention score depends on (n-m), which is the relative distance between the tokens rather than their absolute positions. This allows the model to naturally capture relative positional relationships while preserving the magnitude of the embeddings.

Angular frequency is inspired by sinusoidal positional encoding

$$

\theta_i = \frac{1}{10000^{\frac{2i}{d_{\text{model}}}}}

$$

where 10000 is used to generate different frequencies across dimensions so that different positions can be distinguished. 10000 is not the only number to be taken as base. it is a hyperparameter. Engineers can choose any value which makes sure that no two positions have same values. Some modern LLMs use larger base values such as 50000 to support longer context lengths.

Euler’s Matrix Rotation formula is only for 2D. Positional encoding is in higher dimension. So, the values are split into pairs, where each pair is rotated by multiplying the pair using Euler’s formula. Each pair uses a different angular frequency, allowing the model to capture positional relationships at multiple scales.
![](Pasted%20image%2020261108202248.png)

# Attention Mechanism

Attention Mechanism in Large Language Models are used to find the importance of one token to other tokens in the sequence. It tries to find the relationship between the other token in the sequence. let’s consider the example “The cat is on the mat, and it is eating”, As humans we understand the sentence easily. But LLMs doesn’t understand it easily. It has to understand that “it” refers to object/thing, animal, event and in this context, it is animal and that animal is cat and it is on the mat and it is also eating the food. LLMs has to look at mulitple angles to understand the meaning of the sequence.

Several Attention techniques are used by different large language labs. We are going to see the core attention techniques which eventually lead to the Grouped Query Attention and how OpenAI implemented Attention (GQA) in GPT OSS.

## Self-Attention

Self-Attention mechanism tries to find the relationship between tokens and how important is one token to other tokens in the sequence. Self-Attention tries to represent the input embeddings in a context rich format which the LLM use to understand relation and importance effectively.

Self-Attention uses 3 trainable matrices such ${W_q}, {W_k}, {W_v}$ which tries to get information from 3 perspectives from the input matrices.

Q - What information do I need?
K - What kind of information do I contain?
V - What information should I provide if selected?

![](Pasted%20image%2020261108202256.png)

Dimensions of ${W_q}, {W_k}, {W_v}$ are $d_{\text{model}} \times d_{\text{model}}$ which is ${6 \times 6}$. $d_{\text{model}}$ should be the first dimension to satisy matrix multiplication rule with input matrix last dimension $d_{\text{model}}$ but the second dimension is subjective but preferred to have $d_{\text{model}}$ ; as most of the Large Language Model Labs prefer to have $d_{\text{model}} \times d_{\text{model}}$ as the dimensions for these weight matrices.

${W_q}, {W_k}, {W_v}$ matrices are multiplied with input matrix to get Query, Key, value Matrix respectively. Dimension of Query, Key and Value are ${sequence \space length \times d_{\text{model}}}$ which is ${8 \times 6}$
![](Pasted%20image%2020261108202307.png)

Attention Score is calculated by multiplying Query and Transpose of Key Matrix, Key Matrix is transposed to satisfy matrix multiplication conditions. Dimension of attention score is ${sequence \space length \times sequence \space length}$ which is ${8 \times 8}$ . Attention Scores are having different values, to control the covariance, the Attention scores are divided by $\sqrt{d_k}$ . where, ${d_k}$ refers to the last dimension of the keys Matrix. Attention score should tell how important is one token to other token in the sequence as the the values of attention scores are in different range SoftMax is applied to convert values into probabilities which sum to 1 so the probability makes more sense for interpretation Ex: Token 1 gives 40% importance to Token 3 and 10% importance to Token 10. These mathematical operations convert attention scores into attention weights which preserves dimension, so the dimension remains same as ${sequence \space length \times sequence \space length}$ which is ${8 \times 8}$.

Context Vector is produced by multiplying the Attention Weights with Value Matrix. Attention Matrix tells how important that token in the context and value matrix gives the value of the token which makes context vector. Dimension of Context Vector is ${sequence \space length \times d_{\text{model}}}$ which is ${8 \times 6}$. Context vector matrix has same dimension as Input matrix.

## Causal Attention

In the Self Attention mechanism , it tries find the importance of tokens and represent the relation between one token and the other token in the sequence, it was good for context understanding and tasks like Summarization, Language Translation where the LLM should know the complete context, but for tasks like text generation LLM shouldn’t look at future tokens.

Causal Attention is also known as Masked Attention.

As LLMs are auto regressive models which generates one token at a time.  
Ex:

Hello → How

Hello how → are

Hello how are → you

Let’s look at from the perspective of LLM training which makes the causal attention more insightful. As LLMs are Auto regressive the training data is generated by having the text as input and the output (Target) as the output

Ex: Sentence: Large language Models are neural network trained on large amounts of text data.

Input: Large Language Models are neural

Output: language Models are neural networks.

In the Training Phase when “large Language” is passed to the LLM it has to predict “Model”, when “large language Models are” is passed to the LLM it should predict “neural”.

Transformer Block process all tokens at once so that the input sentence “large Language Models are neural” is processed at once and attention scores are calculated for all tokens in self-attention but the LLM should predict tokens only based on what it knows till the last token of the input but it doesn’t peak into future as the training data contains information about future tokens. So masking is applied for tokens which appears in future.

![](Pasted%20image%2020261108202321.png)

After the calculation of attention scores, masking is performed. Dimension of Attention score is ${sequence \space length \times sequence \space length}$ which is ${8 \times 8}$. Future tokens are masked by setting the values for future tokens (tokens in the upper triangle of the attention score matrix which are values above the main diagonal) to ${-inf}$. As SoftMax is going to applied in the next step ${e^\text{-inf}}$ = 0 makes the attention weights as 0 for future tokens.

Masked Attention preserves dimension so the resultant dimension will be same as attention scores ${sequence \space length \times sequence \space length}$ which is ${8 \times 8}$

## Muti Head Attention

Multi Head Attention is an extension of Self Attention mechanism, where self-Attention captures only one perspective. Multi Head Attention tries to capture multiple perspectives with the help of multiple heads. In the Context of Multi Head attention, self-attention is considered as a single head attention. In the Implementation of Multi Head Attention, it doesn’t just create Multiples Queries, Key and Values it just tries to keep the weights, assigns the weights (split the weights) for each head, perform the attention calculation and combines multiple perspectives to get better understanding of the sequence.  
Like Self Attention ${W_q}, {W_k}, {W_v}$ are having the dimension of $d_{\text{model}} \times d_{\text{model}}$ .

No of heads will capture n perspectives. No of heads is calculated based on the ${d_{\text {model}}}$ and the dimension for each head.  
let’s say ${d_{\text {model}}}$ is 768 and no of heads is 24.

then the head dimension will be $\frac{{d_{\text {model}}}}{head\space dim}$. ->$\frac{768}{24}$ = 32 will the head dimension.

if we look to define the dimension per head then the number of heads will be decided by $\frac{{d_{\text {model}}}}{ No\space of\space Heads}$

let’s look at the following illustration to understand how it looks in practice.

![](Pasted%20image%2020261108202329.png)

Input Embeddings has the dimension ${sequence \space length \times d_{\text{model}}}$. In this example it is ${8 \times 6}$

Dimensions of the ${W_q}, {W_k}, {W_v}$ are $d_{\text{model}} \times d_{\text{model}}$.

No of Heads is defined as 2. then the dimension of heads will be $\frac{{d_{\text {model}}}}{head\space dim}$ = $\frac{6}{2}$ = 3.

Query, key and Value are calculated independently for each head and the resultant Query, Key, value will have the dimension of ${sequence\space length \times head \space dim}$ which is ${8 \times 3}$.
![](Pasted%20image%2020261108202335.png)
Query and Key Transpose (to satisfy matrix multiplication conditions) are multiplied to get attention score which gives information about how one token is important to other tokens. Dimension of Attention Score is ${sequence \space length \times sequence \space length}$ which is ${8 \times 8}$.

Attention Score is calculated for each head using Query and Key independently and then techniques like masking, dividing by $\sqrt{d_k}$ and SoftMax are performed on attention score matrix to convert it into attention weights for each head. Attention weights have the same dimension as attention scores.
![](Pasted%20image%2020261108202346.png)

Attention weights are multiplying with values matrix where the attention matrix has weights which denotes how important is one token to another and value matrix provides value of how much of that value is carried forward which results in context vector matrix for each head which has the dimension of ${sequence \space length \times head \space dimension}$ which is ${8 \times 3}$.

Each head has its context vector which are then combined. In our example we have 2 heads each has the dimension of ${8 \times 3}$ if we combine it will result in ${8 \times 6}$ as combined context vector matrix.

The context vector is again multiplied with a projection matrix ${W_O}$, As perspectives from n heads are combined as a whole matrix but it is not understood the multiple perspectives as a whole, projection matrix tries to understand the multiple perspectives and project it as an abstract. $W_0$ has the dimension of $d_{\text{model}} \times d_{\text{model}}$ Ex: ${6 \times 6}$ and the resultant final context vector matrix will have the dimension of ${8 \times 6}$ which is the same dimension as input. Attention Mechanism will produce the context vector matrix with same dimension as input so that it can proceed to next block without any change in dimensions.

## KV Cache

LLMs are autoregressive models. let’s look at the following example where “Large language” is the input to the LLM, it predicts the word “Model” which is then appended the input and passed to the LLM as “Large Language Model” which then predicts the word “is” then it is appended with the input as “Large Language Model is “ and LLM predicts the word “a” and this process continues to happen till it reaches the End of the sentence token or the LLM reaches the Maximum Token generation Limit. As the same prompt with new token is send to the LLM there will be some computation which are executed in the previous steps are computed again.

![](Pasted%20image%2020261108202353.png)

Let’s look at the internal working to understand how the LLM processes the input. In this example 4 tokens are passed to the LLM each has the embedding dimension of 6. Query, Key and Value matrix are calculated by multiplying the input X with the $W_q , W_k , W_v$ matrices respectively. Dimensions of Query, Key and Value are ${4 \times 6}$. Attention scores, Attention weights and context vector are computed and then context vector will proceed to the next layers and finally it will predict the token.

![](Pasted%20image%2020261108202401.png)

Now a Token is predicted by the LLM which is appended with the input which makes the input as $(5 \times 6)$. For better illustration purpose only, the new tokens are colored where the previous tokens are not colored in X, Q, K and V matrices. $W_q, W_v, W_k$ matrices are trainable matrices which is common for all inputs. Query, Key, Value matrices are computed by multiplying input $X$ with $W_q,W_k,W_v$ matrices respectively. Now look at the Q, K and V matrices only the last row is computed where the other rows are same as values computed in the previous steps. we’re computing the same value again and again which needs more computation and increases latency. In Software development when things are computed repeatedly and it produces the same result then caching is the go-to choice so let’s cache Q, K and V matrices only calculate the Q, K and V for the new token which will reduce the computation significantly. Before, Caching Q, K and V matrices look at the next steps such as attention scores and masked attention. As attention scores are calculated between Q and K^T ,both of these matrices are cached let’s look how it is going to affect the attention score, it turns out that only the last token is compared with all the tokens in the sequence for attention score (the last row) and tokens in the sequence to the last token are compared for attention score(the last column) but after masking the attention scores, masked attention needs the new values of only the new token similarity with other tokens not the other tokens similarity with last token. So only the last row has to be computed again. let’s look at how it is computed, we need last row of Q and complete K Matrix for the computation of last row of masked attention.

![](Pasted%20image%2020261108202410.png)

let’s look at the following math to understand how?

Attention scores calculation for the Input $X (4 \times 6)$ are placed in the left half of the Image. after these calculation LLM will predict the next token Now, X will become $(5 \times 6)$ attention scores calculation is same as the previous pass computation except the last row and last column. But, because of masking the last column values (except the last value in the last column) are set to infinity. now the masked attention has same values as previous pass causal attention, and the last row of causal attention is the row to be calculated. if we look at it, we can see that the values are $q5*k_1, q_5*k_2 , q_5*k_3, q_5*k_4, q_5*k_5$ respectively. As Only the last vector of the new token is used in attention scores, we don’t have to cache Queries. Key Matrix is needed as the calculation needs all K vectors and also the value vector is needed for context vector calculation. So Key and Value Matrix are cached.

![](Pasted%20image%2020261108202417.png)

KV Cache is an inference technique not a training mechanism.

Keys and Values are cached; computation is comparatively lesser which will increase inference speed. KV Cache is an excellent optimization technique. But the issue with KV cache is the storage size.

let’s consider the LLM configuration as:

$KV\space Cache = 2 \times L \times T \times H_{kv} \times d_{head} \times b$

2 - Denotes K and V

L - No of layers (Transformer Block) (32 Layers)

T - Sequence Length (16384 Tokens)

$H_{kv}$ - No of KV Heads (64 heads)

$d_{head}$ - Dimension of each head (128 dimension per head)

b - Bytes Per Parameter (2 bytes- FP16)

MHA: 2 * 32 * 16384* 64* 128 *2 = 17,179,869,184 bytes

approximately 16 GB Storage for KV cache for this configuration.

## Multi Query Attention

Storage size of KV Cache was high; to solve this storage issue researchers come up with the technique of using same key and value for all heads so we have to cache only one key and value, and the storage size will be reduced by a huge amount.

Muti Query attention has the same working principle of multi head attention but here the ${W_k}$ and ${W_v}$ are same across heads which results in same Key and Value across heads. Multi Query Attention is highly efficient in terms of KV Cache storage size, but it doesn’t offer the same performance as Muti Query Attention.

let’s look at the following image for illustration,
![](Pasted%20image%2020261108202646.png)

Same colors represent same weights and Different colors represents different weights.

As ${W_k}$ and ${W_v}$ are same across heads. Wk1, Wk2, Wk3 and Wk4 have same colors (same weights) and Wv1, Wv2, Wv3 and Wv4 have same colors (same weights). As ${W_k}$ and ${W_v}$ are same the resultant Keys and Values are also same across heads which are also represented in same color.Even though the Keys and Values are same. Attention scores, context vector are different for each head because of the different values of Query per head. Attention scores, Attention Weights, Context Vector as calculated as same as MHA.

$KV \space Cache = 2 \times L \times T \times H_{kv} \times d_{head} \times b$

$H_{kv}$(No of KV Heads) becomes 1 in MQA.

MQA: 2 * 32 * 16384* 1 * 128 *2 = 268,435,456bytes

approximately 256 MB of KV Cache Storage. MQA reduces KV Cache significantly.

MQA is not as good as MHA in terms of performance.

## Grouped Query Attention

As MQA lacks the performance like MHA, Grouped Query attention was introduced to find the balance between Storage (MQA) and Performance (MHA).

Instead of having the same ${W_k}$ and ${W_v}$ across heads, Grouped Query Attention introduces grouping concept where the total number of heads are grouped together and heads within a same group will have same ${W_k}$ and ${W_v}$ .

Let’s consider the following image for illustration purpose.

![](Pasted%20image%2020261108202656.png)

Total No of Heads are 4, No of groups are 2, each group contain 2 heads.

Group 1 contains Head 1 and Head 2, Group 2 Contains Head 3 and Head 4. Therefore Head 1 and Head2 in Group 1 will have the same ${W_k}$ and ${W_v}$ and Head 3 and Head 4 in Group 2 will have the same ${W_k}$ and ${W_v}$ which eventually results in same K and V within the groups. Attention scores, Attention Weights, Context Vector as calculated as same as MHA.

Same colors represent same weights and Different colors represents different weights.

Ex: K1 and K2 in Group 1 have same weights which are represented in same color Q1, Q2, Q3, Q4 have different weights which are represented in different colors.

Grouped Query Attention tries to capture multiple perspectives like MHA with grouping approach as different groups capture different perspectives.

Grouped Query Attention tries to save KV cache storage space like MQA by having the Same Key and Value for the heads in a same group.

$KV \space Cache = 2 \times L \times T \times H_{kv} \times d_{head} \times b$

$H_{kv}$(No of KV Heads) becomes no of groups of in GQA = 8

MQA: 2 * 32 * 16384 * 8 * 128 *2 = 2,147,483,648 bytes

approximately 2 GB of KV Cache Storage. GQA KV Cache is lower compared to MHA and but still high compared to MQA.

Storage: MQA < GQA < MHA

Performance: MQA < GQA < MHA

Goal: Storage should be Minimum and Performance should be Maximum.

Grouped Query Attention tries to find a sweet spot between MQA and MHA.

## Sliding Window Attention

Sliding Attention is a masking technique like Causal Attention. Masking is performed in causal attention to make sure that it doesn’t looks into the future. when it comes to sliding window attention the intuition is to have a local attention where most of the tokens are related to nearby tokens rather than the tokens which are located at far. Attention mechanisms like MHA, GQA, MQA are considered as global attention and sliding window attention is considered as local attention.

let’s illustrate the sliding window attention with the following image. Where the attention scores are calculated only for the sliding window where the tokens before the sliding window are not considered for attention scores.
![](Pasted%20image%2020261108202820.png)
Sliding window attention is computationally lower compared to global attention mechanisms. A global attention computes attention score in $n^2$ time complexity where sliding window attention computes in $n*w$ complexity. Most of the LLM architectures doesn’t use sliding window attention for all transformer blocks instead they use alternate Global and Sliding window attention to make sure global attention focuses on complete context and sliding window focuses on nearby tokens.

## Attention Mechanism in GPT Oss

Let’s get into how attention mechanism is implemented in GPT Oss. GPT Oss uses Grouped Query Attention with sliding window attention (Alternate layers of Global and Sliding Attention). Looks same as other LLM architectures right, but use of attention bias and attention sinks makes GPT Oss attention mechanism different from other LLM architectures.

### Attention Bias

In most of the LLM architectures, the bias parameter is set to false in the $W_q, W_k, W_v$ Matrices while initializing, but OpenAI researchers decided to have bias in these matrices. Usage of Bias doesn’t lead to significant performance in any benchmarks, but they prefer to have bias in attention mechanism.

### Attention Sink

Attention sink is really an interesting approach than Attention Bias. Researchers found that attention scores in the deeper layers in LLM gives importance to the initial tokens than other token in the research paper [EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS](https://arxiv.org/pdf/2309.17453).

let’s look at the image from the research paper for a better understanding. If you look at the attention score matrix represented as heatmap, the main diagonal values get more importance because the diagonal values represent the attention score between a token and itself. but if you look at the deeper layers the initial tokens get more importance which is not good. To fix this, researchers came up with an approach of attention sinks. Sink is introduced to absorb the extra importance which was given to the initial tokens than it should get.

![](Pasted%20image%2020261108202830.png)

Source: EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINK

let’s look at the following image for how Attention sink is implemented. Attention Score is calculated between Q and K matrices and then masking, SoftMax are applied to convert into attention weights. As attention score have the issue of initial tokens get more importance attention sink is introduced in this process. you can see that the initial column gets more attention score than the rest of the other columns. To fix this, attention sinks which has a dimension of $(sequence \space length \times Number \space of \space sinks)$ is $(6\times 1)$. Attention sinks are trainable parameters which absorb the extra importance got by the initial tokens. Attention sink is combined with attention scores which absorbs the extra importance, SoftMax is applied to get attention weights. Now the tokens have an understanding of how much important one should give to other tokens. After SoftMax the attention sink is discarded, it carries away the extra importance. Note: The sum of the attention weights is not equal to 1 as the attention sink is removed but the attention weight still carries information of how much importance should the token to other token.

![](Pasted%20image%2020261108202837.png)

Numerical values in the above image are random values used for better illustration. GPT Oss uses GQA with Sliding window attention. sliding attention is not used in the image to not make this complex.

# Normalization

In Deep Neural Network architectures there are multiple layers let’s say there are 100 layers at each layer the weights get multiplied, added and passed through activation functions. After passing many layers,

Weights (Numbers) become very small - Vanishing Gradient Problem

Weights (Numbers) become very large - Exploding Gradient Problem

It leads slower training, unstable network, sometimes no training. To solve this issue normalization is performed which keeps the weights in stable healthy range not too big, not too small which leads to stable learning and faster training.

## Layer Norm

Layer Normalization performs centering and scaling on the values to make the values normalized.


$$ layerNorm(X) = \gamma \frac{x - \mu}{\sqrt{\sigma^2+ \epsilon}} + \beta $$


$\mu$ - Mean (Average of the Vector)

$\sigma^2$ - Variance (Spread of the vector)

$\epsilon$ - Small Number (Avoids Division by Zero Error)

$\gamma$, $\beta$ - Learnable Parameters

$x- \mu$ : performs centering the values around zero

$\sqrt{\sigma^2 + \epsilon}$ : Scales the values by dividing with standard deviation which makes it to have a constant spread

Why Centering and Scaling? Why keep only one of them? → RMS Norm

## RMS Norm

RMS is simpler version of Layer Normalization. It only performs scaling.

RMS Norm - Root Mean square Normalization


$$ RMS(x) = \sqrt{\frac{\sum_{i=1}^{n} x_i^2}{n} + \epsilon} $$

$$RMSNorm(x) = \gamma \frac{x}{RMS(x)} $$

RMS(x) - Root Mean Square of the values of vector

$\gamma$ - Learnable Parameter (As RMS makes it in a fixed range, $\gamma$ makes the values to stretch or shrink)

$\epsilon$ - Small Number (Avoids Division by Zero Error)

Note: RMS Norm is not always best compared to Layer Norm. When it comes to LLMs, residual connections make RMS Norm to perform better because residual connection handles the centering of values.

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

## GPT OSS MoE:

Different Language labs use different approach. when it comes to OpenAI’s GPT OSS they have chosen SwiGLU as their activation function in the Expert’s FFNN. GPT 2 FFNN contains GELU/SWIGLU as activation function.

### SwiGLU:

SWiGLU is an activation function which is combined of Swish and GLU.

#### Swish

$$  
swish(x) = x * \sigma({\beta x})  
$$

$$  
Swish(x) = \frac{x}{1+e^{-\beta x}}  
$$

#### GLU (Gated Linear Unit):

GLU splits the input into two, one is passed through a linear transformation and other is passed through another linear transformation and then a activation function. The output of each layer is a linear projection modulated by the gates $\sigma({xW+b})$. Similar to LSTMs, these gates multiply each element of the matrix and control the information passed on in the hierarchy

$$  
GLU(x,W,V,b,c) = \sigma({xW+b}) *({xV+c})  
$$

SwiGLU uses GLU with Swish as activation function instead of sigmoid.

$$  
SwiGLU(x,W,V,b,c) = swish({xW+b}) *({xV+c})  
$$

![](Pasted%20image%2020261108215133.png)

FC1 of GPT OSS FFNN process the input sequence which has the dimension of $sequence \space length \times d_{\text{model}}$ which produces the output of $sequence \space length \times 2 *d_{\text{model}}$. Typical FFNN produces output as $sequence \space length \times 4 * d_{\text{model}}$ which is the expansion process, but GPT OSS didn’t have an expansion and contraction process. even the $2 * d_{\text{model}}$ is only for the SwiGLU activation not as an approach of expansion. Even DeepSeek use $d_{\text{model}}$ as the hidden dimension

$2 *d_{\text{model}}$ is split into two blocks one for swish activation function and another split is then combined to produce the SwiGLU activation function and now the dimension becomes s$equence \space length \times d_{\text{model}}$ and the FC2 process the sequence which has the dimension $d_{\text{model}} \times d_{\text{model}}$ and produces output of $sequence \space length \times d_{\text{model}}$. FFNN in other experts process the context vectors like these and the expert’s importance and Expert’s FFNN outputs are combined to produce the final output.

As you have completed Positional encoding, attention mechanisms, Normalization and MLP. I highly encourage you to check out the reference section for more detailed explanation of these concepts and more advanced concepts in LLMs.

I also want to mention that this article is mostly inspired by the work of [Raj Abhijit Dandekar](https://www.linkedin.com/in/raj-abhijit-dandekar-67a33118a/). There is a lot of his influence in this article from structuring the article and creating visual representations for explaining the concepts.

# References:

[https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&si=aFUKl5Qr3Jvel7SH](https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&si=aFUKl5Qr3Jvel7SH)

[https://www.youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms)

[https://youtu.be/hBUsySdcA3I?si=cXVi3qkp9qK6XY3s](https://youtu.be/hBUsySdcA3I?si=cXVi3qkp9qK6XY3s)

[https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the)

[https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)

[https://youtu.be/hCzJo4ui1P8?si=h8OO7ns_zs_JN8FF](https://youtu.be/hCzJo4ui1P8?si=h8OO7ns_zs_JN8FF)

[RMSNorm (Root Mean Square Layer Normalization)](https://outcomeschool.com/blog/rmsnorm-root-mean-square-layer-normalization)