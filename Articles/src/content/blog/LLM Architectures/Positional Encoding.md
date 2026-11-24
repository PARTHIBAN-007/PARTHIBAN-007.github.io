---
title: Positional Encoding
description: Attention Mechanism, Mixture of Experts
date: 2026-11-08
category: LLM Architectures
---
Word order dictates semantic meaning, yet transformer attention mechanisms process tokens concurrently as unordered sets. Let's look at the positional encoding techniques required to inject spatial data and preserve sequential syntax.
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
\text{PE}_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$

$$
\text{PE}_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
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
R(\theta) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
$$

let’s consider an example $v = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$ and $\theta = 90^\circ$:

$$
\text{then, } R(90^\circ) = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

$$
R(90^\circ)v = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ -2 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
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

By mapping absolute coordinates via static sinusoidal vectors or encoding relative distances through complex rotation matrices in RoPE, these methods bridge the gap between permutation-invariant attention and sequential language.