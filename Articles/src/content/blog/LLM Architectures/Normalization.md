---
title: Normalization
description: Attention Mechanism, Mixture of Experts
date: 2026-11-08
category: LLM Architectures
---
Maintaining numerical stability across deep transformer layers requires rigorous mathematical transformations of hidden states. Let's examine how normalization mechanisms, including standard LayerNorm and Root Mean Square Normalization (RMSNorm), regulate activation scales during training and inference.
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