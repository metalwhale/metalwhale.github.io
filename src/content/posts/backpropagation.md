---
title: "Backpropagation"
date: 2020-06-09T12:33:36Z
description: "Backpropagation\'s fundamental equations"
tags:
- Neural Network
series:
- Automatic Differentiation
categories:
- Machine Learning
libraries:
- mathjax
---

## Notation

$w^l\_{jk}$: the weight for the connection from the $k^{th}$ neuron in the $(l-1)^{th}$ layer to the $j^{th}$ neuron in the $l^{th}$ layer\
$b^l\_j$: the bias of the $j^{th}$ neuron in the $l^{th}$ layer\
$L$: the number of layers in the network

## Feedforward

##### Intermediate quantity
$$z^l\_j = \sum\_k w^l\_{jk} a^{l-1}\_k + b^l\_j \tag{1}$$

##### Activation
$$a^l\_j = \sigma(z^l\_j) \tag{2}$$

##### Cost
$$C = \frac{1}{2} \sum\_j (y\_j-a^L\_j)^2 \tag{3}$$

## Backpropagate

##### Partial derivatives of the last activations
From $(3)$:
$$\Rightarrow \frac{\partial C}{\partial a^L\_j} = a^L\_j-y\_j$$

##### Partial derivatives of the $z$ quantities
From $(2)$:
$$\Rightarrow \frac{\partial a^l\_j}{\partial z^l\_j} = \sigma\'(z^l\_j) \tag{2\'}$$
Chain rule:
$$\frac{\partial C}{\partial z^l\_j} = \frac{\partial C}{\partial a^l\_j} \frac{\partial a^l\_j}{\partial z^l\_j} \tag{4}$$
From $(2\')$ and $(4)$:
$$\Rightarrow \frac{\partial C}{\partial z^l\_j} = \frac{\partial C}{\partial a^l\_j} \sigma\'(z^l\_j)$$

##### Partial derivatives of the activations in hidden layers
From $(1)$:
$$\Rightarrow \frac{\partial z^l\_j}{\partial a^{l-1}\_k} = w^l\_{jk} \tag{1\'}$$
Chain rule:
$$\frac{\partial C}{\partial a^{l-1}\_k} = \sum\_j \frac{\partial C}{\partial z^l\_j} \frac{\partial z^l\_j}{\partial a^{l-1}\_k} \tag{5}$$
From $(1\')$ and $(5)$:
$$\Rightarrow \frac{\partial C}{\partial a^{l-1}\_k} = \sum\_j \frac{\partial C}{\partial z^l\_j} w^l\_{jk}$$

##### Partial derivatives of the weights
From $(1)$:
$$\Rightarrow \frac{\partial z^l\_j}{\partial w^l\_{jk}} = a^{l-1}\_k \tag{1\'\'}$$
Chain rule:
$$\frac{\partial C}{\partial w^l\_{jk}} = \frac{\partial C}{\partial z^l\_j} \frac{\partial z^l\_j}{\partial w^l\_{jk}} \tag{6}$$
From $(1\'\')$ and $(6)$:
$$\Rightarrow \frac{\partial C}{\partial w^l\_{jk}} = \frac{\partial C}{\partial z^l\_j} a^{l-1}\_k$$

##### Partial derivatives of the biases
From $(1)$:
$$\Rightarrow \frac{\partial z^l\_j}{\partial b^l\_j} = 1 \tag{1\'\'\'}$$
Chain rule:
$$\frac{\partial C}{\partial b^l\_j} = \frac{\partial C}{\partial z^l\_j} \frac{\partial z^l\_j}{\partial b^l\_j} \tag{7}$$
From $(1\'\'\')$ and $(7)$:
$$\Rightarrow \frac{\partial C}{\partial b^l\_j} = \frac{\partial C}{\partial z^l\_j}$$
