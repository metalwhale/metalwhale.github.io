---
title: "Are LLMs not truly \"generative\"?"
date: 2024-02-07T14:08:00Z
description: "Discussion about the generative capability of autoregressive LLMs"
tags:
- LLM
- Generative AI
categories:
- Machine Learning
libraries:
- mathjax
---

**TL;DR**

IMHO, LLMs (Large Language Models) trained with autoregressive process are not "truly" generative, especially those that don't use `BOS` (beginning-of-sentence) token.\
I've asked about this on [`nanoGPT` repo](https://github.com/karpathy/nanoGPT/issues/432), [`transformers` repo](https://github.com/huggingface/transformers/issues/28860) and [HN](https://news.ycombinator.com/item?id=39249301) but haven't yet found the final answer.
This post shows my thoughts on the generative capability of LLMs. Feel free to discuss with me (:

## What is a generative model?
There are two types of models in machine learning: discriminative and generative. Discriminative models learn the relationship between output and input data, while generative models learn the relationship between the components of data.

For example, in terms of computer vision, given a picture of an animal, a discriminative model learns to determine whether the picture is a kitten or a puppy. On the other hand, a generative model learns how every pixel of the picture is related; for instance, one part may describe the eye, and another part the mouth.

Another example with text sequences: given an English sentence, a discriminative model learns to determine whether the topic is sport, economics, or science; a generative model learns about the order, typography, and grammar. While a discriminative model doesn't need to know the "rules" of English, the generative model does; for instance, a sentence cannot start with a comma.

Mathematically, according to [this course of Google](https://developers.google.com/machine-learning/gan/generative), the discriminative model only learns the *conditional probability* `p(y|x)`, while the generative model learns the *joint probability* `p(x,y)`, with `x` as the input and `y` as the output.

Therefore, theoretically, a model is generative if and only if it can learn the joint probability of the data.\
Do LLMs have this ability? Let's figure it out.

## How do LLMs learn?
Nowadays, mostly LLMs are trained using the *autoregressive* process: given a text sequence, they learn how to predict each single token by relying on all tokens that lie before it.

For example: given a sentence "She loves smol kittens", an LLM model needs to learn that:
- The word "loves" likely appears after "She"
- The word "smol" likely appears after "She loves"
- The word "kittens" likely appears after "She loves smol"

The probabilities that an LLM model learns in this case can be written as:\
`p(loves|She)`, `p(smol|She,loves)`, `p(kittens|She,loves,smol)`, respectively.

Generally speaking, given a text sequence with $n$ tokens, LLMs need to learn all the probabilities for every token of that text sequence: $p(x\_{2}|x\_{1})$, $p(x\_{3}|x\_{2},x\_{1})$,... $p(x\_{n-1}|x\_{n-2},x\_{n-3},...x\_{1})$, $p(x\_{n}|x\_{n-1},x\_{n-2},x\_{n-3},...x\_{1})$.

## Why do people keep calling LLMs "generative"?
Now we know how autoregressive helps LLMs learn. But so far, they are only learning *conditional probabilities* for each token given a condition based on all tokens that lie before it. To be called "generative" LLMs need to learn the *joint probability*. How can they achieve that?

It turns out the *joint probability* we are seeking here can be easily calculated by combining all the *conditional probabilities* using the [chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)) of Bayes' formula!

The final probability that LLMs learn is\
= $p(x\_{n}|x\_{n-1},x\_{n-2},x\_{n-3},...x\_{1})$ * $p(x\_{n-1}|x\_{n-2},x\_{n-3},...x\_{1})$ * ... * $p(x\_{3}|x\_{2},x\_{1})$ * $p(x\_{2}|x\_{1})$ * $p(x\_{1})$\
= $p(x\_{n}|x\_{n-1},x\_{n-2},x\_{n-3},...x\_{1})$ * $p(x\_{n-1}|x\_{n-2},x\_{n-3},...x\_{1})$ * ... * $p(x\_{3}|x\_{2},x\_{1})$ * $p(x\_{1},x\_{2})$\
= $p(x\_{n}|x\_{n-1},x\_{n-2},x\_{n-3},...x\_{1})$ * $p(x\_{n-1}|x\_{n-2},x\_{n-3},...x\_{1})$ * ... * $p(x\_{1},x\_{2},x\_{3})$\
= $p(x\_{n}|x\_{n-1},x\_{n-2},x\_{n-3},...x\_{1})$ * $p(x\_{1},...x\_{n-3},x\_{n-2},x\_{n-1})$\
= $p(x\_{1},...x\_{n-3},x\_{n-2},x\_{n-1},x\_{n})$

Neat! By learning the conditional probabilities of every token and then combining them, LLMs have finally learned the joint probability of the entire text sequence! Great, isn't it?

Wait a minute...

## Is it correct to call LLMs "generative"?
We have never talked about one important missing piece of the chain rule: probability of the very first token $p(x\_{1})$.

There is no direct way in the training process of LLMs to learn the probability of the first token. For LLMs to be able to start generating tokens, we need to feed them at least one initial token, or in other words: a "prompt", or "prefix". LLMs are "conditional generative", without this initial condition they are not generative at all, mathematically.

So, is it not correct to call LLMs "generative"?

The interesting thing is, we can apply a "trick" to make LLMs learn the probability of the first token! It is quite simple: by choosing a special token, referred to here as $x\_0$, which has a **probability of 1**, and putting it as the first token even before $x\_1$, **in every text sequence**! And since $p(x\_{0})=1$, we can calculate $p(x\_{1})$ in the same way as all other tokens:

$p(x\_{1})$ = $p(x\_{0})$ * $p(x\_{1}|x\_{0})$

$x\_0$ is often referred to as the `BOS`, or beginning-of-sentence token. More LLMs are trained using this token along with autoregressive process.

## Conclusion
Can LLMs generate texts? Yes, but it's better to use the word "complete". LLMs, as they are trained using autoregressive process, can't *generate* texts from scratch; they can only *complete* them given a certain initial condition (or "prompt").

Are LLMs generative? That depends.\
They are not generative, as we defined them mathematically.\
But they can achieve the generative capability, by using a special token `BOS` that helps them learn the probability of every token in each text sequence of the training data.

LLMs that don't use `BOS` token are not generative.\
LLMs that do use `BOS` token are "tricky" generative.
