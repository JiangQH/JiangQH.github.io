---
layout: post
title: Batch Normalization详解
category: dl
description: Batch Normalization what and how
keywords:  Batch Normalization
author: jiangqh

---
## Batch Normalization why?
在神经网络训练中我们使用基于梯度的优化方法(gradient-based optimization)，最常用的便是结合了Momentum的随机梯度下降。在使用梯度下降法时我们的一般步骤为：1、计算各个weight相对于损失函数loss的梯度。2、假设其它weight不变，对某个weight进行更新。但是在实际更新中我们同时对weight进行了更新。回忆上一篇blog[浅谈优化方法与泰勒展开](http://jiangqh.info/%E6%B5%85%E8%B0%88%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95%E4%B8%8E%E6%B3%B0%E5%8B%92%E5%B1%95%E5%BC%80/)梯度下降法实际上利用的是函数的泰勒一次展开近似，实际上它是会受到高阶影响的，泰勒展开实际为：

$$f(\overrightarrow{w}+\overrightarrow{s}) = f(\overrightarrow{w}) + g(\overrightarrow{w})^{T}\overrightarrow{s} + \frac{1}{2}\overrightarrow{s}^{T}H(\overrightarrow{w})\overrightarrow{s} + ...$$  

而我们做了如下近似：

$$f(\overrightarrow{w}+\overrightarrow{s}) \approx f(\overrightarrow{w}) + g(\overrightarrow{w})^{T}\overrightarrow{s}$$   

考虑一个简单情况：神经网络中不包含激活函数，我们的预测是关于\\(x\\)的线性函数：  

$$y = w_1w_2w_3...w_m x$$    

在这种情况下\\(y\\)是关于\\(x\\)的线性函数，但不是\\(w\\)的线性函数，进一步假设loss关于y的梯度为1。在梯度下降法中我们想要降低loss，有如下更新步骤：

$$w_i = w_i - \epsilon g_i$$

其中\\(g_i\\)为损失函数关于\\(w_i\\)的梯度。那么此时的\\(y\\)变为了：

$$y = (w_1 - \epsilon g_1)(w_2 - \epsilon g_2)...(w_m - \epsilon g_m)x$$  

我们期待此时\\(y\\)会有所减小，因为y与loss成正比(梯度为1)。但实际情况真是如此吗？考察上式可以看到任意其中一项有：

$$+\epsilon ^2 g_1g_2 \prod_{i=3}^m w_i$$  

可以看到的是其它\\(w\\)对整个网络的实际更新效果是有影响的，式子\\(\prod_{i=3}^m w_i\\)中如果\\(w\\)从3到m如果都小于1那么影响便可忽略不计，但是如果它们都大于1那么该整体值就会变成一个大数，超过梯度下降带来的下降效应，导致\\(y\\)不降反升甚至溢出。这种效应使得神经网络的训练中学习速率\\(\epsilon\\)的选择变得尤其困难，需要考虑大量的weight的影响，只能使用很小的学习效率。

一种解决方案是直接利用采用多阶信息的优化方法，对线性的我们可以考虑二次方法比如牛顿法。但是在实际中神经网络包含激活函数并不是线性的，所以其泰勒展开将包含多次的信息，同时神经网络参数十分巨大，这使得使用多阶信息的优化方法根本不可行。

Batch Normalization便是一种解决weight更新中耦合问题的方法。

---未完待续