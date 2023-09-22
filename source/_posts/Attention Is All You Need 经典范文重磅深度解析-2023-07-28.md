---
title: Attention Is All You Need 经典范文重磅深度解析
date: 2023-08-05 14:30:00
categories:
  - 大模型
tags:
  - GPT1
  - GPT2
  - GPT3
  - attention
  - transformer
description: attention的经典之作，引领了文本走向大模型的基础，作为奠基石，自此以后，大模型百花齐放百家争鸣，这都是这篇文章的功劳。
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/Attention%20Is%20All%20You%20Need.jpeg
---


## 引言

### 介绍
序列模型

### transformer 产生的原因

    RNN的缺点
    CNN是否可以代替RNN
    Transformer的优点

![Simple  RNN](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729094433.png)

![transformer 1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729094525.png)

![transformer 2](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729094536.png)


## 模型架构

    Overall Architecture
    Encoder and Decoder Stacks
    Attention
        Scaled Dot-Product Attention
        Multi-Head Attention
        Application of Attention in our Model
    Position-wise Feed-Forward Networks
    Embeddings and Softmax
    Positional Encoding

![模型架构1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729094732.png)

### Encoder and Decoder Stacks

![Encoder and Decoder Stacks](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729094804.png)

### Attention 

- Scaled Dot-Product Attention

![Dot-Product Attention 1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729094911.png)

![Dot-Product Attention 2](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729094923.png)

- Multi-Head Attention

![Multi-Head Attention](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095014.png)

![Multi-Head Attention 1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095032.png)

- Application of Attention in our Model


- 在encoder-decoder attention层中，queries来源于上一个decoder层，而keys和values来源于encoder的输出，以允许每个decoder位置捕获输入序列各个位置的信息，类似sequence-to-sequence模型的encoder-decoder attention机制。
- 在encoder的self-attention层中，每个位置可以捕获上一层所有位置的信息，因为所有keys、values和queries都来自同一处，即上一层encoder的输出。
- 在decoder的self-attention层中，每个位置可以捕获当前位置以及之前所有位置的信息，并通过添加mask来阻止信息向左流动，以保持自回归特性，确保decoder只依赖当前时刻之前的信息进行预测。


### Position-wise Feed-Forward Networks 

![mlp](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095152.png)

![mlp 1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095225.png)

![mlp 2](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095236.png)

### Embeddings and Softmax 

![Embeddings and Softmax ](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095322.png)

### Positional Encoding

![Positional Encoding](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095350.png)

![Positional Encoding](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095402.png)

![Positional Encoding](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095411.png)


## 相关工作

- Why Self-Attention
    - n:序列长度
    - d:向量长度
    - k:卷积核大小
    - r:邻居个数

![Why Self-Attention](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095523.png)


- Training

    - Training Data and Batching

        WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Has 37000 tokens
        larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38].
 
    - Hardware and Schedule

        8 NVIDIA P100 GPUs.
    Base models: each training step took about 0.4 seconds, 10w steps, 12 hours
    Big models:   each training step took about 1 seconds,   30w steps, 3.5 days

    - Optimizer
    - Regularization



## 实验

![实验1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095713.png)

![实验2](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095723.png)


## 总结

![总结1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729095818.png)

## 参考资料

- https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.788&vd_source=5dc536abda18831d58d1dd35d61eee92 
- https://zhuanlan.zhihu.com/p/500569055 
- https://zhuanlan.zhihu.com/p/61494510 
- https://blog.csdn.net/LiRongLu_/article/details/126384067  
- https://blog.csdn.net/weixin_40607428/article/details/105407537 
- https://zhuanlan.zhihu.com/p/497382888 
- https://blog.csdn.net/weixin_60737527/article/details/127141542 



## 我的点评
绝对的经典只做，因为自从这个发表了之后，先是在各种推荐，广告排序模型场景中见到，然后是大模型开始广泛使用。奠基之作。

