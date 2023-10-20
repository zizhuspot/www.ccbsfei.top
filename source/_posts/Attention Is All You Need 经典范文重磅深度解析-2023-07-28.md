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
description: attention的经典之作，引领了文本走向大模型的基础，作为奠基石，自此以后，大模型百花齐放百家争鸣，这都是这篇文章的功劳。 "Attention is a classic work that has led to the foundation of large models for text, serving as a cornerstone. Since then, large models have flourished and blossomed, and all this is thanks to this article."  
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


- In the encoder-decoder attention layer, queries come from the previous decoder layer, while keys and values come from the output of the encoder, allowing each decoder position to capture information from various positions in the input sequence, similar to the encoder-decoder attention mechanism in sequence-to-sequence models.
- In the self-attention layer of the encoder, each position can capture information from all positions above it, because all keys, values, and queries come from the same place, i.e., the output of the previous layer of the encoder.
- In the self-attention layer of the decoder, each position can capture information from both current and previous positions. By adding masks, we prevent information from flowing leftward, maintaining the autoregressive property and ensuring that the decoder only relies on information available up to the current time step for prediction.


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


- Why is self-attention important?

Self-attention is a method in neural networks for learning complex dependencies between input elements. It allows models to focus not only on local information within a sequence but also capture long-range dependencies. This ability is crucial for tasks like natural language processing, speech recognition, and machine translation.

Here are some advantages of self-attention:

1. Parallel Computation: Compared to recurrent neural networks (RNNs), self-attention can process all elements in a sequence in parallel, greatly improving computational efficiency.

2. Long-Range Dependencies: Self-attention can effectively capture long-range dependencies in a sequence, which is essential for understanding the grammatical and semantic structures in natural language.

3. Interpretability: Self-attention mechanisms provide a degree of interpretability, allowing us to examine the parts of the input that the model focuses on when making predictions.

4. Sparse Interaction: Self-attention often exhibits sparsity, meaning the model only attends to a small subset of elements in the input sequence when making predictions. This helps reduce computational complexity and the number of parameters, improving model efficiency and generalization.

5. Adaptation to Different Scales: Self-attention can adapt to different input scales, such as sentences, paragraphs, or documents, without changing the model's structure.

In summary, self-attention is a powerful tool that aids models in better understanding and processing sequential data.





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

It sounds like you're referring to the classic paper that laid the foundation for many applications of attention mechanisms, particularly in recommendation systems and large-scale models. One of the most influential works in this regard is the paper titled "Attention Is All You Need" by Vaswani et al., published in 2017. This paper introduced the Transformer architecture, which marked a significant milestone in the field of natural language processing and machine learning.

The Transformer architecture, as presented in this paper, is indeed a groundbreaking work. It replaced traditional recurrent neural networks (RNNs) in many NLP applications and introduced self-attention mechanisms as a core component. The introduction of the Transformer led to remarkable advancements in various areas, including:

1. **Machine Translation:** Transformers greatly improved the quality of machine translation, and models like "Google's Transformer" (later known as the Transformer model) achieved state-of-the-art performance.

2. **Large Pre-trained Models:** The Transformer architecture paved the way for large pre-trained models like BERT (Bidirectional Encoder Representations from Transformers), GPT (Generative Pre-trained Transformer), and their variants, which have become the basis for numerous NLP applications.

3. **Recommendation Systems:** The attention mechanism introduced in Transformers found applications in recommendation systems, enabling personalized and efficient content recommendation.

4. **Image and Video Processing:** Transformers have been adapted for computer vision tasks, such as image classification and object detection, leading to models like Vision Transformers (ViTs).

5. **Interpretable Attention:** Researchers have explored ways to make attention mechanisms more interpretable and transparent, addressing the need for model explainability.

The paper "Attention Is All You Need" is indeed a classic and pivotal work that triggered a revolution in deep learning and set the stage for many subsequent developments in artificial intelligence.



