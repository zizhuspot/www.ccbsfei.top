---
title: dnn双塔-Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations -youtube
date: 2023-09-20 21:08:00
categories:
  - 排序模型
tags:
  - Neural Networks
  - Information Retrieval
  - 多任务模型
  - dnn 
  - 预估模型 
  - Recommender systems
description: One of the towers is the item tower, which encodes a vast amount of content features related to items.proposed a method to evaluate item frequencies from streaming data. 其中一个塔是商品塔，它编码了与商品相关的大量内容特征。提出了一种从流数据中评估商品频率的方法。  
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_223046.png
---

## 总结

The authors have employed the prevalent dual-tower model and designed a modeling framework using a dual-tower neural network. One of the towers is the item tower, which encodes a vast amount of content features related to items.
作者采用了普遍的双塔模型，并设计了一个双塔神经网络的建模框架。其中一个塔是商品塔，它编码了与商品相关的大量内容特征。

The optimization approach for such dual-tower models typically involves training by negative sampling in mini-batches. However, this approach can lead to issues, especially when there is a significant skew in the sample distribution, which can potentially harm the model's performance.

针对这类双塔模型的优化方法通常涉及通过负采样在小批次中进行训练。然而，这种方法可能会导致问题，特别是当样本分布出现显著偏斜时，可能会损害模型的性能。

To address this challenge, the authors have proposed a method to evaluate item frequencies from streaming data. They have theoretically analyzed and demonstrated through experimental results that this algorithm can produce unbiased estimates without the need for a fixed item corpus. It also adapts to changes in the item distribution through online updates.

为了解决这一挑战，作者提出了一种从流数据中评估商品频率的方法。他们通过理论分析和实验结果证明，该算法可以生成无偏估计，无需固定的商品语料库，还能通过在线更新适应商品分布的变化。

Subsequently, the authors have utilized this "sampling bias correction" method to create a neural network-based large-scale retrieval system for YouTube. This system is used to provide personalized services from a corpus containing millions of videos.

随后，作者利用这一"采样偏差校正"方法创建了一个基于神经网络的YouTube大规模检索系统。该系统用于从包含数百万视频的语料库中提供个性化服务。

Finally, extensive testing on two real datasets and A/B testing has confirmed the effectiveness of the "sampling bias correction" approach.

最后，对两个真实数据集进行了广泛测试和A/B测试，证实了"采样偏差校正"方法的有效性。

## introduction

Initially, the dual-tower model was proposed for document retrieval, with the two towers corresponding to query and document embeddings. Nowadays, in mainstream recommendation systems, the dual-tower architecture typically consists of a user embedding tower and an item embedding tower. The overall structure is as depicted in the following diagram:
最初，双塔模型是用于文档检索的，其中两个塔对应于查询和文档嵌入。如今，在主流的推荐系统中，双塔架构通常由用户嵌入塔和商品嵌入塔组成。总体结构如下图所示：

[Diagram: User Embedding Tower] --> [Recommendation System] --> [Item Embedding Tower]

[图示：用户嵌入塔] --> [推荐系统] --> [商品嵌入塔]

In this setup, the user embedding tower encodes information about the user, while the item embedding tower encodes information about items in the recommendation system. This architecture allows for personalized recommendations by matching user characteristics with item characteristics.

在这个设置中，用户嵌入塔编码了用户的信息，而商品嵌入塔编码了推荐系统中商品的信息。这种架构允许通过将用户特征与商品特征匹配来实现个性化推荐。

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_223342.png)


First, the model is trained based on its architecture. The model training involves learning two vectors, u(x) and v(y), for the user and items, respectively. These vectors are used to compute a dot product to determine the similarity between items. Recommendations are made for items with high similarity scores.
首先，模型是基于其架构进行训练的。模型训练涉及学习用户和商品的两个向量，分别是 u(x) 和 v(y)。这些向量用于计算点积，以确定商品之间的相似性。对于相似性得分较高的商品进行推荐。

There are two main components to this process:

这个过程有两个主要组成部分：

1. The left-side user tower generates the user embedding, which needs to be computed in real-time when a user request is made.

1. 左侧的用户塔生成用户嵌入，需要在用户请求时实时计算。

2. The right-side item tower generates item embeddings during training, which are precomputed and then loaded into a vector retrieval tool. This tool is used to establish an index and transform the problem into a vector retrieval problem.

2. 右侧的商品塔在训练期间生成商品嵌入，这些嵌入是预先计算的，然后加载到一个向量检索工具中。该工具用于建立索引，并将问题转化为向量检索问题。

Many major companies in the recommendation system space have developed their own open-source tools for vector retrieval, such as Faiss and Annoy. These tools facilitate efficient retrieval of items based on their embeddings, making the recommendation process more scalable and practical.


许多主要的推荐系统领域的公司已经开发了自己的开源工具用于向量检索，例如 Faiss 和 Annoy。这些工具有助于根据它们的嵌入高效检索商品，使推荐过程更具可扩展性和实用性。

## core questions 


Here, we use 'x' to represent user and context features, 'y' to represent item features, and 'θ' to denote the parameters of the model. The results can be expressed as follows:


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_223606.png)


Define the log-likelihood function as follows:

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_223638.png)


### hot beats

From the previous discussion, we can see that the fundamental formula is as follows:


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_223717.png)



Looking at the formula above, for a video with high popularity, it implies that the model needs to incorporate the popularity aspect into the parameters of the dual-tower. When the label is fixed, there is always a subtraction of the popularity factor behind the dual-tower similarity, essentially removing the bias introduced by popularity. 

观察上述公式，对于一个具有高流行度的视频，这意味着模型需要将流行度因素纳入到双塔的参数中。当标签被固定时，双塔相似度背后总是存在一个流行度因子的减法，本质上是为了消除流行度引入的偏差。

For positive samples, if you watched a highly popular video, it's very likely that you watched it because it was popular. So, we need to subtract this bias. 

对于正样本，如果您观看了一个非常流行的视频，很可能是因为它很受欢迎，所以我们需要减去这种偏差。

For negative samples, if you didn't watch such a popular video, it indicates that you genuinely didn't like it. Thus, we also subtract this bias. This approach makes a lot of sense!

对于负样本，如果您没有观看这样一个受欢迎的视频，这表明您确实不喜欢它。因此，我们也需要减去这种偏差。这种方法非常合理！

## 结论

In essence, this approach addresses the problem of excessive popularity and inadequate personalization in YouTube recommendations.
从本质上讲，这种方法解决了YouTube推荐系统中普遍存在的问题，即过度流行和个性化不足。

This is a significant challenge in recommendation systems, dating back to the first-generation collaborative filtering (CF) systems, where formulas were introduced to downweight popular items. 

这是推荐系统领域的一个重大挑战，可以追溯到第一代协同过滤（CF）系统，当时引入了公式来降低热门物品的权重。

To illustrate the extreme case, consider the most popular item, which accounts for half of the total system traffic in terms of exposure. 

为了说明极端情况，考虑一下最热门的物品，它占系统总曝光量的一半。在正常的学习情景中，模型会向每个人推荐这个物品，加剧了热门效应。从系统的角度来看，这对长期的推荐生态系统是有害的，因为潜在的高质量但鲜为人知的物品仍然未被发现。从用户的角度来看，他们真正喜欢的物品可能永远不会浮出水面，或者在系统中甚至可能不存在。

In a normal learning scenario, the model would recommend this item to everyone, exacerbating the popularity effect. From a system perspective, this is detrimental to the long-term recommendation ecosystem as potentially high-quality but less discovered items remain undiscovered. From the user's perspective, items they genuinely like might never surface or might not even exist in the system.

On the other hand, this approach is a recall algorithm, and the recall phase is crucial because its primary goal is to retrieve potentially interesting items without necessarily aiming for precision.

另一方面，这种方法是一个召回算法，召回阶段至关重要，因为其主要目标是检索潜在感兴趣的物品，而不一定追求精确性。

Therefore, with the aim of addressing these issues, the approach first directly incorporates popularity downweighting into the definition of item similarity. Additionally, it designs a distributed algorithm capable of streaming updates to adapt to changing item popularity.

因此，为了解决这些问题，该方法首先直接将热门物品的权重降低纳入物品相似性的定义。此外，它设计了一种分布式算法，能够流式传输更新以适应物品热门度的变化。

## 原文link

https://sci-hub.se/10.1145/3298689.3346996  

