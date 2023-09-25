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
description: One of the towers is the item tower, which encodes a vast amount of content features related to items.proposed a method to evaluate item frequencies from streaming data.
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_223046.png
---

## 总结

The authors have employed the prevalent dual-tower model and designed a modeling framework using a dual-tower neural network. One of the towers is the item tower, which encodes a vast amount of content features related to items.

The optimization approach for such dual-tower models typically involves training by negative sampling in mini-batches. However, this approach can lead to issues, especially when there is a significant skew in the sample distribution, which can potentially harm the model's performance.

To address this challenge, the authors have proposed a method to evaluate item frequencies from streaming data. They have theoretically analyzed and demonstrated through experimental results that this algorithm can produce unbiased estimates without the need for a fixed item corpus. It also adapts to changes in the item distribution through online updates.

Subsequently, the authors have utilized this "sampling bias correction" method to create a neural network-based large-scale retrieval system for YouTube. This system is used to provide personalized services from a corpus containing millions of videos.

Finally, extensive testing on two real datasets and A/B testing has confirmed the effectiveness of the "sampling bias correction" approach.


## introduction

Initially, the dual-tower model was proposed for document retrieval, with the two towers corresponding to query and document embeddings. Nowadays, in mainstream recommendation systems, the dual-tower architecture typically consists of a user embedding tower and an item embedding tower. The overall structure is as depicted in the following diagram:

[Diagram: User Embedding Tower] --> [Recommendation System] --> [Item Embedding Tower]

In this setup, the user embedding tower encodes information about the user, while the item embedding tower encodes information about items in the recommendation system. This architecture allows for personalized recommendations by matching user characteristics with item characteristics.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_223342.png)


First, the model is trained based on its architecture. The model training involves learning two vectors, u(x) and v(y), for the user and items, respectively. These vectors are used to compute a dot product to determine the similarity between items. Recommendations are made for items with high similarity scores.

There are two main components to this process:

1. The left-side user tower generates the user embedding, which needs to be computed in real-time when a user request is made.

2. The right-side item tower generates item embeddings during training, which are precomputed and then loaded into a vector retrieval tool. This tool is used to establish an index and transform the problem into a vector retrieval problem.

Many major companies in the recommendation system space have developed their own open-source tools for vector retrieval, such as Faiss and Annoy. These tools facilitate efficient retrieval of items based on their embeddings, making the recommendation process more scalable and practical.



## core questions 


Here, we use 'x' to represent user and context features, 'y' to represent item features, and 'θ' to denote the parameters of the model. The results can be expressed as follows:


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_223606.png)


Define the log-likelihood function as follows:

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_223638.png)


### hot beats

From the previous discussion, we can see that the fundamental formula is as follows:


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_223717.png)



Looking at the formula above, for a video with high popularity, it implies that the model needs to incorporate the popularity aspect into the parameters of the dual-tower. When the label is fixed, there is always a subtraction of the popularity factor behind the dual-tower similarity, essentially removing the bias introduced by popularity. 

For positive samples, if you watched a highly popular video, it's very likely that you watched it because it was popular. So, we need to subtract this bias. 

For negative samples, if you didn't watch such a popular video, it indicates that you genuinely didn't like it. Thus, we also subtract this bias. This approach makes a lot of sense!



## 结论

In essence, this approach addresses the problem of excessive popularity and inadequate personalization in YouTube recommendations.

This is a significant challenge in recommendation systems, dating back to the first-generation collaborative filtering (CF) systems, where formulas were introduced to downweight popular items. 

To illustrate the extreme case, consider the most popular item, which accounts for half of the total system traffic in terms of exposure. 

In a normal learning scenario, the model would recommend this item to everyone, exacerbating the popularity effect. From a system perspective, this is detrimental to the long-term recommendation ecosystem as potentially high-quality but less discovered items remain undiscovered. From the user's perspective, items they genuinely like might never surface or might not even exist in the system.

On the other hand, this approach is a recall algorithm, and the recall phase is crucial because its primary goal is to retrieve potentially interesting items without necessarily aiming for precision.

Therefore, with the aim of addressing these issues, the approach first directly incorporates popularity downweighting into the definition of item similarity. Additionally, it designs a distributed algorithm capable of streaming updates to adapt to changing item popularity.


## 原文link

https://sci-hub.se/10.1145/3298689.3346996  

