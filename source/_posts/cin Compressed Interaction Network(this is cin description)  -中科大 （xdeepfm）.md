---
title: cin Compressed Interaction Network(this is cin description)  -中科大 （xdeepfm）
date: 2023-09-17 13:12:00
categories:
  - 排序模型
tags:
  - 预估模型 
  - cin 
  - Factorization machines
  - neural network
  - deep learning
  - feature interactions
  - Recommender systems
description: the design choices in neural network architectures like CIN aim to balance the model's capacity to capture complex interactions with the risk of overfitting and computational efficiency. The specific choices may vary depending on the problem domain and the goals of the model.
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_125039.png
---

## 介绍


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_125039.png)

In the CIN architecture, there are two matrices referred to as feature maps. The left feature map consists of m feature vectors, each with a dimension of D. The interaction between these two feature maps is calculated by taking the dot product of their corresponding vectors, resulting in a new feature map. The process can be summarized as a way to capture interactions between different depths (D) of features.

The reason for doing this is to enable rich interactions between features of different depths or dimensions. By computing interactions at the vector-wise level and then combining them into a new feature map, the model can capture complex relationships and patterns in the data. This approach allows the model to learn how features of different depths interact with each other, which can be essential for capturing intricate patterns in the data.

As for why interactions between different depths (D) are not directly performed, it's a design choice. Direct interactions between different depths can make the model very complex and may lead to overfitting on the training data. By using the approach described in the paper, the model can learn interactions in a structured and controlled manner, potentially avoiding excessive complexity while still capturing meaningful relationships between features of different depths.

The subsequent sum pooling layer, which combines features from different depths, is another way to capture interactions between different depths in a more simplified manner. It may be a deliberate choice to strike a balance between complexity and the model's ability to capture interactions effectively.

In the end, the design choices in neural network architectures like CIN aim to balance the model's capacity to capture complex interactions with the risk of overfitting and computational efficiency. The specific choices may vary depending on the problem domain and the goals of the model.


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_125336.png)

其中 o 是Hadamard product。

Actually, each row vector of Xk is the weighted sum of all elements in the same layer (k) of CIN. These D layers precisely form a D-dimensional vector.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_125431.png)


The final structure of the entire CIN involves repeating the two steps you mentioned for each Hk and Xk, following a sequential progression because Xk depends on Xk-1. This iterative process allows the network to capture feature interactions gradually, building upon the previous layers' results.

### 架构

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_125602.png)





## 原文link

https://arxiv.org/pdf/1803.05170.pdf  (xdeepfm的一部分 )


## 开源代码：

https://gist.github.com/mpozpnd/d1e90909db94de4c1156e484204c0ce9


