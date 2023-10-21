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
description: the design choices in neural network architectures like CIN aim to balance the model's capacity to capture complex interactions with the risk of overfitting and computational efficiency. The specific choices may vary depending on the problem domain and the goals of the model. 神经网络架构中的设计选择，如CIN，旨在平衡模型捕获复杂相互作用的能力与过拟合和计算效率的风险。具体的选择可能会根据问题领域和模型的目标而有所不同。  
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_125039.png
---

## 介绍


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_125039.png)

In the CIN architecture, there are two matrices referred to as feature maps. The left feature map consists of m feature vectors, each with a dimension of D. The interaction between these two feature maps is calculated by taking the dot product of their corresponding vectors, resulting in a new feature map. The process can be summarized as a way to capture interactions between different depths (D) of features.
在CIN架构中，有两个被称为特征映射的矩阵。左特征映射由m个特征向量组成，每个向量的维度为D。这两个特征映射之间的交互是通过计算它们对应向量的点积来实现的，从而得到一个新的特征映射。这个过程可以总结为一种捕获不同深度（D）特征之间交互的方式。

The reason for doing this is to enable rich interactions between features of different depths or dimensions. By computing interactions at the vector-wise level and then combining them into a new feature map, the model can capture complex relationships and patterns in the data. This approach allows the model to learn how features of different depths interact with each other, which can be essential for capturing intricate patterns in the data.

这样做的原因是为了实现不同深度或维度特征之间的丰富交互。通过在向量级别计算交互，然后将它们组合成一个新的特征映射，模型可以捕获数据中的复杂关系和模式。这种方法使模型能够学习不同深度特征如何相互作用，这对于捕获数据中复杂的模式至关重要。

As for why interactions between different depths (D) are not directly performed, it's a design choice. Direct interactions between different depths can make the model very complex and may lead to overfitting on the training data. By using the approach described in the paper, the model can learn interactions in a structured and controlled manner, potentially avoiding excessive complexity while still capturing meaningful relationships between features of different depths.

至于为什么不直接执行不同深度（D）之间的交互，这是一个设计选择。直接在不同深度之间进行交互可能会使模型非常复杂，并可能导致在训练数据上出现过拟合。通过使用论文中描述的方法，模型可以以有结构且可控的方式学习交互，潜在地避免过于复杂，同时仍然捕获不同深度特征之间的有意义的关系。

The subsequent sum pooling layer, which combines features from different depths, is another way to capture interactions between different depths in a more simplified manner. It may be a deliberate choice to strike a balance between complexity and the model's ability to capture interactions effectively.

随后的求和池化层，它将来自不同深度的特征组合在一起，是另一种以更简化的方式捕获不同深度之间交互的方式。这可能是一种有意选择，以在复杂性和模型有效捕获交互的能力之间取得平衡。

In the end, the design choices in neural network architectures like CIN aim to balance the model's capacity to capture complex interactions with the risk of overfitting and computational efficiency. The specific choices may vary depending on the problem domain and the goals of the model.

最终，神经网络架构中的设计选择旨在平衡模型捕获复杂交互的能力与过拟合和计算效率的风险。具体的选择可能会根据问题领域和模型的目标而有所不同。

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_125336.png)

其中 o 是Hadamard product。

Actually, each row vector of Xk is the weighted sum of all elements in the same layer (k) of CIN. These D layers precisely form a D-dimensional vector.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_125431.png)


The final structure of the entire CIN involves repeating the two steps you mentioned for each Hk and Xk, following a sequential progression because Xk depends on Xk-1. This iterative process allows the network to capture feature interactions gradually, building upon the previous layers' results.
整个CIN的最终结构涉及为每个Hk和Xk重复您提到的两个步骤，遵循了一个顺序的进展，因为Xk依赖于Xk-1。这个迭代过程允许网络逐渐捕获特征交互，基于前一层的结果构建。

### 架构

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_125602.png)





## 原文link

https://arxiv.org/pdf/1803.05170.pdf  (xdeepfm的一部分 )


## 开源代码：

https://gist.github.com/mpozpnd/d1e90909db94de4c1156e484204c0ce9


