---
title: pnn Product-based Neural Networks for User Response Prediction 上海交大&伦敦大学 
date: 2023-09-10 9:00:00
categories:
  - 排序模型
tags:
  - 多任务模型 
  - pnn 
  - dnn
  - Recommender systems
description: Improved Learning of Category Feature Embeddings; Incorporation of Second-Order and High-Order Feature Interactions  改进的类别特征嵌入学习；包括二阶和高阶特征交互的融合。  
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_194956.png
---



## The main contributions of the PNN 
1. **改进的类别特征嵌入学习：** PNN 利用基于产品的结构增强了对类别特征嵌入的学习。这种结构使模型能够更有效地捕获类别特征之间的复杂关系和交互作用。通过使用基于产品的交互，PNN 能够以更丰富和富有表现力的方式表示特征之间的相互作用。

1. **Improved Learning of Category Feature Embeddings:** PNN leverages a product-based structure that enhances the learning of category feature embeddings. This structure allows the model to capture intricate relationships and interactions between category features more effectively. By using product-based interactions, PNN can represent the feature interactions in a richer and more expressive way.

2. **包括二阶和高阶特征交互的融合：** PNN 模型将通过基于产品的结构捕获的二阶特征交互与使用全连接层（密集层）学习的高阶特征交互相结合。这种交互的结合使 PNN 能够同时建模成对的特征交互和高阶特征交互，从而能够捕获数据中的复杂和非线性模式。

2. **Incorporation of Second-Order and High-Order Feature Interactions:** The PNN model combines second-order feature interactions (captured through the product-based structure) with high-order feature interactions learned using fully connected layers (dense layers). This combination of interactions allows PNN to model both pairwise and higher-order feature interactions, making it capable of capturing complex and non-linear patterns in the data.

总之，PNN 通过基于产品的结构改进了类别特征的表示，并成功地将二阶和高阶特征交互结合在一起，使其成为一个强大的模型，能够捕获分类数据中复杂的关系和模式。

In summary, PNN improves category feature representation through product-based structures and successfully combines second-order and high-order feature interactions, making it a powerful model for capturing intricate relationships and patterns in categorical data.





## model 架构

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_194956.png)

PNN（Product-based Neural Network）模型的结构考虑了在工业应用中常见的处理大规模稀疏特征的情况，特别是来自类别特征的 one-hot 或 multi-hot 编码。PNN 模型主要处理类别特征。

The structure of the PNN (Product-based Neural Network) model is designed with the consideration of handling large-scale sparse features commonly encountered in industrial applications, especially those derived from one-hot or multi-hot encoding of category features. The PNN model primarily deals with category features.

1. **输入层：** 这是特征输入层，其中特征被表示为字段，类似于 FFM（Field-aware Factorization Machines）模型中的字段概念。每个字段代表一组相关的特征。

1. **Input Layer:** This is the feature input layer where features are represented as fields, akin to the concept of fields in the FFM (Field-aware Factorization Machines) model. Each field represents a group of related features.

2. **嵌入层成对连接：** 在这一层中，每个字段内的每个特征都被转换为嵌入表示。这一层捕获了字段级别上的特征嵌入。

2. **Embedding Layer Pair-wise Connected:** In this layer, each feature within a field is transformed into an embedding representation. This layer captures the embeddings of features at the field level.

3. **产品层成对连接：** 产品层计算特征嵌入之间的成对交互（二阶特征交互）。这一层计算特征嵌入的逐元素乘积，并捕获特征对之间的交互。

3. **Product Layer Pair-wise Connected:** The Product Layer computes pair-wise interactions (second-order feature interactions) between the embeddings of features. This layer calculates the element-wise product of feature embeddings and captures the interactions between feature pairs.

4. **隐藏层：** 这些是多层感知器（MLP）的隐藏层。隐藏层的目的是捕获高阶特征交互，并模拟数据中的复杂非线性关系。产品层的输出与其他特征连接在一起，然后馈送到这些隐藏层进行进一步处理。

4. **Hidden Layers:** These are the hidden layers of a multi-layer perceptron (MLP). The purpose of the hidden layers is to capture high-order feature interactions and model complex, non-linear relationships within the data. The output of the Product Layer is concatenated with other features and fed into these hidden layers for further processing.

总之，PNN 模型的设计旨在处理稀疏的类别特征，其架构通过产品层计算二阶特征交互，并通过MLP的隐藏层捕获高阶交互。这个设计使PNN能够有效地建模工业应用中常见的大规模稀疏数据集中的复杂特征关系。
In summary, the PNN model is designed to handle sparse category features, and its architecture incorporates the calculation of second-order feature interactions through the Product Layer and captures higher-order interactions through hidden layers of an MLP. This design enables PNN to effectively model intricate feature relationships in large-scale sparse datasets commonly found in industrial applications.


## two explored ways of feature interaction through product operations

1. **输入层（字段）：** 特征被组织成字段，其中每个字段代表一组相关的特征。

1. **Input Layer (Fields):** The features are organized into fields, where each field represents a group of related features.

2. **嵌入层成对连接：** 在每个字段内，特征被转换为嵌入表示。这一层捕获了字段级别上的特征嵌入。

2. **Embedding Layer Pair-wisely Connected:** Within each field, features are transformed into embedding representations. This layer captures the embeddings of features at the field level.

3. **产品层成对连接（内积）：** 在传统的基于产品的交互中，计算特征嵌入之间的内积。对于每一对特征嵌入，计算内积。这个内积操作捕获了二阶特征交互。

3. **Product Layer Pair-wisely Connected (Inner Product):** In the conventional product-based interaction, inner products are calculated between the embeddings of features. For each pair of feature embeddings, an inner product is computed. This inner product operation captures the second-order feature interactions.

4. **隐藏层：** 产品层的输出与其他特征连接在一起，然后馈送到MLP的隐藏层。这些隐藏层捕获高阶特征交互和数据中的非线性关系。

4. **Hidden Layers:** The output from the Product Layer is concatenated with other features and fed into hidden layers of an MLP. These hidden layers capture higher-order feature interactions and non-linear relationships in the data.

现在，让我们讨论使用外积进行特征交互的第二种方式：

Now, let's discuss the second way of feature interaction using outer product:

5. **产品层成对连接（外积）：** 在这种变化中，产品层计算特征嵌入之间的外积。每一对特征嵌入都产生一个外积矩阵。外积操作捕获了特征对之间更丰富和更复杂的交互。


5. **Product Layer Pair-wisely Connected (Outer Product):** In this variation, the Product Layer computes outer products between the feature embeddings. Each pair of feature embeddings results in an outer product matrix. The outer product operation captures richer and more complex interactions between feature pairs.

内积和外积交互都允许PNN模型捕获特征交互的不同方面。内积关注线性交互，而外积捕捉更复杂和非线性的交互。这些交互有助于模型有效地表示和学习特征交互。

Both inner product and outer product interactions allow the PNN model to capture different aspects of feature interactions. Inner product focuses on linear interactions, while outer product captures more complex and non-linear interactions. These interactions contribute to the model's ability to represent and learn from the feature interactions effectively.


## model compare 
1. **FNN模型（Field-aware Neural Network）：** 当从PNN模型中移除LP（Local Product）模块时，PNN本质上等同于FNN模型。FNN模型专注于捕获字段内的特征交互，并使用全连接网络进行高阶交互。

1. **FNN Model (Field-aware Neural Network):** When the LP (Local Product) module is removed from the PNN model, PNN is essentially equivalent to the FNN model. The FNN model focuses on capturing feature interactions within fields and employs a fully connected network for higher-order interactions.

2. **FM模型（Factorization Machine）：** 如果从PNN模型中删除隐藏层，并配置输出层仅执行对产品层输出的加权求和，那么PNN模型将与FM模型非常相似。FM模型主要通过内积来捕获二阶特征交互，不包括高阶交互。

2. **FM Model (Factorization Machine):** If you remove the hidden layers from the PNN model and configure the output layer to simply perform a weighted sum of the outputs from the Product Layer, the PNN model becomes quite similar to the FM model. The FM model primarily captures second-order feature interactions through inner products and does not include higher-order interactions.

3. **IPNN和OPNN的组合：** PNN模型通过允许内积（IPNN）和外积（OPNN）模块的组合来提供灵活性。这意味着您可以选择同时使用内积和外积交互，以更全面的方式建模特征交互，从而创建更强大的PNN模型。

3. **Combination of IPNN and OPNN:** The PNN model offers flexibility by allowing the combination of the inner product (IPNN) and outer product (OPNN) modules. This means that you can choose to use both inner and outer product interactions to model feature interactions in a more comprehensive manner, resulting in a more powerful PNN model.


总之，PNN通过结合内积和外积交互的方式，汇合了不同模型的特点。它在侧重于字段级别交互时类似于FNN，在强调二阶交互时类似于FM。允许组合内积和外积交互的灵活性使PNN变得多才多艺，并能够适应各种建模需求。

In summary, PNN combines aspects of different models by incorporating both inner product and outer product interactions. It can resemble FNN when focusing on field-level interactions and FM when emphasizing second-order interactions. The flexibility to combine inner and outer product interactions makes PNN versatile and adaptable to various modeling needs.


## Performance Comparison
 

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_195701.png)




## 原文link

https://arxiv.org/abs/1611.00144




