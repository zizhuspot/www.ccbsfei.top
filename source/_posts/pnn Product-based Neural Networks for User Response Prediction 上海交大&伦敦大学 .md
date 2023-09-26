---
title: pnn Product-based Neural Networks for User Response Prediction 上海交大&伦敦大学 
date: 2023-09-10 9:00:00
categories:
  - 排序模型
tags:
  - 多任务模型 
  - din 
  - dnn
  - Recommender systems
description: Improved Learning of Category Feature Embeddings; Incorporation of Second-Order and High-Order Feature Interactions 
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_194956.png
---



## The main contributions of the PNN 

1. **Improved Learning of Category Feature Embeddings:** PNN leverages a product-based structure that enhances the learning of category feature embeddings. This structure allows the model to capture intricate relationships and interactions between category features more effectively. By using product-based interactions, PNN can represent the feature interactions in a richer and more expressive way.

2. **Incorporation of Second-Order and High-Order Feature Interactions:** The PNN model combines second-order feature interactions (captured through the product-based structure) with high-order feature interactions learned using fully connected layers (dense layers). This combination of interactions allows PNN to model both pairwise and higher-order feature interactions, making it capable of capturing complex and non-linear patterns in the data.

In summary, PNN improves category feature representation through product-based structures and successfully combines second-order and high-order feature interactions, making it a powerful model for capturing intricate relationships and patterns in categorical data.





## model 架构

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_194956.png)


The structure of the PNN (Product-based Neural Network) model is designed with the consideration of handling large-scale sparse features commonly encountered in industrial applications, especially those derived from one-hot or multi-hot encoding of category features. The PNN model primarily deals with category features.

1. **Input Layer:** This is the feature input layer where features are represented as fields, akin to the concept of fields in the FFM (Field-aware Factorization Machines) model. Each field represents a group of related features.

2. **Embedding Layer Pair-wise Connected:** In this layer, each feature within a field is transformed into an embedding representation. This layer captures the embeddings of features at the field level.

3. **Product Layer Pair-wise Connected:** The Product Layer computes pair-wise interactions (second-order feature interactions) between the embeddings of features. This layer calculates the element-wise product of feature embeddings and captures the interactions between feature pairs.

4. **Hidden Layers:** These are the hidden layers of a multi-layer perceptron (MLP). The purpose of the hidden layers is to capture high-order feature interactions and model complex, non-linear relationships within the data. The output of the Product Layer is concatenated with other features and fed into these hidden layers for further processing.

In summary, the PNN model is designed to handle sparse category features, and its architecture incorporates the calculation of second-order feature interactions through the Product Layer and captures higher-order interactions through hidden layers of an MLP. This design enables PNN to effectively model intricate feature relationships in large-scale sparse datasets commonly found in industrial applications.


## two explored ways of feature interaction through product operations


1. **Input Layer (Fields):** The features are organized into fields, where each field represents a group of related features.

2. **Embedding Layer Pair-wisely Connected:** Within each field, features are transformed into embedding representations. This layer captures the embeddings of features at the field level.

3. **Product Layer Pair-wisely Connected (Inner Product):** In the conventional product-based interaction, inner products are calculated between the embeddings of features. For each pair of feature embeddings, an inner product is computed. This inner product operation captures the second-order feature interactions.

4. **Hidden Layers:** The output from the Product Layer is concatenated with other features and fed into hidden layers of an MLP. These hidden layers capture higher-order feature interactions and non-linear relationships in the data.

Now, let's discuss the second way of feature interaction using outer product:

5. **Product Layer Pair-wisely Connected (Outer Product):** In this variation, the Product Layer computes outer products between the feature embeddings. Each pair of feature embeddings results in an outer product matrix. The outer product operation captures richer and more complex interactions between feature pairs.

Both inner product and outer product interactions allow the PNN model to capture different aspects of feature interactions. Inner product focuses on linear interactions, while outer product captures more complex and non-linear interactions. These interactions contribute to the model's ability to represent and learn from the feature interactions effectively.


## model compare 

1. **FNN Model (Field-aware Neural Network):** When the LP (Local Product) module is removed from the PNN model, PNN is essentially equivalent to the FNN model. The FNN model focuses on capturing feature interactions within fields and employs a fully connected network for higher-order interactions.

2. **FM Model (Factorization Machine):** If you remove the hidden layers from the PNN model and configure the output layer to simply perform a weighted sum of the outputs from the Product Layer, the PNN model becomes quite similar to the FM model. The FM model primarily captures second-order feature interactions through inner products and does not include higher-order interactions.

3. **Combination of IPNN and OPNN:** The PNN model offers flexibility by allowing the combination of the inner product (IPNN) and outer product (OPNN) modules. This means that you can choose to use both inner and outer product interactions to model feature interactions in a more comprehensive manner, resulting in a more powerful PNN model.

In summary, PNN combines aspects of different models by incorporating both inner product and outer product interactions. It can resemble FNN when focusing on field-level interactions and FM when emphasizing second-order interactions. The flexibility to combine inner and outer product interactions makes PNN versatile and adaptable to various modeling needs.


## Performance Comparison
 

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_195701.png)




## 原文link

https://arxiv.org/abs/1611.00144




