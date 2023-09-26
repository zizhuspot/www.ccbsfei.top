---
title: nfm  Neural Factorization Machines for Sparse Predictive Analytics 新加坡国立大学 
date: 2023-09-13 13:00:00
categories:
  - 排序模型
tags:
  - 预估模型 
  - nfm 
  - fm 
  - Recommender systems
description:  a model that could both learn high-order features and capture non-linear relationships effectively, which led to the development of the NFM model.
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_192314.png
---

## introduction 

许多Web应用程序的预测任务需要建模分类变量，例如用户ID以及性别和职业等人口统计信息。

为了应用标准的机器学习技术，这些分类预测变量通常会通过独热编码转换为一组二进制特征，使得生成的特征向量高度稀疏。为了有效地从这样的稀疏数据中学习，必须考虑特征之间的相互作用。

因子分解机（FMs）是用于高效使用二阶特征交互的流行解决方案。

但是，FM模型以线性方式处理特征交互，这可能无法捕捉现实世界数据的非线性和复杂内在结构。虽然深度神经网络最近已被应用于学习行业中的非线性特征交互，例如Google的Wide&Deep和微软的DeepCross，但深层结构同时使它们难以训练。

在本文中，我们提出了一种新颖的模型——神经因子分解机（NFM），用于处理稀疏环境下的预测。

NFM无缝地结合了FM在建模二阶特征交互方面的线性性质和神经网络在建模高阶特征交互方面的非线性性质。

从概念上讲，NFM比FM更具表现力，因为FM可以看作是没有隐藏层的NFM的特殊情况。在两个回归任务的实证结果中，仅使用一个隐藏层的NFM相对于FM显著提高了7.3%。

与最近的深度学习方法Wide&Deep和DeepCross相比，我们的NFM结构更浅，但性能更好，实际中更容易训练和调整。

## 痛点 

NFM (Neural Factorization Machine) can be seen as an improvement over both FM (Factorization Machines) and FNN (Deep Factorization Neural Network), addressing their respective shortcomings as follows:

1. **FM Model Limitation:** While the FM model effectively captures cross-feature interactions, it still models these interactions linearly, which means it cannot capture non-linear relationships between features.

2. **FNN Model Limitation:** The FNN model attempts to overcome the limitations of FM by using FM for feature vector initialization at the lower layers and then using a DNN (Deep Neural Network) to learn high-order non-linear features at the upper layers. However, it relies on concatenating individual feature embeddings and then learning cross-feature interactions in subsequent DNN layers. In practical applications, this approach may not effectively capture cross-feature interactions to the desired extent.

Therefore, there was a need for a model that could both learn high-order features and capture non-linear relationships effectively, which led to the development of the NFM model.


## 目标函数

The objective function of the NFM (Neural Factorization Machine) model is typically represented as follows:

\[ \text{NFM Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right) + \lambda \cdot \Omega(\theta) \]

Where:
- \(N\) is the number of training examples.
- \(y_i\) represents the true label for the \(i\)-th example.
- \(\hat{y}_i\) is the predicted output (probability) for the \(i\)-th example by the NFM model.
- \(\lambda\) is the regularization parameter.
- \(\theta\) represents the model's parameters.
- \(\Omega(\theta)\) is the regularization term used to prevent overfitting. It can be L1 regularization, L2 regularization, or a combination of both.

The goal of training the NFM model is to minimize this loss function, which combines a logistic loss term to measure the prediction error and a regularization term to control the complexity of the model. By minimizing this objective function, the model aims to make accurate predictions while avoiding overfitting.




## network 结构


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_192314.png)


I apologize for the previous response, but I'm unable to view or interpret images or diagrams. However, I can describe the typical architecture of an NFM (Neural Factorization Machine) model in text.

The NFM architecture combines the strengths of Factorization Machines (FMs) and neural networks to capture both linear and non-linear interactions between features. Here's a textual representation of the NFM structure:

1. **Input Layer:** The model takes input features, which are usually one-hot encoded or embedded into continuous vectors.

2. **Embedding Layer:** Each input feature is passed through an embedding layer, which converts them into dense vectors. These dense vectors capture the latent representations of the features.

3. **Pairwise Interaction Layer (Factorization Machine):** The pairwise interaction layer calculates the interactions between all pairs of feature embeddings. It computes the dot product between each pair of embeddings and then aggregates these interactions. This step captures second-order feature interactions and is similar to the FM component.

4. **Fully Connected Neural Network:** In addition to the FM component, the NFM includes a fully connected neural network or deep learning layers. These layers can have multiple hidden layers and neurons, allowing the model to learn complex, high-order feature interactions. This part of the network captures non-linear relationships among features.

5. **Output Layer:** The final layer typically consists of a single neuron with a sigmoid activation function. It produces the predicted output, which represents the probability of the positive class in a binary classification problem.

6. **Objective Function:** The model is trained to minimize an objective function, usually a combination of a logistic loss term (to measure prediction error) and a regularization term (to prevent overfitting).

The NFM architecture leverages both the FM component for capturing pairwise feature interactions and the neural network component for modeling higher-order, non-linear interactions. This combination makes it capable of learning complex patterns in the data.


### nfm输出：

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_192729.png)



## 原文link

https://arxiv.org/abs/1708.05027




