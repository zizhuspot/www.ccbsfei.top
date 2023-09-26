---
title: deepfm  A Factorization-Machine based Neural Network for CTR Prediction 哈工大
date: 2023-09-14 14:10:00
categories:
  - 排序模型
tags:
  - 预估模型 
  - deepfm 
  - wide 
  - Recommender systems
description: DeepFM draws inspiration from the structure of the Wide & Deep model but replaces the Wide component with the Factorization Machine (FM) model. This eliminates the need for manual feature engineering. What makes DeepFM particularly clever is that it shares weights between the second-order part of the FM model and the embedding layer of the neural network. This weight sharing reduces the number of parameters significantly and speeds up the training process.
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_133156.png
---


## brief introduction

This article introduces the highly regarded DeepFM model for click-through rate (CTR) prediction. The model was first introduced in a paper titled DeepFM  A Factorization-Machine based Neural Network for CTR Prediction," published in 2017 by researchers from Harbin Institute of Technology Shenzhen and Huawei Noah's Ark Lab.

DeepFM draws inspiration from the structure of the Wide & Deep model but replaces the Wide component with the Factorization Machine (FM) model. This eliminates the need for manual feature engineering. What makes DeepFM particularly clever is that it shares weights between the second-order part of the FM model and the embedding layer of the neural network. This weight sharing reduces the number of parameters significantly and speeds up the training process.



## main contributions 

1. Feature interactions are crucial for click-through rate (CTR) prediction.
2. Linear models are not suitable for capturing feature interactions, but they can capture some through manual feature engineering.
3. Factorization Machines (FM) are commonly used for learning second-order feature interactions.
4. Neural network models are well-suited for capturing high-order feature interactions, CNN-based models are good at capturing interactions between adjacent features, and RNN-based models are suitable for data with temporal dependencies.
5. FNN: A model that pre-trains an FM model before the DNN.
6. PNN: A model that adds a inner product layer between the embedding layer and the fully connected layer.
7. Wide & Deep: A model combining linear and deep components, which require constructing different inputs for each.

The main contributions of the paper are as follows:

1. DeepFM consists of both an FM component and a Deep component. The FM part is responsible for learning low-order feature interactions, while the Deep part captures high-order interactions. Compared to Wide & Deep, DeepFM can be trained end-to-end without the need for feature engineering.
2. DeepFM shares inputs and embedding vectors.
3. DeepFM achieves an improvement in CTR prediction compared to previous models.


### model 架构 

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_133156.png)


In the figure above, the left side represents the FM (Factorization Machine) model, and the right side represents the Deep model. In this model, the parameters from both the FM component and the Deep neural network component are jointly trained. The model's output is the predicted Click-Through Rate (CTR), as shown below:

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_133453.png)




### FM part

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_133542.png)

The formula for the FM (Factorization Machine) model is as follows:

FM(x) = W_0 + Σ_(i=1)^n (W_i * x_i) + Σ_(i=1)^n (Σ_(j=i+1)^n (〖(V_i)T  (V_j)〗^T * x_i * x_j))

Where:
- FM(x) represents the output of the FM model for input vector x.
- W_0 is the bias term.
- W_i represents the weight parameter for feature x_i.
- V_i represents the embedding vector for feature x_i.
- x_i represents the i-th input feature.
- n is the total number of input features.

This formula describes how the FM model computes interactions between input features, including first-order (linear) interactions and second-order (pairwise) interactions, to make predictions.


### Deep part 

The Deep part is a feedforward neural network used to learn high-order feature interactions.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_133651.png)


### Embedding Layer

The neural network's input should be continuous and dense, while the original data in CTR prediction is typically highly sparse and high-dimensional. 

Therefore, an embedding layer should be added between the raw data and the first hidden layer to transform sparse feature data into dense data.



The paper emphasizes two key characteristics of the embedding layer:

1. The embedding vectors in the embedding layer have the same dimension as the hidden vectors in FM.
2. The embedding vectors in the embedding layer are initialized with the hidden vectors from FM.

I think the first point is somewhat redundant.

In reality, hidden vectors and embedding vectors are conceptually the same, both representing a sparse feature with a vector. The difference lies in how they are used: FM uses the dot product of hidden vectors for pairwise feature interactions, while the embedding layer maps each original feature to a vector, making it easier for neural network models to process.

The paper points out two advantages of sharing feature embeddings between the FM and Deep parts:

1. It allows the model to learn both low-order and high-order feature interactions directly from raw data.
2. It eliminates the need for manual feature engineering.





## Experiments

**1. Datasets**
   - Criteo Dataset
   - Company∗ Dataset: Huawei App Store, 7 days for training, 1 day for testing.

**2. Evaluation Metrics**
   - AUC (Area Under the Receiver Operating Characteristic Curve)
   - Logloss (Logarithmic Loss)

**3. Parameters**
   - Dropout rate: 0.5
   - Network structure: 400-400-400
   - Optimizer: Adam
   - Activation function: tanh for IPNN, relu for other deep models
   - FM latent dimension: 10


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_133842.png)



## 原文link

https://arxiv.org/abs/1703.04247


