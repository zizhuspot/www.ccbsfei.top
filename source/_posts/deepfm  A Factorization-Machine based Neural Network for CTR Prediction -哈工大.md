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
description: DeepFM draws inspiration from the structure of the Wide & Deep model but replaces the Wide component with the Factorization Machine (FM) model. This eliminates the need for manual feature engineering. What makes DeepFM particularly clever is that it shares weights between the second-order part of the FM model and the embedding layer of the neural network. This weight sharing reduces the number of parameters significantly and speeds up the training process. DeepFM从Wide & Deep模型的结构中汲取灵感，但将Wide组件替换为因子分解机（FM）模型。这消除了手动特征工程的需求。DeepFM特别巧妙之处在于它在FM模型的二阶部分和神经网络的嵌入层之间共享权重。这种权重共享显著减少了参数数量并加快了训练过程。 
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_133156.png
---


## brief introduction

This article introduces the highly regarded DeepFM model for click-through rate (CTR) prediction. The model was first introduced in a paper titled DeepFM  A Factorization-Machine based Neural Network for CTR Prediction," published in 2017 by researchers from Harbin Institute of Technology Shenzhen and Huawei Noah's Ark Lab.
本文介绍了备受推崇的DeepFM模型，用于点击率（CTR）预测。该模型首次在一篇名为“DeepFM：一种基于因子分解机的神经网络用于CTR预测”的论文中提出，该论文由哈尔滨工业大学深圳校区和华为诺亚方舟实验室的研究人员于2017年发表。

DeepFM draws inspiration from the structure of the Wide & Deep model but replaces the Wide component with the Factorization Machine (FM) model. This eliminates the need for manual feature engineering. What makes DeepFM particularly clever is that it shares weights between the second-order part of the FM model and the embedding layer of the neural network. This weight sharing reduces the number of parameters significantly and speeds up the training process.


DeepFM汲取了Wide & Deep模型的结构灵感，但用因子分解机（FM）模型替代了Wide组件。这消除了手动特征工程的需求。DeepFM的巧妙之处在于它在FM模型的二阶部分和神经网络的嵌入层之间共享权重。这种权重共享显著减少了参数数量并加快了训练过程。

## main contributions 

1. Feature interactions are crucial for click-through rate (CTR) prediction.
2. Linear models are not suitable for capturing feature interactions, but they can capture some through manual feature engineering.
3. Factorization Machines (FM) are commonly used for learning second-order feature interactions.
4. Neural network models are well-suited for capturing high-order feature interactions, CNN-based models are good at capturing interactions between adjacent features, and RNN-based models are suitable for data with temporal dependencies.
5. FNN: A model that pre-trains an FM model before the DNN.
6. PNN: A model that adds a inner product layer between the embedding layer and the fully connected layer.
7. Wide & Deep: A model combining linear and deep components, which require constructing different inputs for each.
本文的主要贡献如下：

1. DeepFM由FM（因子分解机）组件和深度组件组成。FM部分负责学习低阶特征交互，而深度部分捕捉高阶交互。与Wide & Deep相比，DeepFM可以端到端地进行训练，无需进行特征工程。
2. DeepFM共享输入和嵌入向量。
3. DeepFM在CTR预测方面相对于以前的模型取得了改进。

The main contributions of the paper are as follows:

1. DeepFM consists of both an FM component and a Deep component. The FM part is responsible for learning low-order feature interactions, while the Deep part captures high-order interactions. Compared to Wide & Deep, DeepFM can be trained end-to-end without the need for feature engineering.
2. DeepFM shares inputs and embedding vectors.
3. DeepFM achieves an improvement in CTR prediction compared to previous models.

文章中的关键观点如下：

1. 特征交互对于点击率（CTR）预测至关重要。
2. 线性模型不适合捕捉特征交互，但可以通过手动特征工程捕捉一些交互。
3. 因子分解机（FM）常用于学习二阶特征交互。
4. 神经网络模型适合捕捉高阶特征交互，基于CNN的模型适合捕捉相邻特征之间的交互，而基于RNN的模型适用于具有时间依赖性的数据。
5. FNN：在DNN之前对FM模型进行预训练的模型。
6. PNN：在嵌入层和全连接层之间添加内积层的模型。
7. Wide & Deep：结合了线性和深度组件的模型，需要为每个组件构建不同的输入。

### model 架构 

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_133156.png)


In the figure above, the left side represents the FM (Factorization Machine) model, and the right side represents the Deep model. In this model, the parameters from both the FM component and the Deep neural network component are jointly trained. The model's output is the predicted Click-Through Rate (CTR), as shown below:
在上图中，左侧代表FM（因子分解机）模型，右侧代表深度模型。在这个模型中，来自FM组件和深度神经网络组件的参数是联合训练的。模型的输出是预测的点击率（CTR），如下所示：
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

其中：
- FM(x) 代表输入向量 x 的FM模型的输出。
- W_0 是偏置项。
- W_i 代表特征 x_i 的权重参数。
- V_i 代表特征 x_i 的嵌入向量。
- x_i 代表第i个输入特征。
- n 是输入特征的总数。

This formula describes how the FM model computes interactions between input features, including first-order (linear) interactions and second-order (pairwise) interactions, to make predictions.

这个公式描述了FM模型如何计算输入特征之间的交互，包括一阶（线性）交互和二阶（成对）交互，以进行预测。

### Deep part 

The Deep part is a feedforward neural network used to learn high-order feature interactions.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_133651.png)


### Embedding Layer

The neural network's input should be continuous and dense, while the original data in CTR prediction is typically highly sparse and high-dimensional. 

Therefore, an embedding layer should be added between the raw data and the first hidden layer to transform sparse feature data into dense data.

神经网络的输入应该是连续和密集的，而在CTR预测中的原始数据通常是高度稀疏和高维的。

因此，在原始数据和第一个隐藏层之间应该添加一个嵌入层，将稀疏特征数据转换为密集数据。


The paper emphasizes two key characteristics of the embedding layer:

1. The embedding vectors in the embedding layer have the same dimension as the hidden vectors in FM.
2. The embedding vectors in the embedding layer are initialized with the hidden vectors from FM.

该论文强调了嵌入层的两个关键特点：

1. 嵌入层中的嵌入向量与FM中的隐藏向量具有相同的维度。
2. 嵌入层中的嵌入向量是用FM中的隐藏向量进行初始化的。

I think the first point is somewhat redundant.

我认为第一个观点有些多余。

In reality, hidden vectors and embedding vectors are conceptually the same, both representing a sparse feature with a vector. The difference lies in how they are used: FM uses the dot product of hidden vectors for pairwise feature interactions, while the embedding layer maps each original feature to a vector, making it easier for neural network models to process.

实际上，隐藏向量和嵌入向量在概念上是相同的，都代表一个具有向量表示的稀疏特征。它们的区别在于它们的使用方式：FM使用隐藏向量的点积来进行成对特征交互，而嵌入层将每个原始特征映射到一个向量，使神经网络模型更容易处理。

The paper points out two advantages of sharing feature embeddings between the FM and Deep parts:

1. It allows the model to learn both low-order and high-order feature interactions directly from raw data.
2. It eliminates the need for manual feature engineering.


该论文指出了在FM和深度部分之间共享特征嵌入的两个优点：

1. 它允许模型直接从原始数据中学习低阶和高阶特征交互。
2. 它消除了手动特征工程的需要。



## Experiments

**1. Datasets**
   - Criteo Dataset
   - Company∗ Dataset: Huawei App Store, 7 days for training, 1 day for testing.
**1. 数据集**
   - Criteo数据集
   - 公司∗数据集：华为应用商店，7天用于训练，1天用于测试。

**2. Evaluation Metrics**
   - AUC (Area Under the Receiver Operating Characteristic Curve)
   - Logloss (Logarithmic Loss)

**2. 评估指标**
   - AUC（接收者操作特征曲线下面积）
   - Logloss（对数损失）

**3. Parameters**
   - Dropout rate: 0.5
   - Network structure: 400-400-400
   - Optimizer: Adam
   - Activation function: tanh for IPNN, relu for other deep models
   - FM latent dimension: 10


**3. 参数**
   - 丢弃率：0.5
   - 网络结构：400-400-400
   - 优化器：Adam
   - 激活函数：IPNN使用tanh，其他深度模型使用relu
   - FM潜在维度：10
   
![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_133842.png)



## 原文link

https://arxiv.org/abs/1703.04247


