---
title: dcn模型的详细解析-（交叉网络更有效地学习有界度特征交互） -google
date: 2023-09-15 13:10:00
categories:
  - 排序模型
tags:
  - 预估模型 
  - dcn 
  - 特征工程 
  - Recommender systems
description: In this paper, we introduce the **Deep&Cross Network (DCN)** model, which not only retains the advantages of DNN models but also more effectively learns bounded-degree feature interactions by introducing a Cross Network. DCN demonstrates feature crossing at each layer, eliminating the need for manual feature engineering. This additional complexity compared to DNN models can be negligible.  在本论文中，我们介绍了**Deep&Cross Network (DCN)** 模型，该模型不仅保留了DNN模型的优点，还通过引入交叉网络更有效地学习有界度特征交互。DCN在每一层展示了特征的交叉，消除了手动特征工程的需求。与DNN模型相比，这种附加复杂性可能是可以忽略的。  
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_131407.png
---

## brief introduction

**Feature engineering** has often been a crucial factor in the success of many predictive models. 
**特征工程**通常是许多预测模型成功的关键因素。

However, it often requires manual feature engineering or brute-force search. While deep neural networks (DNNs) can automatically learn feature interactions, they generate all feature interactions implicitly, and not all types of cross-features are useful. 

然而，它通常需要手动特征工程或蛮力搜索。虽然深度神经网络（DNNs）可以自动学习特征交互，但它们隐式生成所有特征交互，而不是所有类型的交叉特征都有用。

In this paper, we introduce the **Deep&Cross Network (DCN)** model, which not only retains the advantages of DNN models but also more effectively learns bounded-degree feature interactions by introducing a Cross Network. 

在本论文中，我们介绍了**Deep&Cross Network (DCN)** 模型，它不仅保留了DNN模型的优点，还通过引入交叉网络更有效地学习有界度特征交互。

Specifically, DCN demonstrates feature crossing at each layer, eliminating the need for manual feature engineering. This additional complexity compared to DNN models can be negligible. 

具体而言，DCN在每一层展示了特征的交叉，消除了手动特征工程的需求。与DNN模型相比，这种额外的复杂性可能是可以忽略的。

Experimental results show that DCN performs well on both CTR prediction datasets and dense classification datasets.


实验结果显示，DCN在CTR预测数据集和稠密分类数据集上表现出色。

## main contributions 

In this paper, we introduce the **Deep&Cross Network (DCN)** model, which is capable of automatically learning features from both sparse and dense inputs. DCN effectively captures valuable feature interactions within bounded degrees, allowing for higher-level non-linear interactions without the need for manual feature engineering or brute-force searching. It also comes with lower computational costs.
在这篇论文中，我们介绍了**Deep&Cross Network (DCN)** 模型，它能够自动从稀疏和稠密输入中学习特征。DCN有效地捕捉有界度内有价值的特征交互，允许高级非线性交互，而无需手动特征工程或蛮力搜索。它还具有较低的计算成本。

The main contributions of this paper are as follows:

本文的主要贡献如下：

1. We propose a **Cross Network** that explicitly applies feature interactions at each layer, effectively learning bounded-degree cross-features without the need for manual feature engineering or brute-force searching.

1. 我们提出了一种**交叉网络**，它在每一层明确应用特征交互，有效地学习了有界度的交叉特征，而无需手动特征工程或蛮力搜索。

2. The Cross Network is simple yet effective. Through its design, the maximum polynomial degree increases layer by layer and is determined by the network depth.

2. 交叉网络简单而有效。通过其设计，多项式的最大次数逐层增加，并由网络深度决定。

3. The Cross Network is memory-efficient and easy to implement.

3. 交叉网络内存高效，易于实现。

4. Experimental results demonstrate that DCN achieves lower log loss compared to DNN models, while having fewer parameters, reducing the complexity by nearly a factor of the data size.

4. 实验结果表明，与DNN模型相比，DCN实现了更低的对数损失，同时参数更少，将复杂性减小了近乎数据规模的倍数。




### model 架构 

In this section, we will describe the network architecture of the DCN model.

A DCN model starts with embedding and stacking layers, followed by parallel Cross Networks and deep networks, and finally, it includes a combination layer that merges the outputs of both networks. The complete DCN model is illustrated in Figure 1.
在本节中，我们将描述DCN模型的网络架构。

DCN模型以嵌入和堆叠层开始，然后是并行的交叉网络和深度网络，最后包括一个合并两个网络输出的组合层。完整的DCN模型如图1所示。

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_131407.png)


Embedding and Stacking Layer
嵌入和堆叠层

We consider input data that consists of both sparse and dense features. In large-scale network recommendation systems, most inputs are categorical features, such as "country=usa." Categorical features are typically one-hot encoded, resulting in excessively high-dimensional feature spaces.

我们考虑的输入数据包括稀疏特征和密集特征。在大规模网络推荐系统中，大多数输入都是分类特征，例如"country=usa"。分类特征通常采用独热编码，导致特征空间维度过高。

To reduce dimensionality, we employ an embedding process to transform binary features into dense real-valued vectors, often referred to as embedding vectors:

为了降低维度，我们使用嵌入过程将二进制特征转换为密集的实值向量，通常称为嵌入向量：

\[E_{i} = \sigma(W_{i} \cdot X_i)\]

Where:
- \(E_{i}\) is the embedding vector.
- \(X_i\) is the binary input for the \(i\)-th category.
- \(W_{i}\) is the corresponding embedding matrix, optimized along with other parameters in the network.
- \(k\) is the embedding size, and \(v\) is the vocabulary size.

其中：
- \(E_{i}\) 是嵌入向量。
- \(X_i\) 是第 \(i\) 个类别的二进制输入。
- \(W_{i}\) 是相应的嵌入矩阵，与网络中的其他参数一起进行优化。
- \(k\) 是嵌入的大小，\(v\) 是词汇表的大小。

Finally, we concatenate the embedding vectors and normalized dense features \(x_{dense}\) into a single vector \(x\) to input into the network:

最后，我们将嵌入向量和归一化的密集特征 \(x_{dense}\) 连接成一个输入网络的单一向量 \(x\)：


\[x = [E_{1}; E_{2}; ...; E_{n}; x_{dense}]\]



### 可视化交叉层

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_131837.png)

High-degree Interaction Across Features. The special structure of the Cross Network results in an increase in the degree of feature interactions as the network depth increases. For the \(l\)-th layer, the highest polynomial degree is \(l+1\). In fact, the Cross Network includes all interaction terms from 1 to \(l+1\). Detailed analysis is provided in Section 3.
跨特征高阶交互。Cross Network的特殊结构导致随着网络深度的增加，特征交互的阶数也增加。对于第 \(l\) 层，最高多项式阶数为 \(l+1\)。实际上，Cross Network包括从1到 \(l+1\) 的所有交互项。详细分析请参见第3节。

Complexity Analysis. Let \(L\) represent the number of layers in the Cross Network, and \(D\) represent the input dimension. Then, the number of parameters in the Cross Network is given by:

复杂性分析。设 \(L\) 表示Cross Network中的层数，\(D\) 表示输入维度。那么，Cross Network中的参数数量为：

\[O(L \cdot D)\]

The time and space complexity of the Cross Network is linear with respect to the input dimension. Therefore, compared to the Deep part, the Cross Network introduces negligible complexity, and the overall complexity of the DCN model remains at the same level as traditional DNN models.

The limited number of parameters in the Cross Network restricts the model's capacity. To capture highly nonlinear interactions, we introduce a deep network in parallel.

Cross Network的时间和空间复杂度与输入维度呈线性关系。因此，与深度部分相比，Cross Network引入的复杂性可以忽略不计，DCN模型的整体复杂性保持在与传统DNN模型相同的水平。

Cross Network中的参数数量有限，限制了模型的容量。为了捕获高度非线性的交互，我们引入了一个深度网络并行使用。



## 实验

### Criteo Display Ads Data

The Criteo Display Ads dataset is used for predicting ad click-through rates. It consists of 13 integer features and 26 categorical features, each with a high cardinality. In this dataset, even a small improvement in log loss, such as 0.001, is considered meaningful. When considering a large user base, a slight increase in prediction accuracy can lead to significant revenue growth for a company. The dataset contains 7 days of user logs, totaling 11 GB (approximately 41 million records). We use data from the first 6 days for training and randomly split the data from the 7th day into equal-sized validation and test sets.
### Criteo展示广告数据

Criteo展示广告数据集用于预测广告点击率。它包含13个整数特征和26个分类特征，每个特征都具有很高的基数。在该数据集中，即使对数损失小幅提高，如0.001，也被认为具有重要意义。在考虑到庞大的用户群体时，预测准确性的轻微提高可以导致公司的显著收入增长。数据集包含7天的用户日志，总计11 GB（约4100万条记录）。我们使用前6天的数据进行训练，并将第7天的数据随机分成相等大小的验证集和测试集。

### Implementation Details

The DCN network is implemented using TensorFlow, and we will briefly discuss some details of training the DCN.

Data processing and embedding: Real-valued features are normalized using a logarithmic transformation. Categorical features are transformed into dense vectors of dimension \(k\) using embedding techniques, and all embedding vectors are concatenated into a 1026-dimensional vector.

Optimization: We use the Adam optimizer for mini-batch stochastic optimization with a batch size of 512. Batch Normalization is applied in the deep network, and the gradient clip norm is set to 100.

Regularization: We use early stopping, as we did not find L2 regularization or dropout to be effective.

Hyperparameters: We perform a grid search to find the optimal number of hidden layers, hidden layer sizes, initial learning rate values, and the number of layers in the Cross Network. The search ranges from 2 to 5 for the number of hidden layers, 32 to 1024 for the hidden layer sizes, 1 to 6 for the Cross Network layers, and the initial learning rate is adjusted from 0.0001 to 0.001 in increments of 0.0001. All experimental results are trained for up to 150,000 iterations with early stopping unless overfitting occurred during the initial stages of training.

### 实现细节

DCN网络使用TensorFlow实现，我们将简要讨论训练DCN的一些细节。

数据处理和嵌入：实值特征通过对数变换进行标准化。分类特征使用嵌入技术转换为维度 \(k\) 的稠密向量，然后将所有嵌入向量串联成一个1026维的向量。

优化：我们使用Adam优化器进行小批量随机优化，批量大小为512。在深度网络中应用批量归一化，梯度剪切范数设置为100。

正则化：我们使用提前停止，因为我们发现L2正则化或丢弃不够有效。

超参数：我们执行网格搜索，以找到最佳的隐藏层数、隐藏层大小、初始学习率值和Cross Network层数。搜索范围为隐藏层数为2到5，隐藏层大小为32到1024，Cross Network层数为1到6，初始学习率从0.0001到0.001，每次增加0.0001。除非在训练的初始阶段出现过拟合，否则所有实验结果都进行了高达150,000次迭代的训练并采用提前停止。

### Models for Comparisons

We compare the DCN model with the following five models: DNN, LR (Logistic Regression), FM (Factorization Machines), Wide & Deep, and Deep Crossing (DC).

### 用于比较的模型

我们将DCN模型与以下五种模型进行比较：DNN、LR（逻辑回归）、FM（因子分解机）、Wide & Deep以及Deep Crossing（DC）。

### Model Performance

In this section, we first show the best results of different models in terms of the log loss metric. Then, we provide a detailed comparison between DCN and DNN, with a focus on the impact of introducing the Cross Network.

### 模型性能

在本节中，我们首先展示了不同模型在对数损失度量方面的最佳结果。然后，我们提供了DCN和DNN之间的详细比较，重点关注引入Cross Network的影响。

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_132031.png)



## 原文link

https://arxiv.org/pdf/1708.05123.pdf





