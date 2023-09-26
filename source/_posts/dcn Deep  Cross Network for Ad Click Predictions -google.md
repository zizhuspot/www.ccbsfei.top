---
title: dcn Deep  Cross Network for Ad Click Predictions -google
date: 2023-09-15 13:10:00
categories:
  - 排序模型
tags:
  - 预估模型 
  - dcn 
  - 特征工程 
  - Recommender systems
description: In this paper, we introduce the **Deep&Cross Network (DCN)** model, which not only retains the advantages of DNN models but also more effectively learns bounded-degree feature interactions by introducing a Cross Network. DCN demonstrates feature crossing at each layer, eliminating the need for manual feature engineering. This additional complexity compared to DNN models can be negligible. 
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_131407.png

## brief introduction

**Feature engineering** has often been a crucial factor in the success of many predictive models. 

However, it often requires manual feature engineering or brute-force search. While deep neural networks (DNNs) can automatically learn feature interactions, they generate all feature interactions implicitly, and not all types of cross-features are useful. 

In this paper, we introduce the **Deep&Cross Network (DCN)** model, which not only retains the advantages of DNN models but also more effectively learns bounded-degree feature interactions by introducing a Cross Network. 

Specifically, DCN demonstrates feature crossing at each layer, eliminating the need for manual feature engineering. This additional complexity compared to DNN models can be negligible. 

Experimental results show that DCN performs well on both CTR prediction datasets and dense classification datasets.


## main contributions 

In this paper, we introduce the **Deep&Cross Network (DCN)** model, which is capable of automatically learning features from both sparse and dense inputs. DCN effectively captures valuable feature interactions within bounded degrees, allowing for higher-level non-linear interactions without the need for manual feature engineering or brute-force searching. It also comes with lower computational costs.

The main contributions of this paper are as follows:

1. We propose a **Cross Network** that explicitly applies feature interactions at each layer, effectively learning bounded-degree cross-features without the need for manual feature engineering or brute-force searching.

2. The Cross Network is simple yet effective. Through its design, the maximum polynomial degree increases layer by layer and is determined by the network depth.

3. The Cross Network is memory-efficient and easy to implement.

4. Experimental results demonstrate that DCN achieves lower log loss compared to DNN models, while having fewer parameters, reducing the complexity by nearly a factor of the data size.





### model 架构 

In this section, we will describe the network architecture of the DCN model.

A DCN model starts with embedding and stacking layers, followed by parallel Cross Networks and deep networks, and finally, it includes a combination layer that merges the outputs of both networks. The complete DCN model is illustrated in Figure 1.


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_131407.png)


Embedding and Stacking Layer

We consider input data that consists of both sparse and dense features. In large-scale network recommendation systems, most inputs are categorical features, such as "country=usa." Categorical features are typically one-hot encoded, resulting in excessively high-dimensional feature spaces.

To reduce dimensionality, we employ an embedding process to transform binary features into dense real-valued vectors, often referred to as embedding vectors:

\[E_{i} = \sigma(W_{i} \cdot X_i)\]

Where:
- \(E_{i}\) is the embedding vector.
- \(X_i\) is the binary input for the \(i\)-th category.
- \(W_{i}\) is the corresponding embedding matrix, optimized along with other parameters in the network.
- \(k\) is the embedding size, and \(v\) is the vocabulary size.

Finally, we concatenate the embedding vectors and normalized dense features \(x_{dense}\) into a single vector \(x\) to input into the network:

\[x = [E_{1}; E_{2}; ...; E_{n}; x_{dense}]\]

Please note that the notation \(E_{i}\) represents the embedding vector for the \(i\)-th categorical feature.



### 可视化交叉层

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_131837.png)

High-degree Interaction Across Features. The special structure of the Cross Network results in an increase in the degree of feature interactions as the network depth increases. For the \(l\)-th layer, the highest polynomial degree is \(l+1\). In fact, the Cross Network includes all interaction terms from 1 to \(l+1\). Detailed analysis is provided in Section 3.

Complexity Analysis. Let \(L\) represent the number of layers in the Cross Network, and \(D\) represent the input dimension. Then, the number of parameters in the Cross Network is given by:

\[O(L \cdot D)\]

The time and space complexity of the Cross Network is linear with respect to the input dimension. Therefore, compared to the Deep part, the Cross Network introduces negligible complexity, and the overall complexity of the DCN model remains at the same level as traditional DNN models.

The limited number of parameters in the Cross Network restricts the model's capacity. To capture highly nonlinear interactions, we introduce a deep network in parallel.




## 实验

### Criteo Display Ads Data

The Criteo Display Ads dataset is used for predicting ad click-through rates. It consists of 13 integer features and 26 categorical features, each with a high cardinality. In this dataset, even a small improvement in log loss, such as 0.001, is considered meaningful. When considering a large user base, a slight increase in prediction accuracy can lead to significant revenue growth for a company. The dataset contains 7 days of user logs, totaling 11 GB (approximately 41 million records). We use data from the first 6 days for training and randomly split the data from the 7th day into equal-sized validation and test sets.

### Implementation Details

The DCN network is implemented using TensorFlow, and we will briefly discuss some details of training the DCN.

Data processing and embedding: Real-valued features are normalized using a logarithmic transformation. Categorical features are transformed into dense vectors of dimension \(k\) using embedding techniques, and all embedding vectors are concatenated into a 1026-dimensional vector.

Optimization: We use the Adam optimizer for mini-batch stochastic optimization with a batch size of 512. Batch Normalization is applied in the deep network, and the gradient clip norm is set to 100.

Regularization: We use early stopping, as we did not find L2 regularization or dropout to be effective.

Hyperparameters: We perform a grid search to find the optimal number of hidden layers, hidden layer sizes, initial learning rate values, and the number of layers in the Cross Network. The search ranges from 2 to 5 for the number of hidden layers, 32 to 1024 for the hidden layer sizes, 1 to 6 for the Cross Network layers, and the initial learning rate is adjusted from 0.0001 to 0.001 in increments of 0.0001. All experimental results are trained for up to 150,000 iterations with early stopping unless overfitting occurred during the initial stages of training.

### Models for Comparisons

We compare the DCN model with the following five models: DNN, LR (Logistic Regression), FM (Factorization Machines), Wide & Deep, and Deep Crossing (DC).

### Model Performance

In this section, we first show the best results of different models in terms of the log loss metric. Then, we provide a detailed comparison between DCN and DNN, with a focus on the impact of introducing the Cross Network.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_132031.png)



## 原文link

https://arxiv.org/pdf/1708.05123.pdf





