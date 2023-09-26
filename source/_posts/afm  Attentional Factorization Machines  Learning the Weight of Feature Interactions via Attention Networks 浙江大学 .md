---
title: afm  Attentional Factorization Machines  Learning the Weight of Feature Interactions via Attention Networks 浙江大学 
date: 2023-09-12 11:00:00
categories:
  - 排序模型
tags:
  - 预估模型 
  - afm 
  - fm 
  - Recommender systems
description:  its most notable feature is the use of an attention network to learn the importance of different combinations of features.assign a weight to each combination feature (cross feature).
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_193155.png
---

## introduction 

AFM stands for Attentional Factorization Machine, and it is developed by the same author as NFM. AFM is an improvement over FM (Factorization Machine), and its most notable feature is the use of an attention network to learn the importance of different combinations of features.

In recommendation systems or click-through rate (CTR) prediction tasks, input data often includes a large number of categorical features. Since these categorical features are not independent, the combinations of these features become crucial. One simple approach is to assign a weight to each combination feature (cross feature). However, the drawback of this cross-feature-based method is that many combination features may not appear in the training dataset, making it challenging to effectively learn their weights.

FM addresses this by learning an embedding vector (also known as a latent vector) for each feature and representing the weight of a combination feature as the inner product of the embedding vectors of its constituent features. However, FM does not consider the fact that some features may be unimportant or even irrelevant in the prediction. These features can introduce noise and interfere with predictions. FM treats all feature combinations equally, assigning them the same weight (implicitly, a weight of 1).

In this paper, AFM introduces the attention mechanism, and it innovatively proposes a solution to assign different levels of importance to different feature combinations. The weights can be automatically learned within the network, eliminating the need for additional domain knowledge. This attention mechanism in AFM allows it to focus on relevant feature combinations while downplaying or ignoring less important ones, ultimately improving the model's predictive accuracy.




## 痛点 

NFM (Neural Factorization Machine) can be seen as an improvement over both FM (Factorization Machines) and FNN (Deep Factorization Neural Network), addressing their respective shortcomings as follows:

1. **FM Model Limitation:** While the FM model effectively captures cross-feature interactions, it still models these interactions linearly, which means it cannot capture non-linear relationships between features.

2. **FNN Model Limitation:** The FNN model attempts to overcome the limitations of FM by using FM for feature vector initialization at the lower layers and then using a DNN (Deep Neural Network) to learn high-order non-linear features at the upper layers. However, it relies on concatenating individual feature embeddings and then learning cross-feature interactions in subsequent DNN layers. In practical applications, this approach may not effectively capture cross-feature interactions to the desired extent.

Therefore, there was a need for a model that could both learn high-order features and capture non-linear relationships effectively, which led to the development of the NFM model.


## 目标函数

FM, which stands for Factorization Machine, can be formalized with the following equation:

\[ \hat{y}(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j \]

Where:
- \(\hat{y}(x)\) represents the predicted output or target variable for input \(x\).
- \(w_0\) is the bias term.
- \(w_i\) represents the weight associated with the \(i\)-th feature.
- \(x_i\) denotes the \(i\)-th feature value in the input \(x\).
- \(v_i\) represents the latent factor vector associated with the \(i\)-th feature.
- \(\langle v_i, v_j \rangle\) denotes the inner product or interaction between the latent factor vectors \(v_i\) and \(v_j\).
- The summation terms capture both linear and pairwise feature interactions.

FM is a model used for recommendation systems and predictive modeling, and it is particularly effective in capturing second-order feature interactions, making it suitable for problems involving categorical data and interactions between features.



In FM (Factorization Machines), there are two main issues:

1. **Shared Latent Vectors for Features:** FM uses the same latent vector for a feature when interacting with all other features. This limitation led to the development of Field-aware Factorization Machines (FFM) to address this issue. FFM allows each feature to have its own set of latent vectors, considering different interactions with other features in different fields.

2. **Equal Weights for All Feature Combinations:** FM assigns the same weight (implicitly, a weight of 1) to all feature combinations. This means that all combinations of features are treated equally, even though in practice, not all feature combinations are equally important. This issue is addressed by Attentional Factorization Machines (AFM), which optimizes the model by assigning different weights to different feature combinations. AFM allows the model to focus on relevant feature combinations while downplaying or ignoring less important ones. This not only improves predictive accuracy but also enhances model interpretability, as it allows for in-depth analysis of important feature combinations in subsequent research and applications.



## The full name of AFM is Attentional Factorization Machine.

The model structure of AFM is as follows:

### Model Structure

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_193155.png)


Pair-wise Interaction Layer in AFM is primarily responsible for modeling the interactions between different feature combinations. It takes the embeddings of the input features, which have been converted into dense vectors through the Embedding Layer, and computes the interactions between them. 

Let's formalize this process:

Assume you have "m" embedded vectors, each with a dimension of "k" (where "k" is the dimension of the embedding).

For each pair of embedded vectors (i, j) where i ≠ j, you calculate the element-wise product between them. This results in m(m-1)/2 combination vectors, each with a dimension of "k."

Mathematically, this can be represented as follows:

For i ≠ j:

\[ \text{Interaction Vector}_{ij} = \text{Embedding Vector}_i \odot \text{Embedding Vector}_j \]

Where:
- \(\text{Interaction Vector}_{ij}\) is the interaction vector between the i-th and j-th embedded vectors.
- \(\text{Embedding Vector}_i\) is the embedding vector for the i-th feature.
- \(\text{Embedding Vector}_j\) is the embedding vector for the j-th feature.
- \(\odot\) represents element-wise multiplication (Hadamard product).

This operation captures the pairwise interactions between features, which is a key component in the AFM model for learning the importance of different feature combinations.



### In AFM, an Attention Network is used to learn the importance of different feature combinations. 

The Attention Network is essentially a one-layer MLP (Multi-Layer Perceptron) with the ReLU activation function. The size of the network is determined by the attention factor, which represents the number of neurons in this layer.

The input to the Attention Network is the result of element-wise product of two embedding vectors, often referred to as the "interaction vector" or "interacted vector." These interaction vectors encode the feature combinations in the embedding space.

The output of the Attention Network is the Attention score corresponding to the feature combination. These scores represent how important or relevant each feature combination is to the final prediction. To ensure that the attention scores are interpretable and sum to 1, a softmax activation is applied to the output.

Mathematically, the Attention Network in AFM can be formalized as follows:

Let \( \text{Interacted Vector}_{ij} \) be the interaction vector between feature embeddings i and j, and \( \text{Attention Score}_{ij} \) be the attention score for this feature combination. The Attention Network can be represented as:

\[ \text{Attention Score}_{ij} = \text{softmax}(\text{ReLU}(\text{W}_a \cdot \text{Interacted Vector}_{ij})) \]

Where:
- \( \text{Attention Score}_{ij} \) is the attention score for the feature combination between embeddings i and j.
- \( \text{W}_a \) represents the weight matrix associated with the Attention Network.
- \(\text{ReLU}\) is the Rectified Linear Unit activation function.
- \( \cdot \) denotes matrix multiplication.
- \(\text{softmax}\) is the softmax activation function applied to normalize the attention scores.

The purpose of the Attention Network is to assign different attention scores to different feature combinations, effectively learning which combinations are more important for the prediction task.


## model train

In AFM, the choice of loss function depends on the specific task you are addressing. Here, let's focus on the case of regression, where the square loss (also known as mean squared error) is used as the loss function. The square loss measures the squared difference between predicted and actual values and is commonly used for regression problems.

Mathematically, the square loss for regression in AFM can be formalized as follows:

\[ \text{Loss} = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 \]

Where:
- \(\text{Loss}\) is the square loss.
- \(N\) is the number of training examples.
- \(\hat{y}_i\) represents the predicted output for the \(i\)-th example.
- \(y_i\) is the actual target value for the \(i\)-th example.

The goal during model training is to minimize this loss by adjusting the model's parameters (including the attention network weights, embedding vectors, etc.) so that the predicted values \(\hat{y}_i\) are as close as possible to the actual target values \(y_i\) for the given regression task.


### 总结

AFM (Attentional Factorization Machine) is an improvement over FM (Factorization Machine) that introduces the attention mechanism to learn the importance of different feature combinations. Compared to other deep learning models like Wide & Deep and DeepCross, which use MLPs to implicitly learn feature interactions, AFM has the advantage of interpretability.

1. **Interpretability:** FM has good interpretability because it learns feature interactions explicitly through the inner product of latent vectors. This makes it easier to understand the importance of different feature combinations. AFM maintains this interpretability by introducing the attention mechanism to assign different weights to feature combinations.

2. **Attention Mechanism:** AFM enhances FM by introducing an attention network, which learns to assign attention scores to different feature combinations. This allows the model to focus on relevant feature interactions while downplaying less important ones. It provides a way to weigh feature combinations based on their relevance to the task.

3. **Performance Improvement:** AFM offers improved performance compared to FM by learning the importance of feature combinations, especially in tasks like click-through rate prediction or recommendation systems. It optimizes the model's ability to capture relevant interactions.

4. **Limitation:** One limitation of AFM is that it still focuses on second-order feature interactions (pairwise interactions) like FM. It doesn't explicitly capture higher-order interactions. If higher-order interactions are crucial for a specific task, more complex models might be needed.

In summary, AFM combines the interpretability of FM with the ability to learn feature interaction importance through the attention mechanism. It strikes a balance between model explainability and predictive performance, making it a valuable tool in tasks where understanding feature interactions is essential. However, it should be noted that AFM, like FM, primarily addresses second-order interactions and may not capture higher-order interactions without further extensions.



## 原文link

https://arxiv.org/abs/1708.04617


## 代码实现

https://link.zhihu.com/?target=https%3A//github.com/gutouyu/ML_CIA



