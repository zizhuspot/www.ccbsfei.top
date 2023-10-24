---
title: afm模型的详细解析-注意力网络来学习不同特征组合- 浙江大学 
date: 2023-09-12 11:00:00
categories:
  - 排序模型
tags:
  - 预估模型 
  - afm 
  - fm 
  - Recommender systems
description:  its most notable feature is the use of an attention network to learn the importance of different combinations of features.assign a weight to each combination feature (cross feature). 它最显著的特点是使用了一个注意力网络来学习不同特征组合的重要性，为每个特征组合（交叉特征）分配权重。 
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_193155.png
---

## introduction 

AFM stands for Attentional Factorization Machine, and it is developed by the same author as NFM. AFM is an improvement over FM (Factorization Machine), and its most notable feature is the use of an attention network to learn the importance of different combinations of features.

AFM代表着“关注性因子分解机”，由与NFM相同的作者开发。AFM是对FM（因子分解机）的一种改进，它最显著的特点是使用注意力网络来学习不同特征组合的重要性。

In recommendation systems or click-through rate (CTR) prediction tasks, input data often includes a large number of categorical features. Since these categorical features are not independent, the combinations of these features become crucial. One simple approach is to assign a weight to each combination feature (cross feature). However, the drawback of this cross-feature-based method is that many combination features may not appear in the training dataset, making it challenging to effectively learn their weights.

在推荐系统或点击率（CTR）预测任务中，输入数据通常包括大量的分类特征。由于这些分类特征并不独立，这些特征的组合变得至关重要。一种简单的方法是为每个组合特征（交叉特征）分配一个权重。然而，这种基于交叉特征的方法的缺点是许多组合特征可能不会出现在训练数据集中，这使得有效学习它们的权重变得具有挑战性。

FM addresses this by learning an embedding vector (also known as a latent vector) for each feature and representing the weight of a combination feature as the inner product of the embedding vectors of its constituent features. However, FM does not consider the fact that some features may be unimportant or even irrelevant in the prediction. These features can introduce noise and interfere with predictions. FM treats all feature combinations equally, assigning them the same weight (implicitly, a weight of 1).

FM通过学习每个特征的嵌入向量（也称为潜在向量），将组合特征的权重表示为其组成特征的嵌入向量的内积，从而解决了这个问题。然而，FM没有考虑到一些特征可能在预测中不重要甚至无关紧要的事实。这些特征可能会引入噪音并干扰预测。FM将所有特征组合平等对待，分配给它们相同的权重（隐含地是1）。

In this paper, AFM introduces the attention mechanism, and it innovatively proposes a solution to assign different levels of importance to different feature combinations. The weights can be automatically learned within the network, eliminating the need for additional domain knowledge. This attention mechanism in AFM allows it to focus on relevant feature combinations while downplaying or ignoring less important ones, ultimately improving the model's predictive accuracy.


在这篇论文中，AFM引入了注意力机制，并创新性地提出了一种方法，为不同的特征组合分配不同重要性级别的解决方案。权重可以在网络内自动学习，无需额外的领域知识。AFM中的这种注意力机制使其能够关注相关的特征组合，同时淡化或忽略不太重要的组合，从而最终提高了模型的预测准确性。

## 痛点 

NFM (Neural Factorization Machine) can be seen as an improvement over both FM (Factorization Machines) and FNN (Deep Factorization Neural Network), addressing their respective shortcomings as follows:
NFM（神经因子分解机）可以被看作是对FM（因子分解机）和FNN（深度因子分解神经网络）的改进，分别解决了它们的缺点，具体如下：

1. **FM Model Limitation:** While the FM model effectively captures cross-feature interactions, it still models these interactions linearly, which means it cannot capture non-linear relationships between features.

1. **FM模型限制：** 虽然FM模型有效地捕捉了交叉特征之间的相互作用，但它仍然线性地建模这些相互作用，这意味着它无法捕捉特征之间的非线性关系。

2. **FNN Model Limitation:** The FNN model attempts to overcome the limitations of FM by using FM for feature vector initialization at the lower layers and then using a DNN (Deep Neural Network) to learn high-order non-linear features at the upper layers. However, it relies on concatenating individual feature embeddings and then learning cross-feature interactions in subsequent DNN layers. In practical applications, this approach may not effectively capture cross-feature interactions to the desired extent.

2. **FNN模型限制：** FNN模型试图通过在较低层使用FM进行特征向量初始化，然后在较高层使用DNN（深度神经网络）来学习高阶非线性特征，以克服FM的限制。然而，它依赖于串联单独的特征嵌入，然后在后续的DNN层中学习交叉特征的相互作用。在实际应用中，这种方法可能无法有效地捕捉跨特征的相互作用到所需的程度。

Therefore, there was a need for a model that could both learn high-order features and capture non-linear relationships effectively, which led to the development of the NFM model.

因此，需要一种模型既能有效学习高阶特征，又能有效地捕捉非线性关系，这促成了NFM模型的发展。

## 目标函数

FM, which stands for Factorization Machine, can be formalized with the following equation:
FM，即因子分解机，可以使用以下方程形式化：

\[ \hat{y}(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j \]

\[ \hat{y}(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j \]

Where:
- \(\hat{y}(x)\) represents the predicted output or target variable for input \(x\).
- \(w_0\) is the bias term.
- \(w_i\) represents the weight associated with the \(i\)-th feature.
- \(x_i\) denotes the \(i\)-th feature value in the input \(x\).
- \(v_i\) represents the latent factor vector associated with the \(i\)-th feature.
- \(\langle v_i, v_j \rangle\) denotes the inner product or interaction between the latent factor vectors \(v_i\) and \(v_j\).
- The summation terms capture both linear and pairwise feature interactions.

其中：
- \(\hat{y}(x)\) 代表了输入 \(x\) 的预测输出或目标变量。
- \(w_0\) 是偏置项。
- \(w_i\) 表示与第 \(i\) 个特征相关的权重。
- \(x_i\) 表示输入 \(x\) 中的第 \(i\) 个特征值。
- \(v_i\) 代表与第 \(i\) 个特征相关的潜在因子向量。
- \(\langle v_i, v_j \rangle\) 表示潜在因子向量 \(v_i\) 和 \(v_j\) 之间的内积或相互作用。
- 求和项捕捉了线性和成对特征交互。

FM is a model used for recommendation systems and predictive modeling, and it is particularly effective in capturing second-order feature interactions, making it suitable for problems involving categorical data and interactions between features.

FM是一种用于推荐系统和预测建模的模型，特别擅长捕捉二阶特征交互，因此非常适用于涉及分类数据和特征之间交互的问题。


In FM (Factorization Machines), there are two main issues:
在FM（因子分解机）中，存在两个主要问题：

1. **Shared Latent Vectors for Features:** FM uses the same latent vector for a feature when interacting with all other features. This limitation led to the development of Field-aware Factorization Machines (FFM) to address this issue. FFM allows each feature to have its own set of latent vectors, considering different interactions with other features in different fields.

1. **特征的共享潜在向量：** FM在与其他特征交互时，对于一个特征使用相同的潜在向量。这一限制促使了Field-aware Factorization Machines（FFM）的发展，以解决这个问题。FFM允许每个特征拥有自己的一组潜在向量，考虑了不同领域中与其他特征的不同交互。

2. **Equal Weights for All Feature Combinations:** FM assigns the same weight (implicitly, a weight of 1) to all feature combinations. This means that all combinations of features are treated equally, even though in practice, not all feature combinations are equally important. This issue is addressed by Attentional Factorization Machines (AFM), which optimizes the model by assigning different weights to different feature combinations. AFM allows the model to focus on relevant feature combinations while downplaying or ignoring less important ones. This not only improves predictive accuracy but also enhances model interpretability, as it allows for in-depth analysis of important feature combinations in subsequent research and applications.


2. **对所有特征组合分配相等权重：** FM分配相同的权重（隐含地是1）给所有特征组合。这意味着所有特征组合都被平等对待，尽管在实践中，并非所有特征组合都同等重要。这个问题由Attentional Factorization Machines（AFM）来解决，通过为不同的特征组合分配不同的权重来优化模型。AFM允许模型聚焦于相关的特征组合，同时淡化或忽略不太重要的组合。这不仅提高了预测准确性，还增强了模型的可解释性，因为它允许在后续研究和应用中深入分析重要的特征组合。

## The full name of AFM is Attentional Factorization Machine.

The model structure of AFM is as follows:

### Model Structure

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_193155.png)


Pair-wise Interaction Layer in AFM is primarily responsible for modeling the interactions between different feature combinations. It takes the embeddings of the input features, which have been converted into dense vectors through the Embedding Layer, and computes the interactions between them. 
在AFM中，Pair-wise Interaction Layer主要负责建模不同特征组合之间的相互作用。它接收通过Embedding Layer转换为稠密向量的输入特征嵌入，并计算它们之间的相互作用。

Let's formalize this process:

我们来形式化这个过程：

Assume you have "m" embedded vectors, each with a dimension of "k" (where "k" is the dimension of the embedding).

假设你有“m”个嵌入向量，每个向量的维度为“k”（其中“k”是嵌入的维度）。

For each pair of embedded vectors (i, j) where i ≠ j, you calculate the element-wise product between them. This results in m(m-1)/2 combination vectors, each with a dimension of "k."

对于每一对嵌入向量（i，j），其中i ≠ j，你计算它们之间的元素级乘积。这将产生m(m-1)/2个组合向量，每个向量的维度为“k”。

Mathematically, this can be represented as follows:

数学上，这可以表示如下：

For i ≠ j:

\[ \text{Interaction Vector}_{ij} = \text{Embedding Vector}_i \odot \text{Embedding Vector}_j \]

Where:
- \(\text{Interaction Vector}_{ij}\) is the interaction vector between the i-th and j-th embedded vectors.
- \(\text{Embedding Vector}_i\) is the embedding vector for the i-th feature.
- \(\text{Embedding Vector}_j\) is the embedding vector for the j-th feature.
- \(\odot\) represents element-wise multiplication (Hadamard product).

对于i ≠ j：

\[ \text{交互向量}_{ij} = \text{嵌入向量}_i \odot \text{嵌入向量}_j \]

其中：
- \(\text{交互向量}_{ij}\) 是第i个和第j个嵌入向量之间的交互向量。
- \(\text{嵌入向量}_i\) 是第i个特征的嵌入向量。
- \(\text{嵌入向量}_j\) 是第j个特征的嵌入向量。
- \(\odot\) 表示元素级乘法（Hadamard积）。

This operation captures the pairwise interactions between features, which is a key component in the AFM model for learning the importance of different feature combinations.

这个操作捕捉了特征之间的成对交互，这是AFM模型中学习不同特征组合重要性的关键组成部分。


### In AFM, an Attention Network is used to learn the importance of different feature combinations. 

The Attention Network is essentially a one-layer MLP (Multi-Layer Perceptron) with the ReLU activation function. The size of the network is determined by the attention factor, which represents the number of neurons in this layer.
注意力网络本质上是一个具有ReLU激活函数的一层MLP（多层感知器）。网络的大小由注意力因子决定，它代表了这一层中的神经元数量。

The input to the Attention Network is the result of element-wise product of two embedding vectors, often referred to as the "interaction vector" or "interacted vector." These interaction vectors encode the feature combinations in the embedding space.

注意力网络的输入是两个嵌入向量的元素级乘积的结果，通常称为“交互向量”或“互动向量”。这些交互向量在嵌入空间中编码了特征组合。

The output of the Attention Network is the Attention score corresponding to the feature combination. These scores represent how important or relevant each feature combination is to the final prediction. To ensure that the attention scores are interpretable and sum to 1, a softmax activation is applied to the output.

注意力网络的输出是对应于特征组合的注意力分数。这些分数表示每个特征组合对最终预测的重要性或相关性。为了确保注意力分数具有可解释性并且总和为1，输出上应用了softmax激活。

Mathematically, the Attention Network in AFM can be formalized as follows:

在AFM中，数学上可以形式化表示注意力网络如下：

Let \( \text{Interacted Vector}_{ij} \) be the interaction vector between feature embeddings i and j, and \( \text{Attention Score}_{ij} \) be the attention score for this feature combination. The Attention Network can be represented as:

\[ \text{Attention Score}_{ij} = \text{softmax}(\text{ReLU}(\text{W}_a \cdot \text{Interacted Vector}_{ij})) \]

Where:
- \( \text{Attention Score}_{ij} \) is the attention score for the feature combination between embeddings i and j.
- \( \text{W}_a \) represents the weight matrix associated with the Attention Network.
- \(\text{ReLU}\) is the Rectified Linear Unit activation function.
- \( \cdot \) denotes matrix multiplication.
- \(\text{softmax}\) is the softmax activation function applied to normalize the attention scores.

设\( \text{互动向量}_{ij} \)是特征嵌入i和j之间的交互向量，\( \text{注意力分数}_{ij} \)是这个特征组合的注意力分数。注意力网络可以表示为：

\[ \text{注意力分数}_{ij} = \text{softmax}(\text{ReLU}(\text{W}_a \cdot \text{互动向量}_{ij})) \]

其中：
- \( \text{注意力分数}_{ij} \) 是特征嵌入i和j之间的特征组合的注意力分数。
- \( \text{W}_a \) 代表与注意力网络相关的权重矩阵。
- \( \text{ReLU} \) 是修正线性单元激活函数。
- \( \cdot \) 表示矩阵乘法。
- \( \text{softmax} \) 是应用于标准化注意力分数的softmax激活函数。

The purpose of the Attention Network is to assign different attention scores to different feature combinations, effectively learning which combinations are more important for the prediction task.

注意力网络的目的是为不同的特征组合分配不同的注意力分数，有效地学习哪些组合对于预测任务更为重要。

## model train

In AFM, the choice of loss function depends on the specific task you are addressing. Here, let's focus on the case of regression, where the square loss (also known as mean squared error) is used as the loss function. The square loss measures the squared difference between predicted and actual values and is commonly used for regression problems.
在AFM中，选择损失函数取决于您正在处理的具体任务。在这里，让我们关注回归的情况，其中使用平方损失（也称为均方误差）作为损失函数。平方损失度量了预测值与实际值之间的差异的平方，并通常用于回归问题。

Mathematically, the square loss for regression in AFM can be formalized as follows:

数学上，AFM中回归的平方损失可以形式化表示如下：

\[ \text{Loss} = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 \]

\[ \text{损失} = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 \]

其中：
- \(\text{损失}\) 是平方损失。
- \(N\) 是训练样本的数量。
- \(\hat{y}_i\) 代表第\(i\)个示例的预测输出。
- \(y_i\) 是第\(i\)个示例的实际目标值。

Where:
- \(\text{Loss}\) is the square loss.
- \(N\) is the number of training examples.
- \(\hat{y}_i\) represents the predicted output for the \(i\)-th example.
- \(y_i\) is the actual target value for the \(i\)-th example.

The goal during model training is to minimize this loss by adjusting the model's parameters (including the attention network weights, embedding vectors, etc.) so that the predicted values \(\hat{y}_i\) are as close as possible to the actual target values \(y_i\) for the given regression task.

在模型训练期间，目标是通过调整模型的参数（包括注意力网络的权重、嵌入向量等），使得预测值\(\hat{y}_i\)尽可能接近给定回归任务的实际目标值\(y_i\)，从而最小化这一损失。

### 总结

AFM (Attentional Factorization Machine) is an improvement over FM (Factorization Machine) that introduces the attention mechanism to learn the importance of different feature combinations. Compared to other deep learning models like Wide & Deep and DeepCross, which use MLPs to implicitly learn feature interactions, AFM has the advantage of interpretability.
AFM（注意力因子分解机）是对FM（因子分解机）的改进，引入了注意力机制来学习不同特征组合的重要性。与其他深度学习模型（如Wide & Deep和DeepCross）使用MLP来隐式学习特征交互相比，AFM具有可解释性的优势。

1. **Interpretability:** FM has good interpretability because it learns feature interactions explicitly through the inner product of latent vectors. This makes it easier to understand the importance of different feature combinations. AFM maintains this interpretability by introducing the attention mechanism to assign different weights to feature combinations.

1. **可解释性：** FM通过潜在向量的内积明确学习特征交互，因此具有良好的可解释性。这使得更容易理解不同特征组合的重要性。AFM通过引入注意力机制来为特征组合分配不同权重，从而保持了这种可解释性。

2. **Attention Mechanism:** AFM enhances FM by introducing an attention network, which learns to assign attention scores to different feature combinations. This allows the model to focus on relevant feature interactions while downplaying less important ones. It provides a way to weigh feature combinations based on their relevance to the task.

2. **注意力机制：** AFM通过引入一个注意力网络来增强FM，该网络学会为不同特征组合分配注意力分数。这使得模型能够专注于相关的特征交互，同时淡化不太重要的交互。它提供了一种基于任务相关性来衡量特征组合的方法。

3. **Performance Improvement:** AFM offers improved performance compared to FM by learning the importance of feature combinations, especially in tasks like click-through rate prediction or recommendation systems. It optimizes the model's ability to capture relevant interactions.

3. **性能提升：** 与FM相比，AFM通过学习特征组合的重要性，在诸如点击率预测或推荐系统等任务中提供了更好的性能。它优化了模型捕捉相关交互的能力。

4. **Limitation:** One limitation of AFM is that it still focuses on second-order feature interactions (pairwise interactions) like FM. It doesn't explicitly capture higher-order interactions. If higher-order interactions are crucial for a specific task, more complex models might be needed.

4. **局限性：** AFM的一个局限性是它仍然侧重于二阶特征交互（成对交互），就像FM一样。它不明确捕捉更高阶的交互。如果对于特定任务高阶交互至关重要，可能需要更复杂的模型。

In summary, AFM combines the interpretability of FM with the ability to learn feature interaction importance through the attention mechanism. It strikes a balance between model explainability and predictive performance, making it a valuable tool in tasks where understanding feature interactions is essential. However, it should be noted that AFM, like FM, primarily addresses second-order interactions and may not capture higher-order interactions without further extensions.

总之，AFM将FM的可解释性与通过注意力机制学习特征交互重要性的能力相结合。它在平衡模型可解释性和预测性能方面发挥了作用，在需要理解特征交互的任务中非常有价值。然而，值得注意的是，AFM像FM一样，主要处理二阶交互，如果没有进一步扩展，可能无法捕捉高阶交互。


## 原文link

https://arxiv.org/abs/1708.04617


## 代码实现

https://link.zhihu.com/?target=https%3A//github.com/gutouyu/ML_CIA



