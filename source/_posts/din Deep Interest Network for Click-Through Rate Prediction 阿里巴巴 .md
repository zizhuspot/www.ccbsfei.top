---
title: din深度兴趣网络的算法思想与框架演进 阿里巴巴 
date: 2023-09-11 10:00:00
categories:
  - 排序模型
tags:
  - 预估模型 
  - din 
  - dnn
  - Recommender systems
description:   Din-model is a term used in the field of computer vision and machine learning. It refers to a type of neural network architecture that is specifically designed for image segmentation tasks. Din-models are known for their efficiency and accuracy in segmenting images into regions of similar pixels. They have been used in a variety of applications, including medical imaging, autonomous driving, and facial recognition.  "Din模型"是计算机视觉和机器学习领域中使用的一个术语。它指的是一种专门设计用于图像分割任务的神经网络架构。Din模型以其在将图像分割成相似像素区域方面的高效性和准确性而闻名。它已经被应用于各种领域，包括医学成像、自动驾驶和人脸识别等应用。 
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_194316.png
---

## introduction 

The DIN (Deep Interest Network) model is a deep learning model introduced by the Alibaba Mama team in 2018. It is used to predict the probability of a user clicking on an advertisement. The core idea of the DIN model is to leverage attention mechanisms to capture user interests, thereby enhancing the model's expressive power and predictive performance. Attention mechanisms are inspired by human attention and allow models to focus on the most relevant parts of a large amount of information while ignoring irrelevant parts. For example, when looking at an image, humans automatically focus on the most meaningful regions while ignoring the background or details.
DIN（Deep Interest Network）模型是由阿里巴巴Mama团队于2018年推出的深度学习模型。它用于预测用户点击广告的概率。DIN模型的核心思想是利用注意力机制来捕捉用户的兴趣，从而增强模型的表达能力和预测性能。注意力机制受人类注意力的启发，使模型能够专注于大量信息中最相关的部分，而忽略不相关的部分。例如，当观看图像时，人类会自动关注最有意义的区域，而忽略背景或细节。

Attention functions can take various forms, such as dot product, additive, scaled dot product, etc. Weight vectors represent the similarity or relevance between query vectors and key vectors, while value vectors contain information from key vectors. By computing weighted sums, an output vector is obtained, which contains the most relevant information.

注意力函数可以采用各种形式，如点积、加性、缩放点积等。权重向量表示查询向量和键向量之间的相似性或相关性，而值向量包含键向量的信息。通过计算加权和，可以获得一个输出向量，其中包含最相关的信息。

The DIN model applies the attention mechanism to user behavioral features, specifically, the products that users have browsed or purchased in the past. The DIN model assumes that user interests in different products are diverse and dynamically changing, and different products have varying impacts on a user's likelihood to click on an advertisement. Therefore, the DIN model dynamically adjusts the weights of user behavioral features based on the relevance between candidate products and the user's historical behavior. This adjustment results in a vector that more accurately reflects the user's interests.

DIN模型将注意力机制应用于用户的行为特征，具体来说，是用户过去浏览或购买的产品。DIN模型假设用户对不同的产品有多样性和动态性的兴趣，不同的产品对用户点击广告的可能性产生不同的影响。因此，DIN模型根据候选产品与用户历史行为之间的相关性动态调整用户行为特征的权重。这种调整导致了一个更准确反映用户兴趣的向量。

The network structure of the DIN model is depicted in the diagram you provided. It uses attention mechanisms to capture the dynamic and diverse interests of users, making it a powerful tool for improving click-through rate prediction in advertising scenarios.

DIN模型的网络结构如您提供的图表所示。它使用注意力机制来捕捉用户的动态和多样化兴趣，使其成为提高广告点击率预测的强大工具。


## model 架构

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_194316.png)

The DIN (Deep Interest Network) model follows a specific process for handling features:
DIN（Deep Interest Network）模型遵循一种特定的特征处理过程：

1. **Embedding:** The model begins by embedding various features, which means it converts them into low-dimensional dense vectors. These embeddings capture the essence of the features in a more compact and meaningful way.

1. **嵌入（Embedding）：** 模型首先对各种特征进行嵌入，这意味着将它们转换为低维稠密向量。这些嵌入以更紧凑和有意义的方式捕获了特征的本质。

2. **Activation Unit:** For user behavioral product features, the DIN model employs an activation unit to calculate the weights between each product and the candidate products. This unit calculates a weighted sum of these products. The activation unit's input consists of two parts: the original product embedding vectors and vectors obtained through outer product calculations. The outer product computation enhances the interaction information between these vectors. The output of the activation unit is a weighted sum vector of user behavioral features.

2. **激活单元（Activation Unit）：** 对于用户行为产品特征，DIN模型使用激活单元来计算每个产品与候选产品之间的权重。该单元计算这些产品的加权和。激活单元的输入由两部分组成：原始产品嵌入向量和通过外积计算获得的向量。外积计算增强了这些向量之间的交互信息。激活单元的输出是用户行为特征的加权和向量。

3. **Concatenation and MLP:** All the features, including the weighted sum vector of user behavioral features, are concatenated together. This concatenated feature vector is then fed into a multi-layer perceptron (MLP). The MLP is a deep neural network with multiple hidden layers, and it is used to make the final click-through rate prediction.

3. **拼接和MLP：** 所有特征，包括用户行为特征的加权和向量，都被拼接在一起。然后，将这个拼接的特征向量馈送到多层感知器（MLP）中。MLP是一个具有多个隐藏层的深度神经网络，用于进行最终的点击率预测。

DIN model's architecture is designed to capture dynamic and diverse user interests by considering interactions between products and candidate products, thanks to the attention mechanism and the activation unit. This approach enhances the model's ability to predict click-through rates accurately in the context of online advertising.

DIN模型的架构旨在通过考虑产品与候选产品之间的交互作用来捕捉动态和多样化的用户兴趣，这得益于注意力机制和激活单元。这种方法增强了模型在在线广告背景下准确预测点击率的能力。

In online experiments within Alibaba Mama's display advertising system, DIN model outperformed benchmark models. It achieved a 10% improvement in click-through rates and a 3.8% increase in revenue. These results demonstrate that the DIN model effectively captures user interests, leading to improved model expressiveness and predictive performance.

在阿里巴巴Mama展示广告系统内的在线实验中，DIN模型胜过了基准模型。它提高了10%的点击率和3.8%的收入。这些结果表明，DIN模型有效地捕捉了用户兴趣，提高了模型的表达能力和预测性能。



## benefits 优势

The advantages of the DIN (Deep Interest Network) model can be attributed to several key innovations and techniques:
DIN（Deep Interest Network）模型的优点可以归因于几项关键的创新和技术：

1. **Dynamic Weight Adjustment:** One of the primary strengths of the DIN model is its ability to dynamically adjust the weights of user behavioral features. It does this based on the relevance between candidate products and user behavioral products. This dynamic weighting allows more relevant products to contribute more significantly to the click-through rate prediction. Unlike traditional models (base models) that assign the same or fixed weights to all user behavioral features, DIN adapts to the diversity and dynamic changes in user interests.

1. **动态权重调整：** DIN模型的主要优势之一是其能够根据候选产品和用户行为产品之间的相关性动态调整用户行为特征的权重。这种动态权重调整使得更相关的产品能够更显著地影响点击率的预测。与将相同或固定权重分配给所有用户行为特征的传统模型（基本模型）不同，DIN适应了用户兴趣的多样性和动态变化。

2. **Outer Product Computation:** The DIN model introduces the use of outer product computation to enhance the interaction information between vectors. Outer product calculation maps two vectors into a matrix, preserving all the information between the two vectors. This allows the activation unit to better capture the relationship between user behavioral products and candidate products. Instead of relying on simple dot products or additive calculations, DIN leverages outer product computation to capture richer interactions.

2. **外积计算：** DIN模型引入了外积计算的使用，以增强向量之间的交互信息。外积计算将两个向量映射成一个矩阵，保留了两个向量之间的所有信息。这允许激活单元更好地捕捉用户行为产品和候选产品之间的关系。DIN不依赖于简单的点积或加法计算，而是利用外积计算来捕捉更丰富的交互。

3. **Training Efficiency and Generalization Techniques:** DIN employs several techniques to improve model training efficiency and generalization. For example, it uses mini-batch aware regularization, which reduces computational overhead in large-scale sparse scenarios by updating gradients only for parts of the parameters that are non-zero within each mini-batch. Additionally, DIN uses an adaptive activation function called Dice, which automatically adjusts the shape of the activation function based on the distribution of input data in each layer. This helps prevent gradient vanishing or exploding issues and contributes to better training stability and generalization.

3. **训练效率和泛化技巧：** DIN采用了几种技术来提高模型的训练效率和泛化能力。例如，它使用了迷你批量感知正则化，通过仅在每个迷你批量中非零参数的部分上更新梯度，减少了大规模稀疏场景中的计算开销。此外，DIN使用了一种自适应激活函数Dice，它根据每一层输入数据的分布自动调整激活函数的形状。这有助于防止梯度消失或爆炸的问题，有助于提高训练稳定性和泛化性能。

Overall, DIN's ability to adapt to user interests, capture rich feature interactions, and leverage efficient training techniques makes it a powerful model for click-through rate prediction in online advertising. Its innovations have led to significant improvements in model performance and user engagement in real-world applications.

总的来说，DIN模型适应用户兴趣、捕捉丰富的特征交互并利用高效的训练技术，使其成为在线广告中点击率预测的强大模型。其创新性使得在实际应用中模型性能和用户参与度有了显著提升。


## Mini-batch aware regularization 

This is a regularization method used in large-scale sparse scenarios to reduce computational overhead and mitigate the risk of overfitting. Its main idea is to update the gradients only for the parts of the parameters that are non-zero within each mini-batch, rather than updating the entire parameter matrix. This approach avoids penalizing features that are not present in the current mini-batch, thereby preserving more information. Additionally, it helps address overfitting issues in data with a long-tailed distribution by applying larger penalties to the tail samples and smaller penalties to the head samples to prevent overfitting to the tail.

这是一种用于大规模稀疏场景的正则化方法，旨在降低计算开销并减轻过拟合风险。其主要思想是仅在每个小批量内对参数矩阵的非零部分进行梯度更新，而不是对整个参数矩阵进行更新。这种方法避免了对不在当前小批量中的特征进行惩罚，从而保留了更多信息。此外，它通过对尾部样本施加更大的惩罚力度，对头部样本施加较小的惩罚力度，有助于解决长尾分布数据中的过拟合问题，防止过度拟合到尾部数据。

Let's assume you have a feature matrix X, which is a high-dimensional sparse binary matrix. Each row represents a sample, and each column represents a feature. The goal is to map this feature matrix to a lower-dimensional dense vector, which involves performing embedding operations. You can define an embedding matrix W, where each column represents the embedding vector for a feature. If you use L2 regularization to prevent overfitting, you would typically include a regularization term in the loss function:

让我们假设您有一个特征矩阵X，这是一个高维稀疏的二进制矩阵。每行代表一个样本，每列代表一个特征。目标是将这个特征矩阵映射到一个低维稠密向量，这涉及执行嵌入操作。您可以定义一个嵌入矩阵W，其中每列代表一个特征的嵌入向量。如果您使用L2正则化来防止过拟合，通常会在损失函数中包含一个正则化项：

\[ \text{Regularization Term} = \frac{\lambda}{2} \sum_{i,j} W_{ij}^2 \]

\[ \text{正则化项} = \frac{\lambda}{2} \sum_{i,j} W_{ij}^2 \]

Where:
- \(\lambda\) is the regularization strength.
- \(W_{ij}\) represents the weight associated with the i-th feature and the j-th embedding dimension.

其中：
- \(\lambda\) 是正则化强度。
- \(W_{ij}\) 表示与第i个特征和第j个嵌入维度相关联的权重。

The issue with this approach is twofold:

这种方法存在两个问题：

1. **High Computational Cost:** Because X is a high-dimensional sparse matrix, W is also a high-dimensional sparse matrix. Updating every element of W can be computationally expensive because you would need to iterate through the entire W matrix, consuming significant time and memory.

1. **高计算成本：** 由于X是高维稀疏矩阵，W也是高维稀疏矩阵。更新W的每个元素可能会产生高计算成本，因为需要遍历整个W矩阵，占用大量时间和内存。

2. **Information Loss:** Since X is sparse, many features may not appear in certain samples. Penalizing the embedding vectors corresponding to these absent features reduces the weights of these features, potentially leading to the loss of valuable information associated with these features.

2. **信息丢失：** 由于X是稀疏的，许多特征可能不会出现在某些样本中。惩罚与这些缺失特征对应的嵌入向量会降低这些特征的权重，可能导致与这些特征相关的宝贵信息丢失。

Mini-batch aware regularization addresses these issues by only updating the parts of the parameter matrix that are relevant within each mini-batch, making the process more efficient and preserving the information contained in the sparse data.

迷你批量感知正则化通过仅在每个小批量内更新参数矩阵的相关部分来解决这些问题，从而使过程更加高效并保留稀疏数据中包含的信息。


## The Dice activation function
 
this is  an improved variant of the ReLU (Rectified Linear Unit) activation function. Its distinctive feature is its ability to adaptively adjust the threshold for the step function based on the data distribution, thereby addressing issues such as the "dying ReLU" problem and insensitivity to small changes. Its definition is as follows:
这是ReLU（修正线性单元）激活函数的改进变体。其独特之处在于它能够根据数据分布自适应地调整步函数的阈值，从而解决了“死亡ReLU”问题和对小变化不敏感等问题。其定义如下：

Let \( x \) be the input to the Dice activation function, and \( y \) be the output. The Dice activation function is defined as follows:

设 \( x \) 为Dice激活函数的输入，\( y \) 为输出。Dice激活函数的定义如下：

\[ y(x) = \frac{2}{1 + e^{-x}} - 1 \]

In this formula:

\[ y(x) = \frac{2}{1 + e^{-x}} - 1 \]

在这个公式中：

- \( e \) is the base of the natural logarithm (approximately 2.71828).
- \( x \) is the input to the activation function.
- \( y \) is the output of the Dice activation function.

- \( e \) 是自然对数的底数（约为2.71828）。
- \( x \) 是激活函数的输入。
- \( y \) 是Dice激活函数的输出。

The Dice activation function introduces a sigmoid-like behavior with a dynamic threshold that adapts to the input data distribution. This adaptability allows the activation function to be more responsive to different patterns in the data, improving the training and generalization of neural networks. It helps mitigate the issues associated with traditional ReLU activations, such as neurons becoming inactive ("dying ReLU") or insensitivity to small input changes.



Dice激活函数引入了类似S形函数的行为，具有根据输入数据分布自适应的动态阈值。这种适应性使激活函数能够更好地响应数据中的不同模式，改善了神经网络的训练和泛化性能。它有助于缓解与传统ReLU激活函数相关的问题，如神经元变得不活跃（“死亡ReLU”）或对小输入变化不敏感的问题。




## 原文  source link 

https://arxiv.org/abs/1706.06978




## 代码 code 

https://link.zhihu.com/?target=https%3A//github.com/i-Jayus/RecSystem-Pytorch



