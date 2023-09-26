---
title: din Deep Interest Network for Click-Through Rate Prediction 阿里巴巴 
date: 2023-09-11 10:00:00
categories:
  - 排序模型
tags:
  - 预估模型 
  - din 
  - dnn
  - Recommender systems
description:   Din-model is a term used in the field of computer vision and machine learning. It refers to a type of neural network architecture that is specifically designed for image segmentation tasks. Din-models are known for their efficiency and accuracy in segmenting images into regions of similar pixels. They have been used in a variety of applications, including medical imaging, autonomous driving, and facial recognition.  
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_194316.png
---

## introduction 

The DIN (Deep Interest Network) model is a deep learning model introduced by the Alibaba Mama team in 2018. It is used to predict the probability of a user clicking on an advertisement. The core idea of the DIN model is to leverage attention mechanisms to capture user interests, thereby enhancing the model's expressive power and predictive performance. Attention mechanisms are inspired by human attention and allow models to focus on the most relevant parts of a large amount of information while ignoring irrelevant parts. For example, when looking at an image, humans automatically focus on the most meaningful regions while ignoring the background or details.

Attention functions can take various forms, such as dot product, additive, scaled dot product, etc. Weight vectors represent the similarity or relevance between query vectors and key vectors, while value vectors contain information from key vectors. By computing weighted sums, an output vector is obtained, which contains the most relevant information.

The DIN model applies the attention mechanism to user behavioral features, specifically, the products that users have browsed or purchased in the past. The DIN model assumes that user interests in different products are diverse and dynamically changing, and different products have varying impacts on a user's likelihood to click on an advertisement. Therefore, the DIN model dynamically adjusts the weights of user behavioral features based on the relevance between candidate products and the user's historical behavior. This adjustment results in a vector that more accurately reflects the user's interests.

The network structure of the DIN model is depicted in the diagram you provided. It uses attention mechanisms to capture the dynamic and diverse interests of users, making it a powerful tool for improving click-through rate prediction in advertising scenarios.



## model 架构

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_194316.png)

The DIN (Deep Interest Network) model follows a specific process for handling features:

1. **Embedding:** The model begins by embedding various features, which means it converts them into low-dimensional dense vectors. These embeddings capture the essence of the features in a more compact and meaningful way.

2. **Activation Unit:** For user behavioral product features, the DIN model employs an activation unit to calculate the weights between each product and the candidate products. This unit calculates a weighted sum of these products. The activation unit's input consists of two parts: the original product embedding vectors and vectors obtained through outer product calculations. The outer product computation enhances the interaction information between these vectors. The output of the activation unit is a weighted sum vector of user behavioral features.

3. **Concatenation and MLP:** All the features, including the weighted sum vector of user behavioral features, are concatenated together. This concatenated feature vector is then fed into a multi-layer perceptron (MLP). The MLP is a deep neural network with multiple hidden layers, and it is used to make the final click-through rate prediction.

DIN model's architecture is designed to capture dynamic and diverse user interests by considering interactions between products and candidate products, thanks to the attention mechanism and the activation unit. This approach enhances the model's ability to predict click-through rates accurately in the context of online advertising.

In online experiments within Alibaba Mama's display advertising system, DIN model outperformed benchmark models. It achieved a 10% improvement in click-through rates and a 3.8% increase in revenue. These results demonstrate that the DIN model effectively captures user interests, leading to improved model expressiveness and predictive performance.




## benefits 优势

The advantages of the DIN (Deep Interest Network) model can be attributed to several key innovations and techniques:

1. **Dynamic Weight Adjustment:** One of the primary strengths of the DIN model is its ability to dynamically adjust the weights of user behavioral features. It does this based on the relevance between candidate products and user behavioral products. This dynamic weighting allows more relevant products to contribute more significantly to the click-through rate prediction. Unlike traditional models (base models) that assign the same or fixed weights to all user behavioral features, DIN adapts to the diversity and dynamic changes in user interests.

2. **Outer Product Computation:** The DIN model introduces the use of outer product computation to enhance the interaction information between vectors. Outer product calculation maps two vectors into a matrix, preserving all the information between the two vectors. This allows the activation unit to better capture the relationship between user behavioral products and candidate products. Instead of relying on simple dot products or additive calculations, DIN leverages outer product computation to capture richer interactions.

3. **Training Efficiency and Generalization Techniques:** DIN employs several techniques to improve model training efficiency and generalization. For example, it uses mini-batch aware regularization, which reduces computational overhead in large-scale sparse scenarios by updating gradients only for parts of the parameters that are non-zero within each mini-batch. Additionally, DIN uses an adaptive activation function called Dice, which automatically adjusts the shape of the activation function based on the distribution of input data in each layer. This helps prevent gradient vanishing or exploding issues and contributes to better training stability and generalization.

Overall, DIN's ability to adapt to user interests, capture rich feature interactions, and leverage efficient training techniques makes it a powerful model for click-through rate prediction in online advertising. Its innovations have led to significant improvements in model performance and user engagement in real-world applications.



## Mini-batch aware regularization 

This is a regularization method used in large-scale sparse scenarios to reduce computational overhead and mitigate the risk of overfitting. Its main idea is to update the gradients only for the parts of the parameters that are non-zero within each mini-batch, rather than updating the entire parameter matrix. This approach avoids penalizing features that are not present in the current mini-batch, thereby preserving more information. Additionally, it helps address overfitting issues in data with a long-tailed distribution by applying larger penalties to the tail samples and smaller penalties to the head samples to prevent overfitting to the tail.


Let's assume you have a feature matrix X, which is a high-dimensional sparse binary matrix. Each row represents a sample, and each column represents a feature. The goal is to map this feature matrix to a lower-dimensional dense vector, which involves performing embedding operations. You can define an embedding matrix W, where each column represents the embedding vector for a feature. If you use L2 regularization to prevent overfitting, you would typically include a regularization term in the loss function:

\[ \text{Regularization Term} = \frac{\lambda}{2} \sum_{i,j} W_{ij}^2 \]

Where:
- \(\lambda\) is the regularization strength.
- \(W_{ij}\) represents the weight associated with the i-th feature and the j-th embedding dimension.

The issue with this approach is twofold:

1. **High Computational Cost:** Because X is a high-dimensional sparse matrix, W is also a high-dimensional sparse matrix. Updating every element of W can be computationally expensive because you would need to iterate through the entire W matrix, consuming significant time and memory.

2. **Information Loss:** Since X is sparse, many features may not appear in certain samples. Penalizing the embedding vectors corresponding to these absent features reduces the weights of these features, potentially leading to the loss of valuable information associated with these features.

Mini-batch aware regularization addresses these issues by only updating the parts of the parameter matrix that are relevant within each mini-batch, making the process more efficient and preserving the information contained in the sparse data.



## The Dice activation function
 
this is  an improved variant of the ReLU (Rectified Linear Unit) activation function. Its distinctive feature is its ability to adaptively adjust the threshold for the step function based on the data distribution, thereby addressing issues such as the "dying ReLU" problem and insensitivity to small changes. Its definition is as follows:

Let \( x \) be the input to the Dice activation function, and \( y \) be the output. The Dice activation function is defined as follows:

\[ y(x) = \frac{2}{1 + e^{-x}} - 1 \]

In this formula:

- \( e \) is the base of the natural logarithm (approximately 2.71828).
- \( x \) is the input to the activation function.
- \( y \) is the output of the Dice activation function.

The Dice activation function introduces a sigmoid-like behavior with a dynamic threshold that adapts to the input data distribution. This adaptability allows the activation function to be more responsive to different patterns in the data, improving the training and generalization of neural networks. It helps mitigate the issues associated with traditional ReLU activations, such as neurons becoming inactive ("dying ReLU") or insensitivity to small input changes.









## 原文link

https://arxiv.org/abs/1706.06978




## 代码实现

https://link.zhihu.com/?target=https%3A//github.com/i-Jayus/RecSystem-Pytorch



