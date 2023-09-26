---
title: dien Deep Interest Evolution Network for Click-Through Rate Prediction --阿里巴巴
date: 2023-09-07 12:11:00
categories:
  - 排序模型
tags:
  - dnn
  - dien
  - din
  - 推荐系统
description: DIEN extracts the user's interest sequence based on their historical behavior. It aims to understand how a user's interests change or evolve when considering a specific item.
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_202120.png
---



## 改进点


DIEN, which stands for "Deep Interest Evolution Network," is a model designed for capturing and modeling the evolution of user interests in the context of recommendation systems. It does this through a two-step process:

1. **Interest Extraction Layer**: In this step, DIEN extracts the user's interest sequence based on their historical behavior. It considers the user's interactions and activities, such as clicks, views, or purchases, over time to create a sequence of user interests.

2. **Interest Evolution Layer**: This layer focuses on modeling the evolution of user interests, particularly in relation to a target item. It aims to understand how a user's interests change or evolve when considering a specific item.

After these two steps, DIEN combines the final interest representation obtained from the Interest Evolution Layer with other relevant information, such as ad information, user profiles, and contextual data. This concatenation of representations serves as input to a multi-layer perceptron (MLP) or a similar neural network structure.

The purpose of this approach is to capture the dynamics of user interests and how they relate to specific items or recommendations. By modeling the evolution of interests, DIEN aims to make more accurate and personalized recommendations for users over time. This makes DIEN suitable for scenarios like personalized advertising and content recommendation where user preferences and interests can change dynamically.





## Interest Extraction Layer

The input data for this layer has dimensions N*T, where N represents the number of users, and T represents the sequence length, as illustrated in the diagram. According to the paper, the authors create N pairs of behavior sequences. Each pair consists of a positive and a negative sequence sample. The length of each sequence in the pair is T.

In this context, the goal of the Interest Extraction Layer is to process these behavior sequences and extract meaningful representations of user interests from the given data. These representations can then be used to understand and model how users' interests evolve over time, which is crucial for making personalized recommendations in recommendation systems.


## Interest Evolution Layer


The "Interest Evolution Layer" addresses a limitation of the Interest Extraction Layer by introducing the concept of attention. The Interest Extraction Layer is often seen as too "uniform" in its modeling of user interests because it treats all user behaviors equally, like a Markov chain. In reality, human interests can be more dynamic and discontinuous. People can quickly change their interests based on a single sentence or a sudden inspiration, which is common in feed-based applications. Therefore, more recent interactions or behaviors should have higher relevance and weighting in modeling user interests.

To capture this dynamic and non-uniform aspect of user interests, the Interest Evolution Layer introduces attention mechanisms. Attention mechanisms allow the model to assign varying degrees of importance to different behaviors or interactions based on their relevance or recency. In this way, the model can focus more on recent behaviors and give them higher weights when predicting a user's next interested item.

By incorporating attention mechanisms, the model becomes more flexible and adaptive to changes in user interests, making it better suited for applications like personalized recommendations, where user preferences can evolve rapidly.



## The computation method 


1. **Input Representation**: Start with the input representation, which includes the user's historical behavior sequence. Each behavior in the sequence is typically represented as an embedding vector.

2. **Query and Key Vectors**: Calculate query vectors and key vectors for the behaviors in the sequence. These vectors are often linear transformations of the input embeddings.

3. **Attention Scores**: Compute attention scores between the query vectors and key vectors. The attention scores measure the relevance or similarity between each behavior and the others in the sequence. Common methods for calculating attention scores include dot product, scaled dot product, or a learned similarity function.

4. **Attention Weights**: Apply a softmax function to the attention scores to obtain attention weights. The softmax operation normalizes the scores to create a probability distribution over the behaviors in the sequence. Behaviors with higher attention weights are considered more important in the context of predicting the user's next interested item.

5. **Weighted Sum**: Multiply the attention weights by the input embeddings to obtain weighted embeddings for each behavior. These weighted embeddings emphasize the importance of each behavior based on the attention mechanism's output.

6. **Contextual Representation**: Sum or concatenate the weighted embeddings to create a contextual representation of the user's historical behaviors. This contextual representation captures the user's evolving interests while giving more weight to recent or relevant behaviors.

7. **Prediction**: The contextual representation is often used as input to a neural network, such as an MLP or a softmax layer, to predict the user's next interested item or make recommendations.

The specific formulas and details of the attention mechanism (e.g., dot product attention, scaled dot product attention, or other variations) can vary based on the architecture and requirements of the DIEN model or similar models. The key idea is to dynamically assign importance to different behaviors in the sequence based on their relevance or recency, enabling the model to capture the evolving nature of user interests.



## AUGRU 

This modification enhances the GRU's ability to focus on specific parts of the input sequence, making it more suitable for tasks where attention to certain elements or features is critical. AUGRU is designed to improve the model's capacity to capture and utilize information in a more context-aware and adaptive manner, which can lead to better performance in various applications, including natural language processing and sequence modeling tasks.


## 实验

1. **BaseModel**: BaseModel follows the same embedding and Multinomial Logistic Regression (MLR) settings as DIEN. It uses a sum pooling operation to integrate behavior embeddings.

2. **Wide&Deep**: Wide & Deep consists of two parts. Its deep model is the same as BaseModel, and its wide model is a linear model. This approach combines both a deep neural network and a linear model for CTR prediction.

3. **PNN (Product-based Neural Network)**: PNN utilizes a product layer to capture interactive patterns between interfield categories. It focuses on modeling interactions between different category features.

4. **DIN (Deep Interest Network)**: DIN uses an attention mechanism to activate related user behaviors. It considers the relevance of different user behaviors when making predictions and incorporates this into the model.

5. **Two-layer GRU with Attention**: This method uses a two-layer Gated Recurrent Unit (GRU) to model sequential behaviors. Additionally, it incorporates an attention layer to activate relative behaviors. This approach is designed to capture sequential patterns in user behavior.

These compared methods represent various approaches to CTR prediction, including traditional linear models (e.g., Wide&Deep), models that focus on capturing interactions between features (e.g., PNN), and models that utilize attention mechanisms (e.g., DIN and Two-layer GRU with Attention). DIEN is evaluated against these methods to assess its performance and effectiveness in modeling the evolution of user interests in the context of CTR prediction.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_202553.png)



## 优化

During the online A/B testing conducted on Taobao from June 7, 2018, to July 12, 2018, DIEN showed significant improvements over the BaseModel, with a 20.7% increase in Click-Through Rate (CTR) and a 17.1% increase in eCPM (effective Cost Per Mille).

The success of DIEN in the online environment can be attributed to various optimization techniques employed to address the dual challenges of algorithmic and engineering aspects. Here's a breakdown of these optimization measures:

1. **Element Parallel GRU & Kernel Fusion**: This optimization involves parallelizing the computation of hidden states for each GRU (Gated Recurrent Unit). It aims to efficiently compute the hidden states in parallel, taking advantage of GPU capabilities. Additionally, it applies kernel fusion techniques to combine as many independent kernels as possible, further improving computational efficiency.

2. **Batching**: Batching is used to group adjacent requests from the same user into a single batch. This batching strategy leverages the processing power of GPUs and allows for more efficient computations by processing multiple requests simultaneously within a batch.

3. **Model Compressing with Rocket Launching**: Rocket Launching is a model compression technique that was applied to reduce the dimensions of the GRU's hidden states. According to the paper, this technique allowed the reduction of the hidden state dimension from 108 to 32 while maintaining performance.

These optimization techniques collectively led to significant improvements in server latency, reducing it from 38.2 to 6.6. Furthermore, the QPS (Queries Per Second) of a single worker increased to 360, showcasing the efficiency gains achieved through these optimizations in an online production environment.



## 原文link

https://arxiv.org/abs/1809.03672



## 代码实现


https%3A//github.com/shenweichen/DeepCTR
