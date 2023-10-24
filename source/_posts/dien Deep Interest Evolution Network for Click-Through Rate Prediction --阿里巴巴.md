---
title: dien深度兴趣演化网络提取用户的兴趣序列 --阿里巴巴
date: 2023-09-07 12:11:00
categories:
  - 排序模型
tags:
  - dnn
  - dien
  - din
  - 推荐系统
description: DIEN extracts the user's interest sequence based on their historical behavior. It aims to understand how a user's interests change or evolve when considering a specific item. DIEN（深度兴趣演化网络）根据用户的历史行为提取用户的兴趣序列，旨在了解用户在考虑特定项目时兴趣如何变化或演化。 
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_202120.png
---



## 改进点


DIEN，全名"深度兴趣演化网络"，是一种专为在推荐系统背景下捕捉和建模用户兴趣演化而设计的模型。它通过以下两个步骤来实现这一目标：

DIEN, which stands for "Deep Interest Evolution Network," is a model designed for capturing and modeling the evolution of user interests in the context of recommendation systems. It does this through a two-step process:

1. **兴趣提取层**：在这一步中，DIEN根据用户的历史行为提取用户的兴趣序列。它考虑用户随时间的交互和活动，例如点击、浏览或购买，以创建用户兴趣的序列。

1. **Interest Extraction Layer**: In this step, DIEN extracts the user's interest sequence based on their historical behavior. It considers the user's interactions and activities, such as clicks, views, or purchases, over time to create a sequence of user interests.

2. **兴趣演化层**：该层着重于建模用户兴趣的演化，特别是与目标项目相关的兴趣演化。其目标是了解用户在考虑特定项目时兴趣如何变化或演化。

2. **Interest Evolution Layer**: This layer focuses on modeling the evolution of user interests, particularly in relation to a target item. It aims to understand how a user's interests change or evolve when considering a specific item.

经过这两个步骤，DIEN将从兴趣演化层获得的最终兴趣表示与其他相关信息（例如广告信息、用户档案和上下文数据）进行组合。这些表示的串联构成了多层感知器（MLP）或类似的神经网络结构的输入。

这种方法的目的是捕捉用户兴趣的动态变化，以及它们与特定项目或推荐的关系。通过建模兴趣的演化，DIEN旨在为用户提供随时间更准确和个性化的推荐。这使得DIEN适用于个性化广告和内容推荐等场景，其中用户的偏好和兴趣可能会动态变化。

After these two steps, DIEN combines the final interest representation obtained from the Interest Evolution Layer with other relevant information, such as ad information, user profiles, and contextual data. This concatenation of representations serves as input to a multi-layer perceptron (MLP) or a similar neural network structure.

The purpose of this approach is to capture the dynamics of user interests and how they relate to specific items or recommendations. By modeling the evolution of interests, DIEN aims to make more accurate and personalized recommendations for users over time. This makes DIEN suitable for scenarios like personalized advertising and content recommendation where user preferences and interests can change dynamically.





## Interest Extraction Layer
在这一层的输入数据维度为N*T，其中N代表用户数量，T代表序列长度，如图所示。根据论文，作者们创建了N对行为序列。每对包括一个正样本和一个负样本的序列样本，每对序列的长度都是T。

The input data for this layer has dimensions N*T, where N represents the number of users, and T represents the sequence length, as illustrated in the diagram. According to the paper, the authors create N pairs of behavior sequences. Each pair consists of a positive and a negative sequence sample. The length of each sequence in the pair is T.

在这个背景下，兴趣提取层的目标是处理这些行为序列，并从给定数据中提取用户兴趣的有意义的表示。然后可以使用这些表示来理解和建模用户兴趣随时间的演化，这对于在推荐系统中进行个性化推荐至关重要。

In this context, the goal of the Interest Extraction Layer is to process these behavior sequences and extract meaningful representations of user interests from the given data. These representations can then be used to understand and model how users' interests evolve over time, which is crucial for making personalized recommendations in recommendation systems.

兴趣提取层的作用是通过处理用户的历史行为数据，捕捉和提取用户兴趣的关键特征，从而帮助系统更好地理解用户的兴趣演化，以实现更加精准的个性化推荐。

## Interest Evolution Layer

"兴趣演化层"通过引入注意机制来解决"兴趣提取层"的一个局限性。"兴趣提取层"通常被认为在建模用户兴趣时过于"均匀"，因为它将所有用户行为等同对待，就像一个马尔可夫链。然而，在现实生活中，人的兴趣可以更加动态和不连续。人们可以根据一句话或突发灵感迅速改变兴趣，这在基于动态信息流的应用中很常见。因此，更近期的互动或行为在建模用户兴趣时应具有更高的相关性和权重。

The "Interest Evolution Layer" addresses a limitation of the Interest Extraction Layer by introducing the concept of attention. The Interest Extraction Layer is often seen as too "uniform" in its modeling of user interests because it treats all user behaviors equally, like a Markov chain. In reality, human interests can be more dynamic and discontinuous. People can quickly change their interests based on a single sentence or a sudden inspiration, which is common in feed-based applications. Therefore, more recent interactions or behaviors should have higher relevance and weighting in modeling user interests.

为了捕捉用户兴趣的这种动态和非均匀特性，"兴趣演化层"引入了注意机制。注意机制允许模型根据行为的相关性或最新性分配不同程度的重要性。通过这种方式，模型可以在预测用户下一个感兴趣的项目时更加关注最近的行为，并赋予它们更高的权重。

To capture this dynamic and non-uniform aspect of user interests, the Interest Evolution Layer introduces attention mechanisms. Attention mechanisms allow the model to assign varying degrees of importance to different behaviors or interactions based on their relevance or recency. In this way, the model can focus more on recent behaviors and give them higher weights when predicting a user's next interested item.

通过整合注意机制，模型变得更加灵活，能够适应用户兴趣的变化，使其更适用于个性化推荐等应用，其中用户偏好可能会迅速演化。

By incorporating attention mechanisms, the model becomes more flexible and adaptive to changes in user interests, making it better suited for applications like personalized recommendations, where user preferences can evolve rapidly.



## The computation method 

1. **输入表示**：从输入表示开始，其中包括用户的历史行为序列。序列中的每个行为通常表示为一个嵌入向量。

1. **Input Representation**: Start with the input representation, which includes the user's historical behavior sequence. Each behavior in the sequence is typically represented as an embedding vector.

2. **查询和键向量**：计算行为序列中的查询向量和键向量。这些向量通常是输入嵌入的线性变换。

2. **Query and Key Vectors**: Calculate query vectors and key vectors for the behaviors in the sequence. These vectors are often linear transformations of the input embeddings.

3. **注意力分数**：计算查询向量和键向量之间的注意力分数。注意力分数度量了序列中每个行为与其他行为之间的相关性或相似性。计算注意力分数的常见方法包括点积、缩放点积或学习的相似性函数。

3. **Attention Scores**: Compute attention scores between the query vectors and key vectors. The attention scores measure the relevance or similarity between each behavior and the others in the sequence. Common methods for calculating attention scores include dot product, scaled dot product, or a learned similarity function.

4. **注意权重**：将注意力分数应用softmax函数，以获得注意力权重。Softmax操作将分数归一化，创建了一个概率分布，用于表示序列中的行为。具有更高注意权重的行为在预测用户下一个感兴趣的项目时被认为更重要。

4. **Attention Weights**: Apply a softmax function to the attention scores to obtain attention weights. The softmax operation normalizes the scores to create a probability distribution over the behaviors in the sequence. Behaviors with higher attention weights are considered more important in the context of predicting the user's next interested item.

5. **加权求和**：将注意力权重乘以输入嵌入，以获得每个行为的加权嵌入。这些加权嵌入强调了每个行为的重要性，基于注意力机制的输出。

5. **Weighted Sum**: Multiply the attention weights by the input embeddings to obtain weighted embeddings for each behavior. These weighted embeddings emphasize the importance of each behavior based on the attention mechanism's output.

6. **上下文表示**：对加权嵌入进行求和或连接，以创建用户历史行为的上下文表示。这个上下文表示捕捉了用户兴趣的演化，同时更重视最近或相关的行为。

6. **Contextual Representation**: Sum or concatenate the weighted embeddings to create a contextual representation of the user's historical behaviors. This contextual representation captures the user's evolving interests while giving more weight to recent or relevant behaviors.

7. **预测**：上下文表示通常用作神经网络的输入，例如多层感知器（MLP）或softmax层，用于预测用户下一个感兴趣的项目或进行推荐。

7. **Prediction**: The contextual representation is often used as input to a neural network, such as an MLP or a softmax layer, to predict the user's next interested item or make recommendations.

具体的注意力机制公式和细节（例如点积注意力、缩放点积注意力或其他变种）可以根据DIEN模型或类似模型的架构和需求而有所不同。关键思想是根据它们的相关性或最新性动态分配不同行为的重要性，使模型能够捕捉用户兴趣的演化特性。

The specific formulas and details of the attention mechanism (e.g., dot product attention, scaled dot product attention, or other variations) can vary based on the architecture and requirements of the DIEN model or similar models. The key idea is to dynamically assign importance to different behaviors in the sequence based on their relevance or recency, enabling the model to capture the evolving nature of user interests.



## AUGRU 
这种修改增强了GRU（门控循环单元）专注于输入序列的特定部分的能力，使其更适用于对某些元素或特征的关注至关重要的任务。AUGRU 旨在提高模型以更具上下文意识和自适应方式捕获和利用信息的能力，这可以在各种应用中提高性能，包括自然语言处理和序列建模任务。

This modification enhances the GRU's ability to focus on specific parts of the input sequence, making it more suitable for tasks where attention to certain elements or features is critical. AUGRU is designed to improve the model's capacity to capture and utilize information in a more context-aware and adaptive manner, which can lead to better performance in various applications, including natural language processing and sequence modeling tasks.


## 实验
1. **BaseModel**：BaseModel遵循与DIEN相同的嵌入和多项式逻辑回归（MLR）设置。它使用总和池化操作来整合行为嵌入。

1. **BaseModel**: BaseModel follows the same embedding and Multinomial Logistic Regression (MLR) settings as DIEN. It uses a sum pooling operation to integrate behavior embeddings.

2. **Wide&Deep**：Wide & Deep 由两部分组成。其深度模型与BaseModel相同，其宽模型是一个线性模型。这种方法结合了深度神经网络和线性模型，用于点击率（CTR）预测。

2. **Wide&Deep**: Wide & Deep consists of two parts. Its deep model is the same as BaseModel, and its wide model is a linear model. This approach combines both a deep neural network and a linear model for CTR prediction.

3. **PNN（基于产品的神经网络）**：PNN 利用产品层来捕捉不同领域类别之间的交互模式。它专注于建模不同类别特征之间的相互作用。

3. **PNN (Product-based Neural Network)**: PNN utilizes a product layer to capture interactive patterns between interfield categories. It focuses on modeling interactions between different category features.

4. **DIN（深度兴趣网络）**：DIN 使用注意力机制来激活相关用户行为。它在进行预测时考虑了不同用户行为的相关性，并将其纳入模型中。

4. **DIN (Deep Interest Network)**: DIN uses an attention mechanism to activate related user behaviors. It considers the relevance of different user behaviors when making predictions and incorporates this into the model.

5. **带注意力机制的双层GRU**：这种方法使用双层门控循环单元（GRU）来建模顺序行为。此外，它还包括一个注意力层，以激活相关行为。这种方法旨在捕捉用户行为中的顺序模式。

5. **Two-layer GRU with Attention**: This method uses a two-layer Gated Recurrent Unit (GRU) to model sequential behaviors. Additionally, it incorporates an attention layer to activate relative behaviors. This approach is designed to capture sequential patterns in user behavior.

这些比较方法代表了CTR预测的各种方法，包括传统线性模型（例如Wide&Deep），专注于捕捉特征之间交互的模型（例如PNN），以及利用注意力机制的模型（例如DIN和带注意力机制的双层GRU）。DIEN被评估与这些方法对比，以评估其在CTR预测背景下捕捉用户兴趣演化的性能和有效性。

These compared methods represent various approaches to CTR prediction, including traditional linear models (e.g., Wide&Deep), models that focus on capturing interactions between features (e.g., PNN), and models that utilize attention mechanisms (e.g., DIN and Two-layer GRU with Attention). DIEN is evaluated against these methods to assess its performance and effectiveness in modeling the evolution of user interests in the context of CTR prediction.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_202553.png)



## 优化
在2018年6月7日至2018年7月12日期间在淘宝进行的在线A/B测试中，DIEN在Click-Through Rate（CTR）上相对BaseModel取得了显著的改善，CTR增长了20.7%，eCPM（每千次展示的有效成本）增长了17.1%。

During the online A/B testing conducted on Taobao from June 7, 2018, to July 12, 2018, DIEN showed significant improvements over the BaseModel, with a 20.7% increase in Click-Through Rate (CTR) and a 17.1% increase in eCPM (effective Cost Per Mille).

DIEN在在线环境中的成功可以归因于采用的各种优化技术，以解决算法和工程两方面的挑战。以下是这些优化措施的详细说明：

The success of DIEN in the online environment can be attributed to various optimization techniques employed to address the dual challenges of algorithmic and engineering aspects. Here's a breakdown of these optimization measures:

1. **元素并行GRU和内核融合**：这个优化涉及并行计算每个GRU（门控循环单元）的隐藏状态。它旨在利用GPU的能力，以并行高效地计算隐藏状态。此外，它应用了内核融合技术，将尽可能多的独立内核组合在一起，进一步提高了计算效率。

1. **Element Parallel GRU & Kernel Fusion**: This optimization involves parallelizing the computation of hidden states for each GRU (Gated Recurrent Unit). It aims to efficiently compute the hidden states in parallel, taking advantage of GPU capabilities. Additionally, it applies kernel fusion techniques to combine as many independent kernels as possible, further improving computational efficiency.

2. **批处理**：批处理用于将来自同一用户的相邻请求分组成一个批次。这种批处理策略充分利用了GPU的处理能力，并通过在一个批次内同时处理多个请求来实现更高效的计算。

2. **Batching**: Batching is used to group adjacent requests from the same user into a single batch. This batching strategy leverages the processing power of GPUs and allows for more efficient computations by processing multiple requests simultaneously within a batch.

3. **使用Rocket Launching进行模型压缩**：Rocket Launching是一种模型压缩技术，用于减小GRU隐藏状态的维度。根据论文，这种技术允许将隐藏状态的维度从108减小到32，同时保持性能不受影响。

3. **Model Compressing with Rocket Launching**: Rocket Launching is a model compression technique that was applied to reduce the dimensions of the GRU's hidden states. According to the paper, this technique allowed the reduction of the hidden state dimension from 108 to 32 while maintaining performance.

这些优化技术共同导致服务器延迟显著降低，从38.2降至6.6。此外，单个工作器的每秒查询数（QPS）增加到360，展示了这些优化在在线生产环境中所取得的效率提升。

These optimization techniques collectively led to significant improvements in server latency, reducing it from 38.2 to 6.6. Furthermore, the QPS (Queries Per Second) of a single worker increased to 360, showcasing the efficiency gains achieved through these optimizations in an online production environment.



## 原文 source link 

https://arxiv.org/abs/1809.03672



## 代码 code 


https%3A//github.com/shenweichen/DeepCTR
