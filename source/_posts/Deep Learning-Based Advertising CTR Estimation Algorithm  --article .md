---
title: Deep Learning-Based Advertising CTR Estimation Algorithm  --article 
date: 2023-08-29 21:24:00
categories:
  - 排序模型 
tags:
  - ctr 
  - deep learning  
  - 推荐系统
description:  Deep Learning-Based Advertising CTR Estimation Algorithm
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230926.png
---



##  brief introduction 

阿里巴巴开发的MLR模型是对线性LR模型的扩展。它使用分段线性方法来拟合数据。基本思想是采用分治策略：如果分类空间本身是非线性的，那么将空间划分为多个区域，每个区域可以线性拟合。最后，MLR的输出成为多个子区域的预测的加权平均。用今天的术语来说，MLR模型可以被视为具有一个隐藏层的神经网络。



The MLR model, developed by Alibaba, is an extension of the linear LR model. It uses a piecewise linear approach to fit the data. The basic idea is to use a divide-and-conquer strategy: if the classification space itself is nonlinear, then divide the space into multiple regions, where each region can be fitted linearly. In the end, the MLR output becomes a weighted average of predictions for multiple subregions. In today's terms, the MLR model can be seen as a neural network with one hidden layer.

DSSM（深度结构化语义模型）是由微软在2013年提出的。原理涉及从查询/文档中提取关键信息（词向量）并执行简单的词哈希。然后，将查询/文档域投影到300维子空间。查询中的每个词对应一个300维向量，查询包含多个向量。这些向量相加以获得一个整合的300维向量，这是一个典型的嵌入操作。提到的数字，如30k（词字典长度）、300（嵌入维度）以及由此产生的大约数千万个参数，强调了DSSM如何探索以密集格式表示大量稀疏ID。


The DSSM (Deep Structured Semantic Model) was proposed by Microsoft in 2013. The principle involves extracting key information (Term Vector) from the query/doc and performing simple Word Hashing. Then, the query/doc domains are projected into 300-dimensional subspaces. Each word in the query corresponds to a 300-dimensional vector, and the query contains multiple vectors. These vectors are summed to obtain a consolidated 300-dimensional vector, which is a typical embedding operation. The numbers mentioned, such as 30k (word dictionary length), 300 (embedding dimensions), and the resulting approximately tens of millions of parameters, highlight how DSSM explores representing a large number of sparse IDs in a dense format.


## Multi-modal signal input

在工业和学术环境中，“组”或“字段”的概念指的是描述用户或项目的不同特征集合。这些组可以包括用户的年龄、性别、最近的浏览历史等特征。这些术语，“组”和“字段”，是可以互换的，它们代表了相同的概念。

In industrial and academic contexts, the concept of "groups" or "fields" refers to different sets of features that describe a user or an item. These groups can include features like a user's age, gender, recent browsing history, and more. These terms, "groups" and "fields," are interchangeable and represent the same concept.

多模态信号输入是指将多种类型的数据或信号用作模型的输入。在提到的上下文中，DIN网络结构利用注意力机制来捕捉广告和用户之间的相关性。它的不同之处在于，它将用户行为ID特征和图像特征无缝地集成在一起，这两种不同类型的数据形式，以解决预测问题。

Multi-modal signal input refers to the use of multiple types of data or signals as input to a model. In the context mentioned, the DIN network structure utilizes an attention mechanism to capture the correlation between ads and users. What makes it different is that it seamlessly integrates user behavior ID features and image features, which represent two different modalities of data, to solve the prediction problem.

处理大规模数据集时会出现挑战。例如，如果有1000亿个样本，每个样本有500个用户行为ID特征和相应的图像，那么这个数据集就会变得非常大。存储和处理这样的数据可能会计算上非常昂贵，并且需要大量的存储资源。为了解决这些挑战，阿里巴巴已经开发出了比传统的参数服务器（PS）架构更先进的AMS深度学习训练架构。这种高级架构有助于有效地处理大规模数据集上的深度学习模型的训练。

Challenges arise when dealing with a large-scale dataset. For instance, if there are 100 billion samples, each with 500 user behavior ID features and corresponding images, the dataset can become massive. Storing and processing such data can be computationally expensive and require significant storage resources. To address these challenges, Alibaba has developed the AMS deep learning training architecture, which is more advanced than traditional Parameter Server (PS) architectures. This advanced architecture helps in efficiently handling the training of deep learning models on massive datasets.




## Multi 场景 迁移 

在机器学习和模型训练的背景下，更多的数据通常会导致模型性能的提高。

然而，在实际的移动淘宝应用中，存在各种广告场景，如主页横幅、购物指南场景等。这些场景中的每一个都代表了不同的用户兴趣。仅仅将来自不同场景的数据合并用于模型训练可能不会产生令人满意的结果，因为这些场景之间的数据分布存在差异。由于这些分布差异，从不同的场景组合样本可能会对模型性能产生负面影响。


In the context of machine learning and model training, having more data often leads to improved model performance. 

However, in practical applications like mobile Taobao, there are various advertising scenarios, such as homepage banners, shopping guide scenes, and more. Each of these scenarios represents different user interests. Simply merging data from different scenarios for model training may not yield satisfactory results because of the differences in data distributions between these scenarios. Combining samples from different scenarios can negatively affect model performance due to these distribution differences.







##  other model 架构




在多任务学习（MTL）的背景下，两个任务通常被分为两个子网络。对于涉及学习嵌入或表示稀疏特征的底层，使用“共享查找表”。这种共享表示学习是有益的，因为它允许大样本子任务帮助小样本子任务，使底层的表示学习更加全面。

In the context of multi-task learning (MTL), two tasks are typically separated into two sub-networks. For the bottom layer, which involves learning embeddings or representing sparse features, a "Shared Lookup Table" is used. This shared representation learning is beneficial as it allows the large-sample sub-task to assist the small-sample sub-task, making the representation learning at the bottom layer more comprehensive.

对于上层子网络，不同的任务被分为不同的子网络。然后每个子网络可以独立地为其特定任务相关的概念分布建模。在更复杂的场景中，考虑不同任务之间的关系是至关重要的，这也有助于提高预测性能。这个方面被称为“标签信息传递”。MTL通过探索不同场景之间的独立性和相关性，为跨场景迁移应用打开了可能性，从而提高了每个场景的性能。


For the upper layer sub-networks, different tasks are separated into different sub-networks. Each sub-network can then independently model the concept distribution relevant to its specific task. In more complex scenarios, it's essential to consider the relationships between different tasks, which can also help improve prediction performance. This aspect is referred to as "Label Information Transfer." MTL opens up possibilities for cross-scenario transfer applications by exploring the independence and correlation between different scenarios to improve performance in each scenario.

“深度个性化质量评分网络”（DQM）通常涉及一个多阶段过程。第一步使用简单的质量评分进行初步筛选（Q评分是每个广告点击率的基本度量）。在第二步，使用DQM模型进行初步排序，这涉及到从数千个广告筛选到数百个广告。最后，使用更复杂和精细的模型，如DIN，从数百个广告中选择少量高度准确的广告。


The "Deep Personalized Quality Score Network" (DQM) typically involves a multi-stage process. The first step uses a simple quality score for initial screening (Qscore is a basic measure of click-through rate for each ad). In the second step, the DQM model is employed for preliminary ranking, which involves filtering down from thousands to hundreds of ads. Finally, a more complex and fine-grained model, such as DIN, is used to select a small number of highly accurate ads from the hundreds.


在第二步，需要在毫秒内对数千个广告进行评分，模型结构不能过于复杂。DQM模型的设计类似于深度结构化语义模型（DSSM），其中不同领域（如用户域、广告域和场景域）表示为向量。最终输出是通过向量之间的简单操作（如内积操作）产生的分数。与传统静态质量评分（Q评分）模型相比，DQM



In the second step, where thousands of ads need to be scored within milliseconds, the model structure cannot be overly complex. The DQM model is designed similarly to the Deep Structured Semantic Model (DSSM), with different domains like user domain, ad domain, and scene domain represented as vectors. The final output is generated by simple operations between vectors, such as inner product operations, to produce scores. Compared to traditional static quality score (Qscore) models, DQM introduces personalization, resulting in significantly improved performance.


