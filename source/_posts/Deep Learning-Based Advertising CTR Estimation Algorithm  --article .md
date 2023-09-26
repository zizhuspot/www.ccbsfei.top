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


The MLR model, developed by Alibaba, is an extension of the linear LR model. It uses a piecewise linear approach to fit the data. The basic idea is to use a divide-and-conquer strategy: if the classification space itself is nonlinear, then divide the space into multiple regions, where each region can be fitted linearly. In the end, the MLR output becomes a weighted average of predictions for multiple subregions. In today's terms, the MLR model can be seen as a neural network with one hidden layer.

The DSSM (Deep Structured Semantic Model) was proposed by Microsoft in 2013. The principle involves extracting key information (Term Vector) from the query/doc and performing simple Word Hashing. Then, the query/doc domains are projected into 300-dimensional subspaces. Each word in the query corresponds to a 300-dimensional vector, and the query contains multiple vectors. These vectors are summed to obtain a consolidated 300-dimensional vector, which is a typical embedding operation. The numbers mentioned, such as 30k (word dictionary length), 300 (embedding dimensions), and the resulting approximately tens of millions of parameters, highlight how DSSM explores representing a large number of sparse IDs in a dense format.


## Multi-modal signal input



In industrial and academic contexts, the concept of "groups" or "fields" refers to different sets of features that describe a user or an item. These groups can include features like a user's age, gender, recent browsing history, and more. These terms, "groups" and "fields," are interchangeable and represent the same concept.

Multi-modal signal input refers to the use of multiple types of data or signals as input to a model. In the context mentioned, the DIN network structure utilizes an attention mechanism to capture the correlation between ads and users. What makes it different is that it seamlessly integrates user behavior ID features and image features, which represent two different modalities of data, to solve the prediction problem.

Challenges arise when dealing with a large-scale dataset. For instance, if there are 100 billion samples, each with 500 user behavior ID features and corresponding images, the dataset can become massive. Storing and processing such data can be computationally expensive and require significant storage resources. To address these challenges, Alibaba has developed the AMS deep learning training architecture, which is more advanced than traditional Parameter Server (PS) architectures. This advanced architecture helps in efficiently handling the training of deep learning models on massive datasets.




## Multi 场景 迁移 


In the context of machine learning and model training, having more data often leads to improved model performance. 

However, in practical applications like mobile Taobao, there are various advertising scenarios, such as homepage banners, shopping guide scenes, and more. Each of these scenarios represents different user interests. Simply merging data from different scenarios for model training may not yield satisfactory results because of the differences in data distributions between these scenarios. Combining samples from different scenarios can negatively affect model performance due to these distribution differences.




##  other model 架构


In the context of multi-task learning (MTL), two tasks are typically separated into two sub-networks. For the bottom layer, which involves learning embeddings or representing sparse features, a "Shared Lookup Table" is used. This shared representation learning is beneficial as it allows the large-sample sub-task to assist the small-sample sub-task, making the representation learning at the bottom layer more comprehensive.

For the upper layer sub-networks, different tasks are separated into different sub-networks. Each sub-network can then independently model the concept distribution relevant to its specific task. In more complex scenarios, it's essential to consider the relationships between different tasks, which can also help improve prediction performance. This aspect is referred to as "Label Information Transfer." MTL opens up possibilities for cross-scenario transfer applications by exploring the independence and correlation between different scenarios to improve performance in each scenario.

The "Deep Personalized Quality Score Network" (DQM) typically involves a multi-stage process. The first step uses a simple quality score for initial screening (Qscore is a basic measure of click-through rate for each ad). In the second step, the DQM model is employed for preliminary ranking, which involves filtering down from thousands to hundreds of ads. Finally, a more complex and fine-grained model, such as DIN, is used to select a small number of highly accurate ads from the hundreds.

In the second step, where thousands of ads need to be scored within milliseconds, the model structure cannot be overly complex. The DQM model is designed similarly to the Deep Structured Semantic Model (DSSM), with different domains like user domain, ad domain, and scene domain represented as vectors. The final output is generated by simple operations between vectors, such as inner product operations, to produce scores. Compared to traditional static quality score (Qscore) models, DQM introduces personalization, resulting in significantly improved performance.

