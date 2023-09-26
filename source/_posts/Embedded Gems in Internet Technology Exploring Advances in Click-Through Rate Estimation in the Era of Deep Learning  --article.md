---
title: Embedded Gems in Internet Technology Exploring Advances in Click-Through Rate Estimation in the Era of Deep Learning  --article 
date: 2023-08-31 22:24:00
categories:
  - 深度学习
tags:
  - 深度学习
  - deep learning  
  - 推荐系统
description:  Exploring Advances in Click-Through Rate Estimation in the Era of Deep Learning
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/8326cffc1e178a823a02564c6b85b58ba877e8a9.png
---



##  brief introduction 


User's historical browsing of products A/B/C and the prediction of ad for product C are combined using a "AND" operation to obtain the feature "ads viewed by the user in the past." Astute readers may easily associate this with later developments such as the DIN model, which uses techniques similar to attention to extend this method.

Models like PNN, DeepFM, DCN, xDeepFM, and others can be summarized as part of a coherent line of thought: using artificially constructed algebraic priors to help the model establish presuppositions about certain cognitive patterns. In the LR model era, this involved cross-combining original discrete features (Cartesian product). In today's DL era, it has evolved into combining features in the projected space after embedding using inner products, outer products, and even polynomial products. In theory, this is more effective than MLP learning arbitrary combinations of features - according to the "No Free Lunch" theorem. In other words, feature engineering based on prior knowledge in the DL field is more effective than direct MLP.

DIN and DIEN are explorations centered around modeling user interests, with a focus on the data characteristics observed in Alibaba e-commerce scenarios. They involve network structure design tailored to these characteristics. This is a higher-order learning paradigm compared to manually crafted algebraic priors: DIN captures the diversity of user interests and their local relevance to the prediction target. DIEN further strengthens the evolution of interests and the projection relationships between interests in different domains.


## esmm model 


The ESMM model, which stands for Entire Space Multi-Task Model, is designed to jointly model CTR (Click-Through Rate) and CVR (Conversion Rate) tasks. It helps address the challenges of sample bias and sparsity in CVR sub-tasks. In the context of advertising systems, it plays a crucial role in various algorithmic stages, including matching, recall, pre-selection, coarse-ranking, fine-ranking, and strategy control. These algorithms are scattered across different engineering modules.

In the existing system, the sorting process involves using a static, non-personalized quality score to complete it. This can be understood as a statistical score at the granularity of advertisements.

As I understand, some industry teams have also used simplified versions of models, such as a lightweight LR (Logistic Regression) model, to accomplish this process. The core challenge here is that the candidate set during retrieval is too large, and the computations must be streamlined to avoid excessive delays.

Figure 5 illustrates the upgraded deep personalized quality score model, referred to as DQM (Deep Quality Model). It constraints the final output to the simplest vector inner product. This model aims to improve the efficiency and effectiveness of sorting in the advertising system.




## DQM 


1) Regarding the pre-selection (coarse-ranking) DQM model, it is important to note that its primary role is to reduce the size of the candidate set, rather than being used for the final ad ranking. Therefore, its precision doesn't need to be as high as that of fine-ranking models. It should also take into consideration system performance and data cyclical perturbations.

2) The DQM model can be applied not only to the sorting process but also to other modules like retrieval, matching, and recall. For instance, many teams have adopted vectorized recall architectures, which align perfectly with DQM's model architecture. However, when applied to recall modules, the modeling signals and training samples are significantly different, with a greater emphasis on generalizing user interests.

When it comes to efficient vectorized computation, both Facebook (F) and Microsoft (M) have open-sourced excellent frameworks, namely faiss and SPTAG.


## model 原理

Figure 6 shows the Rocket Training algorithm, which is a lightweight model compression technique. This illustrates the continuous efforts to optimize and streamline algorithms for efficiency and improved performance.


At the architectural level, a feasible suggestion is to adopt a DQM-style structure, embedding user/ad/query or context into a vector space, and then using vector computation frameworks for online services. The advantage of this approach is that online prediction systems can be kept extremely simple, allowing you to focus your efforts on offline feature/model tuning. Concepts like Rocket and MTL (Multi-Task Learning) are worth exploring. This approach ensures that you can easily achieve the initial business impact.


## conclusion 

In the DL era, the term "model" has become more abstract, and fixed models like LR/MLR from the shallow model era are no longer prevalent. Models are not set in stone; they should adapt to specific scenarios. By following certain principles and understanding the unique characteristics of your data domain, you can absorb the thought processes of methods like DIN, DIEN, ESMM, and customize network architectures that are suitable for your specific problem.


