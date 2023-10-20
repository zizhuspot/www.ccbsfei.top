---
title: Embedded Gems in Internet Technology Exploring Advances in Click-Through Rate Estimation in the Era of Deep Learning  --article 
date: 2023-08-31 22:24:00
categories:
  - 深度学习
tags:
  - 深度学习
  - deep learning  
  - 推荐系统
description:  Exploring Advances in Click-Through Rate Estimation in the Era of Deep Learning 探索在深度学习时代点击率估计的进展 
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/8326cffc1e178a823a02564c6b85b58ba877e8a9.png
---



##  brief introduction 

用户对产品A/B/C的历史浏览以及对产品C广告的预测，通过"AND"操作结合，得到了"用户过去浏览的广告"这一特征。聪明的读者可能很容易将这与后来的发展联系起来，比如使用类似于注意力的技术来扩展这种方法的DIN模型。



User's historical browsing of products A/B/C and the prediction of ad for product C are combined using a "AND" operation to obtain the feature "ads viewed by the user in the past." Astute readers may easily associate this with later developments such as the DIN model, which uses techniques similar to attention to extend this method.

像PNN、DeepFM、DCN、xDeepFM等模型可以总结为一种连贯的思路：利用人工构建的代数先验来帮助模型建立关于某些认知模式的预设。在LR模型时代，这涉及交叉组合原始离散特征（笛卡尔积）。在今天的深度学习时代，它已经发展成在嵌入后的投影空间中组合特征，使用内积、外积，甚至多项式积。理论上，这比MLP学习特征的任意组合更有效 - 根据"没有免费午餐"定理。换句话说，在深度学习领域基于先验知识的特征工程比直接的MLP更有效。

Models like PNN, DeepFM, DCN, xDeepFM, and others can be summarized as part of a coherent line of thought: using artificially constructed algebraic priors to help the model establish presuppositions about certain cognitive patterns. In the LR model era, this involved cross-combining original discrete features (Cartesian product). In today's DL era, it has evolved into combining features in the projected space after embedding using inner products, outer products, and even polynomial products. In theory, this is more effective than MLP learning arbitrary combinations of features - according to the "No Free Lunch" theorem. In other words, feature engineering based on prior knowledge in the DL field is more effective than direct MLP.

DIN和DIEN是围绕建模用户兴趣的探索，重点关注阿里巴巴电子商务场景中观察到的数据特征。它们涉及到根据这些特征量身定制的网络结构设计。与手工制作的代数先验相比，这是一种高阶学习范式：DIN捕获了用户兴趣的多样性以及它们与预测目标的本地相关性。DIEN进一步加强了兴趣的演化以及不同领域兴趣之间的投影关系。 


DIN and DIEN are explorations centered around modeling user interests, with a focus on the data characteristics observed in Alibaba e-commerce scenarios. They involve network structure design tailored to these characteristics. This is a higher-order learning paradigm compared to manually crafted algebraic priors: DIN captures the diversity of user interests and their local relevance to the prediction target. DIEN further strengthens the evolution of interests and the projection relationships between interests in different domains.


## esmm model 

ESMM模型，全称Entire Space Multi-Task Model，旨在共同建模CTR（点击率）和CVR（转化率）任务。它有助于解决CVR子任务中的样本偏差和稀疏性挑战。在广告系统的背景下，它在各种算法阶段发挥关键作用，包括匹配、召回、预选、粗排、精排和策略控制。这些算法分布在不同的工程模块中。


The ESMM model, which stands for Entire Space Multi-Task Model, is designed to jointly model CTR (Click-Through Rate) and CVR (Conversion Rate) tasks. It helps address the challenges of sample bias and sparsity in CVR sub-tasks. In the context of advertising systems, it plays a crucial role in various algorithmic stages, including matching, recall, pre-selection, coarse-ranking, fine-ranking, and strategy control. These algorithms are scattered across different engineering modules.

在现有系统中，排序过程涉及使用静态的非个性化质量评分来完成。这可以被理解为广告粒度的统计评分。


In the existing system, the sorting process involves using a static, non-personalized quality score to complete it. This can be understood as a statistical score at the granularity of advertisements.


据我了解，一些行业团队还使用了模型的简化版本，例如轻量级的LR（逻辑回归）模型，来完成这个过程。这里的核心挑战是检索时的候选集太大，必须简化计算以避免过多的延迟。

As I understand, some industry teams have also used simplified versions of models, such as a lightweight LR (Logistic Regression) model, to accomplish this process. The core challenge here is that the candidate set during retrieval is too large, and the computations must be streamlined to avoid excessive delays.


图5展示了升级后的深度个性化质量评分模型，称为DQM（Deep Quality Model）。这个模型将最终输出限制为最简单的向量内积。该模型旨在提高广告系统中排序的效率和效力。 



Figure 5 illustrates the upgraded deep personalized quality score model, referred to as DQM (Deep Quality Model). It constraints the final output to the simplest vector inner product. This model aims to improve the efficiency and effectiveness of sorting in the advertising system.




## DQM 

1) 关于预选（粗排）的DQM模型，重要的是要注意它的主要作用是减小候选集的大小，而不是用于最终广告排序。因此，它的精度不需要像精排模型那样高。它还应考虑系统性能和数据周期性扰动。


1) Regarding the pre-selection (coarse-ranking) DQM model, it is important to note that its primary role is to reduce the size of the candidate set, rather than being used for the final ad ranking. Therefore, its precision doesn't need to be as high as that of fine-ranking models. It should also take into consideration system performance and data cyclical perturbations.

2) DQM模型不仅可以应用于排序过程，还可以应用于其他模块，如检索、匹配和召回。例如，许多团队采用了向量化的召回架构，这与DQM的模型架构完美契合。然而，在应用于召回模块时，建模信号和训练样本显著不同，更加强调用户兴趣的泛化。



2) The DQM model can be applied not only to the sorting process but also to other modules like retrieval, matching, and recall. For instance, many teams have adopted vectorized recall architectures, which align perfectly with DQM's model architecture. However, when applied to recall modules, the modeling signals and training samples are significantly different, with a greater emphasis on generalizing user interests.

在有效的向量化计算方面，Facebook（F）和Microsoft（M）都开源了出色的框架，分别是faiss和SPTAG。


When it comes to efficient vectorized computation, both Facebook (F) and Microsoft (M) have open-sourced excellent frameworks, namely faiss and SPTAG.


## model 原理

图6展示了Rocket Training算法，这是一种轻量级的模型压缩技术。这说明了为了提高效率和性能而进行的持续优化和算法精简的努力。

Figure 6 shows the Rocket Training algorithm, which is a lightweight model compression technique. This illustrates the continuous efforts to optimize and streamline algorithms for efficiency and improved performance.


在架构层面，一个可行的建议是采用类似DQM的结构，将用户/广告/查询或上下文嵌入到向量空间中，然后使用向量计算框架进行在线服务。这种方法的优势在于在线预测系统可以保持非常简单，使您可以将精力集中在离线特征/模型调优上。像Rocket和MTL（多任务学习）这样的概念值得探索。这种方法可以确保您轻松实现最初的业务影响。

At the architectural level, a feasible suggestion is to adopt a DQM-style structure, embedding user/ad/query or context into a vector space, and then using vector computation frameworks for online services. The advantage of this approach is that online prediction systems can be kept extremely simple, allowing you to focus your efforts on offline feature/model tuning. Concepts like Rocket and MTL (Multi-Task Learning) are worth exploring. This approach ensures that you can easily achieve the initial business impact.


## conclusion 

在深度学习时代，"模型"这个术语变得更加抽象，而像LR/MLR这样的浅层模型时代的固定模型已不再普遍存在。模型不是一成不变的；它们应该适应特定场景。通过遵循某些原则，了解您的数据领域的独特特点，您可以吸收DIN、DIEN、ESMM等方法的思维过程，并定制适合您具体问题的网络架构。


In the DL era, the term "model" has become more abstract, and fixed models like LR/MLR from the shallow model era are no longer prevalent. Models are not set in stone; they should adapt to specific scenarios. By following certain principles and understanding the unique characteristics of your data domain, you can absorb the thought processes of methods like DIN, DIEN, ESMM, and customize network architectures that are suitable for your specific problem.



