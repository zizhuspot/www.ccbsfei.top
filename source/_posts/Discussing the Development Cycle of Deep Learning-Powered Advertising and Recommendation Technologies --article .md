---
title: Discussing the Development Cycle of Deep Learning-Powered Advertising and Recommendation Technologies --article 
date: 2023-08-27 12:30:00
categories:
  - ads system  
tags:
  - ctr 
  - ads system 
description:  Discussing the Development Cycle of Deep Learning-Powered Advertising and Recommendation Technologies
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230926ads.jpg
---



##  brief introduction 

深度学习驱动的广告和推荐技术的发展周期在不同阶段经历了重要的演进：

1. **早期探索（2010年代之前）**：在深度学习出现之前，广告点击率（CTR）预测通常使用传统机器学习技术，如逻辑回归。广告商主要依赖简单的模型和启发式方法。

2. **深度学习兴起（2010年代）**：2010年代标志着深度学习技术的兴起，尤其是深度神经网络，它在计算机视觉和自然语言处理等各个领域展现出卓越的能力。研究人员和从业者开始探索将深度学习应用于广告和推荐系统。

3. **初期应用（2010年代早期）**：在2010年代初期，深度学习开始进入广告和推荐系统。研究人员尝试使用深度神经网络进行CTR预测，早期的成功鼓励了进一步的探索。

4. **规模化和架构改进（2010年代中期）**：随着深度学习的崭露头角，像谷歌和Facebook这样的公司投资大力发展可扩展的深度学习框架和架构。这导致了创建深度推荐系统，能够高效处理大规模数据集。

5. **行业采用（2010年代晚期）**：到了2010年代末，深度学习驱动的推荐系统已经成为广告业的主流。公司认识到深度学习在提高广告定位和个性化方面的潜力。许多商业平台开始采用深度学习技术进行广告投放。

6. **多模态和高级模型（2020年代）**：进入2020年代，焦点转向了多模态推荐系统，可以整合各种数据类型，包括图像、文本和用户行为。高级模型，如transformers、基于注意力的网络和强化学习，在推荐任务中备受推崇。

7. **隐私和伦理考虑（持续进行中）**：当前的发展周期特点是越来越强调隐私和伦理考虑。广告商正在开发尊重用户隐私和遵守伦理准则的AI模型。

8. **未来方向（持续进行中）**：深度学习驱动的广告和推荐技术的未来可能涉及模型架构的持续改进、更高效的训练技术，以及更深入地将AI整合到广告技术中。探索AI的可解释性、公平性和可解释性也将至关重要。

总的来说，深度学习在广告和推荐技术中的发展周期已从早期的实验发展到广泛的行业应用，重点放在改进模型架构、可扩展性和伦理考虑。未来充满了在这一领域进一步的进展和创新的希望。


The development cycle of deep learning-powered advertising and recommendation technologies has witnessed several significant phases over time:

1. **Early Exploration (Pre-2010s)**: Before the emergence of deep learning, traditional machine learning techniques like logistic regression were commonly used for click-through rate (CTR) prediction. Advertisers primarily relied on simple models and heuristics.

2. **Deep Learning Emergence (2010s)**: The 2010s marked the emergence of deep learning techniques, especially deep neural networks, which showed remarkable capabilities in various fields, including computer vision and natural language processing. Researchers and practitioners began to explore the application of deep learning in advertising and recommendation systems.

3. **Initial Applications (Early 2010s)**: In the early 2010s, deep learning started making its way into advertising and recommendation systems. Researchers experimented with deep neural networks for CTR prediction, and early successes encouraged further exploration.

4. **Scaling and Architectural Advancements (Mid-2010s)**: As deep learning gained traction, companies like Google and Facebook invested heavily in developing scalable deep learning frameworks and architectures. This led to the creation of deep recommender systems that could handle large-scale datasets efficiently.

5. **Industry Adoption (Late 2010s)**: By the late 2010s, deep learning-powered recommendation systems had become mainstream in the advertising industry. Companies realized the potential of deep learning in improving ad targeting and personalization. Many commercial platforms started adopting deep learning techniques for ad placement.

6. **Multi-Modal and Advanced Models (2020s)**: In the 2020s, the focus shifted to multi-modal recommendation systems that could incorporate various data types, including images, text, and user behavior. Advanced models like transformers, attention-based networks, and reinforcement learning gained prominence for recommendation tasks.

7. **Privacy and Ethical Considerations (Ongoing)**: The ongoing development cycle is marked by a growing emphasis on privacy and ethical considerations. Advertisers are working on AI models that respect user privacy and adhere to ethical guidelines.

8. **Future Directions (Ongoing)**: The future of deep learning-driven advertising and recommendation systems is likely to involve continued advancements in model architectures, more efficient training techniques, and a deeper integration of AI into ad tech. Exploring AI explainability, fairness, and interpretability will also be critical.

Overall, the development cycle of deep learning in advertising and recommendation technologies has evolved from early experimentation to widespread industry adoption, with a focus on improving model architectures, scalability, and ethical considerations. The future holds promise for further advancements and innovations in this field.




## LR and MLR

1. **特征工程的重点**: 算法工程师在特征工程方面投入了大量精力。他们专注于构建工具，尝试特征离散化方法，调整模型参数，评估模型性能。这个过程涉及构建特征以增强模型的预测能力。

2. **工程解决方案的挑战**: 尽管特征工程起到了关键作用，但它也存在局限性。通过工程解决方案提高模型准确性存在上限，因此需要更复杂的技术。

3. **非线性的出现**: 转向探索非线性方法。在这个转变中出现了两种主要方法：

   - **集成模型**: 一种方法是使用集成模型，如GBDT（梯度提升决策树）+ LR（逻辑回归）组合。在这个范式中，使用GBDT等非线性模型进行特征工程，然后将它们的输出馈入LR等线性模型。

   - **端到端非线性模型**: 另一种方法涉及直接以端到端方式训练大规模非线性模型。这包括FM（分解机）和您的团队开发的MLR（多层回归）模型。

4. **召回通道的挑战**: 在行业中，许多团队大量依赖召回通道，并将附加模型添加到其广告系统中。然而，这导致了一些挑战，如召回通道之间的召回集重叠较大，以及有限配额的低效分配。

5. **转向统一建模**: 您的团队采取了不同的方法。您不仅仅依赖于召回通道，还主张采用统一建模方法。这涉及通过使用基于模型的召回算法覆盖主流流量来加强通道之间的沟通和协作。

总之，广告和推荐系统中的LR时代以特征工程为重点开始，但后来演变为探索非线性技术。行业面临与召回通道和配额相关的挑战，因此出现了创新方法，如统一建模，以提高效率和效果。



1. **Feature Engineering Emphasis**: Algorithm engineers invested significant effort in feature engineering. They focused on constructing tools, experimenting with feature discretization methods, tuning model parameters, and evaluating model performance. This process involved crafting features to enhance the predictive power of the models.

2. **Challenges of Engineering Solutions**: While feature engineering played a crucial role, it had its limitations. Improving model accuracy through engineering solutions had a ceiling, and it became clear that more sophisticated techniques were needed.

3. **Emergence of Non-Linearity**: A shift occurred towards exploring non-linear methods. Two major approaches emerged within this shift:
   
   - **Ensemble Models**: One approach was to use ensemble models, such as the GBDT (Gradient Boosting Decision Trees) + LR combination. In this paradigm, non-linear models like GBDT were used for feature engineering, and their outputs were then fed into a linear model like LR.
   
   - **End-to-End Non-Linear Models**: The other approach involved training large-scale non-linear models directly in an end-to-end fashion. This included models like FM (Factorization Machines) and the MLR (Multi-Layer Regression) model developed by your team.

4. **Recall Channel Challenges**: In the industry, many teams heavily relied on recall channels and appended models to their advertising systems. However, this led to challenges such as significant overlap between recall sets across channels and the inefficient allocation of limited quotas.

5. **Shift towards Unified Modeling**: Your team's approach was different. Instead of relying solely on recall channels, you advocated for a unified modeling approach. This involved strengthening communication and collaboration between channels by using model-based recall algorithms to cover mainstream traffic.

In summary, the LR era in advertising and recommendation systems began with a strong emphasis on feature engineering but later evolved to explore non-linear techniques. The industry faced challenges related to recall channels and quotas, prompting innovative approaches like unified modeling to improve efficiency and effectiveness.



## 粗排 


## 算法、计算和系统架构的协同设计：

在广告和推荐系统不断发展的领域，越来越强调算法、计算能力和系统架构的协同设计。这种方法旨在一起优化这些元素，而不是单独考虑它们。



## Co-design of Algorithm, Compute, and System Architecture:

In the evolving landscape of advertising and recommendation systems, there's a growing emphasis on co-designing algorithms, computing power, and system architecture. This approach aims to optimize these elements together rather than considering them separately.


## 统一建模方法：

传统上，行业采用召回、粗排、精排和重排机制。然而，一种维持候选选择规则一致性的统一建模方法正在变得重要。这种方法被称为“平滑竞标”。



##  Unified Modeling Approach:


The industry has traditionally employed recall, rough ranking, fine ranking, and re-ranking mechanisms. However, a unified modeling approach that maintains consistency in candidate selection rules across all modules is gaining importance. This approach is referred to as "smooth bidding."


## 顺序物品建模（SIM）：

SIM被推荐作为建模用户长期行为序列的解决方案。它涉及用户行为数据的结构化存储和高效检索。SIM与DIN和DIEN等模型的一个关键区别是，SIM对原始输入进行交叉引用，而不是在嵌入后进行。这减少了序列大小，使其适用于建模。



## Sequential Item Modeling (SIM):


SIM is recommended as a solution for modeling long user behavior sequences. It involves the structured storage and efficient retrieval of user behavior data. One key difference between SIM and models like DIN and DIEN is that SIM performs cross-referencing on raw input rather than after embedding. This reduces the sequence size to a more manageable scale, making it suitable for modeling.

## 个性化和联邦学习：

最终目标是为每个用户创建高度个性化的模型。一种方法是联邦学习，其中每个用户都有独立的模型，通过联邦学习技术实现本地和全局信息共享。这种方法旨在平衡个性化和泛化。



## Personalization and Federated Learning:

The ultimate goal is to create highly personalized models for each user. One approach is federated learning, where each user has an independent model, and local and global information sharing occurs through federated learning techniques. This approach aims to balance personalization and generalization.


## 物理与代数先验建模：

建模有两种主要方法：物理先验建模（例如DIN、DIEN、MIMN、SIM），它在数学上解释“用户兴趣”的概念，以及代数先验建模（例如PNN、DeepFM、DCN），它捕捉特征之间的“代数关系”。



## Physical vs. Algebraic Prior Modeling:

There are two main approaches to modeling: physical prior modeling (e.g., DIN, DIEN, MIMN, SIM), which mathematically interprets the concept of "user interest," and algebraic prior modeling (e.g., PNN, DeepFM, DCN), which captures "algebraic relationships" between features.


## 混合模型：

像DeepFM这样的模型通过在嵌入层捕捉特征关系来结合物理和代数建模。然而，通过改变行动点，还有潜力获得更多收益。

您的见解突出了推荐系统建模的动态性，以及协同优化算法、计算和系统架构以获得更好结果的重要性。讨论还强调了建模的多种方法以及在用户特定模型中平衡个性化和泛化的需求。



## Hybrid Models:

Models like DeepFM combine physical and algebraic modeling by capturing feature relationships in the embedding layer. However, there's potential for more gains by shifting the point of action.

Your insights highlight the dynamic nature of recommendation systems and the importance of optimizing algorithm, compute, and system architecture in tandem to achieve better results. The discussion also emphasizes the diverse approaches to modeling and the need to balance personalization and generalization in user-specific models.


## 嵌入的局限性：

1. 嵌入引入了低效性，因为它们需要学习过程将ID转换为向量。这些向量通常是数据驱动的，缺乏内在含义。
2. 嵌入可能不具有内在的泛化属性，不同的随机初始化可能导致不相关的嵌入。
3. 原始的稀疏ID携带确定性信息，但嵌入引入了随机性。



## Limitations of Embeddings:

1. Embeddings introduce inefficiency as they require a learning process to translate IDs into vectors. These vectors are often data-driven and lack inherent meaning.
2. Embeddings may not exhibit inherent generalization properties, and different random initializations can result in unrelated embeddings.
3. Original sparse IDs carry deterministic information, but embeddings introduce randomness.



## 输入端建模：

1. 传统的CTR模型将物品ID、商店ID和类别ID视为独立特征，期望模型学习它们之间的关系。然而，这种方法在有效捕捉这些关系方面存在局限。
2. 明确告知模型ID之间的关系可以显著改善输入阶段可用信息。
3. 改进输入端建模的两种路径探索：
   - 终极路径：利用大型异构图来描述ID之间的关系。图保存在内存中，用于补充样本级的关系信息。
   - 启发式路径：基于图的预选择定义白盒结构，例如物品之间的一阶关系、涉及类别的二阶关系和涉及用户和商店的高阶关系。



## Input-side Modeling:

1. Traditional CTR models treated item IDs, shop IDs, and category IDs as independent features, expecting the model to learn relationships between them. However, this approach had limitations in capturing these relationships effectively.
2. Explicitly informing the model about relationships between IDs can significantly improve the information available at the input stage.
3. Two paths for improving input-side modeling were explored: 
   - Ultimate Path: Utilizing a large heterogeneous graph to depict relationships between IDs. The graph is kept in memory and is used to supplement sample-wise relationship information.
   - Heuristic Path: Defining white-box structures based on the graph's pre-selection, such as first-order relationships between items, second-order relationships involving categories, and higher-order relationships involving users and shops.


## 为什么在输入时使用笛卡尔积交互有效：

1. 在嵌入之前采用笛卡尔积的概念，引入了额外的自由度，对建模有效。与DeepFM等基于嵌入的模型处理嵌入后的交互不同，所提出的参数化方法旨在保留在输入阶段引入的自由度。



## Why Cartesian Product Interactions at Input Work:

1. The concept of taking the Cartesian product of input IDs before embeddings introduces additional degrees of freedom, which can be effective for modeling.
2. While embedding-based models like DeepFM handle interactions post-embedding, the proposed parameterized approach aims to preserve the freedom introduced at the input stage.



## 细粒度建模：

1. 最初，模型将来自不同小场景的样本合并到一个培训集中。然而，人们认识到用户兴趣和场景分布在不同情境下会有所变化。
2. 面临的挑战是允许模型共享一些信息，同时使它们能够表达每个情境的独特特征。


## Fine-grained Modeling:

1. Initially, models combined samples from various small scenarios into a single training set. However, it was recognized that user interests and scenario distributions vary across different scenes.
2. The challenge was to allow models to share some information while enabling them to express unique characteristics for each scenario.


## 适应自动竞标和延迟建模：

1. 广告主越来越多地采用多目标自动竞标策略。
2. 在KDD 2021年提出的延迟建模方法解决了具有延迟约束的建模问题。

您的见解提供了推荐系统建模中挑战和解决方案的综合概述，强调了需要解决嵌入的局限性、优化输入端建模，并实现各种情境的细粒度建模的重要性。


## Adaptation to Auto-bidding and Deferred Modeling:

1. Advertisers have increasingly adopted multi-objective auto-bidding strategies.
2. Deferred modeling, presented at KDD 2021, addresses modeling with latency constraints.

Your insights provide a comprehensive overview of the challenges and solutions in recommendation system modeling, highlighting the importance of addressing limitations in embeddings, optimizing input-side modeling, and achieving fine-grained modeling for various scenarios.


