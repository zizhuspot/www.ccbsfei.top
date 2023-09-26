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

1. **Feature Engineering Emphasis**: Algorithm engineers invested significant effort in feature engineering. They focused on constructing tools, experimenting with feature discretization methods, tuning model parameters, and evaluating model performance. This process involved crafting features to enhance the predictive power of the models.

2. **Challenges of Engineering Solutions**: While feature engineering played a crucial role, it had its limitations. Improving model accuracy through engineering solutions had a ceiling, and it became clear that more sophisticated techniques were needed.

3. **Emergence of Non-Linearity**: A shift occurred towards exploring non-linear methods. Two major approaches emerged within this shift:
   
   - **Ensemble Models**: One approach was to use ensemble models, such as the GBDT (Gradient Boosting Decision Trees) + LR combination. In this paradigm, non-linear models like GBDT were used for feature engineering, and their outputs were then fed into a linear model like LR.
   
   - **End-to-End Non-Linear Models**: The other approach involved training large-scale non-linear models directly in an end-to-end fashion. This included models like FM (Factorization Machines) and the MLR (Multi-Layer Regression) model developed by your team.

4. **Recall Channel Challenges**: In the industry, many teams heavily relied on recall channels and appended models to their advertising systems. However, this led to challenges such as significant overlap between recall sets across channels and the inefficient allocation of limited quotas.

5. **Shift towards Unified Modeling**: Your team's approach was different. Instead of relying solely on recall channels, you advocated for a unified modeling approach. This involved strengthening communication and collaboration between channels by using model-based recall algorithms to cover mainstream traffic.

In summary, the LR era in advertising and recommendation systems began with a strong emphasis on feature engineering but later evolved to explore non-linear techniques. The industry faced challenges related to recall channels and quotas, prompting innovative approaches like unified modeling to improve efficiency and effectiveness.



## 粗排 


## Co-design of Algorithm, Compute, and System Architecture:

In the evolving landscape of advertising and recommendation systems, there's a growing emphasis on co-designing algorithms, computing power, and system architecture. This approach aims to optimize these elements together rather than considering them separately.

##  Unified Modeling Approach:


The industry has traditionally employed recall, rough ranking, fine ranking, and re-ranking mechanisms. However, a unified modeling approach that maintains consistency in candidate selection rules across all modules is gaining importance. This approach is referred to as "smooth bidding."

## Sequential Item Modeling (SIM):


SIM is recommended as a solution for modeling long user behavior sequences. It involves the structured storage and efficient retrieval of user behavior data. One key difference between SIM and models like DIN and DIEN is that SIM performs cross-referencing on raw input rather than after embedding. This reduces the sequence size to a more manageable scale, making it suitable for modeling.

## Personalization and Federated Learning:

The ultimate goal is to create highly personalized models for each user. One approach is federated learning, where each user has an independent model, and local and global information sharing occurs through federated learning techniques. This approach aims to balance personalization and generalization.

## Physical vs. Algebraic Prior Modeling:

There are two main approaches to modeling: physical prior modeling (e.g., DIN, DIEN, MIMN, SIM), which mathematically interprets the concept of "user interest," and algebraic prior modeling (e.g., PNN, DeepFM, DCN), which captures "algebraic relationships" between features.

## Hybrid Models:

Models like DeepFM combine physical and algebraic modeling by capturing feature relationships in the embedding layer. However, there's potential for more gains by shifting the point of action.

Your insights highlight the dynamic nature of recommendation systems and the importance of optimizing algorithm, compute, and system architecture in tandem to achieve better results. The discussion also emphasizes the diverse approaches to modeling and the need to balance personalization and generalization in user-specific models.


## Limitations of Embeddings:

1. Embeddings introduce inefficiency as they require a learning process to translate IDs into vectors. These vectors are often data-driven and lack inherent meaning.
2. Embeddings may not exhibit inherent generalization properties, and different random initializations can result in unrelated embeddings.
3. Original sparse IDs carry deterministic information, but embeddings introduce randomness.

## Input-side Modeling:

1. Traditional CTR models treated item IDs, shop IDs, and category IDs as independent features, expecting the model to learn relationships between them. However, this approach had limitations in capturing these relationships effectively.
2. Explicitly informing the model about relationships between IDs can significantly improve the information available at the input stage.
3. Two paths for improving input-side modeling were explored: 
   - Ultimate Path: Utilizing a large heterogeneous graph to depict relationships between IDs. The graph is kept in memory and is used to supplement sample-wise relationship information.
   - Heuristic Path: Defining white-box structures based on the graph's pre-selection, such as first-order relationships between items, second-order relationships involving categories, and higher-order relationships involving users and shops.

## Why Cartesian Product Interactions at Input Work:

1. The concept of taking the Cartesian product of input IDs before embeddings introduces additional degrees of freedom, which can be effective for modeling.
2. While embedding-based models like DeepFM handle interactions post-embedding, the proposed parameterized approach aims to preserve the freedom introduced at the input stage.

## Fine-grained Modeling:

1. Initially, models combined samples from various small scenarios into a single training set. However, it was recognized that user interests and scenario distributions vary across different scenes.
2. The challenge was to allow models to share some information while enabling them to express unique characteristics for each scenario.

## Adaptation to Auto-bidding and Deferred Modeling:

1. Advertisers have increasingly adopted multi-objective auto-bidding strategies.
2. Deferred modeling, presented at KDD 2021, addresses modeling with latency constraints.

Your insights provide a comprehensive overview of the challenges and solutions in recommendation system modeling, highlighting the importance of addressing limitations in embeddings, optimizing input-side modeling, and achieving fine-grained modeling for various scenarios.

