---
title: mmoe-Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts -谷歌
date: 2023-09-22 10:08:00
categories:
  - 排序模型
tags:
  - 多任务模型
  - MMOE 
  - 预估模型 
  - Recommender systems
description: 提出了一种名为MMOE模型。将专家混合（MoE）结构引入多任务学习，使专家子模型在所有任务之间显式共享。
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/v2-9aa7c9716c09a7decec10abfe765e53e_1440w.png
---

## 总结 

The incorporation of the Mixture-of-Experts (MoE) structure into multi-task learning enabled the explicit modeling of task relationships and the learning of task-specific functionalities from shared representations.

Modulation and gating mechanisms enhance trainability in non-convex deep neural networks.

## 框架

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/v2-9aa7c9716c09a7decec10abfe765e53e_1440w.png)

Shared-bottom Multi-task Model：


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_135135.png)


### Impact of Task Relatedness:

- As task relatedness decreases, the effectiveness of sharing a common underlying model diminishes.

- Traditional multi-task models are sensitive to task relatedness.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_135259.png)



The Original Mixture-of-Experts (MoE) Model

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_135328.png)



## 模型

### MoE Layer:

- The MoE layer shares the same structure as the MoE model but takes the output of the previous layer as input and forwards it to a subsequent layer.

- For each input example, the model can selectively activate a subset of experts through the gating network, which is conditioned on the input.

- Multi-gate Mixture-of-Experts: Each task-specific gating mechanism independently determines the degree to which the results of different experts are utilized.


n represents the number of experts, and w is an n x d-dimensional matrix.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_135452.png)




### Performance on Data with Different Task Correlations:

Multi-task models perform better on tasks with high correlations.

The MMoE model outperforms the OMoE model on tasks with varying degrees of correlation.


## 实验

### Census-Income Data Validation:

Dataset: https://archive.ics.uci.edu/ml/databases/census-income/

Experiment Setup (1) - Absolute Pearson correlation: 0.1768:

Task 1: Predict whether the income exceeds $50K.
Task 2: Predict whether this person's marital status is "never married."

Experiment Setup (2) - Absolute Pearson correlation: 0.2373:

Task 1: Predict whether the education level is at least college.
Task 2: Predict whether this person's marital status is "never married."

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_135640.png)


## 改进：

Open Source Implementation Link: https://github.com/drawbridge/keras-mmoe

Application Scenarios:

1. Allocate experts based on features.
2. Allocate experts based on structure.
3. Utilize a combination of task-specific experts and common experts.


## 结论

Overall, this paper serves as an extension of multi-task learning by introducing the mechanism of gate control networks to balance multiple tasks. This approach has practical implications in real-world business scenarios. Below, I will provide additional information about one of the dataset configurations described in the paper and analyze the comparative results of different models in the experiments.

**Dataset Configuration and Experiment Results:**

In the paper, one of the dataset configurations involved the use of a specific dataset. The details of this configuration, such as the dataset source and task definitions, were provided. 

The experimental results presented in the paper included a comparative analysis of various models. These models were likely evaluated based on performance metrics, and the results demonstrated how different models performed under various conditions. The paper may have discussed the advantages or limitations of each model and provided insights into which model performed best for the given tasks.

Overall, this approach of experimenting with different models on specific datasets and analyzing their performance contributes to a better understanding of the effectiveness of multi-task learning in real-world applications.




