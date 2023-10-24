---
title: mmoe-专家混合（MoE）结构引入多任务学习 -谷歌
date: 2023-09-22 10:08:00
categories:
  - 排序模型
tags:
  - 多任务模型
  - MMOE 
  - 预估模型 
  - Recommender systems
description: 提出了一种名为MMOE模型。将专家混合（MoE）结构引入多任务学习，使专家子模型在所有任务之间显式共享。 A model called MMOE (Mixture of Experts) is proposed. It introduces the Mixture of Experts (MoE) structure into multi-task learning, enabling expert sub-models to explicitly share information across all tasks. 
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/v2-9aa7c9716c09a7decec10abfe765e53e_1440w.png
---

## 总结 

The incorporation of the Mixture-of-Experts (MoE) structure into multi-task learning enabled the explicit modeling of task relationships and the learning of task-specific functionalities from shared representations.

Modulation and gating mechanisms enhance trainability in non-convex deep neural networks.

将“专家混合”（Mixture-of-Experts，MoE）结构引入多任务学习，实现了对任务关系的明确建模，并从共享表示中学习任务特定的功能。

调制和门控机制增强了非凸深度神经网络的可训性。


## 框架

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/v2-9aa7c9716c09a7decec10abfe765e53e_1440w.png)

Shared-bottom Multi-task Model：


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_135135.png)


### Impact of Task Relatedness:

- As task relatedness decreases, the effectiveness of sharing a common underlying model diminishes.

- Traditional multi-task models are sensitive to task relatedness.

- 随着任务相关性的降低，共享通用底层模型的有效性减弱。

- 传统的多任务模型对任务相关性敏感。

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_135259.png)



The Original Mixture-of-Experts (MoE) Model

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_135328.png)



## 模型

### MoE Layer:

- The MoE layer shares the same structure as the MoE model but takes the output of the previous layer as input and forwards it to a subsequent layer.

- For each input example, the model can selectively activate a subset of experts through the gating network, which is conditioned on the input.

- Multi-gate Mixture-of-Experts: Each task-specific gating mechanism independently determines the degree to which the results of different experts are utilized.


n represents the number of experts, and w is an n x d-dimensional matrix.

- MoE层与MoE模型具有相同的结构，但以前一层的输出作为输入，并将其传递给后续层。

- 对于每个输入示例，模型可以通过取决于输入的门控网络有选择地激活专家的子集。

- 多门控的专家混合：每个任务特定的门控机制独立确定不同专家结果的利用程度。

n代表专家的数量，w是一个n x d维度的矩阵。


![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_135452.png)




### Performance on Data with Different Task Correlations:

Multi-task models perform better on tasks with high correlations.

The MMoE model outperforms the OMoE model on tasks with varying degrees of correlation.

多任务模型在相关性高的任务上表现更好。

在具有不同相关性程度的任务中，MMoE模型优于OMoE模型。


## 实验

### Census-Income Data Validation:

Dataset: https://archive.ics.uci.edu/ml/databases/census-income/

Experiment Setup (1) - Absolute Pearson correlation: 0.1768:
实验设置（1） - 绝对皮尔逊相关性：0.1768：

Task 1: Predict whether the income exceeds $50K.
Task 2: Predict whether this person's marital status is "never married."

任务1：预测收入是否超过5万美元。
任务2：预测这个人的婚姻状况是否是"从未结婚"。

Experiment Setup (2) - Absolute Pearson correlation: 0.2373:

Task 1: Predict whether the education level is at least college.
Task 2: Predict whether this person's marital status is "never married."

实验设置（2） - 绝对皮尔逊相关性：0.2373：

任务1：预测教育水平是否至少大学程度。
任务2：预测这个人的婚姻状况是否是"从未结婚"。

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-25_135640.png)


## 改进：

Open Source Implementation Link: https://github.com/drawbridge/keras-mmoe

Application Scenarios:

1. Allocate experts based on features.
2. Allocate experts based on structure.
3. Utilize a combination of task-specific experts and common experts.
应用场景：

1. 基于特征分配专家。
2. 基于结构分配专家。
3. 利用特定任务专家和通用专家的组合。

## 结论

Overall, this paper serves as an extension of multi-task learning by introducing the mechanism of gate control networks to balance multiple tasks. This approach has practical implications in real-world business scenarios. Below, I will provide additional information about one of the dataset configurations described in the paper and analyze the comparative results of different models in the experiments.
总的来说，本文通过引入门控网络机制来平衡多个任务，作为多任务学习的延伸，具有实际的商业场景应用意义。以下，我将提供有关论文中描述的数据集配置之一的附加信息，并分析实验中不同模型的比较结果。

**Dataset Configuration and Experiment Results:**

**数据集配置和实验结果:**

In the paper, one of the dataset configurations involved the use of a specific dataset. The details of this configuration, such as the dataset source and task definitions, were provided. 

在本文中，其中一个数据集配置涉及使用特定数据集。提供了此配置的详细信息，例如数据集来源和任务定义。

The experimental results presented in the paper included a comparative analysis of various models. These models were likely evaluated based on performance metrics, and the results demonstrated how different models performed under various conditions. The paper may have discussed the advantages or limitations of each model and provided insights into which model performed best for the given tasks.


论文中呈现的实验结果包括各种模型的比较分析。这些模型可能是基于性能指标进行评估的，结果展示了不同模型在各种条件下的性能表现。论文可能讨论了每个模型的优点或局限性，并提供了哪个模型在给定任务下表现最佳的见解。

Overall, this approach of experimenting with different models on specific datasets and analyzing their performance contributes to a better understanding of the effectiveness of multi-task learning in real-world applications.


总的来说，在特定数据集上尝试不同模型并分析它们的性能，有助于更好地了解多任务学习在实际应用中的有效性。


