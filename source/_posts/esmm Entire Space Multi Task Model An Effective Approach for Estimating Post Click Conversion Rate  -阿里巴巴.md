---
title: esmm Entire Space Multi Task Model An Effective Approach for Estimating Post Click Conversion Rate  -阿里巴巴
date: 2023-09-16 12:12:00
categories:
  - 排序模型
tags:
  - 预估模型 
  - esmm 
  - post-click conversion rate
  - multi-task learning
  - sample selection bias
  - data sparsity
  - entire-space modeling 
  - Recommender systems
description: 通过充分利用用户行为的顺序模式，即印象→点击→转化，以全新的视角对CVR进行建模. By effectively leveraging the sequential patterns of user behavior, specifically impressions → clicks → conversions, we approach modeling CVR from a completely new perspective.  
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_130110.png
---

## 摘要

准确估计后点击转化率（CVR）对于工业应用中的排名系统非常关键，例如推荐和广告。
Accurately estimating Click-Through Rate (CVR) is crucial for industrial ranking systems such as recommendation and advertising.

传统的CVR建模应用流行的深度学习方法，达到了最先进的性能。

Traditional CVR modeling leverages popular deep learning approaches and has achieved state-of-the-art performance.

然而，在实践中，CVR建模面临一些特定任务的问题，使其具有挑战性。

However, in practice, CVR modeling faces challenges specific to certain tasks. For instance, traditional CVR models are trained using samples with clicks, but during inference, they use samples with impressions across the entire space, leading to the problem of sample selection bias. Additionally, extreme data sparsity poses significant challenges for model fitting.

例如，传统的CVR模型是使用点击的样本进行训练的，而在整个空间上进行推断时使用所有印象的样本。这会导致样本选择偏差的问题。此外，还存在极端的数据稀疏问题，使模型拟合变得相当困难。

在本文中，我们通过充分利用用户行为的顺序模式，即印象→点击→转化，以全新的视角对CVR进行建模。

In this paper, we take a fresh perspective on CVR by effectively utilizing the sequential patterns of user behavior, specifically impressions → clicks → conversions.

提出的整体空间多任务模型（ESMM）可以通过以下两种方式同时解决这两个问题：

i）直接在整个空间上建模CVR，

ii）采用特征表示迁移学习策略。

The proposed Entire Space Multi-Task Model (ESMM) addresses both of these issues simultaneously in two ways:

i) Directly modeling CVR across the entire space,

ii) Employing a feature representation transfer learning strategy.

我们通过从淘宝推荐系统的流量日志中收集的数据集进行实验，结果表明ESMM明显优于竞争方法。

Through experiments using a dataset collected from the traffic logs of Taobao's recommendation system, the results demonstrate that ESMM outperforms competing methods significantly.

我们还发布了这个数据集的采样版本，以促进未来的研究。据我们所知，这是第一个包含点击和转化标签的样本具有顺序依赖性的CVR建模的公开数据集。


We have also released a sampled version of this dataset to promote future research. To our knowledge, this is the first publicly available dataset for CVR modeling with samples containing sequential dependencies and both click and conversion labels.

## 提出的方法

### 符号表示

我们假设观察到的数据集为S = {(xi, yi → zi)}|N i=1，其中样本(x, y → z)来自分布D，其定义域为X × Y × Z，其中X表示特征空间，Y和Z表示标签空间，N表示总的印象数量。x表示观察到的印象的特征向量，通常是一个高维稀疏向量，具有多个字段[8]，例如用户字段、物品字段等。y和z是二进制标签，其中y = 1或z = 1表示是否发生点击或转化事件。y → z表示点击和转化标签之间的顺序依赖关系，即在发生转化事件时总会有一个先前的点击事件。
We assume the observed dataset is denoted as S = {(xi, yi → zi)} | N i=1, where samples (x, y → z) are drawn from the distribution D, with the domain defined as X × Y × Z. Here, X represents the feature space, and Y and Z represent the label spaces, while N represents the total number of impressions. 

In this context, x represents the feature vector of the observed impression, typically a high-dimensional sparse vector with multiple fields [8], such as user fields, item fields, etc. y and z are binary labels, where y = 1 or z = 1 indicates the occurrence of a click or conversion event. The notation y → z signifies the sequential dependency between click and conversion labels, implying that a prior click event always precedes a conversion event.

后点击CVR建模是为了估计pCVR = p(z = 1|y = 1, x)的概率。与之相关的两个概率是：后观看点击率（CTR）pCTR = p(z = 1|x)和后观看点击与转化率（CTCVR）pCTCVR = p(y = 1, z = 1|x)。在给定印象x的情况下，这些概率遵循以下公式（1）：

Post-click CVR modeling aims to estimate the probability pCVR = p(z = 1|y = 1, x). Two related probabilities are post-view click-through rate (CTR) pCTR = p(z = 1|x) and post-view click-to-conversion rate (CTCVR) pCTCVR = p(y = 1, z = 1|x). In the context of a given impression x, these probabilities follow the formula (1):

This modeling approach takes into account the sequential relationships between viewing, clicking, and conversion events and is critical for various applications, including online advertising and recommendation systems.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_130354.png)

### CVR建模和挑战

最近，基于深度学习的方法已经被提出用于CVR建模，并取得了最先进的性能。其中大多数方法都采用了类似的嵌入和MLP网络架构，如[3]中所介绍的。图2的左侧部分展示了这种架构，出于简化的目的，我们将其称为BASE模型。
Recently, deep learning-based approaches have been proposed for CVR modeling and have achieved state-of-the-art performance. Most of these methods adopt similar embedding and MLP network architectures, as described in [3]. The left part of Figure 2 illustrates this architecture, which we refer to as the BASE model for simplicity.

简而言之，传统的CVR建模方法直接估计后点击转化率p(z = 1|y = 1, x)。它们使用点击印象的样本进行模型训练，即Sc = {(xj, zj)|yj = 1}|M j=1。


In essence, traditional CVR modeling methods directly estimate the post-click conversion rate p(z = 1|y = 1, x). They train their models using samples of click impressions, denoted as Sc = {(xj, zj)|yj = 1}|M j=1}.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_130110.png)

M是所有印象中的点击次数。显然，Sc是S的一个子集。注意，在Sc中，（被点击的）印象没有转化被视为负样本，而具有转化（也被点击）的印象被视为正样本。在实践中，CVR建模面临一些特定任务的问题，使其具有挑战性。
M represents the total number of clicks in all impressions. Clearly, Sc is a subset of S. It's important to note that in Sc, impressions (those that were clicked) without conversions are treated as negative samples, while impressions with conversions (and clicks) are treated as positive samples. In practice, CVR modeling faces specific challenges for certain tasks.

 **样本选择偏差（SSB）[12]。**
 
 事实上，传统的CVR建模通过引入辅助特征空间Xc来对p(z = 1|y = 1, x)进行近似估计，即p(z = 1|y = 1, x) ≈ q(z = 1|xc)。Xc表示与Sc相关的有限2的空间。对于Xc中的任何xc，都存在一对（x = xc，yx = 1），其中x ∈ X且yx是x的点击标签。这样，q(z = 1|xc)是使用Sc的点击样本在空间Xc上训练的。在推断阶段，对整个空间X的p(z = 1|y = 1, x)的预测被计算为q(z = 1|x)，假定对于任何一对（x，yx = 1），其中x ∈ X，x属于Xc。这一假设很可能被违反，因为Xc只是整个空间X的一小部分。它受到极少发生的点击事件的随机性的影响，该事件的概率在空间X的不同区域变化。此外，在实践中，由于观察不足，空间Xc可能与X相差很大。这将导致训练样本的分布漂移离真正的底层分布，并影响CVR建模的泛化性能。

**Sample Selection Bias (SSB):**

In fact, traditional CVR modeling approximates p(z = 1|y = 1, x) by introducing an auxiliary feature space Xc, which is denoted as p(z = 1|y = 1, x) ≈ q(z = 1|xc). Xc represents a finite space related to Sc. For any xc in Xc, there exists a pair (x = xc, yx = 1), where x ∈ X, and yx is the click label of x. Thus, q(z = 1|xc) is trained on click samples from Sc in the space Xc. During inference, the prediction for p(z = 1|y = 1, x) over the entire space X is computed as q(z = 1|x), assuming for any pair (x, yx = 1) where x ∈ X, x belongs to Xc. This assumption is likely to be violated because Xc is only a small part of the entire space X. It is influenced by the randomness of rare click events, which have varying probabilities in different regions of space X. Moreover, in practice, due to limited observations, Xc may differ significantly from X. This leads to a distribution shift in training samples from the true underlying distribution and affects the generalization performance of CVR modeling.

 **数据稀疏性（DS）。**
 
**Data Sparsity (DS):**

Traditional methods use click samples from Sc to train CVR models. The rarity of click events results in extremely sparse training data for CVR modeling. Intuitively, this data is typically 1-3 orders of magnitude sparser than related CTR tasks, which are trained on the dataset S containing all impressions. Table 1 presents the statistical data for our experimental dataset, where the number of samples for the CVR task is only 4% of the CTR task.

 
 传统方法使用Sc的点击样本来训练CVR模型。点击事件的罕见发生导致CVR建模的训练数据极度稀疏。从直观上看，通常比相关的CTR任务要稀疏1-3个数量级，CTR任务是在包含所有印象的S数据集上训练的。表1显示了我们实验数据集的统计数据，其中CVR任务的样本数仅为CTR任务的4%。


**Entire Space Multi-Task Model (ESMM):**

The proposed ESMM, as shown in Figure 2, effectively leverages the sequential patterns of user behavior. Borrowing ideas from multi-task learning [9], ESMM introduces two auxiliary tasks: CTR and CTCVR, simultaneously addressing the problems mentioned earlier in CVR modeling.

In essence, ESMM simultaneously outputs pCTR, pCVR, and pCTCVR relevant to a given impression. It primarily consists of two subnetworks: the CVR network shown on the left in Figure 2 and the CTR network on the right. Both CVR and CTR networks adopt the same structure as the BASE model. CTCVR multiplies the outputs of the CVR and CTR networks to generate its output. ESMM introduces several key features that have a significant impact on CVR modeling and distinguish it from traditional methods.

**整个空间多任务模型（ESMM）**

所提出的ESMM如图2所示，充分利用了用户行为的顺序模式。借鉴了多任务学习的思想[9]，ESMM引入了CTR和CTCVR两个辅助任务，并同时消除了CVR建模中前面提到的问题。
总体上，ESMM同时输出了与给定印象相关的pCTR、pCVR以及pCTCVR。它主要包括两个子网络：图2左侧显示的CVR网络和右侧的CTR网络。CVR和CTR网络都采用了与BASE模型相同的结构。CTCVR将CVR和CTR网络的输出相乘作为输出。ESMM中有一些亮点，对CVR建模产生了显著影响，并使ESMM与传统方法有所区别。
全空间建模。公式（1）给了我们一些提示，可以转化成公式（2）。

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_130644.png)


**Modeling the Entire Space:**

Formula (1) suggests that it can be transformed into Formula (2).

[Formula (2) description and details are expected here.]

## **实验**

**实验设置**

**数据集**。

在我们的调查中，没有在CVR建模领域找到具有点击和转化顺序标签的公共数据集。为了评估所提出的方法，我们收集了淘宝推荐系统的流量日志，并发布了整个数据集的1%随机抽样版本，其大小仍然达到38GB（未经压缩）。在本文的其余部分，我们将发布的数据集称为公共数据集，将整个数据集称为产品数据集。表1总结了这两个数据集的统计信息。详细描述可以在公共数据集的网站上找到。
**Datasets:**

In our investigation, we couldn't find any publicly available dataset in the field of CVR modeling with click and conversion sequence labels. To evaluate the proposed methods, we collected traffic logs from the Taobao recommendation system and released a 1% random sample of the entire dataset, which still amounts to 38GB (uncompressed). Throughout the rest of this paper, we will refer to the released dataset as the "public dataset" and the full dataset as the "product dataset." Table 1 summarizes the statistical information for these two datasets. Detailed descriptions can be found on the public dataset's website.

**竞争对手**。

我们在CVR建模上进行了几种竞争方法的实验。 

(1) BASE是在第2.2节中介绍的基线模型。 

(2) AMAN [6] 应用了负采样策略，并报告了在采样率搜索中获得的最佳结果 {10％，20％，50％，100％}。

(3) OVERSAMPLING [11] 复制正样本以减少稀疏数据训练的难度，采样率在 {2, 3, 5, 10} 中搜索。 

(4) UNBIAS遵循[10]，通过拒绝抽样从观察中拟合真正的底层分布。 pCTR被视为拒绝概率。 

(5) DIVISION使用单独训练的CTR和CTCVR网络估计pCTR和pCTCVR，并通过公式(2)计算pCVR。 

(6) ESMM-NS是ESMM的轻量版，不共享嵌入参数。前四种方法是基于最先进的深度网络直接建模CVR的不同变种。DIVISION、ESMM-NS和ESMM都采用了相同的思想，模型CVR覆盖整个空间，涉及CVR、CTR和CTCVR三个网络。

**Competitors:**

We conducted experiments on several competing methods in CVR modeling.

1. **BASE:** This is the baseline model introduced in Section 2.2.

2. **AMAN [6]:** AMAN applies negative sampling and reports the best results obtained in the sampling rate search {10%, 20%, 50%, 100%}.

3. **OVERSAMPLING [11]:** This method replicates positive samples to reduce the training difficulty of sparse data, with sampling rates in {2, 3, 5, 10}.

4. **UNBIAS [10]:** Following the UNBIAS approach, it fits the true underlying distribution by rejecting sampling from observations. pCTR is considered as the rejection probability.

5. **DIVISION:** DIVISION uses separately trained CTR and CTCVR networks to estimate pCTR and pCTCVR and computes pCVR using Formula (2).

6. **ESMM-NS:** This is a lightweight version of ESMM that doesn't share embedding parameters. The first four methods are different variants of state-of-the-art deep networks directly modeling CVR. DIVISION, ESMM-NS, and ESMM all follow the same idea of modeling CVR over the entire space involving three networks: CVR, CTR, and CTCVR.

ESMM-NS和ESMM同时训练这三个网络，并采用CVR网络的输出进行模型比较。为了公平起见，包括ESMM在内的所有竞争对手都与BASE模型共享相同的网络结构和超参数，其中 

i) 使用ReLU激活函数，

ii) 将嵌入向量的维度设置为18，

iii) 将MLP网络中每个层的维度设置为360×200×80×2，

iv) 使用带有参数β1 = 0.9、β2 = 0.999、ϵ = 10−8的adam求解器。

ESMM-NS and ESMM simultaneously train these three networks and compare models using the output of the CVR network. For fairness, all competitors, including ESMM, share the same network structure and hyperparameters as the BASE model, where

i) ReLU activation function is used,

ii) Embedding vector dimension is set to 18,

iii) MLP network dimensions for each layer are set to 360×200×80×2,

iv) The Adam solver is used with parameters β1 = 0.9, β2 = 0.999, and ϵ = 10^-8.

**指标**。

比较是针对两个不同的任务进行的：

(1) 传统CVR预测任务，该任务估计具有点击印象的数据集上的pCVR，

(2) CTCVR预测任务，该任务估计所有印象数据集上的pCTCVR。任务(2)旨在比较不同的CVR建模方法在整个输入空间上，这反映了与SSB问题相对应的模型性能。在CTCVR任务中，

所有模型都通过pCTR×pCVR计算pCTCVR，

其中：

i) 分别由每个模型估算pCVR，

ii) pCTR是由独立训练的相同CTR网络估算的（与BASE模型相同的结构和超参数）。这两个任务中，前1/2的数据按时间顺序分为训练集，其余数据为测试集。

面积下的受试者工作特征曲线（AUC）被采用作为性能指标。所有实验重复进行10次，并报告平均结果。
**Metrics:**

The comparison is conducted for two different tasks:

1. Traditional CVR prediction task, which estimates pCVR on the dataset with impressions that have clicks.
2. CTCVR prediction task, which estimates pCTCVR on the entire dataset with all impressions. Task (2) is designed to compare different CVR modeling methods across the entire input space, reflecting model performance associated with the SSB problem. In the CTCVR task, all models calculate pCTCVR by pCTR × pCVR, where:

   i) pCVR is estimated separately by each model,
   ii) pCTR is estimated by an independently trained identical CTR network (with the same structure and hyperparameters as the BASE model).

In both of these tasks, the first 1/2 of the data is split into a training set in chronological order, and the remaining data is used as the test set.

The Area Under the Receiver Operating Characteristic Curve (AUC) is adopted as the performance metric. All experiments are repeated 10 times, and the average results are reported.

![](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/2023-09-26_130923.png)



## 结论与未来工作

在本文中，我们提出了一种用于CVR建模任务的新方法ESMM。ESMM充分利用了用户行为的顺序模式。借助CTR和CTCVR两个辅助任务的帮助，ESMM巧妙地解决了实际中CVR建模面临的样本选择偏差和数据稀疏性等挑战。

对真实数据集的实验表明，所提出的ESMM方法具有卓越的性能。

这种方法可以轻松推广到具有顺序依赖性的情境中的用户行为预测。未来，我们打算设计全局优化模型，用于具有多个阶段行动的应用，例如请求 → 曝光 → 点击 → 转化。

In this paper, we introduced a new method called ESMM for CVR modeling tasks. ESMM makes full use of the sequential patterns in user behavior. With the help of two auxiliary tasks, CTR and CTCVR, ESMM cleverly addresses challenges such as sample selection bias and data sparsity commonly encountered in practical CVR modeling.

Experiments on real datasets demonstrate that the proposed ESMM method exhibits outstanding performance.

This approach can be readily extended to scenarios involving sequential dependencies in user behavior prediction. In the future, we plan to design a global optimization model for applications with multiple stages of actions, such as request → exposure → click → conversion.


## 原文link

https://arxiv.org/pdf/1804.07931.pdf





