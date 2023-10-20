---
title: Attention大合集讲解，一定要给你讲透
date: 2023-08-04 23:00:00
categories:
  - 大模型
tags:
  - attention
  - nlp 
description: attention的大合集，包含所有attention知识  A comprehensive collection of knowledge about "attention," including all aspects of attention.  
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/Attention.png
---


Attention mechanisms have been widely used in the field of machine learning, particularly in natural language processing and computer vision. Here's a big collection of attention-related knowledge:

1. **Introduction to Attention Mechanisms:**
   - Attention mechanisms are a fundamental concept in machine learning and neural networks that allow models to focus on specific parts of the input data.

2. **Types of Attention:**
   - **Self-Attention:** In self-attention mechanisms, the model can weigh the importance of different elements within the same input sequence. It's commonly used in Transformer models.
   - **Multi-Head Attention:** Multi-head attention combines several self-attention mechanisms to capture different aspects of the input data.

3. **Applications of Attention Mechanisms:**
   - **Transformer Models:** The Transformer architecture introduced the self-attention mechanism and revolutionized natural language processing tasks like machine translation and text generation.
   - **BERT (Bidirectional Encoder Representations from Transformers):** BERT is a pre-trained model that uses bidirectional self-attention to understand the context of words in a sentence.
   - **GPT (Generative Pre-trained Transformer):** GPT models use unidirectional self-attention for generating coherent text, making them suitable for text generation tasks.
   - **Vision Transformers (ViT):** Vision Transformers use attention mechanisms for image classification tasks.
   - **Object Detection:** Attention mechanisms have been applied to improve object detection in images.
   - **Machine Translation:** Attention has greatly improved the quality of machine translation by allowing models to focus on relevant parts of the source text when generating translations.

4. **Scaled Dot-Product Attention:**
   - This is a key component in self-attention mechanisms, where dot products of query and key vectors are scaled and used to calculate the attention scores.

5. **Positional Encoding:**
   - Positional encoding is added to the input embeddings to give the model information about the positions of elements in a sequence, compensating for the lack of order information in self-attention.

6. **Attention in RNNs and LSTMs:**
   - Attention mechanisms can also be integrated into recurrent neural networks and long short-term memory networks to improve their performance in tasks like sequence-to-sequence modeling.

7. **Attention in Reinforcement Learning:**
   - Attention mechanisms have found applications in reinforcement learning, where agents use attention to focus on important parts of the environment.

8. **Interpretable Attention:**
   - Researchers have worked on making attention mechanisms more interpretable and explainable, which is crucial for model transparency and trustworthiness.

9. **Sparse Attention:**
   - To reduce computational complexity, sparse attention mechanisms have been developed, limiting the number of elements a model can attend to in each step.

10. **Attention-Based Recommender Systems:**
    - Attention mechanisms have been applied in recommender systems to help the model focus on relevant user behaviors and item features.

11. **Cross-Modal Attention:**
    - In applications like image captioning, attention can be used to align information from different modalities, such as images and text.

12. **Attention in Speech Processing:**
    - Attention mechanisms are also used in automatic speech recognition and speech synthesis tasks.

13. **Attention in Time Series Forecasting:**
    - Time series forecasting models often employ attention mechanisms to capture dependencies between different time steps.

14. **Attention in Graph Neural Networks:**
    - Attention can be used in graph neural networks to weigh the importance of neighboring nodes in graph data.

15. **Limitations and Challenges:**
    - Attention mechanisms can be computationally expensive and may require substantial training data. Handling long sequences and ensuring model robustness are ongoing challenges.

This collection provides a broad overview of attention mechanisms and their applications in various domains within machine learning and artificial intelligence.


## attention分类

“Neural machine translation by jointly learning to align and translate” 论文首次提出 attention mechanism 

注意力机制核心目标也是从众多信息中选择出对当前任务目标更关键的信息  The core objective of attention mechanisms is to select the most crucial information from a vast pool of data for the current task or goal.  



![attention分类](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729230812.png)


## Soft Attention模型:

对于特征的注意力是不一样的，如果都一样，那么注意力是不集中的。
类似于经过编解码翻译时候每个单词给的注意力大小不同，注意力大小对应着不同的源语句子单词的注意力分配概率分布

所谓Soft，意思是在求注意力分配概率分布的时候，对于输入的任意一个特征都给出个概率，是个概率分布. Soft Attention是所有的数据都会注意，都会计算出相应的注意力权值，不会设置筛选条件


For different features, attention is not uniform. If it were uniform, attention would not be focused. Similar to when translating through encoding and decoding, the attention given to each word varies in magnitude, with attention sizes corresponding to probability distribution assignments for different source language words.

The term "Soft" implies that when calculating the probability distribution of attention allocation, a probability is assigned to any input feature. It forms a probability distribution. Soft Attention means that all the data will be attended to, and corresponding attention weights will be calculated for each, without setting any filtering criteria.


## Hard Attention

会在生成注意力权重后筛选掉一部分不符合条件的注意力，让它的注意力权值为0，即可以理解为不再注意这些不符合条件的部分。一般用soft attention

After generating attention weights, a portion of them may be filtered out, setting their attention weights to 0, effectively implying that those elements are no longer being attended to. This process is typically referred to as "soft" attention.

## Attention:-需要用到目标

- 将输出与输入的各个特征的相似性作为权重（attention权重 归一化，得到直接可用的权重），加权融合归一化的权重与输入特征作为输出（attention输出）
- 区分输入的不同部分对输出的影响。相当于给每个输入加了权重。

- Calculate the similarity between the output and each feature of the input to obtain weights (attention weights), normalize them to obtain directly usable weights, and then use these normalized weights to combine and normalize the input features as the output (attention output).

- This process distinguishes the impact of different parts of the input on the output, effectively assigning a weight to each input.


## Self-Attention

- key=value=query.所有都来自一个输入文本。
- 将输入的每个特征作为query, 加权融合所有输入特征的信息，得到每个特征的增量语义向量。也可以理解为Target=Source这种特殊情况下的注意力计算机制
- 好处：寻找原文内部的关系; 在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来。对于时间排序的序列特征，不加self attention,需要不断的累计，self attention随机捕获，更容易捕获句子中长距离的相互依赖的特征。


- Key, Value, and Query all originate from the same input text.
- Each feature of the input is used as a query, and the information from all input features is weighted and fused, resulting in an incremental semantic vector for each feature. This mechanism can also be understood as the case where Target equals Source in attention computation.

- Benefits: It helps uncover relationships within the original text, and during the calculation process, it directly links any two words in a sentence through a single computational step. For sequentially ordered sequence features, without self-attention, you would need constant accumulation. Self-attention, on the other hand, randomly captures dependencies, making it easier to capture long-range dependencies between features in a sentence.


## Multi-head self attention:

- 为了增强Attention的多样性,多个不同的self attention模块都获得了每个特征的增强语义向量，将每个特征的多个增强语义向量线下组合，获得最终的输出。Multi-head self attention 的每个self attention关注的语义场景不同。 相当于重复做多次单层attention
- Multi-head Self-Attention可以理解为考虑多种语义场景下目标字与文本中其它字的语义向量的不同融合方式
- 举例不同语义：在不同语义场景下对这句话可以有不同的理解：“南京市/长江大桥”，或“南京市长/江大桥”

- To enhance the diversity of attention, multiple different self-attention modules generate enhanced semantic vectors for each feature. These multiple enhanced semantic vectors for each feature are then combined to obtain the final output. Each self-attention module in Multi-head Self-Attention focuses on a different semantic context. Essentially, it's like performing single-layer attention multiple times.

- Multi-head Self-Attention can be understood as considering different fusion methods for the semantic vectors of the target word and other words in the text under various semantic contexts.

- For example, in different semantic contexts, the same sentence can be understood differently: "Nanjing City / Yangtze River Bridge," or "Nanjing City long / Jiang Bridge."


![Multi-head self attention:](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729231101.png)

## 多层Attention

层次化文本模型使用多层attention来处理文本数据，首先计算句子级别的句向量，然后通过文档级别的attention计算文档向量，最后用文档向量进行任务处理。


Hierarchical text models use multiple layers of attention to process text data. They begin by computing sentence-level embeddings, followed by document-level attention to calculate a document-level vector. Finally, this document-level vector is utilized for task-specific processing.


