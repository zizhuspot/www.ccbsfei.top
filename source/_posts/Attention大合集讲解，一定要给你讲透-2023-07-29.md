---
title: Attention大合集讲解，一定要给你讲透
date: 2023-07-29 23:00:00
categories:
  - 大模型
tags:
  - attention
  - nlp 
description: attention的大合集，包含所有attention知识
---

## attention分类

“Neural machine translation by jointly learning to align and translate” 论文首次提出 attention mechanism 

注意力机制核心目标也是从众多信息中选择出对当前任务目标更关键的信息



![attention分类](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729230812.png)


## Soft Attention模型:

对于特征的注意力是不一样的，如果都一样，那么注意力是不集中的。
类似于经过编解码翻译时候每个单词给的注意力大小不同，注意力大小对应着不同的源语句子单词的注意力分配概率分布

所谓Soft，意思是在求注意力分配概率分布的时候，对于输入的任意一个特征都给出个概率，是个概率分布. Soft Attention是所有的数据都会注意，都会计算出相应的注意力权值，不会设置筛选条件

## Hard Attention

会在生成注意力权重后筛选掉一部分不符合条件的注意力，让它的注意力权值为0，即可以理解为不再注意这些不符合条件的部分。一般用soft attention

## Attention:-需要用到目标

- 将输出与输入的各个特征的相似性作为权重（attention权重 归一化，得到直接可用的权重），加权融合归一化的权重与输入特征作为输出（attention输出）
- 区分输入的不同部分对输出的影响。相当于给每个输入加了权重。

## Self-Attention

- key=value=query.所有都来自一个输入文本。
- 将输入的每个特征作为query, 加权融合所有输入特征的信息，得到每个特征的增量语义向量。也可以理解为Target=Source这种特殊情况下的注意力计算机制
- 好处：寻找原文内部的关系; 在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来。对于时间排序的序列特征，不加self attention,需要不断的累计，self attention随机捕获，更容易捕获句子中长距离的相互依赖的特征。

## Multi-head self attention:

- 为了增强Attention的多样性,多个不同的self attention模块都获得了每个特征的增强语义向量，将每个特征的多个增强语义向量线下组合，获得最终的输出。Multi-head self attention 的每个self attention关注的语义场景不同。 相当于重复做多次单层attention
- Multi-head Self-Attention可以理解为考虑多种语义场景下目标字与文本中其它字的语义向量的不同融合方式
- 举例不同语义：在不同语义场景下对这句话可以有不同的理解：“南京市/长江大桥”，或“南京市长/江大桥”

![Multi-head self attention:](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729231101.png)

## 多层Attention

一般用于文本具有层次关系的模型，假设我们把一个document划分成多个句子，在第一层，我们分别对每个句子使用attention计算出一个句向量（也就是单层attention）；在第二层，我们对所有句向量再做attention计算出一个文档向量（也是一个单层attention），最后再用这个文档向量去做任务。

