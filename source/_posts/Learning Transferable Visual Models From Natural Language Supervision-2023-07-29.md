---
title: Learning Transferable Visual Models From Natural Language Supervision
date: 2023-07-29 22:41:00
categories:
  - AI绘画
tags:
  - 文生图
  - ai画图
  - clip
description: 提出来一个非常简单的方法叫clip，也就是 contrast language image pre training
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/Visual.jpg
---


## 摘要

- 有限制性的监督训练，限制了模型本身泛化性，无法识别新的类别
- 直接从自然语言这边，从这个文本里去得到一些监督信号，是一个看起来非常有前途的办法，因为它的这个监督信号涵盖的范围就太广了。只要是你语言描述过的物体，你就有可能让你这个视觉模型去识别到这个物体
- 选择一种自监督的训练方式去预训练一个大模型
- 模型的迁移学习效果，就是说对大多数任务来说它的效果都是非常好的


## 介绍和论文动机

- 直接从这种原始的文本数据里去预训练一个模型。已经在过去几年里在 NLP 领域取得了革命性的成功，是一种自监督的训练方式，它的目标函数是跟下游任务无关的
- 这种大规模的没有标注的数据，其实是要比那些手工标注的质量非常高的那种数据集反而是要更好使的
- 尝试把这个图片和文字结合起来，去学得一个更好的特征
- 17 年visual n-grams提出zero shot 迁移学习，但是那时候既没有transformer，也没有大规模高质量数据集
- 跟 CLIP 最相似的三种方法VirTex, ICMLM, and ConVIRT 是基于 transformer ，但在几十万张图片上训练。
- 在大数据加大模型双重的加持之下，提出来一个非常简单的方法叫clip，也就是 contrast language image pre training
- 迁移学习的效果是跟这个模型的大小基本上是呈正相关的

## 方法

- 自然语言监督
- 创建一介足够大的数据集
- 选择一种有效的预训练方法
- 选择和度量模型
- 训练

### 自然语言监督

- 核心是从自然语言中包含的监督中学习
- 用于描述该领域工作的术语是多种多样的，甚至看似矛盾，陈述的动机也是多种多样的，无监督- 的、自监督的、弱监督的和有监督的
- 不需要再去标注数据，监督信号是一个文本，不是N选1，模型的输入输出自由度大了很多
- 学习特征不再只是一个视觉特征了，而是一个多模态特征。当和语言联系在一起以后，很容易去- 做 zero shot 种迁移学习


### 创建一介足够大的数据集

- MS-COCO、Visual Genome 和这个 YFCC 100 million，前两个数据集标注质量非常高，但是数据量太少，只有大概 10 万个训练样本，跟modern standard 相比而言，小太多了。
- YFCC 100  million 数据集有一个亿多，但是标注质量很差，清洗以后只有 1500 万
- 创建了 WIT，也就是 web image text 数据集， 50万次搜索获取4 亿图片文本对，总字数与用于 训练GPT-2的WebText数据集差不多 
- 孕育了 clip 这篇工作，还孕育了DALL.E

### 选择一种有效的预训练方法
- 初始方法类似于VirTex，从头开始联合训练图 像CNN和文本transformer来预测图像的标题 ，非常困难
- 对比目标比与其等效的预测目标有更好的表示
- 只预测哪个文本作为一个整体与哪个图像配对，而不是预测准确的文本单词 


```
给定一批N (图像，文本)对，CLIP被训练以预测在一个批处理中实际发生的N × N可能(图像，文本)对中的哪一对是真实的。
为了做到这一点，CLIP通过联合训练图像编码器 和文本编码器来学习多模态嵌入空间，以最大化批处理中N 对真实的图像和文本嵌入的余弦相似性，同时最小化N2 − N对错误的嵌入余弦相似性。
最后优 化这些相似度分数的对称交叉熵损失。 
```

![选择一种有效的预训练方法](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729224724.png)


- 没有使用imagenet的初始化权重，从头开始训练
- 没有使用对比学习中最常用的非线性投影，只使用一个线性投影，在clip上没有差异
- 不用采样单句，因为clip训练数据只有单句
- 随机裁剪是唯一数据增强方式
- 温度参数t通过学习得到，避免成为超参

![超参1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729224759.png)

### 选择和调整模型

**视觉:**

Resnet-50:
rect-2 blur pooling 
global average pooling layer -> attention pooling mechanism. 
vision transformer:
adding an additional layer normalization to the combined patch and position embeddings 

**文本**

Transformer: 
63M-parameter 12- layer 512-wide model with 8 attention heads. BPE 49,152   ,the max sequence length was capped at 76.

### 训练

- 一共训练了 8 个模型，有 5 个这个Resnet，有 3 个这个 vision transformer
- Resnet 里面有 Resse 50 和 Resse 101、 Resnet 50* 4、 Resnet 50* 16 和 Resnet 50* 64
- vision transformer 有vit base 32， vit base 16 和 vit large 14，这里的 32、16、14 分别指的是这个 patch 的大小
- 对于所有的超参数，简单做了自动超参搜索，有grade search、random search 和手动调整。为了让调参快一点，在做超参搜索时，用的最小resnet 50 ，只训练了一个epoch，对于更大的模型，没再调参
- batch size 是 3 万多，非常的大
- 混合精度加速训练和节省显存

## 实验


### zero shot 迁移

####   动机
        零样本学习通常指的是在图像分类中对未见过的物体类别进行泛化的研究
        无监督学习领域的许多研究都集中在特征表示学习上，应用到下游任务的时候，还是需要有标签的数据去做微调
        Visual N-Grams (Li et al., 2017)首先使用通用预训练模型研究了零样本迁移到标准图像分类数据集
        研究零样本迁移作为任务学习的评估，是受 到NLP领域中展示任务学习的工作的启发(gpt1,gpt2)


####  用 clip 做 zero shot 迁移

        clip 预训练好之后，其实它就有两个编码器，
        一个是这个图像编码器，一个是这个文本编码器
        任意给定一张照片，通过图片编码器会得到一个图片特征
        文本的输入是你感兴趣的标签有哪些，将所有标签 prompt engineering 变成N个句子，通过文本编码器得到 N 个文本特征，
        N 个文本特征跟一个图像特征算 cosine  similarity，再通过一层 Softmax 得到一个概率分布。
####  和VISUAL N-GRAMS 对比

        imagenet 上这个的迁移效果：
        Visual n-grams在top1准确率11.5%
        clip在top1准确率76.2%，接近ResNet-50， top5准确率95%， 接近Inception-V4 
        这不是一个公平的对比
        clip 比Visual n-grams用的数据集大了十倍，模型多 100 倍计算量，相当于在训练上超过 1000 倍资源。
        模型的架构上，clip 用的是transformer，2017 年Visual n-grams这篇论文发表的时候， transformer 还没出现

####  prompt engineering 和 ensembling

        做微调或者直接做推理的时候用的一种方法，不是在预训练阶段，不需要非常多的计算资源
        为什么要做 Prompt engineering
        第一个问题是缺少上下文导致单词多义性（imagenet: construction cranes and cranes , Oxford-IIIT Pet dataset: boxer ），
        第二个问题是distribution gap
        使用 “A photo of a {label}.” 准确度提升1.3%。提供更多信息效果更好
        prompt ensembling
        多用一些提示的模板，做多次推理，然后把这个结果综合起来会有更好的结果。
        使用了80多个模板
        https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb

####  ZERO-SHOT CLIP 性能分析

        在 27 个数据集上衡量clip 做 zero shot迁移的效果，16/27效果提升
        对于普通的给物体进行分类的数据集来说， clip 一般都表现得比较好
        CLIP在一些专门的、复杂的或抽象的任务上非常弱，如卫星图像分类、淋巴结肿瘤检测、物体计数、交通标志识别等
        更难的数据集， few shot 会比 zero shot 的衡量更合理
        clip 不用任何训练样本，直接就和你最好的这个 BiT打成平手

--- 

### 表示学习

度量方法

- 从模型中提取的表示上拟合线性分类模型，并测量其在各种数据集上的性能
- 测量模型的端到端微调的性能
- 微调在大多数图像分类数据集上优于线性分类，仍然选择基于线性分类器的评估，微调有更大超参空间，有失公平，计算成本很高

![度量方法](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729225336.png)

### Robustness to Natural Distribution Shift

与在ImageNet上预训练的模型相比，CLIP对任务转换更鲁棒 

![Shift1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729225415.png)

## Comparison to Human Performance

- 我们可以看到 zero shot clip 比这个 zero shot 的人表现要好得多。
- 人类在one shot的情况下平均性能从54%上升到76%，这表明人类“知道他们不知道的东西”，并且能够根据单个示例更新他们对最不确定的图像的先验知识
- 人类在two shot的情况下与one shot相比性能没有显著提升，需要系统知识的学习
- CLIP的这些少样本评估没有有效地利用先验知识（找到一种将先验知识适当地整合到少样本学习中的方法是改进CLIP算法的重要一步）
- 对于 clip 难的类对于人来说也很难，对于 clip 简单的类对于人也简单


![Comparison to Human Performance 1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729225516.png)

## Data Overlap Analysis

```
预训练的一个问题是会无意中与下游值评估数据重叠
1）对于每个评估数据集，检测重叠数据，分成all、重叠(包含高阈值相似)和清洁(包含低阈值相似)三份数据进行比较
2 ）计算CLIP RN50x64在三种划分上的零样本精度
3 ）重叠的数量通常很小，还进行二项式显著性检验
结论：总体效果影响很少，几乎没有差异

```

![Data Overlap Analysis 1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729225611.png)

## 局限性

- clip 的效果很好，但也还是有差距，弥补zero shot的差距需要增加1000倍的计算量
- 细粒度分类上的性能较差 ，不在预训练数据集中的新任务效果接近随机
- zero shot对训练数据分布外(out of distribution )的数据泛化效果仍然很差 （如:mnist）
- 可以做 zero shot 的分类任务，但还是从给定的类别里做选择，无法生成，做成生成式的模型- （一个简单的想法是把对比学习的目标函数和生成式的目标函数合在一起）
- 较差的数据效率问题 ，需要大量的数据去投，需要极大的计算量
- 评估数据不全面，有偏
- 数据来自互联网，会学习到许多社会偏见
- 复杂的任务和视觉概念可能很难通 过文本指定 
- zero shot到few shot，有导致有违直觉的性能下降 

## 影响力 

### 偏见

算法决策、训练数据和类设计都可以促进和放大人工智能系统的使用造成的社会偏见和不平等 

一个模型在不同的子组上具有更高的准确性和更低的性能差异，这并不意味着它将 具有更低的影响差异 

![偏见1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729225741.png)

![偏见2](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729225759.png)


### Surveillance

动作识别、物体分类、地理定位到面部表情识别， 这些被测量的能力都可以用于监控。 
CLIP确实解锁了视频监控的可用性。可以创建类似CLIP类似视频模型，降低了构建此类应用程序的技能要求 。



### 未来工作

- 更早期识别模型的潜在有益下游用 途，使其他研究人员能够考虑应用。 
- 面临具有重大敏感性和大量社会利益相关者的任 务，可能需要决策者的干预。 
- 更好地描述模型中的偏差，提醒其他研究人员需要关注和干预的领域。 
- 创建测试套件能全面评估像CLIP这样的系统，可以在开发周期的早期更好地描述模型的功能。 
- 识别潜在的故障模式和需要进一步加强的领域



