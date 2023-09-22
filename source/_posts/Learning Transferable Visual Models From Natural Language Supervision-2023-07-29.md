---
title: Learning Transferable Visual Models From Natural Language Supervision
date: 2023-07-26 22:41:00
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

- 通过自监督训练，我们可以直接从原始文本数据中构建模型，这种方法在NLP领域取得了重大突破，而且不依赖于具体的下游任务。
- 大规模的未标记数据通常比手工标记的高质量数据更适用，因为它可以为模型提供更多信息。
- 尝试结合图像和文本数据，以获得更强大的特征表示。
- 在2017年，提出了"zero shot"迁移学习方法，但当时还没有Transformer模型和大规模高质量数据集。
- 最相似于CLIP的三种方法，包括VirTex、ICMLM和ConVIRT，都基于Transformer模型，并在数十万张图片上进行了训练。
- 基于大数据和大型模型的支持，我们提出了一种名为CLIP（对比语言图像预训练）的简单方法。
- 迁移学习的效果通常与模型的规模正相关。

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


- 核心思想是从自然语言中获得监督信号并进行学习。
- 该领域的术语多种多样，有时看起来似乎相互矛盾，而动机也有多种，包括无监督、自监督、弱监督和有监督的方法。
- 不再需要耗费大量精力去标注数据，监督信号可以是文本，而不仅仅是N选1的问题。这使得模型的输入输出变得更加灵活多样。
- 特征学习不再局限于视觉特征，而是变成了一种多模态特征。当与自然语言结合时，更容易实现零-shot迁移学习。



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

CLIP是一个训练模型，用于判断一组图像和文本配对中，哪些是真实的。为了达到这个目标，CLIP同时训练了图像编码器和文本编码器，让它们一起学习一个多模态嵌入空间。在这个空间中，CLIP试图让真实的图像和文本嵌入的余弦相似度最大化，同时最小化错误匹配的嵌入的余弦相似度。最终，CLIP使用对称交叉熵损失来优化这些相似度分数。

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

        clip预训练后，实际上它拥有两个编码器，一个用于图像，一个用于文本。对于任何给定的图像，通过图像编码器可以获得图像的特征表示。而对于文本输入，你可以提供你感兴趣的标签或关键词。这些标签和关键词会被处理成N个句子，然后通过文本编码器转化为N个文本特征。接着，这N个文本特征会与图像特征计算余弦相似度，然后通过一个额外的Softmax层得到一个概率分布。

        这种方法的关键思想是将图像和文本都编码成特征表示，然后通过度量它们之间的相似性来得出一个概率分布，用以表征它们之间的关系。

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


        在 27 个数据集上评估 CLIP 在零迁移条件下的性能，结果显示其中 16 个数据集表现有所提升。
        对于一般的物体分类任务，CLIP通常表现出色。
        然而，在一些专门的、复杂的或抽象的任务上，如卫星图像分类、淋巴结肿瘤检测、物体计数以及交通标志识别等，CLIP的性能相对较弱。
        对于更具挑战性的数据集，零样本学习（few-shot learning）作为性能衡量标准可能更加合适。
        CLIP的独特之处在于，它无需任何训练示例，就能够与最先进的模型媲美。

--- 

### 表示学习


        通过在模型中提取的表示上进行线性分类模型的拟合，并在各种数据集上测试其性能，以及在端到端微调的情况下测量模型的性能。尽管微调在大多数图像分类数据集上表现更好，但我们仍然选择使用基于线性分类器的评估方法。这是因为微调涉及到更广泛的超参数空间，这可能会引入不公平因素，而且计算成本较高。

![度量方法](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729225336.png)

### Robustness to Natural Distribution Shift

与在ImageNet上预训练的模型相比，CLIP对任务转换更鲁棒 

![Shift1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729225415.png)

## Comparison to Human Performance

- 相对于这个零样本的人来说，零样本的CLIP表现要更出色。
- 当人们只有一个示例时，他们的平均性能从54%上升到76%。这意味着人类在某种程度上了解他们对于未知图像的不确定性，并能够根据单个示例来调整他们的先验知识。
- 当人们有两个示例时，与只有一个示例相比，他们的性能没有明显提高，这可能需要更多系统性的学习和知识。
- CLIP的这些少样本评估没有充分地利用先验知识。改进CLIP算法的一个重要步骤是找到一种方法，能够适当地将先验知识整合到少样本学习中。
- 与CLIP中困难的类别相比，人们也会在困难的类别上遇到困难，而与CLIP中简单的类别相比，人们在简单的类别上也会表现得更轻松。


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

从动作识别、物体分类、地理定位到面部表情识别，这些测量能力可以应用于监控领域。CLIP的出现确实提高了视频监控的实用性，使得开发类似CLIP的视频模型变得更加容易，降低了构建这类应用所需的技能门槛。


### 未来工作

- 更早期识别模型的潜在有益下游用 途，使其他研究人员能够考虑应用。 
- 面临具有重大敏感性和大量社会利益相关者的任 务，可能需要决策者的干预。 
- 更好地描述模型中的偏差，提醒其他研究人员需要关注和干预的领域。 
- 创建测试套件能全面评估像CLIP这样的系统，可以在开发周期的早期更好地描述模型的功能。 
- 识别潜在的故障模式和需要进一步加强的领域



