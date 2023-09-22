---
title: DALL.E2 模型从0到1解密
date: 2023-08-03 8:30:00
categories:
  - AI绘画
tags:
  - 文生图
  - ai画图
  - clip模型
  - diffusion model
  - DALL.E2 
description: 这意味着利用Clip模型来理解和生成文本描述的视觉概念，然后借助Diffusion Models的技术来生成高质量的图像。这种结合可以实现文本描述到图像的高保真生成，通过Clip模型，文本描述被转化为视觉概念，并且通过Diffusion Models，这些视觉概念被映射成逼真的图像。
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/DALL.E2.jpg
---


## 背景

### CLIP模型：

构建了文本到图像的联合空间，打通文本图像壁垒，实现了文本描述到图像特征的映射，拓展了文本和图像的关联性。

### Diffusion models生成建模框架
利用一种guidance技术，以样本多样性为代价提高了样本保真度。

**本文思想：将二者结合，来解决 text-conditional image generation的问题**

## 方案

### DALL.E2:Prior+decode

### 模型架构


![模型架构](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729084052.png)


### CLIP的训练过程
输入文本图像对
正样本：文本和图像编码器提取文本和图像特征
负样本：文本特征与其他图像特征
对比学习，训练文本和图像编码器
最终：合并的多模态特征空间


### DALL·E 2的训练过程
#### Prior-第一阶段
通过clip模型得到文本特征，输入prior模型，得到图像特征。
CLIP图像编码器生成的图像特征辅助训练

#### Decode-第二阶段
扩散模型得到图片。

## 数学含义

### 生成式模型

![生成式模型](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729084327.png)

补充：
P (x|y)根据文本生成图像； 
P (x, zi|y)根据文本生成图像特征和图像

### prior模型


**作用：把文本特征变成图像特征**

训练过程：图片文本对（x,y）
Clip模型文本编码器得到文本y的特征Zt
Clip模型图像编码器得到图片特征Zi
令Zt进入Prior模型的预测图片特征Zi(t)
期望Zi(t)与Zi接近，更新prior模块。


**目标函数：**
损失：预测值与图片特征二范数

![目标函数](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729090458.png)

**自回归模型-AR prior**

输入文本特征+CLIP的图像特征去做生成，不断去预测。

**扩散模型-Diffusion prior (效果更好）**

训练的一个Transformer，用来处理序列信息。

**输入：** 有文本、CLIP文本的embedding，扩散模型的time step embedding，加入噪声后的CLIP图像embedding以及transformer自身的embedding 

**输出：** 没有加入噪声的CLIP图像embedding

### Decode

作用：图像特征传给DALL·E 2的解码器,生成图像

方法：
改进版GLIDE ：CLIP guidance + classifier-free guidance。
随机设10%的时间令CLIP的特征为0，并且训练的时候有50%的时间把文本直接丢弃。


 
级联生成：
    DALL·E 2先生成一个64×64的图像，再使用一个模型上采样生成一个256×256，继续上采样到1024×1024，所以最后DALL·E2生成出来的是一个1024×1024的高清大图。

## 图像处理

### 图像变化

给定一张图，图像具有相同的基本内容，但在其他方面，如形状和方向不同

![图像变化](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729090907.png)


### 插值（图像间）

两张图像，在图像特征间做内插，插入特征更偏向于某个图像时，所生成的图像也就更多地具有该图像的特征。

![插值（图像间）](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729090924.png)


### 插值（文本间）

利用CLIP优势：将图像和文本嵌入到同一空间中，指导图像生成

![插值（文本间）](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729090939.png)

## 探测 CLIP Latent Space

### 可视化CLIP latent空间结构

PCA重构
提取了少量源图像的CLIP图像嵌入
逐渐增加PCA维数重建
用解码器对重建的图像嵌入进行可视化

![可视化CLIP](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729091032.png)

## 文生图

### 先验重要性

从字幕生成CLIP图像嵌入的先验并不是严格必要的


### 人工评价

对unCLIP和GLIDE进行评估，比较照片真实感、标题相似度和样本多样性。

![人工评价](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729091121.png)

### 通过引导改进保真度和多样性的平衡

增加unCLIP和GLIDE的引导比例
     (提示符“桌子上有一个盛满红玫瑰的绿色花瓶)

## 局限性

无法将物体和属性结合起来。CLIP太关注物体间的相似性，不能识别上下左右等方位信息。

![局限性](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729091224.png)


当生成的图像里有文字时，文字是错误的（有可能是文本编码器使用了BPE编码）。

![局限性2](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729091300.png)

不能生成特别复杂场景的图像，细节缺失特别严重
时代广场上广告牌都是像素块

![局限性3](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729091321.png)


## 我的点评
自从chatgpt3大火之后，ai绘画也开始爆火。很多团队瞄准了这个方向去创业。这篇文章算是比较经典的文生图的范例，很值得细细研究阅读。

