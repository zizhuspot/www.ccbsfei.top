---
title: 文本转视频（text-to-video, T2V），另一种方式解读原理，带你走进不一样的世界
date: 2023-07-28 16:30:00
categories:
  - AI绘画
tags:
  - 文生图
  - ai画图
  - clip模型
  - diffusion model
  - text-to-video
  - T2V
description: 文生图的一些技术方案的探讨与总结
---

## 概念

合成长视频的制作流程如下所示，往往与二次编辑平台组合在一起，比如腾讯智影。
合成短视频是另一个单独的方向（一句话合成几秒）。

![概念](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729100644.png)

## 技术方案

### 检索（系统涉及很多方面，不只是算法）

    - 优点：可解释，生成结果稳定可控，适用于长视频。
    - 缺点：无法凭空生成视频素材，依赖现有内容。

- VidPress （百度研究院的智能视频合成平台）
    - （文本）输入图文内容，例如某条新闻事件的链接
    - （文本）用NLP模型分析文字内容，提取摘要，作为解说词
    - （语音）百度文字转音频服务(TTS)合成解说词语音
    - （标签）语义理解模型，平台识别故事中的关键信息，包括主题、段落主旨、核心人物或者机构- 等。
    - （搜索）通过自有视频库和精准搜索能力，智能化聚合最新最适合呈现的内容，以及从同一主题相- 关新闻里抽取更多的素材及其语义表征
    - （视频）基于图像识别、视频内容理解等技术，自动剪切和精选视频素材。
    - （标签）音视频对齐剪辑，用户真正关注的是故事中的关键点，也称兴趣锚点（anchor - point）。通过VidPress特有的时间轴对齐算法，选取出兴趣锚点。将媒体片段与兴趣锚点进行相- 关度打分，将优质媒体片段优先放入时间轴，保证视频的整体观感和用户兴趣的持续激发。
    - 当时间轴生成完毕之后，数据转交给渲染器，从而生成一个完成的视频。
- 易车（与百度类似，除了红字部分）另外还有：
    - （标签）音乐节奏分析，音视频对齐实现“踩点”视频。
    - （特效）图层管理，图片动态呈现


- 端到端（特定算法：一句话生成内容，适合发论文）
    - 共同缺点：不可解释，生成结果不稳定
    - 文本生成图片（T2I, text-to-image）    
        - 优点：技术上比生成视频更容易实现
        - 缺点：生成的图片差异很大，无法组成连贯的视频
            - 连贯视频需要每秒至少24帧，相邻两帧的图片差异微弱
    - 文本生成视频（T2V, text-to-video）
        - 优点：可以凭空生成视频素材
        - 缺点：大部分模型只能用一句话生成几秒短视频。

#### 相关度打分

Wang et al., 2019: Write-A-Video Computational Video Montage from Themed Text

![相关度打分](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101049.png)

Wang et al., 2019: Write-A-Video Computational Video Montage from Themed Text

![相关度打分 2](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101136.png)


    1，文字通过模型识别，获得标签
    2，视频通过关键词搜索、模型识别，获得标签
    3，先找出与文字标签相同的视频（可能有多个）

    4，参考VSE++ (Faghri et al., 2018)，用MSCOCO Captions训练模型（图片用ResNet，文本用GRU，提取特征计算内积），分别计算文本、图片的embedding。

    对于多个候选视频，每个抽取几帧图片，用于计算embedding。
    最终选择与文本embedding距离最近的视频。

CLIP: Radford et al., 2021. Learning Transferable Visual Models From Natural Language Supervision

4亿高质量的文本图像对（大力出奇迹）
通过Text Encoder和Image Encoder得到文本和图像的表征
文本：CBOW和Transformer
图像：5个ResNets模型和3个Vision Transformer模型
拉近同一文本图像对的表征相似度

![相关度打分 3](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101220.png)

### 端到端生成图片

    GAN的过河方式：
    从先验分布随机采样一个Z（在左岸随便找一个码头），直接通过对抗损失的方式强制引导船开到右岸，要求右岸下船的码头和真实数据点在分布层面上比较接近。
    VAE的过河方式：
    考虑右岸的数据到达河左岸会落在什么样的码头。如果知道大概落在哪些码头，我们直接从这些码头出发就可以顺利回到右岸了（右岸的样本到左岸是高斯分布）
    Flow的过河方式：
    类似VAE，先看从右到左。区别是到左岸的是一个固定位置，而且双向可逆。
    Diffusion的过河方式：
    借鉴VAE和Flow，区别是不只看从右到左的端点，也看路线中间的点。从左到右也要逐个经过这些点。（马尔科夫链）


![端到端生成图片1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101313.png)

![端到端生成图片2](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101326.png)


    模型 DPM: 2015
    模型 DDPM: 2020
    文本引导、图像引导 Diffusion Models Beat GANs on Image Synthesis
    爬取4亿数据，实现引导 CLIP: 2021
    ......
    DALL·E: 2021.1. dVAE+Transformer+CLIP
    应用 Dream by WOMBO: 2021.11. CLIP
    应用 Disco Diffusion: 2021.10 Latent Diffusion
    GLIDE: 2021.12. guided Diffusion
    DALL·E 2: 2022.4
    应用 Midjourney: 2022.7
    Imagen: 2022.5
    应用 Stable Diffusion: 2022.8

![端到端生成图片3](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101406.png)


**Make-A-Video:** Singer et al., 2022: Make-A-Video: Text-to-Video Generation without Text-Video Data
不再依赖“文本-视频”对，所以不需要大量的这种标注数据

    文本生成图片(T2I)模型 (Ramesh et al., 2022)
    先验的网络P：输入文本embedding、BPE分词，输出图片embedding
    decoder网络D：生成64*64像素RGB图片
    两个超分辨率网络SRl, SRh
    256*256，768*768
    时空卷积和注意力层，扩展到时间维度
    基于U-Net的扩散模型，Dt生成16帧
    帧插值网络：内插和外推
    各个模块分别训练，
    只有P需要输入文本，用文本-图片对训练

![端到端生成图片4](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101501.png)


**Imagen Video:** Ho et al., 2022: Imagen Video: High Definition Video Generation with Diffusion Models

生成高清1280×768（宽×高）视频，每秒24帧，共128帧（~5.3秒）

    级联架构，7个子模型（基于U-Net），共116亿个参数
    1 个T5文本编码器
    将文本prompt编码为text_embedding
    1 个基础视频扩散模型
    以文本为条件，生成初始视频
    16帧，24*48像素，每秒3帧
    3 个 SSR扩散模型
    提高视频的分辨率
    3 个 TSR扩散模型
    提高视频的帧数
    级联架构的优点，每个模型都可以独立训练

![级联架构1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101559.png)

![级联架构2](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101608.png)


**Phenaki:** Villegas et al., 2023. Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions

能够用长文本生成长镜头视频

    训练数据：大量文本-图像，少量文本-视频
    文本-图像数据多：LAION-5B, FFT4B等
    文本-视频数据少：WebVid等
    编码器-解码器：C-ViViT
    提取视频的压缩表征（token）
    支持任意长度的视频
    双向Transformer
    同时预测多个视频token
    保证视频的连贯性
    在1500万8FPS的文本-视频对，5000万个文本-图像对，以及4亿混合语料库LAION-400M上进行训练，最终Phenaki模型参数量为18亿。

![能够用长文本生成长镜头视频1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101707.png)

## 参考资料

### 检索
- AI自动生成国风视频海外引关注，百度文心大模型助力AIGC智能创作https://baijiahao.- baidu.com/s?id=1729537094778602568
- Wang et al., 2019: Write-A-Video Computational Video Montage from Themed Text 
- https://dl.acm.org/doi/10.1145/3355089.3356520 
- CLIP: Radford et al., 2021. Learning Transferable Visual Models From Natural - Language Supervision  https://arxiv.org/pdf/2103.00020.pdf 
- 如何评价OpenAI最新的工作CLIP：连接文本和图像，zero shot效果堪比ResNet50？ - 王思若- 的回答 - 知乎  https://www.zhihu.com/question/438649654/answer/2521781785

### 文本生成图像
- What are Diffusion Models?  Lil'Log.html  https://lilianweng.github.io/posts/- 2021-07-11-diffusion-models/ 
- DALL·E: Ramesh et al., 2021: Zero-Shot Text-to-Image Generation  https://arxiv.- org/abs/2102.12092 
- Dsico Diffusion: https://github.com/alembics/disco-diffusion 
- https://docs.google.com/document/d/- 1l8s7uS2dGqjztYSjPpzlmXLjl5PM3IGkRWI3IiCuK7g/ 
- Latent Diffusion: Rombach et al., 2021. High-Resolution Image Synthesis with - Latent Diffusion Models  https://arxiv.org/abs/2112.10752
- GLIDE: Nichol et al., 2021. GLIDE: Towards Photorealistic Image Generation and - Editing with Text-Guided Diffusion Models https://arxiv.org/abs/2112.10741 
- DALL·E 2: Ramesh et al., 2022: Hierarchical Text-ConditionalImage Generation - with CLIP Latents  https://arxiv.org/abs/2204.06125 
- 上线一个月成为准独角兽、上万人排队注册，AI Art是下一个NFT？ _ 全球行业mapping-36氪.- html  https://36kr.com/p/1936392560380553
- Imagen: Saharia et al., Photorealistic Text-to-Image Diffusion Models with - Deep Language Understanding  https://openreview.net/forum?id=08Yk-n5l2Al
- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- Latent Diffusion: Rombach et al., 2021. 
 
### 文本生成视频
- Make-A-Video: Singer et al., 2022: Make-A-Video: Text-to-Video Generation - without Text-Video Data  https://makeavideo.studio/Make-A-Video.pdf
- Imagen Video: Ho et al., 2022: Imagen Video: High Definition Video Generation - with Diffusion Models  https://arxiv.org/abs/2210.02303 
- Phenaki: Villegas et al., 2023. Phenaki: Variable Length Video Generation from - Open Domain Textual Descriptions https://arxiv.org/abs/2210.02399 



## 我的点评

本篇算是一个综述的文章吧，总结性质的。