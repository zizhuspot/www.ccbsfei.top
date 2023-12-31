---
title: 文本转视频（text-to-video, T2V），另一种方式解读原理，带你走进不一样的世界
date: 2023-08-06 16:30:00
categories:
  - AI绘画
tags:
  - 文生图
  - ai画图
  - clip模型
  - diffusion model
  - text-to-video
  - T2V
description: 文生图的一些技术方案的探讨与总结  Discussion and Summary of Some Technical Solutions for Generative Text Modeling.  
cover:  https://cdn.jsdelivr.net/gh/1oscar/image_house@main/T2V.jpg
---

## 概念

合成长视频的制作流程如下所示，往往与二次编辑平台组合在一起，比如腾讯智影。
合成短视频是另一个单独的方向（一句话合成几秒）。

The process for creating long-form videos often involves integration with secondary editing platforms, such as Tencent Video Intelligence. Creating short videos is a separate direction, typically involving the synthesis of a few seconds of content.


![概念](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729100644.png)

## 技术方案

### 检索（系统涉及很多方面，不只是算法）

    - 优点：可解释，生成结果稳定可控，适用于长视频。
    - 缺点：无法凭空生成视频素材，依赖现有内容。

- Advantages: Interpretable, generates stable and controllable results, suitable for long videos.
- Disadvantages: Cannot generate video content from scratch; it relies on existing material.



VidPress是百度研究院的智能视频合成平台，通过以下步骤创建视频内容：


VidPress is an intelligent video synthesis platform developed by Baidu Research Institute. It creates video content through the following steps:


用户输入文本或图文内容，如新闻链接。
使用NLP模型分析文字内容，提取摘要，并合成解说词的语音。
进行语义理解，识别故事中的关键信息，如主题、段落主旨、核心人物或机构等。
利用自有视频库和精准搜索能力，智能聚合最新适合呈现的内容，并抽取相关素材及其语义表征。
基于图像识别和视频内容理解等技术，自动剪切和精选视频素材。
进行音视频对齐剪辑，选取关键兴趣锚点，根据时间轴对齐算法，将媒体片段与兴趣锚点关联，以确保视频整体观感和用户兴趣的持续激发。
生成完整的视频并渲染。

1. Users input text or text-image content, such as news links.
2. NLP models are used to analyze the text, extract summaries, and synthesize voiceovers for narration.
3. Semantic understanding identifies key information in the story, such as the theme, main ideas of paragraphs, core characters, or institutions.
4. Utilizing an in-house video library and precise search capabilities, it intelligently aggregates the most relevant content for presentation, extracting related materials and their semantic representations.
5. Automatic cutting and selection of video materials based on technologies like image recognition and video content understanding.
6. Audio-video alignment editing involves selecting key interest points, associating media clips with interest points based on a timeline alignment algorithm to ensure the overall visual experience and sustained user engagement.
7. The platform generates the complete video and renders it.



易车与百度类似，不同之处在于它还包括以下功能：
Yiche, similar to Baidu, includes the following additional features:


进行音乐节奏分析，实现“踩点”视频。
管理图层，动态呈现图片效果。
这些平台都旨在帮助用户自动生成高质量的视频内容。

- Rhythm analysis for music synchronization, creating "synced" videos.
- Layer management for dynamic presentation of image effects.

These platforms aim to assist users in automatically generating high-quality video content.




- 端到端（特定算法：一句话生成内容，适合发论文）
    - 共同缺点：不可解释，生成结果不稳定
    - 文本生成图片（T2I, text-to-image）    
        - 优点：技术上比生成视频更容易实现
        - 缺点：生成的图片差异很大，无法组成连贯的视频
            - 连贯视频需要每秒至少24帧，相邻两帧的图片差异微弱
    - 文本生成视频（T2V, text-to-video）
        - 优点：可以凭空生成视频素材
        - 缺点：大部分模型只能用一句话生成几秒短视频。


- End-to-End (Specific Algorithm: One-sentence content generation, suitable for research papers):
  - Common drawbacks: Lack of interpretability, unstable generation results.

- Text-to-Image (T2I):
  - Advantages: Technically easier to implement than video generation.
  - Disadvantages: Generated images exhibit significant differences, making it challenging to create coherent videos. Coherent videos require at least 24 frames per second, with minimal differences between adjacent frames.

- Text-to-Video (T2V):
  - Advantages: Capable of generating video content from scratch.
  - Disadvantages: Most models can only generate a few seconds of short video using a single sentence.


#### 相关度打分

Wang et al., 2019: Write-A-Video Computational Video Montage from Themed Text

![相关度打分](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101049.png)

Wang et al., 2019: Write-A-Video Computational Video Montage from Themed Text

![相关度打分 2](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101136.png)


    1，文字通过模型识别，获得标签
    2，视频通过关键词搜索、模型识别，获得标签
    3，先找出与文字标签相同的视频（可能有多个）

    4，参考VSE++ (Faghri et al., 2018)，用MSCOCO Captions训练模型（图片用ResNet，文本用GRU，提取特征计算内积），分别计算文本、图片的embedding。

1. Text is processed through a model to obtain labels.
2. Videos are labeled through keyword searches and model recognition.
3. Videos with labels matching the text are identified (there may be multiple matching videos).

4. Reference to VSE++ (Faghri et al., 2018), a model is trained on MSCOCO Captions using ResNet for images and GRU for text. Features are extracted and inner products are calculated to compute text and image embeddings.

    对于多个候选视频，每个抽取几帧图片，用于计算embedding。
    最终选择与文本embedding距离最近的视频。

For multiple candidate videos, a few frames are extracted from each for calculating embeddings. The final selection is made based on the video whose embedding is closest to the text embedding.

CLIP: Radford et al., 2021. Learning Transferable Visual Models From Natural Language Supervision

4亿高质量的文本图像对（大力出奇迹）
通过Text Encoder和Image Encoder得到文本和图像的表征
文本：CBOW和Transformer
图像：5个ResNets模型和3个Vision Transformer模型
拉近同一文本图像对的表征相似度


Additionally, the CLIP model (Radford et al., 2021) is used, which involves 4 billion high-quality text-image pairs. It obtains representations for text and images through a Text Encoder (using CBOW and Transformer) and an Image Encoder (consisting of 5 ResNets and 3 Vision Transformer models). The goal is to bring the representations of the same text-image pair closer in similarity.


![相关度打分 3](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101220.png)

### 端到端生成图片


GAN使用对抗损失从左岸到右岸引导船，强制生成数据与真实数据分布接近。VAE考虑右岸数据的码头分布，从合适的码头出发返回右岸，以高斯分布为模型。Flow类似VAE但具有双向可逆性，而Diffusion借鉴VAE和Flow，需考虑路线中间点，形成双向的马尔可夫链过河方式。

GANs use adversarial loss to guide the boat from the left bank to the right bank, forcing generated data to closely resemble real data distributions. VAE considers the distribution of the docks on the right bank and departs from the appropriate dock, returning to the right bank with a Gaussian distribution as the model. Flow is similar to VAE but possesses bidirectional reversibility. Diffusion combines elements from VAE and Flow and considers intermediate points in the journey, forming a bidirectional Markov chain crossing approach.


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

No longer reliant on "text-video" pairs, thus reducing the need for extensive annotation data.

Text-to-Image (T2I) model (Ramesh et al., 2022):
- Prior network P: Takes input text embedding and BPE tokenization, and outputs image embedding.
- Decoder network D: Generates 64x64 pixel RGB images.
- Two super-resolution networks SRl and SRh for 256x256 and 768x768 resolutions.
- Temporal convolutions and attention layers are extended to the time dimension.
- Based on a U-Net diffusion model, Dt generates 16 frames.
- Frame interpolation network: Handles interpolation and extrapolation.
- Each module is trained separately, and only P requires input text for training, using text-image pairs.


![端到端生成图片4](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729101501.png)


**Imagen Video:** Ho et al., 2022: Imagen Video: High Definition Video Generation with Diffusion Models


生成高清1280×768（宽×高）视频，每秒24帧，共128帧（~5.3秒）的级联架构，包括7个子模型（基于U-Net，共116亿个参数）、1个T5文本编码器将文本prompt编码为text_embedding、1个基础视频扩散模型生成初始视频（16帧，24*48像素，每秒3帧）、3个SSR扩散模型提高视频分辨率、3个TSR扩散模型提高视频帧数。级联架构的优点在于每个模型都可以独立训练。

Generate a high-definition 1280x768 (width x height) video with 24 frames per second, comprising 128 frames (approximately 5.3 seconds). This is achieved using a cascading architecture consisting of 7 sub-models based on U-Net, totaling 116 billion parameters. The architecture also includes:

1. A T5 text encoder for encoding text prompts into text embeddings.
2. A base video diffusion model for generating the initial video (16 frames, 24x48 pixels, 3 frames per second).
3. Three SSR diffusion models to enhance video resolution.
4. Three TSR diffusion models to increase the number of video frames.

The advantage of a cascading architecture is that each model can be trained independently.



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

The system is capable of generating long-shot videos from lengthy text inputs.

Training Data:
- Abundant text-image pairs, including LAION-5B, FFT4B, etc.
- Limited text-video data, such as WebVid.

Encoder-Decoder: Utilizes the C-ViViT model.
- Extracts compressed representations (tokens) from videos.
- Supports videos of arbitrary lengths.
- Employs a bidirectional Transformer that simultaneously predicts multiple video tokens to maintain video coherence.

The model is trained on a dataset of 15 million text-video pairs at 8 frames per second (FPS), 50 million text-image pairs, and a mixed corpus of 4 billion data points called LAION-400M. This extensive training results in the Phenaki model with 1.8 billion parameters.


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

This article can be considered a review, providing a summary of the topic.

