---
title: InstructGPT 模型深度解密
date: 2023-07-27 11:30:00
categories:
  - 大模型
tags:
  - GPT1
  - GPT2
  - GPT3
  - InstructGPT
  - 强化学习
description: 使用PPO来微调SFT模型。输入一个prompt期望得到一个输出。给定一个prompt和response，生成奖励分数。除此之外，增加了KL散度降低奖励模型的过度优化。我们称这个模型为PPO。作者把预训练的梯度加入到PPO的梯度中，为了缓和模型在公开数据集中的性能损失。我们称这个模型为PPO-ptx。
cover: https://cdn.jsdelivr.net/gh/1oscar/image_house@main/InstructGPT.jpg
---

## 摘要&介绍

### ChatGPT 发展时间线

![ChatGPT发展时间线](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729092614.png)

增大语言模型并不一定保证其输出与人类意图一致。较大的语言模型可能生成不真实、有害或无效的答案。换句话说，模型可能与用户的预期不一致。在这项研究中，作者采取了一种微调方法，利用用户的反馈来改善模型性能。

Increasing the size of a language model does not necessarily guarantee that its output will align with human intent. Larger language models can generate unreal, harmful, or ineffective responses. In other words, the model may be inconsistent with user expectations. In this study, the authors employed a fine-tuning approach using user feedback to enhance model performance.

首先，他们使用一组经过人工标注的提示词来进行监督微调，然后收集了模型生成的有序数据集。接着，他们利用强化学习（RLHF）方法，从人类反馈中进一步微调经过监督学习的模型。这一经过微调的模型被称为"instruct GPT"。

First, they used a set of manually annotated prompt words for supervised fine-tuning and collected an ordered dataset of model-generated responses. Then, they further fine-tuned the model using Reinforcement Learning from Human Feedback (RLHF) methods based on human feedback. This fine-tuned model is referred to as "instruct GPT."

人类评估结果显示，相较于拥有175B参数的gpt-3模型，拥有1.3B参数的"instruct GPT"模型的输出更加优越。尽管"instruct GPT"模型偶尔会出现一些简单的错误，但这个研究结果表明，利用人类反馈进行微调是朝着使模型输出与人类意图一致的正确方向迈出的一步。

Human evaluation results show that the 1.3-billion-parameter "instruct GPT" model provides superior outputs compared to the gpt-3 model with 175 billion parameters. While "instruct GPT" occasionally makes minor errors, this research suggests that fine-tuning with human feedback is a step in the right direction to make model outputs align more closely with human intent.


### 人类在不同模型上的评估，
GPT（prompt）：是GPT3在prompt上做比较多的调整

SFT：第一个模型

PPO：当γ=0时，这个模型叫做PPO；

PPO-ptx：当γ不为0时，这个模型叫做PPO-ptx。

![人类在不同模型上的评估，](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729092703.png)


语言模型经常编造事实，生成带有偏见和有毒的文本或者没有遵循用户的指令的原因是语言模型的目标函数用于预测下一个token-这与我们的目标不同，我们的目标是生成遵循用户指令的有用的和安全的结果。

The reason why language models often fabricate facts, generate biased or toxic text, or fail to follow user instructions is that the objective function of language models is designed for predicting the next token. This objective function does not align with our goals, which are to generate useful and safe results that adhere to user instructions.

作者aligning语言模型方法（three models）：

The authors propose an approach to align language models, which involves three models:


1)从人类反馈中使用增强学习（RLHF）来微调GPT-3，具体的做法是人工写了很多prompt，用标注工具把答案写出来，这样就标注了一个数据集，然后对GPT3模型做微调，结果是SFT（Supervised Fine-Tuning)。

1) They use Reinforcement Learning from Human Feedback (RLHF) to fine-tune GPT-3 based on human feedback. The specific process involves creating a dataset by manually writing numerous prompts and generating answers using an annotation tool. This dataset is used for supervised fine-tuning (SFT).


2) 使用SFT模型，输出多个结果，对结果进行排序(构建新的数据集)，训练一个奖励模型（RM）（人工标注太贵）。

2) Using the SFT model, multiple responses are generated and ranked (constructing a new dataset), and a reward model (RM) is trained. Since manual annotation is costly, the RM helps assess the quality of responses.


3)使用RM作为奖励函数，fine-tune有监督学习（SFT），使用PPO算法来最大化奖励函数。这个模型称为instruct GPT。

3) They employ the RM as a reward function and fine-tune the model using supervised learning (SFT) with the Proximal Policy Optimization (PPO) algorithm to maximize the reward function. This model is referred to as "instruct GPT."



![语言模型方法](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729092746.png)

作者训练了三个模型（1.3B，6B，175B），这些模型都使用GPT-3框架。

作者发现如下：

    与GPT-3的输出相比，标注者明显更喜欢InstructGPT的输出
    与GPT-3相比，InstructGPT模型的真实性要好一些。
    与GPT-3相比，InstructGPT在毒性方面略有改善，因为它可以说我不想回答你这个问题，但是偏见上没太多提升。
    微调的时候都是针对某些任务做微调，可能使得你在一些别的任务上性能会下降。
    没造过训练数据的雇员觉得InstructGPT效果更好；
    公开的NLP数据集不能反应我们的语言模型是如何被使用的；
    InstructGPT模型展示了更好的泛化性在那些RLHF微调之外的指令上。
    InstructGPT仍然会犯一些简单的错误。

The authors trained three models (1.3B, 6B, 175B), all based on the GPT-3 framework, and made the following observations:

- Compared to GPT-3's outputs, human labelers significantly preferred the outputs of InstructGPT.
- InstructGPT exhibited slightly better reliability compared to GPT-3.
- InstructGPT showed some improvement in toxicity relative to GPT-3 because it could respond by saying it didn't want to answer a particular question. However, there wasn't a significant reduction in bias.
- Fine-tuning for specific tasks might lead to performance drops on other tasks.
- Labelers who had not seen training data felt that InstructGPT's performance was better.
- Public NLP datasets might not reflect how language models are used in practice.
- InstructGPT demonstrated better generalization to instructions beyond those used in RLHF fine-tuning.
- InstructGPT still made some simple errors.


## 相关工作

    从人的反馈中学习和对齐
    作者使用基于人类反馈的强化学习(RLHF)进行微调
    训练语言模型来遵循指令。
    在公共NP数据集微调，在不同的NLP数据集进行评估。结论是在一系列NLP任务上微调LMs，提高了他们在执行任务时的表现，无论是在zero-shot还是few-shot上。
    评估语言模型的危害；
    有一个新兴的不断发展的领域，旨在建立基准，具体评估这些危害，但取得进展很难。
    修改语言模型的行为以减轻危害。
    在过滤的数据集上训练后，LMs产生的有害文本较少，但代价是语言建模性能略有下降。Xu使用多种方法来提高聊天机器人的安全性，包括数据过滤、在生成过程中阻止某些单词或 n-gram、指定token， 和human-in-the-loop 数据收集等。


- Learning from Human Feedback and Alignment with Human Intent:
The authors used Reinforcement Learning from Human Feedback (RLHF) to fine-tune language models (LMs) to follow instructions.
- Fine-Tuning LMs on Public NLP Datasets:
The LMs were fine-tuned on public NLP datasets and evaluated on various NLP tasks, demonstrating improved performance in executing tasks in both zero-shot and few-shot settings.
- Evaluating the Harms of Language Models:
Evaluating the harms caused by language models is a challenging and evolving field, but efforts are being made to establish benchmarks for assessing these harms.
- Modifying LM Behavior to Mitigate Harms:
Training LMs on filtered datasets reduced the generation of harmful text but at the cost of slightly reduced language modeling performance. Various methods, including data filtering, blocking certain words or n-grams during generation, specifying tokens, and human-in-the-loop data collection, have been used to enhance the safety of chatbots, as demonstrated by Xu.


## 方法和实验细节

    作者的方法遵循了Ziegler et al.和Stiennon et al.的方法，从一个预训练的语言模型、一个希望模型产生align输出的提示语和一组经过培训的标注者开始。执行下面三个步骤
    Step1：收集数据，并训练一个监督模型。使用监督学习在这些数据上微调一个预训练好的GPT-3模型。
    Step2：收集比较数据，并训练一个奖励模型。收集一些模型输出之间的比较数据，其中标注者指出他们对于给定输入更喜欢哪个输出。然后训练一个奖励模型来预测人类偏好的输出。
    Step3：使用PPO算法优化奖励模型下的策略。把奖励模型的输出作为一个标量奖励，用PPO算法在这个奖励下微调监督策略。第二步和第三步可以不断迭代；收集更多当前最佳策略下的比较数据，用它们来训练一个新的奖励模型和一个新的策略。

The author's approach follows the methods of Ziegler et al. and Stiennon et al., starting with a pre-trained language model, a prompt intended to elicit aligned outputs, and a group of trained annotators. It involves three steps:

**Step 1:** Data Collection and Supervised Model Training
- Collect data and fine-tune a pre-trained GPT-3 model using supervised learning on this data.

**Step 2:** Comparative Data Collection and Reward Model Training
- Gather comparative data where annotators indicate their preference for the outputs given a specific input.
- Train a reward model to predict human-preferred outputs.

**Step 3:** Policy Optimization Using PPO Algorithm
- Utilize the output of the reward model as a scalar reward and fine-tune the supervised policy using the Proximal Policy Optimization (PPO) algorithm.
- Steps two and three can be iterated continuously: collect more comparative data under the current best policy to train a new reward model and a new policy.


--- 
    提示语数据集主要是由提交到open AI API 上的文本提示语组成。没有使用来自使用API的客户的数据，对数据集删除重复的提示词，限制提示词的长度为200，过滤训练集中个人身份信息等。训练InstructGPT模型时，要求标注员写下面三类提示词：
    Plain:我们只是要求标注员写一个任意的问题，同时确保问题有足够的多样性。 
    Few-shot:我们要求标注者提出一个指令，以及该指令的多个查询/响应对。 
    User-based：我们在OpenAI的候选名单应用程序中列出了许多用例API。我们要求标注者提出与这些用例相对应的提示词。
    SFT数据集：13K条数据；RM数据集：33K条数据；PPO数据集：31K条数据

The prompt dataset primarily consists of text prompts submitted to the OpenAI API. It does not include data from customers using the API, and the dataset has undergone preprocessing steps such as removing duplicate prompts, limiting the length of prompts to 200 characters, and filtering out personal information in the training set. When training the InstructGPT model, annotators were required to provide prompts falling into three categories:

1. **Plain:** An arbitrary question is requested from the annotator, ensuring that the questions exhibit sufficient diversity.

2. **Few-shot:** An instruction prompt, along with multiple query/response pairs for that instruction, is requested.

3. **User-based:** Annotated prompts corresponding to the use cases listed in OpenAI's candidate applications.

The dataset statistics are as follows:
- SFT dataset: 13,000 entries.
- RM dataset: 33,000 entries.
- PPO dataset: 31,000 entries.

---

    训练任务来自两个来源:(1)由我们的标注者编写的提示数据集和（2）在我们的API上提供给早期InstructGPT模型的提示语数据集。
    提示语非常多样化，包括生成、问答、对话、摘要、摘录等其他自然语言任务。我们的数据集超过96%是英语。
    尽管任务很复杂，但我们发现标注者之间的一致率相当高：训练标注者在 72.6 ± 1.5% 的时间内彼此一致，而对于hold-out者，这个数字是 77.3 ± 1.3%。为了比较，研究人员之间的一致性为 73 ± 4%。
    我们从GPT-3预训练语言模型开始。我们用三个不同的技术来训练模型:
    Supervised fine-tuning（SFT）。监督微调(SFT)。我们在我们的标注数据上微调GPT-3使用监督学。我们训练了16个epoch，使用余弦学习率衰减，残差为0.2。我们根据验证集上的RM分数进行最终的SFT模型选择。我们发现我们的SFT模型在1 epoch后的验证集的损失上过拟合;然而，我们发现尽管存在过拟合，但更多epoch的训练对RM分数和人类偏好评级都有帮助。

The training data for InstructGPT is sourced from two main channels: 

1. A dataset of prompt data created by our annotators.
2. A dataset of prompt data provided to the early InstructGPT models on our API.

The prompts are highly diverse, encompassing various natural language tasks, including generation, question-answering, dialogues, summaries, excerpts, and more. Over 96% of the dataset consists of prompts in English.

Despite the complexity of the tasks, the agreement among annotators is relatively high: training annotators exhibit approximately 72.6% ± 1.5% mutual agreement, while for hold-out annotators, this figure is 77.3% ± 1.3%. For comparison, inter-rater reliability among researchers is around 73% ± 4%.

The training process initiates with the GPT-3 pretrained language model. Three distinct techniques are used to train the model:

1. **Supervised Fine-Tuning (SFT):** In SFT, GPT-3 is fine-tuned on our annotated data using supervised learning. Training is conducted over 16 epochs, employing cosine learning rate decay with a residual weight of 0.2. The final SFT model selection is based on RM scores from the validation set. It's observed that the SFT model tends to overfit on the validation loss after one epoch, but further training beyond this point is helpful for both RM scores and human preference ratings.


**Reward modeling（RM）奖励建模(RM)**

SFT模型去掉最后的umembedding layer，输入提示语和答案，输出一个标量结果（reward）。在这篇论文中我们仅使用了6B RMs, 这节省了很大的计算量。我们发现175B RM训练不稳定因此不适合用于值函数，在RL期间。
奖励模型的损失函数是：

The loss function for the reward model (RM) in the context of InstructGPT is as follows:

The RM's loss function comprises two primary components:

1. **RM Cross-Entropy Loss:** This component measures the agreement between the RM's predicted probabilities and the human ranker's preferences. It is calculated using the cross-entropy loss between the predicted probabilities assigned to different model outputs and the actual human preferences provided in the dataset.

2. **Rank-Corrected Loss:** This component is designed to ensure that the RM assigns higher scores to responses that are ranked more favorably by humans. It helps in penalizing the RM when its ranking does not match the human preferences.

The combination of these components in the loss function helps the reward model learn to predict responses that align with human preferences, which is crucial for reinforcement learning from human feedback (RLHF). This methodology is used to fine-tune the language model and make its output more in line with user intent.

![奖励建模(RM)](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729093037.png)


r_𝜃 (𝑥,𝑦)是奖励模型对于提示词x和完成y的标量输出，y_w是y_w和y_l中更受欢迎的补全，D是人类比较的数据集。
为了加快比较收集，取K = 9，36对排序进行优化

**Reinforcement learning (RL)  强化学习(RL)**

作者使用PPO来微调SFT模型。输入一个prompt期望得到一个输出。给定一个prompt和response，生成奖励分数。除此之外，增加了KL散度降低奖励模型的过度优化。我们称这个模型为PPO。
作者把预训练的梯度加入到PPO的梯度中，为了缓和模型在公开数据集中的性能损失。我们称这个模型为PPO-ptx。
我们在强化学习中最大化这个目标函数。

The authors use the Proximal Policy Optimization (PPO) algorithm to fine-tune the Supervised Fine-Tuned (SFT) model, which is referred to as the PPO model. The PPO model aims to maximize the following objective:

1. **Expected Reward Maximization:** In this step, given a prompt and response, the model generates a reward score. The PPO model seeks to maximize the expected reward by optimizing its policy to produce responses that align with human preferences.

To mitigate potential performance losses of the model on public datasets during reinforcement learning, the authors combine the pre-trained gradients with the PPO gradients. This model is referred to as PPO-ptx, where "ptx" represents the incorporation of the pre-trained gradients. The objective in both cases is to maximize expected reward by optimizing the model's policy through reinforcement learning.

![强化学习(RL)](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729093148.png)

**Baselines：**

我们对PPO模型、SFT模型和GPT-3的性能进行比较。我们也对比了给gpt-3的提示词提供few-shot 前缀的模型。前缀附加在用户指定的指令之前。

还对比了InstructGPT与在FLAN和T0上的微调175B GPT-3模型。两个数据集都由各种NLP任务组成，结合自然语言指令。我们分别在大约100万个例子上对它们进行微调并选择在验证集上获得最高奖励模型分数选为checkpoint。

评估我们的模型是如何“对齐”的，定量评估分为两个独立的部分

    API评价：主要的评价标准是人类偏好胜率
    对公共NLP数据集的评估

We compared the performance of the PPO model, the SFT model, and GPT-3. We also compared models that used few-shot prefixes for the prompts given to GPT-3. These prefixes are added before the user-specified instructions.

We further compared InstructGPT to the fine-tuned 175B GPT-3 models on the FLAN and T0 datasets. Both datasets consist of various NLP tasks combined with natural language instructions. We fine-tuned them separately on approximately one million examples and selected the models with the highest reward model scores on the validation set as checkpoints.

The evaluation of our models on how well they "align" is quantitatively assessed in two independent parts:

1. **API Evaluation:** The primary evaluation criterion is the human preference win rate.
2. **Evaluation on Public NLP Datasets:** This evaluation involves assessing model performance on common NLP datasets.

## 结果

    图3:我们的模型的偏好结果，通过对比175B SFT模型的胜率。
    我们从对提交给GPT-3模型(左)的提示的评估中省略了GPT(prompt) 
    因为这些提示已经设计得很好，可以适用 于GPT-3， 和InstructGPT 模型得提示恰好相反（右）。

![结果](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729093256.png)

    图4中，展示了标注者也对InstructGPT输出进行了更具体的评价轴。具体来说，与GPT-3相比，InstructGPT 输出更接近一个客户助理，更经常遵循指令中明确定义的约束，更少的有害性。

![结果1](https://cdn.jsdelivr.net/gh/1oscar/image_house@main/20230729093424.png)



## 讨论

    在这项工作中，我们的对齐研究方法是迭代的:我们正在改进的是现有的AI系统的对齐而不是抽象地专注于调整尚不存在的人工智能系统。
    作者在研究中吸取的经验教训：
        增加模型对齐的成本相对于预训练是更划算的。
        我们已经看到一些证据表明，InstructGPT将“遵循指令”泛化到我们不对其进行监督的设置，例如非英语语言任务和与代码相关的任务。
        我们能够减轻通过微调带来的大多数性能下降。
        我们已经从现实世界的研究中验证了对齐技术。

In this work, our approach to alignment research is iterative. We are focused on improving existing AI systems' alignment rather than abstractly tuning AI systems that do not yet exist.

Here are some key lessons and insights the authors have learned in the course of this research:

1. **Increasing alignment in post-training is more cost-effective than during pre-training.**
   
2. There is some evidence that InstructGPT generalizes "following instructions" to settings where it is not supervised, such as non-English language tasks and code-related tasks.

3. Most of the performance degradation introduced by fine-tuning can be mitigated.

4. Alignment techniques have been validated in real-world research.

---

    局限性
    InstructGPT模型的行为部分是由人类的反馈决定的（标注员）。一些标签任务依赖于价值判断，这可能是受我们承包商的身份，信仰，文化背景和个人经历的影响。 
    我们的模型既不完全对齐，也不完全安全;他们仍然产生有毒或偏见输出，编造事实，在没有明确提示的情况下产生性和暴力内容。
    开放问题
    可以使用对抗设置来减少有害的输出；
    比较也不一定是提供对齐信号的最有效方式。
    另一个可能改进我们方法的修改是过滤预训练混合数据中的有毒内容或使用合成指令扩充此数据。

Limitations:
- The behavior of the InstructGPT model is partly determined by human feedback (annotators), and some tasks rely on value judgments that may be influenced by the identity, beliefs, cultural background, and personal experiences of the contractors.
- Our models are neither fully aligned nor completely safe; they still produce toxic or biased outputs, fabricate facts, and generate sexual or violent content without explicit prompts.

Open Questions:
- Adversarial setups can potentially reduce harmful outputs.
- Comparison may not necessarily be the most effective way to provide alignment signals.
- Another potential improvement to our approach is to filter out toxic content from the pretraining mixture data or use synthetic prompts to augment this data.

## 附录

### 标注员写提示词的三种类型
Plain：我们只是要求标记者提出一个任意任务，同时确保任务的多样性。
Few-shot：我们要求标注者提出一条指令，以及该指令的多个查询/响应对。
User-based：我们在 OpenAI API 的应用程序中陈述了许多用例。 我们要求标注者提出与这些用例相对应的提示。

### Three Types of Prompts Provided by Annotators
Plain: Annotators were asked to come up with any task, ensuring diversity in the tasks suggested.
Few-shot: Annotators were tasked with providing an instruction and multiple query/response pairs for that instruction.
User-based: Annotators were required to suggest prompts corresponding to the various use cases presented in the OpenAI API application.


### API 用户提示词
使用Instruct GPT模型早期版本用户提交的提示词。为了保证提示词的多样性，我们会删除重复的提示词，限制提示词的长度为200。除此之外，我们基于组织id来划分训练集，测试和验证集。
图表展示了一些用户的prompts：生成(generation)、开放式(open) QA、封闭式(closed) QA、头脑风暴、聊天(chat)、重写、总结、分类、提取或其他。
Table6 展示了用于训练/验证 SFT、RM 和 RL 模型的数据集的大小
Table7展示了数据集的多样性。
Table8展示了每个用户的平均prompts数量；
Table9展示了数据集分类的提示词长度，类别分类的提示词长度

### API User Prompts
User prompts were drawn from early versions of the Instruct GPT model submitted by users. To ensure diversity in prompts, duplicates were removed, and the length of prompts was limited to 200 characters. Additionally, the training, test, and validation sets were organized based on organization IDs.

The figures show some user prompts in different categories: generation, open QA, closed QA, brainstorming, chat, rewriting, summarization, classification, extraction, and others.

Table 6 presents the sizes of the datasets used for training/validation of SFT, RM, and RL models.
Table 7 illustrates the diversity of the datasets.
Table 8 shows the average number of prompts per user.
Table 9 provides information on prompt length distribution by category.


### 模型细节

#### Training Details for All Models
All models were built using the GPT-3 architecture and trained using the Adam optimizer.



#### 所有模型都使用了GPT-3架构。所有模型训练使用Adam优化器。
1.SFT训练细节
SFT模型训练16个epoch，dropout是0.2，学习率是原始值的10%，没有学习率预热。1.3B和6B模型，使用LR值9.65e-6，batchsize=32；175B使用LR值5.03e-6，batchsize=8。最终模型根据RM分数进行选择的。

1. **SFT Training Details**
   - SFT models were trained for 16 epochs.
   - Dropout rate was set to 0.2.
   - The learning rate started at 10% of the base value and had no learning rate warm-up.
   - For 1.3B and 6B models, the learning rate was 9.65e-6, and the batch size was 32.
   - For the 175B model, the learning rate was 5.03e-6, and the batch size was 8.
   - The final model was chosen based on RM scores.

#### RM训练细节
选择6B RM模型的原因：大模型不稳定。RM模型训练过程中，对epoch敏感，多个epoch很快就过拟合，这里作者只训练了一个epoch，学习率9e-6，batchsize=64。每个prompt选择k=4和k=9标签补全，结果会有(K¦2)个对比项，因此单批最多可以包含64*(K¦2)≤2304 个对比项。

2. **RM Training Details**
   - The 6B RM model was chosen due to the instability of larger models.
   - RM models were trained for one epoch.
   - The learning rate was set to 9e-6 with a batch size of 64.
   - Each prompt had k=4 and k=9 label completions, leading to (K choose 2) comparison items in each batch, allowing for a maximum of 2304 comparison items per batch.

#### RLHF模型的初始化细节
从预训练的GPT-3模型初始化RLHF模型，在数据集上进行监督微调 2个epoch。
微调期间混入了10%的预训练数据，因为发现这对PPO训练很有帮助。
使用了1.3B和6B的batchsize=32；175B的batchsize=8。作者对比了每个模型的不同的峰值的学习率，选择了在演示和预训练验证集上损失都比较小的那个。

3. **Initialization Details for RLHF Models**
   - The RLHF models were initialized from the pre-trained GPT-3 model and supervised fine-tuned for 2 epochs on the dataset.
   - During fine-tuning, 10% of the pre-training data was mixed in as it was found to be helpful for PPO training.
   - For 1.3B and 6B models, batch sizes were set to 32. For 175B, the batch size was 8.
   - Various peak learning rates were tested for each model, and the one with the smallest loss on the demo and pre-training validation sets was chosen.


#### RLHF训练的细节
所有的PPO模型使用6B RM和 6B值函数。值函数1.3B和6B的学习率是9e-6，175B的学习率是5e-6；
预训练示例是 RL 训练集数的 8 倍，  预训练数据是从用于训练 GPT-3 模型的数据集中随机抽取的。
对于每个batch，我们在连续的步骤中计算 PPO 梯度和预训练梯度，并将它们都累积到梯度缓冲区中。
我们将预训练梯度乘以一个系数 γ = 27.8，以控制来自 PPO 和预训练分布的梯度的相对强度。

4. **Training Details for RLHF**
   - All PPO models used the 6B RM and value functions.
   - The learning rate for value functions was 9e-6 for 1.3B and 6B models, and 5e-6 for the 175B model.
   - The pre-training dataset was eight times the size of the RL training set, randomly sampled from the dataset used to train the GPT-3 model.
   - For each batch, PPO and pre-training gradients were computed in consecutive steps and accumulated in a gradient buffer.
   - The pre-training gradient was scaled by a factor γ = 27.8 to control the relative strength of gradients from PPO and pre-training distributions.


#### FLAN和T0模型
 我们通过在 FLAN 和 T0 数据集上微调 175B GPT-3 模型来获得我们的 FLAN 和 T0 基线。 将 T0 数据集下采样到 1M个数据点，以使每个模型的训练数据量具有可比性。
FLAN checkpoint：使用6B RM模型对prompt验证集进行评分。最后选择了4e-6 的学习率和 896k 个示例进行训练的checkpoint。
T0 checkpoiint。在一个实验中，我们使用了batch大小为128，学习率为4e-6，样本数为128万。另一个实验使用批大小为64，学习率为6e-6，样本数为100万。再次使用奖励模型分数，我们在学习率为4e-6 ,896k个训练样本后从前一个实验中选择了。


5. **FLAN and T0 Models**
   - The FLAN baseline was obtained by fine-tuning the 175B GPT-3 model on the FLAN dataset.
   - The T0 baseline was achieved by fine-tuning on the T0 dataset, downsampled to 1 million data points, making training data comparable.
   - FLAN checkpoint: A learning rate of 4e-6 and 896k examples were used for training.
   - T0 checkpoint: Two experiments were conducted, one with a batch size of 128, a learning rate of 4e-6, and 1.28 million samples. The other used a batch size of 64, a learning rate of 6e-6, and 1 million samples. A model checkpoint was selected from the first experiment based on RM scores, using a learning rate of 4e-6 and 896k training samples.


## 我的点评
自从chatgpt3大火之后，ai绘画也开始爆火。很多团队瞄准了这个方向去创业。这篇文章算是比较经典的文生图的范例，很值得细细研究阅读。

Since the rise of ChatGPT-3, AI-generated art has also gained popularity. Many teams are venturing into this field for entrepreneurial opportunities. This article can be considered a classic example of text-to-image generation and is certainly worth a closer examination and study.

