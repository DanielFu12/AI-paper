# AI Paper

## 1、Adam: A Method for Stochastic Optimization
* 作者：Kingma等
* 时间：2015
* 意义：这篇论文提出了Adam算法变体，作为当时流行的随机梯度下降优化算法的改进，可以快速收敛神经网络，加快训练效率。Adam已经成为训练神经网络的默认优化算法。
* 链接：https://arxiv.org/abs/1412.6980

## 2、Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
* 作者：Ioffe、Szegedy
* 时间：2015
* 意义：通过归一化输入特征的简单方法，让神经网络能更快地训练，获得更稳定的表现。该方法也被认为是深度神经网络性能进一步提升的关键点。
* 链接：https://arxiv.org/abs/1502.03167

## 3、Faster R-CNN: towards real-time object detection with region proposal networks
* 作者：Ren等人
* 时间：2015
* 意义：Faster R-CNN 是人工智能视觉识别在工业领域大规模应用的起点，安防摄像头、自动驾驶和各种图像识别程序中都使用了这套算法。
* 链接：https://arxiv.org/abs/1506.01497

## 4、Neural Machine Translation by Jointly Learning to Align and Translate
* 作者：Bahdanau等人
* 时间：2015, cited by 16866
* 意义：神经网络第一次使用注意力机制进行机器翻译，让AI翻译不再受限于RNN网络数据处理长度。
* 链接：https://arxiv.org/abs/1409.0473

## 5、Human-level control through deep reinforcement learning
* 作者：Mnih等人
* 时间：2015年
* 意义：这篇论文引入了强化学习算法DQN，该算法在许多游戏中实现了人类水平的表现，也推动了诸多软件程序从硬编码，转向了强化学习，取代了传统手工编码的软件自动化策略。
* 链接：https://www.nature.com/articles/nature14236

## 6、Explaining and Harnessing Adversarial Examples
* 作者：Goodfellow等人
* 时间：2015
* 意义：第一次推出神经网络对抗学习算法，也提出了对抗训练的基本思路。该研究还表明，此前的机器学习存在的“鲁棒性”问题，如同样的图片，轻微修改几个像素，AI的识别结果就会发生大幅改变
* 链接：https://arxiv.org/abs/1412.6572

## 7、Deep Residual Learning for Image Recognition
* 作者： Kaiming He（何恺明）等
* 时间：2015
* 意义：这篇论文基于数学原理，提出了加快深度神经网络训练的方法，在视觉识别领域取得了显著效果，让人工智能可以更快提取物体特征。该研究激发了谷歌员工提出了Transformers模型。
* 链接：https://arxiv.org/abs/1512.03385

## 8、Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
* 作者：Radford等人
* 时间：2016
* 意义：这篇论文提出了DCGAN，一种用于GAN模型生成器的深度卷积神经网络体系结构，首次让AI画出了不存在的图片，成为AI绘画变革的重要节点。
* 链接：https://arxiv.org/abs/1511.06434

## 9、Attention Is All You Need
* 作者：Ashish Vaswani等
* 时间：2017
* 意义：注意力机制首次被完整论述，并提出了对应的Transformer架构，成为所有AI大模型的底层技术。这篇研究的作者们此后成为了硅谷众多AI企业的创业者。
* 链接：https://arxiv.org/abs/1706.03762

## 10、BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding
* 作者：Jacob Devlin等
* 时间：2018
* 意义：提出了一种新的机器学习训练方法，显著提高了语言相关任务中的性能，例如使用上下文来确定某个词的含义。OpenAI的GPT系列均采用了该方法。
* 链接：https://arxiv.org/abs/1810.04805

## 11、Language Models Are Few-Shot Learners
* 作者：Tom B. Brown等
* 时间：2020
* 意义：OpenAI GPT-3的研究论文，这篇论文提出使用足够的算力和数据，对大型语义模型进行训练，无需进行具体的数据分隔，最终它就能获得多方面的泛化能力，可以完成翻译、回答等任务。
* 链接：https://arxiv.org/abs/2005.14165

## 12、Learning Transferable Visual Models From Natural Language Supervision
* 作者：Alec Radford等
* 时间：2021
* 意义：这篇论文提出了文字-图片预训练神经网络（CLIP），能够基于图片的文字标注，学习图片与对应概念间的关系，与OpenAI的Dall-E模型共同成为了领域的重要基础。
* 链接：https://arxiv.org/abs/2103.00020

## 13、High-Resolution Image Synthesis With Latent Diffusion Models
* 作者：Robin Rombach等
* 时间：2021
* 意义：提出了图像生成领域知名的扩散模型，极大提高了图片生成的效率，成为刺激图片生成领域大爆发的关键技术。
* 链接：https://arxiv.org/abs/2112.10752

## 14、Highly Accurate Protein Structure Prediction With AlphaFold
* 作者：John Jumper等
* 时间：2021
* 意义：DeepMind研究人员在AI应用的又一次开创研究，解决了约10万种独特蛋白质的三维结构问题，为科学家开发新药物和治疗疾病提供可能性。
* 链接：https://www.nature.com/articles/s41586-021-03819-2

## 15、Human-Level Play in the Game of Diplomacy by Combining Language Models With Strategic Reasoning
* 作者：Anton Bakhtin等
* 时间：2022
* 意义：该论文提出了名为Cicero的机器学习算法，让AI在多人对话游戏中，具备了沟通、协作，推理他人意图的能力。
* 链接：https://www.science.org/doi/10.1126/science.ade9097

## 16、Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners
* 作者：Renrui Zhang等
* 时间：2023
* 意义：Combine diverse prior knowledge、DINO visioncontrastive info、CLIP lang-contrastive info、DALL-E vision info GPT-3 lang info、Ensembles via cache model、State-of-the-art few-shot prediction
![image](https://user-images.githubusercontent.com/79930083/227792482-d6963daf-e652-42ce-8361-3df21fbf7366.png)
* 链接：https://arxiv.org/abs/2303.02151

## 17、Toolformer: Language Models Can Teach Themselves to Use Tools
* 作者：Timo Schick等
* 时间：2023
* 意义：篇论文提出了一种新的语言模型：Toolformer。该模型的特别之处是可以训练自己来使用各种工具，例如调用 API、做数值计算、请求网页内容，或者是其他任何操作。
* 链接：https://arxiv.org/abs/2302.04761

## 18、Language Is Not All You Need: Aligning Perception with Language Models
* 作者：Shaohan Huang
* 时间：2023
* 意义：提出了一种多模态模型
* 链接：https://arxiv.org/abs/2302.14045

## 19、Sparks of Artificial General Intelligence:Early experiments with GPT-4
* 作者：Sébastien Bubeck等
* 时间：2023
* 意义：论证了GPT4是通往AGi之路的里程碑。GPT-4的能力具有普遍性，它的许多能力跨越了广泛的领域，而且它在广泛的任务中的表现达到或超过了人类水平，这两者的结合使可以说GPT-4是迈向AGI的重要一步。
* 链接：https://arxiv.org/abs/2303.12712

## 20、Training language models to follow instructions with human feedback
* 作者：Long Ouyang等
* 时间：2022
* 意义：使语言模型更大并不能从本质上使它们更好地遵循用户的意图。例如，大型语言模型可能会生成不真实的、有毒的或对用户没有帮助的输出。换句话说，这些模型与其用户不一致。在本文中，我们展示了一种途径，可以通过根据人类反馈进行微调，使语言模型与用户对各种任务的意图保持一致。
* 链接：https://arxiv.org/abs/2203.02155

## 21、Reflexion: an autonomous agent with dynamic memory and self-reflection
* 作者：Noah Shinn等
* 时间：2023
* 意义：让AI能够自我反思：Reflexion，一种赋予代理动态记忆和自我反思能力的方法，以增强其现有的推理轨迹和特定任务的行动选择能力。
* 链接：https://arxiv.org/abs/2303.11366
