# 第 18 章 UC Berkeley（BAIR）：强化学习的故乡，与"开源通才策略"的起点

> **一句话定位**：Berkeley 人工智能研究实验室（BAIR）是深度强化学习与机器人学习的发源地之一——Pieter Abbeel、Sergey Levine、Jitendra Malik、Ken Goldberg、Trevor Darrell、Angjoo Kanazawa 等人的实验室，培养了 OpenAI、PI、Covariant、Skild（Pathak 曾为 Malik 学生/合作者）等公司的大量核心成员。在人形机器人大脑领域，Berkeley 贡献了三类关键思想：**开源通才策略（Octo）**、**真机强化学习的实用化（HIL-SERL）**、以及**以 Transformer 序列建模统一人形运动控制（Radosavovic & Malik 的人形 RL 系列）**。

---

## 0. 速览卡片

| 维度 | 内容 |
|---|---|
| 核心实验室 | RAIL（Sergey Levine）、Abbeel 组、Malik 组、Hybrid Robotics（Koushil Sreenath）、Kanazawa 组、AUTOLAB（Ken Goldberg）|
| 代表工作（全集）| TRPO/PPO/SAC（RL 基础算法）、QT-Opt（与 Google）、BridgeData V2（2023）、**Octo（2024）**、CrossFormer（2024）、**HIL-SERL（2024）**、**人形 RL 系列（2023–2024）**、Berkeley Humanoid（2024）、VideoMimic（2025）、Pi 系列（Levine 参与）|
| 人才输出 | OpenAI（Schulman 等）、PI（Levine 联合创始人）、Covariant/Amazon FAR（Abbeel、Chen、Duan）、众多中国具身公司研究负责人 |
| 本章三篇代表作 | ① Octo：开源通才机器人策略（2024）② HIL-SERL：人在环路的高效真机强化学习（2024）③ 人形运动的 Transformer 化：真实世界人形 RL 与"人形运动即下一个 token 预测"（2023–2024）|
| 身位 | 强化学习与开源通才策略的思想源头；对 PI 的方法论（RECAP）、人形 RL 范式有直接影响 |

---

## 1. 故事的开端："能不能让机器人自己学？"

Berkeley 的机器人学习传统，从一开始就带着"**让机器人自主学习**"的执念：Abbeel 早年的直升机特技学徒学习、Levine 2015 年的端到端视觉运动策略（GPS）、2016–2018 年与 Google 合作的 QT-Opt 大规模抓取 RL……这些工作都在回答"如何减少人的干预、让机器人从经验中变强"。

2022 年后，当"大模型 + 模仿学习"成为主流时，Berkeley 一方面积极参与（BridgeData、Octo、OpenVLA、OXE），另一方面坚持追问：**模仿学习的天花板在哪里？RL 如何在真机上真正可用？**——这条思路最终在 PI 的 π*0.6/RECAP 中开花结果。

---

## 2. 代表作一：Octo（2024）——开源通才机器人策略

> 论文：Octo Model Team（Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, ..., Sergey Levine 等），*Octo: An Open-Source Generalist Robot Policy*（RSS 2024，arXiv 2405.12213）——Berkeley、斯坦福、CMU 等合作。

### 2.1 要解决的问题

2023 年 OXE 数据集发布后，一个问题摆在面前：**能否训练一个开源的、能适配不同机器人（不同相机、不同动作空间）的"通才策略"，作为大家微调的起点？** RT-1-X/RT-2-X 未完全开源，且输入输出格式固定。

### 2.2 设计：灵活的 Transformer + 扩散动作头

```
  任务 token（语言指令 或 目标图像）
  观察 token（多路相机图像 patch、本体状态）     ← 可按需增减输入模态
  读出 token（readout tokens）                  ← 类似 BERT 的 [CLS]，聚合信息
            │
            ▼
   Transformer 主干（分块注意力：观察只能看任务与过去，读出 token 可看所有）
            │
            ▼
   读出 token 的嵌入 → 扩散动作头 → 动作块
```

关键设计：
- **模块化输入输出**：新机器人有新相机或新动作空间时，只需加新的 tokenizer / 动作头并微调，主干不变；
- **扩散动作头**：借鉴 Diffusion Policy，以读出 token 为条件生成连续动作块；
- **规模小**：Octo-Small 约 2700 万参数，Octo-Base 约 **9300 万参数**——刻意保持小，以便学术界在普通 GPU 上使用；
- **数据**：在 OXE 中约 **80 万条**轨迹（25 个数据集）上预训练。

### 2.3 结果与意义

- 开箱即用可控制多种机器人（WidowX、UR5、Franka 等）；
- 在新机器人、新观察/动作空间上仅需少量数据微调即可达到良好性能，优于从零训练；
- **意义**：Octo 是第一个真正"**可插拔**"的开源通才策略，确立了"预训练通才 + 少量数据微调"的开源工作流。它与 OpenVLA 形成互补：Octo 小、快、灵活但语义弱；OpenVLA 大、语义强但慢。π0 后来把两者优点结合（大 VLM 语义 + 扩散/流匹配动作专家）。

后续的 **CrossFormer（2024）** 进一步把同一个 Transformer 扩展到操作、导航、四足、无人机等 20 多种本体，验证了"跨本体通才"的可行性。

---

## 3. 代表作二：HIL-SERL（2024）——让真机强化学习"真正可用"

> 论文：Jianlan Luo, Charles Xu, Jeffrey Wu, Sergey Levine，*Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*（HIL-SERL，2024；Science Robotics 2025）

### 3.1 要解决的问题

真机 RL 长期被认为"不实用"：需要数十万次试错、需要精心设计奖励、训练不稳定、会损坏机器人。但模仿学习也有天花板——它最多学到示范者的水平，而且在精密装配这类任务上，几十条示范往往只能达到 50%–70% 的成功率。

### 3.2 方法：三个关键组件

1. **少量示范 + 奖励分类器**：先采集少量成功/失败样本，训练一个视觉二分类器作为奖励（"任务成功了吗？"）——不需要手工设计奖励函数；
2. **样本高效的离线到在线 RL**：基于 RLPD（利用先验数据的强化学习）等高 UTD（更新/数据比）的离策略算法，把示范数据与在线经验混合训练；
3. **人在环路纠正（Human-in-the-Loop）**：当机器人在训练中犯错时，人用遥操作设备（如 SpaceMouse）实时接管纠正，纠正数据被加入回放缓冲区——**人的干预既保证了安全，又提供了高价值的"在失败处如何做对"的数据**。

### 3.3 结果

在一系列精密、灵巧、动态任务上——**仪表盘装配、正时皮带安装、主板 RAM 插装、IKEA 家具组装、双臂物体交接、"抽打"叠叠乐中的一块（Jenga whipping）、翻煎锅中的鸡蛋**等——HIL-SERL 在**约 1 到 2.5 小时**的真机训练后达到**接近 100% 的成功率**，并且执行速度明显快于人类示范（周期时间缩短）。相比之下，同等数据量的模仿学习基线成功率明显更低。

### 3.4 意义：通往 RECAP 的桥梁

HIL-SERL 的"**示范 + 自主经验 + 人工纠正**"三元组，几乎就是一年后 PI π*0.6 中 RECAP 的数据配方（演示数据、自主 rollout、专家纠正）。区别在于：HIL-SERL 是"每个任务训练一个小策略"，RECAP 把这一思想推广到"大 VLA + 优势条件化"。**Berkeley 的真机 RL 思想，就这样通过 Sergey Levine 进入了 PI 的大模型体系。**

---

## 4. 代表作三：人形运动的 Transformer 化（2023–2024）

> 论文：Ilija Radosavovic, Tete Xiao, Bike Zhang, Trevor Darrell, Jitendra Malik, Koushil Sreenath，*Real-World Humanoid Locomotion with Reinforcement Learning*（Science Robotics, 2024）；Radosavovic et al., *Humanoid Locomotion as Next Token Prediction*（NeurIPS 2024）。

### 4.1 第一篇：用 Transformer 做 RL 运动控制

**问题**：传统 RL 运动策略用 MLP，只看当前状态（或短历史），难以适应不同地形、负载与扰动。

**方法**：
- 策略是一个**因果 Transformer**，输入**过去一段时间的本体感知与动作历史**，输出下一步关节指令；
- 在仿真中用大规模 RL（含大量地形、动力学随机化）训练，借助"教师-学生"框架：先训练能看到特权信息（地形、摩擦等）的教师，再蒸馏给只能看本体感知的 Transformer 学生；
- **零样本 sim2real** 部署到 Agility 的全尺寸人形 **Digit** 上。

**结果**：Digit 在旧金山的街道、草地、人行道、斜坡等多种真实户外地形上稳定行走，能抵抗推搡、负重、踩到障碍物等扰动——而且**全部在没有真机微调的情况下完成**。

**洞察**：Transformer 的"上下文"在这里扮演了**隐式的系统辨识器**：通过观察过去的状态-动作历史，策略在推理时"推断"出当前地形和自身动力学特性，并相应调整——这与 Skild 后来宣称的"上下文中适应新身体"（第 4 章）同源，也与 Pathak/Malik 的 RMA 一脉相承。

### 4.2 第二篇：人形运动即"下一个 token 预测"

**问题**：能否像训练 GPT 一样，用**自回归下一个 token 预测**在多源数据上训练人形运动模型？

**方法**：
- 把传感器观察与动作都 token 化，用一个因果 Transformer 做**下一个 token 预测**；
- 数据来源多样：**神经网络策略的轨迹、基于模型的控制器的轨迹、人类动作捕捉数据、YouTube 视频中提取的人类运动**；
- 对于缺少某种模态的数据（例如人类视频没有机器人动作），**把缺失模态用掩码 token 代替**，模型仍可从中学习。

**结果**：仅用约 27 小时的行走数据训练的模型，就能零样本部署到真实 Digit 上在旧金山街头行走，且能泛化到训练中未见过的指令（如后退行走）；加入不完整数据（视频、动捕）能进一步提升表现。

### 4.3 意义

这两项工作把 LLM 的方法论（Transformer、上下文学习、下一个 token 预测、多源异构数据）系统地带进了**人形底层运动控制**，为 2025–2026 年"运动控制基础模型"（SONIC、Figure S0、Agility 全身控制模型、BFM-Zero）奠定了思想基础。

---

## 5. 其他重要工作

- **BridgeData V2（2023）**：WidowX 低成本平台上约 6 万条轨迹的多环境数据集，是 OXE 与 Octo/OpenVLA 的重要组成；
- **Berkeley Humanoid（2024，Hybrid Robotics）**：低成本、面向学习的中尺寸人形研究平台；
- **VideoMimic（2025，Kanazawa、Malik 等）**：从普通手机视频中重建人和场景，训练人形学会"坐椅子、爬楼梯"等与环境交互的情境化动作——"看视频学人形技能"的代表作之一；
- **Levine 组的 VLA 与 RL 研究**：大量参与 OpenVLA、π 系列、RL 微调 VLA 的研究。

---

## 6. 横向对比与身位

| 维度 | Berkeley | 斯坦福 | CMU |
|---|---|---|---|
| 核心特长 | RL（算法与真机）、开源通才策略、人形运动 Transformer | 开源 VLA、低成本数据、家庭基准 | 人形全身控制、sim2real |
| 对 PI 的影响 | Levine（RL、RECAP 思想）| Finn（模仿学习、数据）| — |
| 对人形的影响 | Digit 上的 RL 运动、VideoMimic | HumanPlus | H2O/ASAP/BFM-Zero |

**身位**：Berkeley 是"**让机器人从经验中学习**"这一思想流派的大本营。在 2026 年"VLA + RL 后训练"成为热点的背景下，Berkeley 的思想影响力处于顶峰。

---

## 7. 未来展望

1. **RL 规模化**：从"每任务一个小策略"（HIL-SERL）到"大模型的车队级 RL"，Berkeley 的研究者将继续在学术界推进这一方向；
2. **视频到人形技能**：VideoMimic 一类工作代表了"看互联网视频学人形技能"的学术前沿，与特斯拉、Figure 的人类视频路线相呼应；
3. **人才持续外流到产业**：Berkeley 的学生与教授持续流向 PI、OpenAI、Amazon 等。

---

## 8. 延伸阅读与核验来源

- Octo：[arXiv 2405.12213](https://arxiv.org/pdf/2405.12213)
- HIL-SERL：Luo et al., arXiv 2410.21845；Science Robotics 2025
- Radosavovic et al., *Real-World Humanoid Locomotion with Reinforcement Learning*，Science Robotics 2024（arXiv 2303.03381）
- Radosavovic et al., *Humanoid Locomotion as Next Token Prediction*，NeurIPS 2024（arXiv 2402.19469）
- CrossFormer（arXiv 2408.11812）；BridgeData V2（arXiv 2308.12952）；VideoMimic（arXiv 2505.03729）

> **核验说明**：Octo 的规模（约 2700 万/9300 万参数）、约 80 万条 OXE 轨迹来自原论文；HIL-SERL 的任务列表、1–2.5 小时训练、接近 100% 成功率来自原论文；人形 RL 系列的方法与"约 27 小时数据"来自原论文；本章未引用 2026 年新论文的具体数据。
