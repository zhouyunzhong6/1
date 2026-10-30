# 第 15 章 丰田研究院（TRI）：扩散策略的发源地，与"大行为模型"的严谨科学

> **一句话定位**：如果说 RT-2 定义了 VLA 的"语义继承"，那么 TRI 参与提出的 **Diffusion Policy（扩散策略，2023）** 则定义了 VLA 的"动作生成"——此后几乎所有主流 VLA 的动作头（π0 的流匹配、GR00T 的 DiT、Atlas LBM、Helix……）都是扩散/流匹配家族的后代。TRI 随后提出"**大行为模型（Large Behavior Model, LBM）**"概念，并在 2025 年发表了行业中**统计上最严谨**的 VLA 评测研究（约 1700 小时数据、1800 次真机盲测 + 数万次仿真评估），回答了"多任务预训练到底有没有用"这个根本问题。2025 年，TRI 的 LBM 被 Boston Dynamics 搬上 Atlas；2026 年，前 TRI LBM 负责人 Russ Tedrake 带队分拆创办 **Walden Robotics**（3 亿美元种子轮），丰田则宣布从 2028 年起在工厂部署约 40 万台机器人。TRI 是"**学术严谨性 + 汽车工业落地**"结合的代表。

---

## 0. 速览卡片

| 维度 | 内容 |
|---|---|
| 机构 | Toyota Research Institute（丰田研究院），2015 年成立，总部加州 Los Altos，另有剑桥（马萨诸塞）等研究点 |
| 关键人物 | Gill Pratt（CEO）；**Russ Tedrake**（MIT 教授，曾任 TRI 机器人研究负责人、LBM 高级副总裁，2026 年分拆创办 Walden Robotics）；Ben Burchfiel、Siyuan Feng、Eric Cousineau 等；**Cheng Chi**（Diffusion Policy 一作，当时为哥伦比亚大学博士生，与 TRI 合作，后创办 Sunday）；**宋舒然 Shuran Song**（Diffusion Policy/UMI 通讯作者）|
| 代表工作 | **Diffusion Policy**（2023）；UMI（2024，与斯坦福/哥伦比亚合作）；**LBM 研究：A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation**（2025.07，arXiv 2507.05331）；**Atlas LBM**（与 Boston Dynamics，2025.08）；Punyo（柔性人形研究平台）|
| 2026 动态 | Walden Robotics（2026.07.15 走出隐身，约 3 亿美元种子轮、估值约 11 亿美元，丰田、NVIDIA、波音等投资，RaaS 模式，丰田为早期生产客户）；丰田计划从 2028 年起在自有及供应商工厂部署约 40 万台机器人（每年最高约 1 万亿日元，以"匠人"示范训练机器人）|
| 本章三篇代表作 | ① Diffusion Policy（2023）② LBM 严谨评测研究（2025）③ 从 LBM 到工业：Atlas LBM、Walden 分拆与丰田 40 万台机器人计划（2025–2026）|
| 身位 | 动作生成范式的奠基者之一；方法论与评测严谨性的标杆；产业化通过 BD、Walden 与丰田工厂实现 |

---

## 1. 故事的开端：一家汽车公司为什么要研究"机器人大脑"？

2015 年，丰田投资 10 亿美元成立 TRI，聘请 DARPA 机器人挑战赛的组织者 Gill Pratt 担任 CEO。最初的重点是自动驾驶与"老龄化社会的家用辅助机器人"——日本社会的人口结构，让丰田比任何车企都更早意识到"机器人照护"的必要性。

TRI 机器人团队的独特之处在于它的**MIT 基因**：Russ Tedrake（MIT 欠驱动机器人与 Drake 仿真器的作者）长期领导 TRI 的机器人研究，带来了严格的控制理论与"**像物理学家一样做实验**"的科学文化。这种文化在 2025 年的 LBM 研究中体现得淋漓尽致。

---

## 2. 代表作一：Diffusion Policy（2023）——把扩散模型变成机器人的"手"

> 论文：Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song, *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*（RSS 2023；IJRR 扩展版 2024）——哥伦比亚大学、TRI、MIT 合作。

### 2.1 要解决的问题：人类示范的"多模态"

假设人类示范"把 T 形块推到目标位置"：有时人从左边绕过去推，有时从右边绕过去。这两种方式都对，但它们的"平均"——从中间直接撞上去——是错的。

传统行为克隆用均方误差回归动作，学到的恰恰是这个"错误的平均"。这就是**动作分布的多模态问题**。此前的解决方案（高斯混合、离散化分类、隐式能量模型 IBC）要么表达能力有限，要么训练不稳定。

### 2.2 核心思想：把动作生成看成"去噪"

Diffusion Policy 把扩散模型（当时在图像生成中刚刚大获成功）用于动作：

- **训练**：对示范中的动作序列逐步加噪声，训练网络 ε_θ(O, A^k, k) 在给定观察 O 的条件下预测所加的噪声；
- **推理**：从纯高斯噪声出发，以观察为条件，迭代 K 步去噪（训练用 DDPM 约 100 步，推理用 DDIM 约 10 步加速），得到动作序列。

扩散模型天然能表达任意复杂的多模态分布——它不会"平均"两种方式，而是在每次采样时"选择"其中一种。

### 2.3 三个关键工程设计

1. **动作序列预测 + 滚动时域控制（receding horizon）**：
   - 观察时域 T_o = 2（用最近 2 帧观察）；
   - 预测时域 T_p = 16（一次预测未来 16 步动作）；
   - 执行时域 T_a = 8（执行前 8 步后重新规划）。
   这与 ACT 的动作分块思想异曲同工：预测序列带来时间一致性，部分执行带来闭环反应能力。
2. **视觉条件化而非联合建模**：观察只作为条件（通过 FiLM 注入），不参与去噪，因此视觉编码只需算一次，推理更快。
3. **两种网络骨干**：
   - **CNN 版**：1D 时间卷积 U-Net + FiLM 条件化，稳定、调参少，是大多数任务的默认选择；
   - **Transformer 版**：时间序列 Transformer，对高频、需要精确动作变化的任务更好，但更难调。
   视觉编码器采用 ResNet-18（用空间 softmax 替代全局平均池化、用 GroupNorm 替代 BatchNorm），端到端训练。

另一个重要发现：**位置控制（输出目标位姿）比速度控制更适合扩散策略**——这与此前许多工作的经验相反。

### 2.4 结果

- 在 4 个基准（Robomimic、Push-T、Block Pushing、Franka Kitchen）的 12 个任务上，相比此前最好方法**平均提升约 46.9%**；
- 真机任务：Push-T、翻转杯子、倒酱汁并用勺子抹匀（涉及液体与可形变物体的连续操作）等，表现稳定。

### 2.5 影响：动作生成范式的奠基

| 后续工作 | 与 Diffusion Policy 的关系 |
|---|---|
| Octo（2024）| 扩散动作头 + Transformer 主干 |
| UMI（2024）| 用 Diffusion Policy 作为策略 |
| π0（2024）| 把扩散换成更简洁的流匹配，动作专家与 VLM 共享注意力 |
| RDT-1B（2024，清华）| 10 亿参数扩散 Transformer |
| GR00T N1（2025）| DiT + 流匹配 |
| Atlas LBM（2025）| 4.5 亿参数 DiT + 流匹配 |
| Xiaomi-Robotics-0、WALL-OSS、GO-1……| 都使用扩散/流匹配动作专家 |

可以说，**2024–2026 年"VLM + 扩散/流匹配动作专家"的主流 VLA 架构，是 RT-2（上半身）与 Diffusion Policy（下半身）的结合**。

---

## 3. 代表作二：大行为模型的严谨评测（2025.07）——"多任务预训练到底有没有用？"

> 论文：*A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation*（TRI LBM 团队，arXiv 2507.05331）

### 3.1 背景：TRI 的"大行为模型"愿景

2023 年 9 月，TRI 宣布用 Diffusion Policy 快速教会机器人 60 多种技能（倒液体、用工具、操作软物体），并提出目标：到 2024 年底达到 1000 种技能，进而训练"**大行为模型（LBM）**"——一个能执行数百种任务、并能快速学会新任务的通用模型。

但到了 2025 年，行业中充斥着各种 VLA 的"惊艳演示"，却很少有人认真回答：**多任务预训练真的比单任务训练好吗？好多少？在统计上可信吗？**

### 3.2 实验设计：像做临床试验一样评测机器人

这篇论文最值得学习的是它的**方法论**：

| 设计 | 内容 |
|---|---|
| 模型 | 扩展 Diffusion Policy 范式的多任务 DiT 策略（LBM），以多路相机图像、本体状态与语言为输入 |
| 数据 | 约 **1700 小时**示范数据（TRI 自有双臂遥操作数据 + 仿真数据 + 公开数据集）|
| 平台 | 双臂操作平台，**真实与仿真**同时评测 |
| 评测规模 | **约 1800 次真实世界试验** + 数万次仿真试验 |
| 盲测 | 真机评测采用**盲法 A/B 测试**：评测者不知道当前运行的是哪个模型 |
| 指标 | 二元成功率 + **基于里程碑的任务完成度评分**（衡量部分进展）|
| 统计 | 使用严格的统计分析（置信区间、假设检验），明确报告结论的统计显著性 |

### 3.3 关键发现

1. **多任务预训练 + 微调 > 单任务训练**：预训练过的 LBM 在微调后，成功率更高、对分布偏移（光照、位置、物体变化）更鲁棒；
2. **数据效率**：二手解读称微调后的 LBM 达到同等表现所需的任务数据约为单任务模型的 **1/3 到 1/5**（论文原文的具体数字有待核实）；
3. **不微调的多任务模型并不总是更好**：直接使用（不经任务微调）的多任务模型，**并不能稳定超过单任务模型**——这是一个对"通用模型开箱即用"叙事的重要冷水；
4. **评测的方差非常大**：同一模型在不同评测批次、不同评测者下的结果差异显著，许多论文中"几个百分点的提升"在统计上可能并不显著。

### 3.4 意义

这篇论文对行业的贡献，与其说是一个模型，不如说是一种**科学态度**：
- 它提醒整个行业，**真机评测的统计噪声远大于大多数论文报告的提升幅度**；
- 它为"预训练-微调"范式提供了迄今最严谨的正面证据，同时诚实地揭示了"开箱即用"的局限；
- 它推动了后续第三方评测平台（RoboArena、RoboChallenge）对"盲测""大样本""统计显著性"的重视。

---

## 4. 代表作三：从 LBM 到工业——Atlas、Walden 与丰田的 40 万台机器人（2025–2026）

### 4.1 Atlas LBM（2025.08）

TRI 与 Boston Dynamics 合作，把 LBM 搬到了全尺寸人形 Atlas 上：4.5 亿参数 DiT + 流匹配、30Hz、语言条件化、单一策略控制全身（详见第 9 章）。这是 LBM 第一次在双足人形上验证"手脚一体"的全身控制。

### 4.2 Walden Robotics（2026.07）

2026 年 7 月 15 日，**Walden Robotics** 走出隐身，宣布完成约 **3 亿美元**种子轮融资，估值约 **11 亿美元**：
- **创始团队**：Russ Tedrake（联合创始人兼 CEO，前 TRI LBM 高级副总裁，MIT 教授）及多位 TRI 核心成员（报道提及 Ben Burchfiel、Siyuan Feng、Adrien Gaidon 等）；
- **投资方**：丰田（丰田汽车、Toyota Invention Partners、Toyota Ventures）、Deviation Capital 领投，NVIDIA、波音等参与；
- **定位**：全栈物理 AI 公司，打造"**在真实工作中持续学习与改进**"的通用机器人；
- **商业模式**：至少初期不直接销售硬件，采用**机器人即服务（RaaS）**；丰田既是股东也是早期生产客户；
- **形态**：报道说法不一——有的描述为"部分人形"的 AI 机器人系列，有的称其刻意不走标准人形路线（如不用腿）。

Walden 的出现意味着 TRI 的 LBM 技术栈正式走向商业化，"持续学习"的定位也与 PI 的 RECAP、Skild 的上下文学习等 2026 年热点一致。

### 4.3 丰田的 40 万台机器人计划（2026.09）

据日经亚洲等 2026 年 9 月报道：
- 丰田估计从 **2028 年起每年最多投入约 1 万亿日元（约 64 亿美元）** 实现全球工厂自动化；
- 在自有工厂与主要供应商工厂中**逐步部署约 40 万台机器人**（约 15 万台在丰田自有工厂，约 25 万台在集团公司及其他设施）；
- 训练方式：**由工人示范，而不是逐个动作编程**——丰田全球约 60 家工厂中有约 1.8 万名被称为"**匠（takumi）**"的资深技工，他们将提供示范、评估与专业知识；
- 中心化学习的技能数据将在全球工厂间共享，避免各地重复训练。

报道指出，该计划使用的机器人程序以 TRI 的扩散策略 LBM 框架为核心之一（此关联在部分报道中有提及，但并非所有来源都明确确认）。

**分析**：如果这一计划落地，丰田将成为全球**最大规模的"示范学习"工业应用**之一——它把 Diffusion Policy/LBM 从实验室带到了真正的大规模制造，并以"匠人示范"这一极具日本特色的方式解决数据来源问题。

---

## 5. 技术路线总结

| 阶段 | 时间 | 核心贡献 |
|---|---|---|
| 动作生成范式 | 2023 | Diffusion Policy：多模态动作分布、动作序列、滚动时域 |
| 数据采集 | 2024 | UMI（与斯坦福/哥大合作）：无本体、硬件无关数据采集 |
| 规模化与科学评测 | 2025 | LBM 严谨评测：预训练-微调有效，但"开箱即用"有限；评测统计噪声极大 |
| 人形验证 | 2025 | Atlas LBM：单策略全身控制 |
| 产业化 | 2026 | Walden 分拆（RaaS、持续学习）；丰田 40 万台机器人计划 |

TRI 的路线有一个贯穿始终的主题：**"做可验证的科学"**。它不追求最大的模型或最炫的演示，而追求"这个结论在统计上站得住吗"。

---

## 6. 横向对比与身位

| 维度 | TRI | PI | DeepMind |
|---|---|---|---|
| 动作生成贡献 | Diffusion Policy（奠基）| 流匹配动作专家、FAST | 动作 token 化（RT-2）|
| 评测严谨性 | **最高**（盲测、统计显著性）| 高 | 高（安全评测系统）|
| VLM 语义 | 较弱（LBM 不以大 VLM 为核心）| 强 | 最强 |
| 产业化 | BD、Walden、丰田工厂 | API/授权 | 平台授权 |

**身位判断**：
- **学术影响力**：Diffusion Policy 是近五年机器人学习领域被引用最多的论文之一，TRI 在动作生成范式上的贡献是奠基性的；
- **模型前沿性**：在"大 VLM + 大数据"的规模竞赛中，TRI 不是领先者；
- **产业化路径**：通过 BD（人形）、Walden（RaaS）、丰田（自有工厂）三条线落地，路径清晰、客户确定。

---

## 7. 未来展望与风险

**展望**：
1. **丰田工厂作为数据飞轮**：40 万台机器人 + 1.8 万名匠人示范，可能形成全球最大的制造业示范学习数据集；
2. **Walden 的持续学习**：如果 Walden 实现"部署中持续改进"，将是 LBM 范式的重要升级；
3. **严谨评测的推广**：TRI 的评测方法论可能成为行业标准。

**风险**：
1. **人才分流**：核心团队分拆到 Walden，TRI 自身的研究连续性面临考验；
2. **语义能力短板**：在需要复杂语言理解与推理的场景，LBM 需要与大 VLM（如 Gemini）结合；
3. **40 万台计划的不确定性**：成本数字为估算而非承诺，部署将随工厂改造逐步推进，可靠性与标准化挑战巨大。

---

## 8. 延伸阅读与核验来源

**三篇代表作**
- Diffusion Policy：Chi et al., RSS 2023（arXiv 2303.04137）；IJRR 2024 扩展版
- LBM 评测：[arXiv 2507.05331](https://arxiv.org/html/2507.05331v1)；[数据集（Dryad）](https://datadryad.org/dataset/doi:10.5061/dryad.xd2547dxc)
- 产业化：[Boston Dynamics LBM 博客](https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/)；[AI Business：Walden 走出隐身](https://aibusiness.com/robotics/walden-robotics-emerges-from-stealth)；[Boston Globe：Walden Robotics](https://www.bostonglobe.com/2026/07/15/business/walden-robotics-cambridge-toyota-humanoid/)；[Automotive World：丰田计划从 2028 年部署 40 万台工厂机器人](https://www.automotiveworld.com/news/toyota-plans-to-deploy-400000-factory-robots-from-2028/)；[TechRepublic：丰田 40 万台机器人、64 亿美元现代化](https://www.techrepublic.com/article/news-toyota-400000-factory-robots-apac-japan/)

**其他**
- [The Robot Report：TRI 预训练 LBM 加速机器人学习](https://www.therobotreport.com/tri-pretrained-large-behavior-models-accelerate-robot-learning/)
- [TechTimes：丰田押注 40 万台机器人捕获匠人技能](https://www.techtimes.com/articles/327745/20260919/toyota-bets-400000-robots-can-capture-craftspeople-skills-that-took-decades-build.htm)

> **核验说明**：Diffusion Policy 的技术细节与 46.9% 平均提升来自原论文；LBM 研究的 1700 小时、约 1800 次真机试验来自论文摘要与数据集页面，"3–5 倍数据效率"为二手解读；Walden 的融资、估值、投资方与 RaaS 模式来自多家媒体（投资方表述略有差异）；丰田 40 万台计划来自日经亚洲等报道，"以 TRI LBM 为核心"的关联并非所有来源都确认。
