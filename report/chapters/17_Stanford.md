# 第 17 章 斯坦福大学：机器人大脑的"黄埔军校"

> **一句话定位**：斯坦福是全球机器人学习领域人才与思想最密集的源头之一——Physical Intelligence 的联合创始人 Chelsea Finn、Sunday 的创始人 Tony Zhao 与 Cheng Chi、UMI/Diffusion Policy 的通讯作者宋舒然、BEHAVIOR 的主导者李飞飞与吴佳俊都在这里。它为行业贡献了第一个被广泛使用的**开源 7B VLA（OpenVLA）**、第一个"**看人即学**"的低成本人形全身影子系统（HumanPlus），以及家庭机器人领域规模最大的**仿真基准与年度挑战赛（BEHAVIOR）**。斯坦福的角色不是"做产品"，而是"**定义问题、提供基线、输出人才**"。

---

## 0. 速览卡片

| 维度 | 内容 |
|---|---|
| 核心实验室 | IRIS（Chelsea Finn）、REAL（宋舒然 Shuran Song）、SVL（李飞飞、吴佳俊）、ILIAD（Dorsa Sadigh）、Wetzstein 计算成像组等 |
| 代表工作（全集）| ALOHA/ACT（2023）、Mobile ALOHA（2024）、VoxPoser（2023）、ReKep（2024）、**OpenVLA（2024）**、OpenVLA-OFT（2025）、DROID（2024，多机构）、UMI（2024，与哥大/TRI）、**HumanPlus（2024）**、**BEHAVIOR-1K 与 BEHAVIOR Challenge（2024–2026）**、斯坦福 AI Index |
| 人才输出 | Physical Intelligence（Chelsea Finn 等）、Sunday（Tony Zhao、Cheng Chi）、World Labs（李飞飞）、多家中国具身公司的创始人/核心成员 |
| 本章三篇代表作 | ① OpenVLA（2024）与 OpenVLA-OFT（2025）② HumanPlus：人形影子跟随与模仿（2024）③ BEHAVIOR：家庭机器人的"ImageNet 时刻"尝试（2024–2026）|
| 身位 | 学术思想与人才的顶级源头；开源基线的主要提供者之一 |

---

## 1. 故事的开端：从 ImageNet 到 BEHAVIOR

2009 年，李飞飞团队发布了 ImageNet，十多年后深度学习革命由此点燃。斯坦福机器人学者们心中一直有一个问题：**机器人领域的 ImageNet 在哪里？**

答案不是一个，而是三条并行的探索：
- **开源模型**：让任何人都能下载、微调一个强大的机器人基础模型（→ OpenVLA）；
- **低成本数据**：让任何人都能用便宜的设备采集数据（→ ALOHA、UMI、HumanPlus）；
- **标准化基准**：让所有方法在同一套任务上比较（→ BEHAVIOR）。

---

## 2. 代表作一：OpenVLA（2024）与 OpenVLA-OFT（2025）——开源 VLA 的"第一块基石"

> 论文：Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, ..., Percy Liang, Sergey Levine, Chelsea Finn 等，*OpenVLA: An Open-Source Vision-Language-Action Model*（CoRL 2024，arXiv 2406.09246）——斯坦福、Berkeley、TRI、Google DeepMind、PI 等合作；*Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success*（OpenVLA-OFT，2025）。

### 2.1 为什么需要 OpenVLA

2023 年的 RT-2 证明了 VLA 的威力，但它**闭源**、**55B 参数**、**只能在 Google 的云 TPU 上运行**。学术界无法复现、无法微调、无法研究。OpenVLA 的目标是：**用公开数据、公开 VLM，训练一个可下载、可微调、能在单张消费级 GPU 上运行的 VLA。**

### 2.2 架构

```
        图像（224×224）
            │
   ┌────────┴────────┐
   ▼                 ▼
DINOv2 编码器      SigLIP 编码器      ← 两个视觉编码器的特征按通道拼接
（空间细节）       （语义理解）
   └────────┬────────┘
            ▼
     小型 MLP 投影器
            ▼
   Llama 2 7B 语言模型  ← 输入：视觉 token + 语言指令
            ▼
   7 个动作 token（每维 256 桶离散化）→ 反量化为末端位移、旋转、夹爪
```

- **底座**：Prismatic-7B VLM（斯坦福 Siddharth Karamcheti 等人的工作）——其研究发现"DINOv2 + SigLIP 双编码器融合"比单一 CLIP/SigLIP 在空间推理上更好；
- **动作表示**：沿用 RT-2 的离散化——每个动作维度分成 256 个桶，用 Llama 词表中最少使用的 256 个 token 表示；
- **总参数**：约 70 亿。

### 2.3 数据与训练

- 数据：从 Open X-Embodiment 中精选约 **97 万条**机器人示范（经过清洗与重新加权）；
- 训练：在 64 张 A100 上训练约 14 天；
- 视觉编码器在 VLA 训练中**解冻微调**（论文发现这对机器人任务很重要，与 VLM 训练的常规做法相反）。

### 2.4 结果

- 在 29 个任务、多种机器人上，OpenVLA 的**绝对成功率比 55B 的 RT-2-X 高 16.5%**，而参数量只有其约 1/8；
- 在新任务上用 LoRA 等参数高效方法微调，可在消费级 GPU 上完成，且显著优于从零训练的 Diffusion Policy（在多任务、语言条件化场景中）。

### 2.5 OpenVLA-OFT（2025）：从 5Hz 到约 130Hz

OpenVLA 的最大问题是**慢**——自回归逐 token 生成，约 5Hz。2025 年的 **OFT（Optimized Fine-Tuning）** 配方做了三项关键改动：
1. **并行解码**：一次前向同时输出所有动作 token，不再逐个生成；
2. **动作分块**：一次输出多步动作（借鉴 ACT）；
3. **连续动作 + L1 回归**：用 MLP 动作头直接回归连续动作，替代离散 token。

结果：推理吞吐提升到约 **130Hz** 级别（相比自回归约 5Hz），在 LIBERO 仿真基准上平均成功率达到当时最高水平（约 97%），并在双臂 ALOHA 真机上完成灵巧任务。OFT 说明：**VLA 的"慢"不是 VLM 主干的宿命，而是动作解码方式的问题。**

### 2.6 意义

OpenVLA 是 2024–2025 年**被使用最广泛的开源 VLA 之一**：大量学术论文以它为基线或起点，大量中国公司的早期 VLA 实验也从它开始。它与 Octo（Berkeley）、π0（PI 开源）共同构成了开源 VLA 的"三大基线"。

---

## 3. 代表作二：HumanPlus（2024）——"人做什么，人形就学什么"

> 论文：Zipeng Fu, Qingqing Zhao, Qi Wu, Gordon Wetzstein, Chelsea Finn, *HumanPlus: Humanoid Shadowing and Imitation from Humans*（CoRL 2024）

### 3.1 要解决的问题

人形机器人的遥操作极其困难：需要动捕服、VR、外骨骼等昂贵设备，且操作员需要大量训练。能否**只用一个普通 RGB 摄像头**，让人形实时"影子跟随"操作员的全身动作，并以此采集数据？

### 3.2 两阶段系统

**阶段一：影子跟随（Shadowing）**
- 在仿真中，用约 **40 小时**人类动作数据（AMASS）训练一个**人形影子 Transformer（Humanoid Shadowing Transformer）**——一个底层全身控制策略，输入目标人体姿态，输出人形关节指令；
- 部署时，用一个 RGB 摄像头实时估计操作员的**身体姿态与手部姿态**（现成的人体/手部姿态估计模型），重定向后作为影子 Transformer 的目标；
- 结果：操作员在摄像头前做什么，人形就实时做什么——走路、蹲下、拿东西、打拳。
- 硬件：宇树 H1 + 每臂一只因时（Inspire-Robots）RH56DFX 灵巧手，共 **33 自由度**（19 自由度身体 + 两只 6 自由度手 + 两个 1 自由度手腕）；头部装两个 RGB 网络摄像头构成**双目视觉**；据报道整机改装成本约 10.8 万美元。

**阶段二：模仿（Imitation）**
- 用影子跟随采集任务示范（每个任务最多约 **40 条**——论文强调"40 条示范即可学会双目感知 + 高自由度控制"）；
- 训练一个**人形模仿 Transformer（Humanoid Imitation Transformer）**：输入人形自身的双目第一视角图像与本体状态，输出目标姿态（再交给影子 Transformer 执行）；
- 结果：人形可以**自主**完成穿鞋站起并行走、从仓库货架卸货、叠运动衫、整理物品、打字、与另一台机器人打招呼等任务，成功率在 60%–100% 之间。

### 3.3 意义

HumanPlus 在 2024 年就提出了今天行业的标准分层：**"上层视觉模仿策略输出目标姿态 + 底层全身追踪控制器执行"**——这与 Figure Helix 02（S1 + S0）、NVIDIA（GR00T + SONIC）、Amazon FAR（ResMimic）的架构同构。它也证明了**低成本 RGB 摄像头即可完成人形全身遥操作采集**。第一作者 Zipeng Fu（Mobile ALOHA 共同一作）后来继续在人形与移动操作领域深耕。

---

## 4. 代表作三：BEHAVIOR——家庭机器人的"ImageNet 时刻"尝试（2024–2026）

> 论文：Chengshu Li, Ruohan Zhang, ..., Jiajun Wu, Li Fei-Fei 等，*BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation*（CoRL 2022 / 2024 扩展）；BEHAVIOR Challenge 2025（NeurIPS 2025）与 2026。

### 4.1 BEHAVIOR-1K：以人为中心定义任务

BEHAVIOR 的独特之处在于**任务从哪里来**：团队调查了大量美国普通人，询问"**你最希望机器人帮你做的日常家务是什么**"，据此选出 **1000 种日常活动**（如收拾餐桌、做烤面包、整理房间、洗碗），并用一种逻辑语言（BDDL）精确定义每个活动的初始状态与目标状态。

配套的仿真环境 **OmniGibson**（基于 NVIDIA Omniverse/Isaac Sim）支持：
- 50 个完整的室内场景、上万个带物理与语义属性的物体；
- 刚体、可形变物体（布料）、流体、温度变化（烹饪）、开关状态等复杂物理与状态；
- 照片级渲染。

### 4.2 BEHAVIOR Challenge 2025：一次"冷水实验"

2025 年，斯坦福在 NeurIPS 2025 上举办了第一届 BEHAVIOR Challenge：
- **50 个**完整的家务任务（从做吐司到整理房间）；
- 统一的仿真机器人平台（双臂移动操作机器人），以及约 **1 万条**遥操作专家示范（逾 1200 小时）作为训练数据；
- **结果**：获胜方案在 50 个任务上的 q-score（含部分进展）约 **26%**；而**完整完成整个任务的成功率仅约 12.4%**。参赛方案中有不少基于 openpi（π0/π0.5）微调。

斯坦福 2026 年 AI 指数报告也引用了类似的结论：**人形机器人在家务任务上的失败率约为 88%**。

### 4.3 BEHAVIOR Challenge 2026

第二届挑战赛提交截止于 2026 年 10 月 16 日，获胜者将于 2026 年 11 月 4 日公布（截至本报告撰写日 2026 年 10 月 30 日尚未公布结果）。

### 4.4 意义

BEHAVIOR 的价值在于**它给行业泼了一盆必要的冷水**：在各家公司纷纷发布"机器人做家务"演示视频的 2025–2026 年，一个标准化、长时程、完整任务成功率的评测显示，**最先进的方法在真实复杂度的家务上，完整成功率仍只有约 12%**。这与 Figure Helix 2.5 在 30 户真实家庭中约 56% 的结果（任务更少、更聚焦）共同勾勒出 2026 年家务机器人能力的真实区间。

---

## 5. 技术路线总结：斯坦福的三条线

| 线索 | 代表工作 | 对行业的贡献 |
|---|---|---|
| 模型 | OpenVLA / OFT | 开源 VLA 基线；证明"动作解码方式"决定速度 |
| 数据 | ALOHA、Mobile ALOHA、UMI、HumanPlus、DROID | 低成本、可规模化的数据采集范式 |
| 评测 | BEHAVIOR-1K / Challenge、AI Index | 以人为中心的任务定义；真实能力的"温度计" |
| 推理 | VoxPoser、ReKep（李飞飞组）| 用 VLM/LLM 生成 3D 价值图与关键点约束，零样本操作 |

---

## 6. 横向对比与身位

| 学术机构 | 最突出贡献 | 与产业的连接 |
|---|---|---|
| **斯坦福** | OpenVLA、ALOHA/ACT、UMI、HumanPlus、BEHAVIOR | PI、Sunday、World Labs |
| UC Berkeley | Octo、HIL-SERL、人形 RL、BridgeData | PI、Covariant/Amazon |
| CMU | 人形全身控制（H2O/ASAP/BFM-Zero）、Skild 一脉 | Skild、NVIDIA GEAR、OpenAI |
| UCSD | 人形遥操作与人类数据（TeleVision、HAT）| 多家人形公司 |
| 清华 | RDT、VPP、Spirit 一脉 | 星动纪元、千寻、星海图 |

**身位**：斯坦福在"**开源基线 + 低成本数据 + 标准化评测**"三个维度上都处于全球学术界最前列，是机器人大脑领域最重要的思想源头之一。

---

## 7. 未来展望

1. **BEHAVIOR 2026 结果**：11 月公布的第二届结果将显示家务能力一年间的进步幅度；
2. **人才持续外溢**：斯坦福培养的博士生仍将是全球具身创业公司的核心来源；
3. **学术与产业的分工**：在算力与数据规模上，学术界已难以与 PI、DeepMind、Generalist 竞争，斯坦福的优势将更多体现在"问题定义、评测与新范式探索"上。

---

## 8. 延伸阅读与核验来源

**三篇代表作**
- OpenVLA：[arXiv 2406.09246](https://arxiv.org/pdf/2406.09246)；[PMLR 版本](https://proceedings.mlr.press/v270/kim25c.html)；OpenVLA-OFT（arXiv 2502.19645）
- HumanPlus：Fu et al., CoRL 2024（arXiv 2406.10454）
- BEHAVIOR：BEHAVIOR-1K（arXiv 2403.09227）；[斯坦福 HAI：BEHAVIOR 挑战赛为家庭机器人指明方向](https://hai.stanford.edu/news/behavior-challenge-charts-the-way-forward-for-domestic-robotics)；[Openpi Comet：2025 BEHAVIOR 挑战赛方案（arXiv 2512.10071）](https://arxiv.org/pdf/2512.10071v1.pdf)

**其他**
- [eWeek：人形机器人炒作遭遇 88% 家务失败率](https://www.eweek.com/de/news/humanoid-robots-fail-88-percent-household-tasks/)
- [Emergent Mind：OpenVLA 主题汇总](https://www.emergentmind.com/topics/openvla)

> **核验说明**：OpenVLA 的架构、97 万条数据、比 RT-2-X 高 16.5% 来自原论文；OFT 约 130Hz 与 LIBERO 约 97% 来自 OFT 论文及二手汇总（具体数字以论文为准）；HumanPlus 的 40 小时 AMASS、最多 40 条示范、33 自由度来自原论文；BEHAVIOR Challenge 2025 的 26% q-score 与 12.4% 完整成功率来自挑战赛官方与参赛方报告；"88% 失败率"为媒体对斯坦福 AI Index 2026 的转述。补充核验：HumanPlus 的硬件构成（19+6+6+1+1=33 自由度、Inspire RH56DFX、双目网络摄像头）、40 条示范、60%–100% 成功率来自论文，改装成本约 10.8 万美元来自媒体报道。
