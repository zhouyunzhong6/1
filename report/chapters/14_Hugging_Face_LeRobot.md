# 第 14 章 Hugging Face（LeRobot）：机器人领域的"Transformers 库"与开源民主化

> **一句话定位**：Hugging Face 不追求"最强的机器人大脑"，而是追求"**让每个人都能训练机器人大脑**"。它的 LeRobot 项目（2024.05 起）为机器人学习提供了统一的数据格式、模型实现、训练与部署工具，以及低成本的开源硬件（SO-100/SO-101 机械臂、LeKiwi、HopeJR 人形、Reachy Mini）；它的 SmolVLA（2025.06）证明了一个 4.5 亿参数的小 VLA 在社区数据上也能达到大模型的性能；2026 年的 LeRobot v0.6 则把世界模型策略、奖励模型、NVIDIA GR00T N1.7 等前沿能力纳入开源工具链。它与 Dexmal（原力灵机）共同发起的 RoboChallenge，成为 2026 年最被引用的真机评测平台之一。它是人形大脑生态的"**基础设施与公共品提供者**"。

---

## 0. 速览卡片

| 维度 | 内容 |
|---|---|
| 机构 | Hugging Face（纽约/巴黎），LeRobot 团队 |
| 关键人物 | Rémi Cadène（LeRobot 创始负责人，前特斯拉 Optimus 团队）；Thomas Wolf（HF 联合创始人，力推开源机器人）；LeRobot 研究团队（SmolVLA 作者群）|
| 收购 | 2025 年 4 月收购法国 Pollen Robotics（Reachy 系列人形机器人）|
| 开源硬件 | Koch v1.1、**SO-100 / SO-101**（与 The Robot Studio 合作的低成本机械臂）、LeKiwi（移动底盘）、**HopeJR**（低成本开源人形，2025 年发布）、**Reachy Mini**（桌面机器人，2025.07）、Reachy 2 |
| 软件 | LeRobot 库（数据集格式 LeRobotDataset、策略实现、训练/评估/部署）；Hub 上数千个社区机器人数据集 |
| 模型 | 自研 **SmolVLA**（2025.06）；集成 ACT、Diffusion Policy、TDMPC、VQ-BeT、π0/π0-FAST/π0.5（PyTorch 移植）、GR00T N1.5 → **N1.7**、MolmoAct2、EO-1 等 |
| 评测 | 与原力灵机等共同推出 **RoboChallenge**（2025 年，真机评测平台，Table30 等榜单）|
| 本章三篇代表作 | ① LeRobot：开源机器人学习库与数据标准（2024）② SmolVLA：社区数据训练的高效小 VLA（2025）③ LeRobot v0.6：世界模型策略、奖励模型与前沿模型整合（2026）|
| 身位 | 开源机器人生态的中枢；不以模型能力竞争，以生态与标准取胜 |

---

## 1. 故事的开端：一位特斯拉 Optimus 工程师的"开源梦"

2024 年初，Rémi Cadène 从特斯拉 Optimus 团队加入 Hugging Face。他看到的问题是：**机器人学习领域的碎片化极其严重**——每个实验室有自己的数据格式、自己的训练代码、自己的硬件，一篇论文的方法很难在另一个实验室复现；而真正好用的开源机器人硬件动辄数万美元，学生与爱好者无法参与。

Hugging Face 在 NLP 领域用 Transformers 库解决过同样的问题：统一的模型接口 + 统一的 Hub + 开放的社区，让"下载一个模型、跑一行代码"成为现实。Cadène 和 Thomas Wolf 的目标，就是**在机器人领域复刻这个成功**。

2024 年 5 月，LeRobot 正式发布。

---

## 2. 代表作一：LeRobot（2024）——统一数据、模型与硬件

### 2.1 三层设计

**（1）数据层：LeRobotDataset**
- 统一格式：表格化的状态/动作数据（Parquet）+ 视频压缩的图像流（MP4），既节省存储又便于流式加载；
- 直接托管在 Hugging Face Hub 上，任何人都可以一行代码下载、可视化、训练；
- 提供数据可视化工具，可在浏览器中逐帧查看示范。

到 2025–2026 年，Hub 上已有数千个社区上传的机器人数据集，主要来自 SO-100/SO-101 等低成本平台——这构成了一个**去中心化、众包式的机器人数据生态**。

**（2）模型层：统一的策略实现**
- 首批集成 ACT、Diffusion Policy、TDMPC、VQ-BeT 等经典模仿学习/强化学习方法；
- 2025 年起集成 π0、π0-FAST（从 JAX 移植到 PyTorch）、自研 SmolVLA，后续加入 π0.5、GR00T N1.5/N1.7 等；
- 统一的训练、评估、部署接口，让研究者可以在同一数据集上公平比较不同方法。

**（3）硬件层：低成本开源机器人**
- **SO-100 / SO-101**：与 The Robot Studio 合作设计的 6 自由度低成本机械臂，采用总线舵机，物料成本约百美元级，可 3D 打印，支持"主从臂"遥操作采集（继承 ALOHA 的 puppeteering 思想）；
- **LeKiwi**：给 SO 臂加上全向移动底盘；
- **HopeJR**：低成本开源人形（2025 年发布，宣称成本约数千美元）；
- **Reachy Mini**：桌面级可编程机器人（2025 年 7 月，价格数百美元）；
- 收购 Pollen Robotics 后获得 Reachy 2 人形平台。

### 2.2 意义

LeRobot 对行业最大的贡献是**降低门槛**：一名学生用几百美元的 SO-101、一台带 GPU 的电脑，就能采集数据、训练 ACT/π0/SmolVLA、部署到真机。这让全球成千上万的个人与小团队进入机器人学习领域，也为"社区数据训练基础模型"提供了数据来源。

---

## 3. 代表作二：SmolVLA（2025.06）——社区数据训练的高效小 VLA

> 论文：*SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*（Hugging Face，arXiv 2506.01844）

### 3.1 要解决的问题

2025 年的主流 VLA 都很大：π0 约 33 亿参数、OpenVLA 70 亿、GR00T N1 22 亿。它们需要高端 GPU 训练与推理，且主要在机构自有的大规模数据上预训练。问题是：
1. **能否用一个足够小的模型，在消费级 GPU 甚至 CPU 上运行？**
2. **能否只用社区贡献的、杂乱的开源数据，训练出有竞争力的 VLA？**

### 3.2 架构：剪裁过的 VLM + 流匹配动作专家

| 组件 | 设计 |
|---|---|
| VLM 主干 | SmolVLM2（约 5 亿参数级的小型 VLM）；**只使用 LLM 的前一半层（约 16 层）的特征**——借鉴 GR00T"取中间层"的发现，既提速又不损性能 |
| 视觉 token | 每帧图像压缩到较少的 token（约 64 个），不做图像切块 |
| 动作专家 | 约 1 亿参数，**流匹配**生成动作块 |
| 注意力 | 动作专家中**交错使用交叉注意力（关注 VLM 特征）与因果自注意力（动作 token 之间）** |
| 总规模 | 约 **4.5 亿参数** |
| 训练 | VLM 主干冻结，只训练动作专家 |

### 3.3 数据：481 个社区数据集

- 预训练数据来自 Hugging Face Hub 上约 **481 个社区公开数据集**（不同报道为 481 或 487 个），约 **2.3 万条示范、约 1000 万帧**，主要采集自 SO-100 机械臂；
- 社区数据的一个大问题是**任务描述混乱**（有的写"pick"，有的写"task1"，有的为空）。SmolVLA 用一个 VLM 自动**重写/标准化任务描述**；
- 相机视角命名也不统一（"top""front""cam_1"），同样做了标准化映射。

### 3.4 异步推理

SmolVLA 提出了一个实用的**异步推理栈**：把"机器人客户端"与"策略服务器"解耦——机器人在执行当前动作块的同时，服务器已经在根据最新观察计算下一个动作块。报道称这使任务完成时间缩短约 30%，或在固定时间内完成约 2 倍的任务。（这与 PI 的实时分块 RTC 解决的是同一类问题。）

### 3.5 结果

- 在仿真基准 LIBERO 与 Meta-World 上，SmolVLA 的表现与规模大得多的 π0（33 亿）、OpenVLA（70 亿）相当或更好；
- 在真实 SO-100/SO-101 任务上，有预训练时平均成功率约 **78.3%**，无预训练时约 **51.7%**——说明社区数据预训练确实有效；
- 可在消费级 GPU 上训练，甚至在 CPU 上推理。

### 3.6 意义

SmolVLA 是"**小模型 + 社区数据**"路线的代表。它证明了：VLA 的能力并不完全依赖规模；对于低成本机械臂上的桌面任务，一个精心设计的小模型配合多样的社区数据，足以达到实用水平。它也为边缘端部署（低功耗机器人、教育机器人）提供了现实选项。

---

## 4. 代表作三：LeRobot v0.6（2026.07）——世界模型策略、奖励模型与前沿模型整合

> 来源：LeRobot v0.6.0 发布说明（2026.07）；Hugging Face × NVIDIA 博客 *NVIDIA Isaac Teleop and GR00T 1.7 Open VLA Model Available in LeRobot*；v0.6.1（2026.08.03）。

### 4.1 v0.6 的新增内容

| 新能力 | 意义 |
|---|---|
| **世界模型策略（world-model policies）** | 把 2026 年最热的"世界模型 → 策略"路线纳入开源工具链 |
| **奖励模型 API** | 为真机强化学习（如 RECAP 类方法）、成功检测、数据筛选提供标准接口 |
| **仿真基准** | 统一的仿真评估 |
| **rollout CLI** | 标准化的部署与评估命令行 |
| **深度数据** | 支持 RGB-D 数据 |
| **HF Jobs 训练** | 在 Hugging Face 云端直接训练 |
| **新 VLA 模型** | **GR00T N1.7**（替代 N1.5，且 N1.5 不再支持）、MolmoAct2（AI2）、EO-1、EVO1、Multitask DiT 等 |
| **NVIDIA Isaac Teleop** | 开源遥操作工具 |

### 4.2 与 NVIDIA 的合作

2026 年 7 月，NVIDIA 与 Hugging Face 宣布把 **Isaac Teleop** 与 **GR00T 1.7**（"面向通用人形机器人的最新开放、可商用 VLA 基础模型"）引入 LeRobot，并计划未来整合 **Cosmos 3**。这意味着：
- 开源社区可以直接在 LeRobot 中使用 NVIDIA 最新的人形基础模型；
- NVIDIA 借助 Hugging Face 的社区触达数十万开发者；
- LeRobot 成为"开源人形大脑"的事实标准入口之一。

### 4.3 RoboChallenge：开源评测的"中立场"

2025 年，Hugging Face 与原力灵机（Dexmal，见第 45 章）等共同推出 **RoboChallenge**——一个**真机评测平台**：研究者提交模型，平台在标准化的真实机器人与任务上运行评测，公开排行榜（如 Table30 榜单）。2026 年 1 月千寻 Spirit v1.5 在其上超过 π0.5 登顶，引发广泛关注；后来也出现过榜单成绩因评测数据异常被撤销的争议（见第 41 章）。RoboChallenge 的意义在于：**它把"机器人模型谁更强"从各家自说自话，变成了可以在同一真机环境中比较的问题**——尽管其可信度仍在建设中。

---

## 5. 技术路线总结

Hugging Face 的"路线"不是一种模型架构，而是一种**生态哲学**：

```
  低成本开源硬件（SO-101、HopeJR、Reachy）
           ↓ 让更多人采集数据
  统一数据格式 + Hub（LeRobotDataset）
           ↓ 让数据可以汇聚
  统一模型实现（ACT → π0 → SmolVLA → GR00T N1.7 → 世界模型策略）
           ↓ 让方法可以复现与比较
  开放评测（RoboChallenge）
           ↓ 让能力可以被客观衡量
  社区反馈 → 更多数据、更多模型
```

---

## 6. 横向对比与身位

| 维度 | Hugging Face LeRobot | NVIDIA Isaac/GR00T | PI openpi | 智元/宇树开源 |
|---|---|---|---|---|
| 定位 | 中立的开源生态中枢 | 平台 + 硬件销售 | 模型开源 | 本体公司的开源生态 |
| 数据 | 社区众包（数千数据集）| 合成 + 合作伙伴 | 自有 | 自有大规模数据集 |
| 硬件 | 低成本开源 | 不做本体（参考设计）| 不做 | 自有本体 |
| 模型 | 小模型 + 集成第三方 | GR00T | π 系列 | GO-1/UnifoLM |

**身位判断**：Hugging Face 是全球开源机器人生态的"**中枢节点**"——它不必拥有最强模型，因为所有最强的开源模型最终都会出现在 LeRobot 里。这种"生态位"在 NLP 领域已被证明极具价值。

---

## 7. 未来展望与风险

**展望**：
1. **社区数据的规模效应**：随着 SO-101 等低成本硬件的普及，社区数据可能达到数万小时级别，为"开源基础模型"提供燃料；
2. **开源人形**：HopeJR、Reachy 2 等平台若成熟，可能让人形机器人研究像今天的桌面机械臂一样普及；
3. **评测标准化**：RoboChallenge 若建立起足够的公信力，可能成为机器人领域的"LMSYS Arena"。

**风险**：
1. **数据质量**：社区数据质量参差、本体单一（以低成本桌面臂为主），对人形全身控制的帮助有限；
2. **商业模式**：开源生态的变现路径（云训练、硬件销售、企业服务）仍在探索；
3. **评测公信力**：真机评测的标准化、防作弊机制仍需完善。

---

## 8. 延伸阅读与核验来源

**三篇代表作**
- LeRobot：[GitHub huggingface/lerobot](https://github.com/huggingface/lerobot)；[LeRobot 发布索引](https://releases.sh/hugging-face/lerobot)
- SmolVLA：[arXiv 2506.01844（Emergent Mind）](https://www.emergentmind.com/papers/2506.01844)；[Pure AI 报道](https://pureai.com/Articles/2025/06/10/Hugging-Face-Releases-SmolVLA.aspx)
- LeRobot v0.6：[The AI Feed：LeRobot 0.6 闭合机器人学习闭环](https://theaifeed.com/en/ai-news/hugging-face-lerobot-06-robot-learning-loop-2026/)；[Hugging Face 博客：Isaac Teleop 与 GR00T 1.7 进入 LeRobot](https://huggingface.co/blog/nvidia/nvidia-isaac-teleop-and-gr00t17-in-lerobot)

**核验来源**
- [NVIDIA 博客：NVIDIA 与 Hugging Face 为 LeRobot 带来新模型与框架](https://blogs.nvidia.com/blog/hugging-face-lerobot-models-frameworks-open-robotics/)
- [AIwire：NVIDIA 与 Hugging Face 为开源机器人社区带来新模型](https://www.hpcwire.com/aiwire/2026/07/08/nvidia-and-hugging-face-bring-new-models-and-frameworks-to-lerobot-for-the-open-robotics-community/)
- [Wikipedia：Hugging Face](https://en.wikipedia.org/wiki/Hugging_Face)

> **核验说明**：SmolVLA 的架构（约 4.5 亿参数、约 1 亿动作专家、前约 16 层特征、冻结 VLM）、数据（481/487 个数据集、约 1000 万帧）与成功率（78.3% vs 51.7%）来自论文及多篇一致的二手解读，数据集数量各来源略有出入；LeRobot v0.6 内容来自发布说明的报道；HopeJR、Reachy Mini 价格为 2025 年发布时公开信息的概数。
