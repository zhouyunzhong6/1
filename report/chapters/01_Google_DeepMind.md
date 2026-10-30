# 第 1 章 Google DeepMind：从"把动作当成语言"到"全身智能"

> **一句话定位**：机器人基础模型（VLA）范式的发明者与定义者。DeepMind 的打法是"以最强的通用多模态大模型（Gemini）为母体，把机器人当作 Gemini 的又一个输出模态"，用"推理大脑（ER）+ 动作模型（VLA）+ 端侧模型（On-Device）"的三件套，向所有本体厂商输出"安卓式"的机器人大脑。截至 2026 年 10 月，它在"通用性 + 推理能力 + 生态开放度"三项上仍处在全球第一梯队的最前列。

---

## 0. 速览卡片

| 维度 | 内容 |
|---|---|
| 机构 | Google DeepMind（Alphabet 旗下；2023 年 Google Brain 与 DeepMind 合并） |
| 机器人负责人 | Carolina Parada（机器人负责人），Vikas Sindhwani、Kanishka Rao 等长期核心；早期代表人物 Vincent Vanhoucke、Karol Hausman、Brian Ichter、Chelsea Finn（已离开创办 PI 的多位成员原属 Google） |
| 代表模型谱系 | QT-Opt → BC-Z → SayCan → RT-1 → PaLM-E → RT-2 → RT-X → ALOHA Unleashed → Gemini Robotics 1.0 / ER → On-Device → Gemini Robotics 1.5 / ER 1.5 → ER 1.6 → **Gemini Robotics 2 / ER 2 / On-Device 2（2026.07）** |
| 技术路线 | 大 VLM 母体 + 动作解码；"思考型 VLA"；"分层智能体"（ER 做规划/调用工具，VLA 做执行）；跨本体 Motion Transfer |
| 开放程度 | ER 系列通过 Gemini API / AI Studio 对外开放；VLA 与 On-Device 仅限可信测试伙伴（Trusted Tester）；早期 RT-1、OXE 数据集开源 |
| 本体伙伴 | Apptronik（Apollo / Apollo 2，战略合作）、Boston Dynamics（2026 CES 宣布合作，新一代 Atlas）、Agile Robots、Agility Robotics、Enchanted Tools、Franka、Trossen、Dexmate、SO-101 等 |
| 本报告选取的三篇代表作 | ① RT-2（2023）② Gemini Robotics 1.0 → 1.5（2025）③ Gemini Robotics 2 家族（2026） |
| 身位评估 | 通用推理与跨本体泛化：全球第一档；全身控制/灵巧手：2026 年才补齐，仍在追赶 Figure/特斯拉的"整机端到端"；商业化：走平台授权路线，营收贡献尚小 |

---

## 1. 故事的开端：Google 的机器人十年

要理解 DeepMind 今天的机器人大脑，必须回到 2016 年的 Google 山景城——那里有一排被员工戏称为"手臂农场（arm farm）"的 KUKA 机械臂，日夜不停地尝试抓取散落在箱子里的杂物。

**第一幕：暴力规模化的强化学习（2016–2019）。** Sergey Levine 等人在 Google Brain 做的"手眼协调"抓取实验，以及 2018 年的 **QT-Opt**（7 台机械臂、约 58 万次真实抓取尝试、离线 Q 学习），证明了"规模化的真机数据 + 深度学习"能让抓取成功率达到 96% 左右。但这条路的代价是：一个技能就要几十万次试错，无法扩展到成百上千种任务。

**第二幕：语言进入机器人（2021–2022）。** 2022 年 4 月的 **SayCan**（"Do As I Can, Not As I Say"）是第一个让大语言模型（PaLM）给真实机器人做任务规划的重要工作：LLM 负责回答"这一步说得通吗"（say），每个底层技能的价值函数负责回答"这一步做得到吗"（can），两者相乘决定下一步执行哪个技能。同期的 Inner Monologue、Code as Policies 把"语言 = 机器人的思维载体"推向主流。但这时的 LLM 只是一个"上层调度器"，底层动作仍由独立训练的小模型完成。

**第三幕：Transformer 吃下动作（2022.12）。** **RT-1**（Robotics Transformer 1）是转折点：Everyday Robots 的 13 台移动机械臂在 17 个月里采集了约 13 万条示范、覆盖 700+ 条指令，训练出一个约 3500 万参数的 Transformer。它的输入是 6 帧历史图像 + 一句语言指令（经 Universal Sentence Encoder 编码、通过 FiLM 注入 EfficientNet-B3），每张图被 TokenLearner 压成 8 个 token，最后输出 11 维动作（7 维手臂 + 3 维底盘 + 1 维模式切换），每一维离散成 256 个桶，以约 3 Hz 运行。RT-1 在已见任务上达到 97% 成功率，并首次显示出"数据多样性比数据量更影响泛化"。

**第四幕：动作就是语言（2023.07）——RT-2。** 这是本章第一篇代表作，也是整个 VLA 时代的起点，下一节详述。

**第五幕：开放与联邦（2023.10）——Open X-Embodiment。** DeepMind 牵头 21 家机构（后扩到 30+）整合出 Open X-Embodiment（OXE）数据集：22 种机器人本体、100 多万条轨迹、527 种技能。在其上训练的 RT-1-X 在小数据实验室的任务上平均比各家原方法高约 50%，RT-2-X 在"涌现技能"上约为 RT-2 的 3 倍。OXE 后来成为 Octo、OpenVLA、π0 等几乎所有开源 VLA 的预训练底座——**这是 DeepMind 对行业最大的"公共品"贡献之一**。

**第六幕：低成本双臂与灵巧操作（2024）。** 斯坦福的 ALOHA 低成本双臂平台被 DeepMind 升级为 ALOHA 2，并做出了 **ALOHA Unleashed**（用扩散策略 + 大规模示范学会系鞋带、挂衬衫）和 **DemoStart**（仿真中用少量示范 + 自动课程强化学习训练灵巧手）。ALOHA 2 车队后来成为 Gemini Robotics 的主力数据采集平台。

**第七幕：Gemini 化（2025–2026）。** 2025 年 3 月，DeepMind 用 Gemini 2.0 为底座推出 **Gemini Robotics** 与 **Gemini Robotics-ER**；6 月推出端侧版；9 月推出带"思考"与跨本体 Motion Transfer 的 1.5；2026 年 4 月推出 ER 1.6；7 月 30 日推出 **Gemini Robotics 2 家族**，第一次把"全身控制 + 五指灵巧手 + 多机协作"纳入一个统一的模型家族。

与此同时，Alphabet 内部的另一条线——2023 年被关闭的 Everyday Robots、以及工业软件公司 Intrinsic——让 DeepMind 在 2023 年后彻底转向"不自己造整机、只做大脑"的平台路线。这一点决定了它和特斯拉、Figure 的根本区别。

---

## 2. 代表作一：RT-2（2023）——"把动作当成另一种语言"

> 论文：*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*（Brohan, Brown, ... , Zitkovich 等，2023.07，CoRL 2023）

### 2.1 它要解决的问题

2023 年上半年，行业面对一个尴尬局面：一边是在互联网数据上训练、懂得"什么是恐龙""哪个是能量饮料"的视觉语言模型（VLM，例如 PaLI-X、PaLM-E）；另一边是只在几万条机器人示范上训练、对新物体"两眼一抹黑"的控制模型（RT-1）。SayCan 式的"LLM 做规划 + 小模型做技能"只能让语义知识影响"选哪个技能"，却无法让它影响"手怎么动"。

RT-2 的提问非常直接：**能不能让同一个大模型，既回答"图里是什么"，又直接输出"手该往哪动"？**

### 2.2 核心思想：动作 token 化

RT-2 的关键洞察只有一句话：**如果把机器人的动作写成一串文本 token，那么动作预测就只是 VLM 的又一种"看图说话"。**

具体做法：

1. **动作空间定义**：每一步动作由 8 个数组成——1 个"是否终止"的离散标志、末端执行器 6 自由度的位移与旋转增量（Δx, Δy, Δz, Δroll, Δpitch, Δyaw）、1 个夹爪开合量。
2. **离散化**：除终止位外，每个连续维度均匀切成 256 个桶，于是一步动作就是 8 个整数，例如 "1 128 91 241 5 101 127 217"。
3. **映射到词表**：
   - 对 **PaLI-X**，其分词器本身给 0–1000 的每个整数都分配了独立 token，所以直接用数字 token 表示动作桶；
   - 对 **PaLM-E**，没有现成的整数 token，就把词表里 256 个最少使用的 token"征用"为动作符号。
4. **输出约束**：在机器人任务上解码时，把采样空间限制在合法的动作 token 内，保证输出一定可执行。

这样，"VLA（Vision-Language-Action）"这个词第一次被正式提出：一个模型、一个词表、一个自回归解码器，同时承担视觉问答与机器人控制。

### 2.3 架构

RT-2 并没有发明新网络，它是两个现成 VLM 的"再训练"：

| 变体 | 底座 | 视觉编码器 | 语言主干 | 参数量 |
|---|---|---|---|---|
| RT-2-PaLI-X | PaLI-X | ViT-22B（较小版本用更小 ViT） | UL2 式编码器-解码器 | 5B 与 55B |
| RT-2-PaLM-E | PaLM-E | ViT-4B | PaLM 仅解码器 | 12B |

输入：当前一帧机器人相机图像 + 文本提示（"Q: what action should the robot take to [task]? A:"）。输出：动作 token 串，解码后反量化为连续控制指令。

### 2.4 数据与训练：协同微调（co-fine-tuning）

RT-2 最容易被忽视、但最关键的工程选择是 **co-fine-tuning**：不是只用机器人数据微调 VLM，而是把**原始的网络视觉语言数据（WebLI 等图文对、VQA、图像描述）与 RT-1 的机器人示范数据混在同一个批次里一起训练**，并对机器人数据做上采样。

为什么？因为只用机器人数据微调，会让模型迅速"忘掉"它在互联网上学到的概念（灾难性遗忘）；而混合训练让模型在学"怎么动"的同时，保留"这是什么、那是什么"的语义知识。论文的消融实验显示：co-fine-tuning 的泛化效果显著好于纯机器人数据微调，而从头训练（不用预训练 VLM）的效果最差。

机器人数据来自 RT-1 数据集（13 台机器人、17 个月、厨房办公环境）。

### 2.5 推理部署：大模型怎么跑在机器人上

55B 参数的模型显然塞不进机器人。RT-2 的做法是**云端推理**：模型部署在多 TPU 的云服务上，机器人通过网络发送图像、接收动作。实测 55B 版本能达到 1–3 Hz，5B 版本约 5 Hz。这个频率对于"移动到某处并抓取"这类准静态任务足够，但对于需要连续力控、动态平衡的任务远远不够——这正是两年后 π0、Helix 等工作要解决的问题。

### 2.6 实验结果

DeepMind 做了超过 6000 次真机评估试验：

- **已见任务**：RT-2 与 RT-1 相当（说明换成大模型没有损失原有能力）；
- **未见场景的泛化**（新物体、新背景、新环境）：RT-2 的成功率约 **62%**，RT-1 约 **32%**，接近翻倍；
- **涌现能力**：RT-2 能完成训练数据中从未出现过的语义任务，例如"把香蕉放到 2+1 的和上"（符号理解）、"捡起已经灭绝的动物"（从一堆玩具中选出恐龙）、"把可乐罐移到泰勒·斯威夫特的照片旁"（人物识别）、"拿一个可以当临时锤子用的东西"（选石头）。这些语义知识全部来自网络预训练，而不是机器人数据。
- **思维链变体**：基于 PaLM-E 的 RT-2 还试验了"先输出一句自然语言计划、再输出动作"的 chain-of-thought 版本，例如面对"我累了，给我拿点喝的"，先说出"Plan: pick rxbar chocolate / energy drink"，再执行。这一思想在 2025 年被 Gemini Robotics 1.5 发展为"思考型 VLA"。

### 2.7 局限

RT-2 论文本身坦率列出了局限：

1. **不会学新动作**：泛化体现在"语义"层面——认得新物体、理解新指令——但机器人的**物理技能集合并没有扩大**，它只会做 RT-1 数据里出现过的那些动作（抓、放、推、开抽屉等）。
2. **太慢太贵**：55B 模型 1–3 Hz、依赖云端，无法做高频灵巧操作。
3. **离散化损失精度**：256 桶的均匀量化对精细操作不友好；自回归逐 token 生成也难以输出高频、平滑的动作序列。

### 2.8 为什么它是"代表作"

RT-2 给出了此后三年整个行业的"第一性公式"：

> **VLA = 预训练 VLM（语义与常识）+ 机器人动作数据（技能）+ 共同训练（防遗忘）**

OpenVLA（2024，斯坦福等）几乎是 RT-2 的开源复刻（Llama 2 + DINOv2/SigLIP + 256 桶动作 token）；π0 把离散动作 token 换成了流匹配（flow matching）连续动作专家；GR00T、Helix 把单一大模型拆成"慢系统 + 快系统"；但它们都继承了 RT-2 的基本判断：**机器人大脑的语义能力，应当从互联网规模的视觉语言预训练中"继承"，而不是从机器人数据中"重新学"。**

---

## 3. 代表作二：Gemini Robotics 1.0 → 1.5（2025）——把机器人变成 Gemini 的一个"模态"

> 技术报告：*Gemini Robotics: Bringing AI into the Physical World*（2025.03，arXiv 2503.20020）；*Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer*（2025.09/10，arXiv 2510.03342）；以及 *Gemini Robotics On-Device*（2025.06 博客与 SDK）

### 3.1 背景：两年之间发生了什么

从 RT-2 到 Gemini Robotics 之间的两年，行业发生了三件事：

- **Physical Intelligence 的 π0（2024.10）**证明了"VLM + 流匹配动作专家"能以 50 Hz 输出连续动作块，完成叠衣服这类高灵巧任务；
- **Figure 的 Helix（2025.02）**证明了"慢系统 7B VLM + 快系统 80M 策略"的双系统能在人形上跑起来；
- **DeepMind 自己的 Gemini 2.0** 在空间理解（2D 指点、3D 框、多视角对应）上远超同代模型。

DeepMind 的结论是：**机器人大脑的瓶颈有两个，一是"具身推理（Embodied Reasoning）"——看懂三维物理世界；二是"灵巧执行"——把理解转成高频平滑的动作。** 于是 Gemini Robotics 从一开始就是**两个模型**。

### 3.2 Gemini Robotics-ER：会"空间思考"的 Gemini

ER（Embodied Reasoning）模型是在 Gemini 2.0 Flash 基础上强化空间与物理推理能力得到的 VLM，它**不直接输出电机指令**，而是输出机器人可用的"中间表征"：

- **2D 指点（pointing）**：输出图像中某个物体或部位的像素坐标（"指出杯子的把手"）；
- **2D/3D 检测框**：从单张或多张图像中给出物体的三维包围盒；
- **轨迹预测**：画出从当前位置到目标的末端运动轨迹；
- **抓取预测**：给出抓取位姿；
- **多视角对应**：在不同相机视角间匹配同一个点；
- **代码生成控制**：在零样本模式下，ER 可以生成调用机器人 API 的代码直接完成任务；在少样本模式下，通过上下文中的几条示范做 in-context learning。

为了衡量这种能力，DeepMind 发布了 **ERQA（Embodied Reasoning Question Answering）** 基准，约 400 道多选题，覆盖空间推理、轨迹推理、动作推理、状态估计、多视角推理等。

### 3.3 Gemini Robotics（VLA）：云端大脑 + 本地小脑

1.0 版 VLA 的系统架构是一个非常务实的"两段式"：

```
相机图像 + 语言指令
        │
        ▼
┌──────────────────────────────┐
│ 云端：Gemini Robotics Backbone │  ← 由 Gemini Robotics-ER 蒸馏而来，
│ （优化后查询-响应延迟 < 160ms）  │    负责理解与"潜在动作意图"
└──────────────────────────────┘
        │（潜在表征 / 动作意图）
        ▼
┌──────────────────────────────┐
│ 机器人本地：Action Decoder      │  ← 小型解码器，补偿网络延迟，
│                                │    输出高频动作块（action chunk）
└──────────────────────────────┘
        │
        ▼
   端到端延迟约 250ms，有效控制频率约 50Hz（动作块平滑执行）
```

这一设计与 Figure 的 Helix、NVIDIA 的 GR00T 在思想上同源（慢-快分离），但 DeepMind 的特点是：**慢系统就是 Gemini 本身（蒸馏版），因而直接继承了 Gemini 的世界知识、多语言理解和空间推理**。

**数据**：DeepMind 在 12 个月里用一个 ALOHA 2 双臂机器人车队采集了"数千小时"的专家遥操作示范，覆盖数千种任务；再加上网络图文、视频、代码数据和 ER 数据混合训练。

**三大能力维度（DeepMind 自己的框架）**：
1. **通用性（Generality）**：在新物体、新指令、新环境、新视觉干扰下保持成功率，论文报告其在泛化基准上的平均表现是此前最强 VLA 的两倍以上；
2. **交互性（Interactivity）**：理解自然语言（含多语言）对话式指令，能在执行中被打断、改口、实时调整；
3. **灵巧性（Dexterity）**：叠折纸狐狸、装午餐盒（拉链袋 + 放面包 + 盖盖）、打包行李等长时程、多阶段任务。对于最难的长程任务，用 2000–5000 条专门数据做"专精化（specialization）"后能显著提升成功率；对新任务，约 100 条示范即可快速适配。

**跨本体**：同一模型被微调到双臂 Franka 和 Apptronik 的 Apollo 人形上，显示出一定的本体迁移能力。

### 3.4 Gemini Robotics On-Device（2025.06）：把大脑塞进机器人

云端方案在工厂、仓库、家庭等网络不稳定的场景是硬伤。2025 年 6 月，DeepMind 发布 **On-Device** 版本：一个针对机载算力优化的 VLA，**完全本地运行**，并首次开放 **Gemini Robotics SDK**（含 MuJoCo 仿真评测）。官方称只需 **50–100 条示范**即可把它适配到新任务，并演示了适配到双臂 Franka FR3 与 Apollo 人形。这是 DeepMind 第一次让外部开发者亲手微调它的机器人模型——平台化的关键一步。

### 3.5 Gemini Robotics 1.5（2025.09）：会"先想后做"、能"跨身迁移"

1.5 版本是 DeepMind 机器人路线的成熟形态，包含两个模型：

**(1) Gemini Robotics 1.5（VLA）**——两大创新：

- **Thinking VLA（思考型 VLA）**：模型在输出动作之前，先用自然语言生成"内部思考轨迹"（thinking traces），例如"我需要先把蓝色毛衣分到深色衣物堆，所以先抓起它，移动到右边的篮子……"。思考可以把一个长指令拆成若干个短的、可执行的子步骤，并在执行中根据视觉反馈修正。这本质上是把 RT-2 时代的"chain-of-thought"变成了模型的原生能力，并且**思考与动作交错（interleave）**进行，而不是一次性规划完。DeepMind 报告"思考模式"显著提升了多步骤任务的进度完成率，同时让机器人的行为对人类更可解释。
- **Motion Transfer（MT，运动迁移）**：传统 VLA 换一个本体就要重新收集大量数据。1.5 提出了一种新的模型架构与训练配方，使单一模型能从 **ALOHA（双臂桌面）、双臂 Franka（工业臂）、Apollo（人形）** 这三类形态差异巨大的机器人的异构数据中联合学习，抽取跨本体共享的"运动与物理"规律。效果是：在 ALOHA 上学会的技能，**零样本**就能在 Apollo 人形或 Franka 上执行（报告中称之为跨本体技能迁移）。这对于"数据稀缺"的人形机器人意义重大——人形可以"借用"成本低得多的双臂桌面平台的数据。

**(2) Gemini Robotics-ER 1.5（推理/编排模型）**——它被定位为"机器人的高层大脑"：

- 在 **15 个具身推理学术基准**（Point-Bench、RefSpatial、RoboSpatial-Pointing、Where2Place、BLINK、CV-Bench、ERQA、EmbSpatial、MindCube、RoboSpatial-VQA、SAT 等）上取得最高的综合分数，DeepMind 博客的对比图显示其综合得分高于 GPT-5 与 ER 1.0；
- **可以原生调用工具**：例如为了完成"按照我所在城市的垃圾分类规则把这些垃圾分好"，ER 1.5 会先调用 Google 搜索查当地规则，再把分类任务拆成子任务，逐条以自然语言指令下发给 VLA 执行；
- **思考预算可调**：开发者可在延迟与准确度之间权衡；
- **安全**：在 DeepMind 自建的 ASIMOV 语义安全基准上取得最好成绩。

**智能体化的系统架构**：

```
       用户："把这些衣服按颜色分类，深色放左边篮子"
                       │
                       ▼
     ┌─────────────────────────────────────┐
     │ Gemini Robotics-ER 1.5（编排者）       │
     │ · 理解场景、拆解任务、调用工具/搜索     │
     │ · 监控进度、判断成功与否                │
     └─────────────────────────────────────┘
                       │ 逐条自然语言子指令
                       ▼
     ┌─────────────────────────────────────┐
     │ Gemini Robotics 1.5（VLA，带"思考"）    │
     │ · 把子指令转成短步骤 → 输出动作块        │
     │ · Motion Transfer：同一权重驱动多本体    │
     └─────────────────────────────────────┘
                       │
                       ▼
              ALOHA / Franka / Apollo
```

**开放策略**：ER 1.5 通过 Gemini API（Google AI Studio）向所有开发者开放；VLA 1.5 仅向精选伙伴开放。这形成了"推理层开放、动作层受控"的分层商业模式。

### 3.6 局限与争议

- **数据主要来自 ALOHA 2 双臂平台**，在真正的人形全身任务（行走 + 操作）上，1.5 时代的展示仍以上半身操作为主；
- **云-端分离依赖网络**，On-Device 版能力弱于云端版；
- **评测多为自家任务集**，第三方难以复现（VLA 权重不开源），与 π0/π0.5 开源 openpi 形成鲜明对比；
- **灵巧手**：1.5 时代主要是平行夹爪，与 Figure/特斯拉强调的五指灵巧手路线差距明显。

---

## 4. 代表作三：Gemini Robotics 2 家族（2026.07）——"全身智能"

> 官方博客：*Gemini Robotics 2 brings whole body intelligence to robots*（2026-07-30）；配套 *Gemini Robotics 2 安全技术报告*；前序：*Gemini Robotics-ER 1.6*（2026-04-14）

### 4.1 过渡：ER 1.6（2026.04）的"智能体视觉"

在 2 代发布前三个半月，DeepMind 先发布了 **Gemini Robotics-ER 1.6**。它的关键词是 **agentic vision（智能体式视觉）**：模型不再"看一眼就回答"，而是把视觉推理和**代码执行**结合起来——例如读一个工业仪表盘时，模型会写代码对图像做裁剪、放大、几何测量，再给出精确读数。这种"会用工具看图"的能力，直接服务于工厂巡检、仪表读取、精密定位等工业场景，也为 2 代 ER 的"任务进度追踪"打下基础。

### 4.2 Gemini Robotics 2 家族的三个成员

| 模型 | 定位 | 关键能力 | 开放程度（截至 2026.10） |
|---|---|---|---|
| **Gemini Robotics 2（VLA）** | 动作模型 | **全身控制**：把行走、下蹲、弯腰、抓取作为一个统一问题求解；支持五指灵巧手 | 早期合作伙伴 |
| **Gemini Robotics-ER 2** | 推理/编排模型 | 用机器人相机画面**追踪任务进度**；**把长任务拆分给多台机器人协作**；在共享空间中协调多机 | Gemini API / AI Studio 公开预览；企业平台私有预览 |
| **Gemini Robotics On-Device 2** | 端侧模型 | 本地运行，用"几小时数据、不到 200 条示范"即可适配新本体 | 早期合作伙伴 |

### 4.3 全身控制：从"桌面双臂"走向"人形全身"

2 代最重要的变化是**动作空间从上半身扩展到全身**。展示平台包括：

- **Apptronik Apollo 2**，配 **SharpaWave**（22 自由度五指灵巧手）与 **Inspire（因时）** 五指手；
- 双臂 **Franka Duo** + Robotiq 夹爪；
- Dexmate、Trossen 以及 Hugging Face 生态里的低成本 **SO-101** 机械臂。

公开报道的官方数字（来自 DeepMind 7 月 30 日发布内容的二手转述，hand 配置细节各报道略有出入）：

**Apollo 2 全身拾取任务成功率**

| 物体位置 | 成功率 |
|---|---|
| 货架上 | 76.3% |
| 桌面上 | 68.4% |
| 地面上（需深蹲/弯腰 + 平衡） | 45.7% |

**五指灵巧手任务成功率**

| 任务 | 成功率 |
|---|---|
| 拧下灯泡 | 92% |
| 拧上灯泡 | 36% |
| 系垃圾袋 | 44% |
| 使用簸箕 | 32% |
| 封自封袋 | 40% |

**解读**：这组数据非常"诚实"，也非常有信息量：

1. **高度越低、全身协调需求越大，成功率越低**（货架 76% → 地面 46%），说明"边平衡边操作"的全身协同仍是硬骨头；
2. **"拆"比"装"容易得多**（拧下 92% vs 拧上 36%）：装配需要对准、旋入、力反馈判断，这正是当前 VLA 缺乏精细力控与触觉闭环的体现；
3. **可形变物体（垃圾袋、自封袋）在 40% 上下**，距离商用（通常要求 95%+）还有很长的路。

DeepMind 选择公开这些"不好看"的数字，本身就是一种姿态：它们把 Gemini Robotics 定位为**通用研究平台与开发者平台**，而不是某个场景的"交付产品"。

### 4.4 ER 2：从"单机大脑"到"机器人团队的调度员"

ER 2 的两个新能力，标志着 DeepMind 把"多智能体编排"作为下一阶段的重点：

- **任务进度追踪（progress tracking）**：ER 2 能持续观看机器人相机画面，判断"任务完成了几成""刚才那一步是否成功"，这相当于为机器人提供了一个**通用的奖励模型/成功检测器**——这对于后续的强化学习自我提升（类似 PI 的 π*0.6 RECAP 中的价值函数）至关重要；
- **多机分工**：把一个长任务拆成若干并行子任务，分配给共享空间中的多台机器人（可以是不同形态），并协调它们避免冲突。

配套的 **ASIMOV-Agentic** 安全基准评测三件事：ER 是否能**拒绝不安全的 VLA 工具调用**、能否**预判任务是否可行**、在不确定时能否**主动请求人类介入**。DeepMind 同时发布了 2 代的**安全技术报告**，涵盖安全约束遵循和人机近距离测试——在全行业中，DeepMind 是把"机器人安全评测"做得最系统的一家。

### 4.5 On-Device 2：生态扩张的"楔子"

"不到 200 条示范、几小时数据适配一个新本体"——这句话的商业含义是：**任何一家本体厂商，都可以以很低的成本把 Gemini 大脑装进自己的机器人**。学术界已出现基于 On-Device 做人形专精化微调的工作（例如 2026 年 7 月的 CLIFT：用非侵入式闭环迭代微调把 Gemini Robotics On-Device 变成人形专用模型），说明 On-Device 正在成为一个被外部研究者使用的"底座"。

### 4.6 伙伴生态：Apptronik 与 Boston Dynamics

- **Apptronik**：自 2024 年底与 DeepMind 结成战略合作，Apollo/Apollo 2 是 Gemini Robotics 的"官方人形"；Apptronik 自身专注硬件与量产，大脑主要依赖 Gemini。
- **Agile Robots**：据报道，2026 年 3 月这家德国工业机器人公司宣布在其工业平台中集成 Gemini Robotics 的 VLA 模型——成为继 Apptronik、Boston Dynamics 之后又一家深度接入 Gemini Robotics 的硬件厂商；在 Gemini Robotics 2 发布时，VLA 与 On-Device 模型仍仅向 Apptronik、Boston Dynamics、Franka、Agile Robots 等合作伙伴开放（有单一来源称 ER 2 基于 Gemini 3.5 架构，未经官方确认）。
- **Boston Dynamics**：2026 年 1 月 5 日 CES 上，在现代汽车集团发布会中宣布与 DeepMind 建立 AI 合作，**新一代量产版电动 Atlas 将以 Gemini Robotics 为大脑研究对象**，首批量产 Atlas 交付给现代汽车与 Google DeepMind，计划在现代汽车工厂测试。这是一次"回归"——Google 在 2013 年收购、2017 年把 Boston Dynamics 卖给软银。对 BD 来说，这意味着它在 TRI 的 LBM 合作之外，又接入了全球最强的通用推理大脑。

---

## 5. 技术路线总结：DeepMind 大脑的五条"设计哲学"

1. **母体论**：机器人大脑不应从零训练，而是"长在"最强的通用多模态模型上。Gemini 每一次升级（2.0 → 2.5 → 3），机器人大脑都"免费"获得更强的世界知识与推理能力。
2. **分层论**：推理（ER）与执行（VLA）分开训练、分开发布、分开迭代。ER 可以做大做慢、调用工具；VLA 必须做快做稳。两者通过自然语言接口连接——这是一种"可解释的分层"，与 Figure Helix 用潜在向量连接两层形成对照。
3. **思考论**：动作之前先用语言思考，让 VLA 能处理多步骤任务并具备可解释性。这条路线把 LLM 领域"推理模型（o1/R1）"的成功迁移到了机器人。
4. **跨本体论**：Motion Transfer 让一套权重驱动多种形态，把数据便宜的平台（ALOHA 双臂）的能力迁移给数据昂贵的平台（人形）。
5. **平台论**：不造整机，做"机器人界的 Android/Gemini API"。ER 开放给所有人，VLA/On-Device 分级授权。

---

## 6. 横向对比与身位

| 对比维度 | Google DeepMind | Physical Intelligence | NVIDIA GR00T | Figure Helix | 特斯拉 Optimus |
|---|---|---|---|---|---|
| 母体 | Gemini（闭源，最强之一） | PaliGemma/Gemma 系（开源 VLM 起步） | Eagle/Cosmos Reason（自研）→ N2 改用世界模型 | 自研 7B 级 VLM | 自研（FSD 体系） |
| 层次结构 | ER（推理）+ VLA（执行）+ On-Device | 单一 VLA + 高层语言子任务预测（π0.5） | System 2 + System 1 → 世界动作模型 | S2 + S1 + S0（三层） | 端到端单网络（公开信息有限） |
| 层间接口 | 自然语言 | 自然语言（π0.5 的高层推理） | 潜在 token | 潜在向量 | 不详 |
| 动作生成 | 动作解码器（细节未完全公开） | 流匹配动作专家 | DiT 流匹配 → 视频-动作联合生成 | 视觉运动 Transformer | 不详 |
| 全身控制 | 2026.07 才在 2 代纳入 | 主要为移动操作臂 | 依赖 WBC（HOVER/SONIC 等） | S0 1kHz 全身控制（2026.01） | 自研 |
| 开放程度 | ER 开放 API；VLA 受限 | openpi 开源权重 | 开源权重（N1 系列） | 闭源 | 闭源 |
| 安全评测 | 最系统（ASIMOV 系列） | 一般 | 一般 | 不公开 | 不公开 |

**身位判断**：

- **领先之处**：具身推理（ER 系列在 15+ 基准上领先）、自然语言交互与多语言、跨本体迁移、安全体系、生态开放（API）。在"机器人通用常识"这一项上，2026 年仍无人能撼动。
- **相对落后之处**：①**高频全身控制与灵巧操作**——2 代的灵巧手成功率（32%–44%）显示其在接触丰富任务上仍落后于专门优化的系统；②**真实部署规模**——相比特斯拉/Figure 的工厂部署、Generalist 的"数十万小时真实交互数据"、PI 的大量真实家庭测试，DeepMind 的数据飞轮主要依赖自有车队与伙伴；③**开源影响力**——开发者社区的心智被 openpi、GR00T、LeRobot 占据。
- **一句话**：DeepMind 是"机器人大脑的上层建筑"最强者，但"肌肉记忆"仍需伙伴和更多真机数据来补。

---

## 7. 未来展望与风险

**展望（2026Q4–2027）**：

1. **与 Gemini 3.x/4 同步升级**：DeepMind 的节奏表明，机器人模型会跟随 Gemini 主线迭代（2.0 → 2.5 → 3）。下一代很可能引入更长的视频记忆、更强的 3D 世界表征。
2. **世界模型合流**：DeepMind 拥有 Genie 3（实时可交互世界模型）与 Veo（视频生成）两张王牌。将世界模型用于策略评估、合成数据与"想象中训练"，几乎是确定方向——这与 NVIDIA（Cosmos/DreamZero）、1X（世界模型）殊途同归。
3. **Atlas + Gemini 的工厂验证**：2026–2027 年现代汽车工厂里的 Atlas 将是 Gemini Robotics 的第一块"工业试金石"，其结果将决定 DeepMind 平台路线能否在严苛工业场景里成立。
4. **从 ER 到"机器人 Agent 平台"**：ER 2 的多机编排 + 任务追踪 + 工具调用，正在把 Gemini 变成"机器人团队的操作系统"。

**风险**：

- **平台 vs 整机的路线风险**：如果特斯拉/Figure 的"整机端到端 + 自有数据飞轮"跑出显著优势，平台型大脑可能被边缘化为"上层调度器"；
- **数据风险**：自有 ALOHA 车队的数据规模与真实世界多样性，可能不及 Generalist（50 万小时+）、它石（百万小时人类数据）等以数据为核心的新玩家；
- **组织风险**：Google 机器人团队历史上多次人员流失（PI 的大量创始成员、Skild 合作者等都出自 Google 机器人体系），人才持续外流是 DeepMind 机器人业务的长期隐忧。

---

## 8. 延伸阅读与核验来源

**三篇代表作**
- RT-2：Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023（arXiv 2307.15818）
- Gemini Robotics：*Gemini Robotics: Bringing AI into the Physical World*（arXiv 2503.20020）；*Gemini Robotics 1.5*（arXiv 2510.03342）
- Gemini Robotics 2：DeepMind 博客 [Gemini Robotics 2 brings whole body intelligence to robots](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)（2026-07-30）

**前序与相关**
- RT-1（arXiv 2212.06817）；Open X-Embodiment（arXiv 2310.08864）；SayCan（arXiv 2204.01691）；PaLM-E（arXiv 2303.03378）；ALOHA Unleashed（2024）
- [Gemini Robotics-ER 1.6 博客](https://deepmind.google/blog/gemini-robotics-er-1-6/)（2026-04）；[Gemini Robotics ER 2 介绍](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)
- [Gemini Robotics 1.5 博客](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)（2025-09-25）

**2026 动态核验来源**
- [SiliconANGLE：Gemini Robotics 2 系列](https://siliconangle.com/2026/07/30/google-deepmind-debuts-gemini-robotics-2-model-series-humanoid-robots/)
- [The Robot Report：Gemini Robotics 2 enables full body control](https://www.therobotreport.com/google-deepmind-says-gemini-robotics-2-enables-full-body-control/)
- [SiliconANGLE：ER 1.6](https://siliconangle.com/2026/04/15/deepmind-launches-gemini-robotics-er-1-6-meet-precise-physical-ai-demands/)
- [TechCrunch：Boston Dynamics 新一代 Atlas 将有 DeepMind 基因](https://techcrunch.com/2026/01/05/boston-dynamicss-next-gen-humanoid-robot-will-have-google-deepmind-dna/)；[Boston Dynamics 官方博客](https://bostondynamics.com/blog/boston-dynamics-google-deepmind-form-new-ai-partnership/)
- [packetnebula：Gemini Robotics 2 官方基准解读](https://packetnebula.com/articles/gemini-robotics-2-google-own-benchmarks/)（灵巧手与全身拾取成功率数字来源，为对官方发布内容的二手转述）

> **核验说明**：Gemini Robotics 2 的具体成功率数字来自对 DeepMind 7 月 30 日发布内容的二手转述，部分报道对"Apollo 2 所配手型"表述不一；Gemini Robotics 2 尚无完整架构论文公开，本章对其内部结构不做臆测。RT-2、RT-1、OXE、Gemini Robotics 1.0/1.5 的数字来自原论文与官方博客。
