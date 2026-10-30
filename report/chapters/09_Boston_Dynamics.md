# 第 9 章 Boston Dynamics：运动控制之王的"大脑移植手术"

> **一句话定位**：Boston Dynamics（波士顿动力）是人形机器人"身体"的传奇——液压 Atlas 的后空翻与跑酷定义了公众对人形机器人的想象。但在"大脑"时代，它做了一个关键决定：**放弃手写的模型预测控制（MPC）作为唯一路线，转向学习——与丰田研究院（TRI）合作训练"大行为模型（LBM）"，与 RAI 研究所合作做强化学习运动生成，并在 2026 年 CES 上宣布与 Google DeepMind 合作，把 Gemini Robotics 装进量产版电动 Atlas。** 它是"顶级身体 + 外部顶级大脑"组合的代表，背后是现代汽车集团"到 2028 年部署数万台 Atlas"的工业野心。

---

## 0. 速览卡片

| 维度 | 内容 |
|---|---|
| 成立 | 1992 年，从 MIT Leg Lab 分拆（创始人 Marc Raibert）|
| 股东变迁 | Google（2013）→ 软银（2017）→ **现代汽车集团（2020/2021，控股）** |
| 产品 | Spot（四足，商用）、Stretch（仓储搬运）、Atlas（液压版 2013–2024 退役 → **电动版 2024.04 发布，2026 年 CES 发布量产版**）|
| 大脑合作方 | **丰田研究院 TRI**（2024.10 起，LBM）；**RAI 研究所**（Marc Raibert 创办的 AI 研究所，RL）；**NVIDIA**（Jetson Thor、Isaac Lab）；**Google DeepMind**（2026.01 起，Gemini Robotics）|
| 量产计划 | 2026 年产量全部分配给现代汽车与 Google DeepMind；现代汽车计划到 2028 年建立年产约 3 万台机器人的体系、部署数万台 Atlas；Atlas 计划 2028 年起进入现代佐治亚 Metaplant 产线；2026 年 9 月在佐治亚开设机器人 Metaplant 应用中心（2027 年将建约 10 倍规模的设施）|
| 本章三篇代表作 | ① Atlas 大行为模型（与 TRI，2025.08）② 基于动捕的强化学习全身运动（与 RAI 研究所，2025）③ 量产 Atlas + Gemini Robotics + 现代工厂训练体系（2026）|
| 身位 | 硬件与运动控制全球顶尖；大脑主要依靠合作；工业落地路径最清晰（母公司即客户）|

---

## 1. 故事的开端：从"预编程的奇迹"到"学习的必要"

在 2013–2023 年的十年里，Boston Dynamics 的液压 Atlas 做出了无数令人惊叹的动作：后空翻、跑酷、体操。但内行都知道，这些动作背后是**精心设计的模型预测控制（MPC）+ 离线优化的轨迹库**：工程师为每一个动作设计目标、约束与代价函数，机器人在线求解最优控制。这种方法的优点是精确、可解释、动态性能极强；缺点是**每一个新动作、每一种新任务都需要工程师手工设计**，无法扩展到成千上万种日常操作。

2022 年后，随着 RT-2、Diffusion Policy、ALOHA 等学习方法的突破，Boston Dynamics 内部也意识到：**要让 Atlas 从"表演者"变成"工人"，大脑必须从"编程"转向"学习"。** 2024 年 4 月，液压 Atlas 退役，全电动 Atlas 发布——它的关节可以 360° 旋转，"身体的可能性"远超人类，这本身也需要学习方法来探索。

Boston Dynamics 选择的路径是**合作**而非从零自研：它找到了两家在机器人学习领域最强的研究机构——丰田研究院（Diffusion Policy 与 LBM 的发源地）和 RAI 研究所（由 Boston Dynamics 创始人 Marc Raibert 创办）。

---

## 2. 代表作一：Atlas 大行为模型（与 TRI，2025.08）

> 官方博客：*Large Behavior Models and Atlas Find New Footing*（bostondynamics.com，2025-08-20）；TRI 同步发布。

### 2.1 核心主张：手和脚"几乎一视同仁"

这项工作最重要的一句话是：

> **单一大行为模型直接控制整个机器人，对待手和脚几乎完全相同。**

过去的人形机器人（包括 Figure Helix 1 代、早期 GR00T）都把"行走与平衡"与"手臂操作"分开：下半身由专门的控制器负责，上半身由学习策略负责。Atlas LBM 则让**一个策略同时输出全身所有关节的动作**——迈步是动作，伸手也是动作；为了够到远处的物体，策略自己决定"迈一步"还是"弯腰"。这比 Figure Helix 02（2026.01）的全身自主早了约 5 个月。

### 2.2 架构

| 组件 | 内容 |
|---|---|
| 主干 | **4.5 亿参数的扩散 Transformer（DiT）**，基于 TRI 的 LBM（大规模化的 Diffusion Policy 类架构）|
| 训练目标 | **流匹配（flow matching）**|
| 输入 | 头部相机图像、本体感知（关节状态）、**语言提示** |
| 输出 | 控制**整个 Atlas** 的动作，频率 **30 Hz** |
| 形式 | 单一、语言条件化的多任务策略 |

### 2.3 数据管线：四步循环

Boston Dynamics 与 TRI 描述的开发流程是一个典型的"数据-训练-评估"闭环：

1. **采集**：通过 VR 遥操作系统，在真实 Atlas（以及只有上半身的 Atlas 操作测试台 MTS）和仿真中采集具身行为示范数据；
2. **处理**：对数据进行标注、整理、筛选；
3. **训练**：在全部任务数据上训练一个统一的神经网络策略；
4. **评估**：在一套测试任务上评估，根据失败模式决定下一轮采集什么数据。

**遥操作系统的重要性**：要让一个策略学会全身动作，示范者必须能用全身"附身"机器人——Boston Dynamics 为此开发了基于 VR 与身体追踪的全身遥操作系统，操作员的手、躯干、腿部动作都能映射到 Atlas 上。

### 2.4 演示内容

- **工厂零件拣放**：一段**连续 2 分钟不中断**的工厂拣放任务——机器人从容器中取出零件、走到另一处、放入指定位置，期间处理物体掉落等意外；
- **新颖任务**：**系绳结、折叠椅子、翻转轮胎**、展开桌布等——这些任务涉及可形变物体、大物体全身搬运，是传统 MPC 极难手工编程的；
- **意外恢复**：人为干扰（例如关上容器盖、把物体挪走）后，策略能自主调整、重新完成任务。

### 2.5 意义

1. **"一个策略控制全身"的早期证明**：在双足人形上，用单一学习策略同时完成行走与灵巧操作，这在 2025 年中是领先的；
2. **TRI LBM 研究的工业验证**：TRI 的 LBM 研究（见第 15 章）在固定双臂平台上验证了"多任务预训练提升样本效率"；Atlas LBM 把这一结论推到了全尺寸人形；
3. **Boston Dynamics 文化的转变**：从"动作工程师"到"数据工程师"。

### 2.6 局限

- 30Hz 对于高动态全身动作而言偏低，高动态行为仍依赖底层控制器；
- 4.5 亿参数且不含大规模 VLM 预训练，语义泛化能力有限（这正是后来引入 Gemini 的原因）。

---

## 3. 代表作二：基于动捕的强化学习全身运动（与 RAI 研究所，2025）

> 来源：Boston Dynamics 与 RAI 研究所发布的 Atlas 强化学习系列视频与博客（2025）；RAI 研究所《强化学习加速人形行为生成》；相关学术论文（如 ZEST，arXiv 2602.00401）。

### 3.1 问题：如何让一个 360° 关节的机器人"像人一样动"？

电动 Atlas 的关节活动范围远超人类，这既是优势也是难题——**手工设计"自然的人类动作"几乎不可能**。RAI 研究所给出的方案是：**用人类动作捕捉与动画作为参考，用强化学习训练机器人去"追踪"这些参考动作。**

### 3.2 方法

- **参考动作**：人类动作捕捉数据，以及动画师制作的动画；先做**运动重定向（retargeting）**，把人体动作映射到 Atlas 的骨架上；
- **仿真训练**：在仿真中用强化学习训练一个控制策略，奖励是"与参考动作的接近程度"+ 物理约束（平衡、接触、关节限制）；
- **规模**：据 RAI 研究所介绍，**每一个动作技能都基于约 1.5 亿次仿真运行**生成；
- **零样本迁移**：训练好的策略**零样本**部署到真实 Atlas 上，无需真机微调。

### 3.3 成果

Atlas 展示了行走、跑步、爬行（"匍匐前进"）、侧手翻、霹雳舞等动作——其中多接触动作（手、肘、膝同时着地）对传统 MPC 来说极难建模。学术论文 ZEST 也报告了在 Atlas 上从动作捕捉中学习匍匐、霹雳舞等动态多接触技能。

### 3.4 意义

这项工作与 NVIDIA SONIC、Figure Helix 02 S0 共同印证了 2025–2026 年的一个行业共识：**"动捕追踪 + 仿真 RL + 零样本迁移"是人形底层运动控制的标准范式**。对于 Boston Dynamics 来说，它意味着 Atlas 的"小脑"也完成了从 MPC 到学习的转型——上层由 LBM/Gemini 负责，下层由 RL 运动策略负责。

---

## 4. 代表作三：量产 Atlas + Gemini Robotics + 工厂训练体系（2026）

> 来源：CES 2026（2026-01-05，现代汽车集团发布会）；Boston Dynamics 官方博客 *Boston Dynamics & Google DeepMind Form New AI Partnership*；现代汽车 AI 机器人战略；2026 年 9 月佐治亚机器人应用中心开幕报道。

### 4.1 CES 2026：三件事同时发生

1. **量产版 Atlas 亮相**：面向工业部署的设计；
2. **与 Google DeepMind 建立 AI 合作**：DeepMind 机器人负责人 Carolina Parada 称 Atlas 将是合作的"第一个测试案例"，双方将在接下来数月启动联合研究，在 Atlas 上使用 Gemini Robotics 基础模型；
3. **现代汽车的 AI 机器人战略**：现代计划到 2028 年建立年产约 3 万台机器人的生产体系；Atlas 将从 2028 年起分阶段部署到集团各工厂，首先是佐治亚的现代汽车美国 Metaplant。**2026 年的 Atlas 产量已全部分配给现代汽车与 Google DeepMind。**

### 4.2 "双大脑合作"的含义

Boston Dynamics 现在同时拥有两个外部大脑合作方：

| 合作方 | 提供 | 优势 |
|---|---|---|
| TRI（LBM） | 多任务行为模型，扩散/流匹配策略 | 精细操作、样本效率、学术严谨 |
| Google DeepMind（Gemini Robotics） | 具身推理（ER）+ VLA + 端侧模型 | 语义理解、推理、跨本体、语言交互 |

结合 DeepMind Gemini Robotics 2（2026.07）新增的全身控制能力，可以推测（本报告分析）：未来 Atlas 的大脑栈可能是"**Gemini Robotics-ER（任务理解与编排）+ Gemini Robotics / LBM（全身视觉运动策略）+ RL 运动控制器（高频平衡与运动）**"的三层组合。

### 4.3 工厂即训练场：机器人 Metaplant 应用中心

2026 年 9 月 21 日，Boston Dynamics 在现代佐治亚园区开设了**机器人 Metaplant 应用中心（Robotics Metaplant Application Center）**，用于训练 Atlas 完成零部件装配与重物搬运等任务；计划 2027 年建设约 10 倍规模的设施。按照现代的"韩国式预训练模式"，机器人先在训练中心做"预训练"，再部署到产线，在生产现场持续迭代学习。

**大脑视角的意义**：这是一个**"工厂内数据飞轮"**——母公司现代汽车既是客户、又是数据源、又是测试场。相比 Figure 需要说服宝马、特斯拉只能用自家工厂，Boston Dynamics 拥有一个**集团内部、规模巨大、场景标准化**的汽车制造体系作为训练场。

### 4.4 冷静看待

- 多家媒体（包括一些低可信度来源）声称 Atlas"已在 Metaplant 自主工作"，但较一致的说法是**正式产线部署从 2028 年开始**；
- "到 2028 年数万台"属于集团规划，实际节奏取决于大脑能力能否满足汽车产线的可靠性要求（通常 99.9% 以上）。

---

## 5. 技术路线总结：从"编程身体"到"学习身体 + 借用大脑"

| 层级 | 2013–2023（液压 Atlas） | 2025–2026（电动 Atlas） |
|---|---|---|
| 高层任务 | 人工编排 | Gemini Robotics-ER / 语言指令 |
| 视觉运动策略 | 无（预设轨迹 + 感知模块） | LBM（4.5 亿参数 DiT，30Hz，全身）/ Gemini Robotics |
| 底层运动控制 | MPC + 离线轨迹优化 | 动捕参考 + 仿真 RL + 零样本迁移（与 MPC 并存） |
| 数据来源 | 工程师设计 | VR 全身遥操作 + 仿真 + 工厂运行数据 |

Boston Dynamics 的战略可以概括为：**把自己最强的东西（身体与运动控制）做到极致，把大脑交给最强的合作者，把母公司的工厂变成数据飞轮。**

---

## 6. 横向对比与身位

| 维度 | Boston Dynamics | Figure | 特斯拉 | Agility |
|---|---|---|---|---|
| 大脑 | TRI LBM + DeepMind Gemini + RAI RL | 自研 Helix | 自研（FSD 体系） | 自研全身控制 + 上层系统 |
| 全身单策略 | 2025.08（LBM） | 2026.01（Helix 02） | 不详 | 全身控制基础模型（底层） |
| 工业客户 | 母公司现代（内生） | 宝马（外部） | 自家工厂 | GXO、舍弗勒等 |
| 硬件水平 | 全球顶尖（360° 关节） | 顶尖 | 顶尖 | 成熟、务实 |

**身位判断**：
- **身体**：全球第一档；
- **大脑**：自身研发能力中上，但通过与 TRI、DeepMind 的合作获得了**第一梯队的大脑资源**；
- **落地**：拥有最确定的大客户（现代汽车），但大规模部署时间点（2028）晚于部分竞争对手的宣称。

---

## 7. 未来展望与风险

**展望**：
1. **Gemini + Atlas 的首批成果**：2026 年下半年至 2027 年，DeepMind 与 BD 的联合研究成果将是 Gemini Robotics 全身控制能力在顶级人形硬件上的首次大规模验证；
2. **工厂数据飞轮**：Metaplant 应用中心的扩建意味着 Atlas 将在真实汽车制造场景中积累海量数据；
3. **Spot 的复用**：有报道称 Gemini Robotics 也可能运行在 Spot 上，形成"四足 + 人形"的统一大脑。

**风险**：
1. **大脑依赖外部**：如果 DeepMind 或 TRI 的合作关系变化，Boston Dynamics 将面临核心能力缺口；
2. **成本**：Atlas 的高端硬件成本与中国厂商（宇树等）的低价策略形成鲜明对比；
3. **时间表**：2028 年的大规模部署节点，相对于特斯拉、Figure、中国厂商的激进时间表，可能在市场竞争中显得保守。

---

## 8. 延伸阅读与核验来源

- [Boston Dynamics 博客：Large Behavior Models and Atlas Find New Footing](https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/)（2025-08-20）
- [Toyota 新闻：BD 与 TRI 的 AI 机器人迈向通用人形的关键一步](https://pressroom.toyota.com/ai-powered-robot-by-boston-dynamics-and-toyota-research-institute-takes-a-key-step-towards-general-purpose-humanoids/)
- [RAI 研究所：强化学习加速人形行为生成](https://rai-inst.com/resources/videos/reinforcement-learning-accelerates-humanoid-behavior-production)；[i-programmer：Atlas 强化学习](https://www.i-programmer.info/news/169-robotics/17911-atlas-in-action-with-reinforcement-learning.html)
- ZEST（arXiv 2602.00401）：Atlas 上从动捕学习多接触技能
- [Boston Dynamics 博客：与 Google DeepMind 建立新 AI 合作](https://bostondynamics.com/blog/boston-dynamics-google-deepmind-form-new-ai-partnership/)；[TechCrunch](https://techcrunch.com/2026/01/05/boston-dynamicss-next-gen-humanoid-robot-will-have-google-deepmind-dna/)；[The Robot Report](https://www.therobotreport.com/boston-dynamics-google-reunite-on-next-gen-atlas-humanoid/)
- [A3 Automate：Boston Dynamics 2026 年开始生产重新设计的 Atlas](https://www.automate.org/robotics/industry-insights/boston-dynamics-to-begin-production-on-redesigned-atlas-humanoid-in-2026)
- [Korea Herald：Atlas 2028 年进入现代佐治亚 Metaplant](https://www.koreaherald.com/article/10649929)
- [CBT News：Boston Dynamics 开设机器人训练中心](https://www.cbtnews.com/boston-dynamics-opens-robot-training-center/)

> **核验说明**：LBM 的参数量、训练目标、输入输出与频率来自 Boston Dynamics 官方博客；"每个动作约 1.5 亿次仿真"来自 RAI 研究所官方页面；DeepMind 合作与产量分配来自 CES 2026 官方发布与多家主流媒体；"已在 Metaplant 自主工作"的说法来自低可信来源，本章未采信；4.2 节三层大脑栈为本报告分析推测。
