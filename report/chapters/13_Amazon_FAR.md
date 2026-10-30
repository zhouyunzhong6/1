# 第 13 章 Amazon（FAR / Covariant 一脉）：全球最大的机器人用户，如何打造人形"小脑"工具链

> **一句话定位**：亚马逊是全球部署机器人数量最多的企业（其履约网络中运行着上百万台各类机器人），也是人形机器人最具潜力的"超级买家"。2024 年 8–9 月，亚马逊以"人才收购 + 非独家授权"的方式吸纳了 Covariant 的三位创始人——Pieter Abbeel、Peter Chen、Rocky Duan——以及其机器人基础模型 RFM-1，并组建 **Amazon FAR（Frontier AI & Robotics）**。FAR 在 2025–2026 年发布的工作（OmniRetarget、ResMimic、Holosoma）几乎都聚焦于一个问题：**如何把人类动作数据高质量地转化为人形全身"移动+操作"能力**——这是人形大脑中"小脑"层最核心的数据与训练工具链。

---

## 0. 速览卡片

| 维度 | 内容 |
|---|---|
| 机构 | Amazon FAR（Frontier AI & Robotics）；亚马逊机器人（Amazon Robotics）|
| 核心人物 | Pieter Abbeel（UC Berkeley 教授，深度强化学习与机器人学习先驱）、Peter Chen、Rocky Duan（Covariant 联合创始人）；合作者包括 Guanya Shi（CMU）、C. Karen Liu（斯坦福）、Angjoo Kanazawa（Berkeley）、Carmelo Sferrazza 等 |
| 历史 | Covariant（2017 年创立，仓储拣选 AI）→ RFM-1（2024.03）→ 亚马逊雇佣创始人并获得非独家模型授权（2024.08–09）→ Amazon FAR |
| 代表工作 | RFM-1（Covariant，2024）；**OmniRetarget**（2025.09/10）；**ResMimic**（arXiv 2510.05070）；**Holosoma**（2025.12，全栈 sim2real 代码库）；2026 年的后续工作 |
| 亚马逊其他相关 | Vulcan（带触觉的仓储拣选机器人，2025）；DeepFleet（机器人车队调度基础模型，2025）；人形机器人送货测试等 |
| 开放程度 | FAR 的研究以论文 + 开源代码形式发布（如 GitHub amazon-far/ResMimic、Holosoma）|
| 本章三篇代表作 | ① RFM-1：Covariant 的多模态机器人基础模型 ② OmniRetarget：保留交互关系的动作重定向与数据生成 ③ ResMimic + Holosoma：从通用动作追踪到全身移动操作的残差学习与全栈工具链 |
| 身位 | "人形小脑工具链"的重要贡献者；亚马逊作为超级用户的场景与数据优势巨大；尚未公开发布自有人形大脑产品 |

---

## 1. 故事的开端：从 OpenAI 到仓库

Pieter Abbeel 是深度强化学习的奠基人之一：他的学生 John Schulman 发明了 TRPO/PPO，Sergey Levine 曾在他的组做博士后，Chelsea Finn 也出自 Berkeley 体系。Peter Chen 与 Rocky Duan 都是 Abbeel 的学生，曾在 OpenAI 早期工作。2017 年，三人创办了 Covariant，目标是"**让机器人在真实仓库里可靠地拣选任意物品**"。

Covariant 的路线非常务实：它把 AI 部署在全球各地的仓储拣选站上，让机器人每天处理数以百万计的真实拣选——这些真实世界的成功与失败，构成了 Covariant 最宝贵的数据资产。到 2024 年，Covariant 宣称其系统已在多个国家的仓库中运行，积累了海量的真实操作数据。

2024 年 3 月，Covariant 发布了 RFM-1。半年后，亚马逊以"雇佣创始人 + 非独家授权模型"的方式完成了一次典型的"人才收购"（这种模式与微软-Inflection、谷歌-Character.AI 类似）。亚马逊发言人称，Covariant 的模型将用于驱动其履约网络中的部分机器人操作系统。

---

## 2. 代表作一：RFM-1（Covariant，2024.03）——"任意模态到任意模态"的机器人基础模型

> 发布：Covariant 官方博客与技术介绍（2024 年 3 月）。

### 2.1 设计思想

RFM-1（Robotics Foundation Model 1）是较早的"商用机器人基础模型"之一。据 Covariant 介绍，它是一个约 **80 亿参数**的 Transformer，在**文本、图像、视频、机器人动作与各种传感器读数**上联合训练，形成一个"**任意到任意（any-to-any）**"的多模态序列模型：

- **输入/输出可以是任意模态的组合**：给图像 + 文本 → 输出动作；给图像 + 动作 → 输出预测的未来图像（视频预测）；给图像 → 输出文本描述……
- **语言交互**：机器人可以接受文本或图像形式的指令、回答问题，并在不确定时**主动请求进一步的指示**（例如"我不确定该抓哪一个，请指出"）；
- **物理预测**：模型可以生成"如果执行这个抓取动作，接下来会发生什么"的视频预测，用于判断动作结果——这是一种早期的"世界模型"能力。

### 2.2 数据优势

RFM-1 的核心竞争力在于其训练数据来自**真实仓库中长期运行的拣选系统**：大量真实物品、真实失败、真实的长尾情况（反光包装、变形袋子、缠绕物品）。这与学术界基于实验室遥操作数据的 VLA 形成鲜明对比。

### 2.3 意义与局限

**意义**：RFM-1 在 2024 年初就提出了"多模态 any-to-any + 视频预测 + 语言交互"的机器人基础模型构想，在概念上领先于同期多数工作。
**局限**：主要面向固定工作站的拣选（吸盘/夹爪），不涉及人形全身控制；技术细节公开有限；亚马逊收购后，Covariant 公司本身的后续发展受到质疑（有媒体报道了相关举报与"空壳公司"指控，未经证实）。

---

## 3. 代表作二：OmniRetarget（2025.09/10）——高质量重定向是人形 RL 的"地基"

> 论文：*OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction*（Amazon FAR 等，arXiv 2509.26633）；作者包括 Angjoo Kanazawa、Pieter Abbeel、Carmelo Sferrazza、C. Karen Liu、Rocky Duan、Guanya Shi 等。

### 3.1 被低估的问题：重定向的"伪影"

2025 年，"用人类动捕数据训练人形 RL 策略"已成为共识（见第 7、3、9 章）。但这里有一个经常被忽视的环节——**运动重定向（retargeting）**：人类和机器人的骨骼比例、关节结构、质量分布都不同，必须把人类动作"翻译"成机器人能执行的动作。

传统重定向常产生各种伪影：
- **脚滑（foot skating）**：脚在着地时滑动；
- **穿地（ground penetration）**：脚陷入地面；
- **交互关系丢失**：人搬箱子时手贴着箱子，重定向后机器人的手可能离箱子几厘米或穿进箱子；人坐在椅子上，重定向后机器人"悬空坐"。

Abbeel 在发布时一针见血地指出：**对于接近可行的轨迹，控制策略的优化地形会好得多；带伪影的轨迹会让 RL 事倍功半**。

### 3.2 方法：保留交互关系的重定向

OmniRetarget 的核心是一个**交互网格（interaction mesh）**：
- 把人体关键点、物体、地形表面上的点共同构造成一个网格，**显式建模人-物-环境之间的空间与接触关系**；
- 重定向被表述为一个优化问题：在满足机器人运动学约束（关节限位、无穿透、脚不滑）的前提下，**最小化交互网格的形变**（即尽量保持"手与箱子的相对关系""脚与台阶的相对关系"）；
- 除了重定向单段动作，它还能通过改变物体位置、尺寸、地形高度等，**从一段人类示范生成大量多样化的、物理一致的机器人训练轨迹**——这就是"数据生成"的含义。

### 3.3 结果

- 用 OmniRetarget 生成的数据训练 RL 策略，**只需极简的奖励设计**（据报道仅约 5 项奖励与 4 项域随机化），就能学会复杂的全身移动操作与场景交互技能，例如搬运箱子、攀爬平台、在复杂地形上移动并与物体交互；
- 在宇树 G1 等人形上实现长时程、多接触的技能执行。

### 3.4 意义

OmniRetarget 揭示了一个工程真理：**在"人类数据 → 人形技能"这条管线中，数据质量（重定向质量）往往比算法更重要**。它把重定向从"预处理小步骤"提升为"核心数据生成技术"。

---

## 4. 代表作三：ResMimic + Holosoma（2025.10–12）——从通用动作追踪到精确全身移动操作

> 论文：*ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning*（arXiv 2510.05070；代码 github.com/amazon-far/ResMimic）；Holosoma 全栈代码库（2025.12）。

### 4.1 问题：通用动作追踪器"不够准"

到 2025 年下半年，业界已有多个"通用动作追踪器"（General Motion Tracking，GMT）——一个策略可以追踪任意人类动作（如 NVIDIA SONIC、CMU/Berkeley 的多项工作）。但当任务涉及**与物体的精确交互**（搬起一个重箱子、把椅子推到桌下），通用追踪器往往"动作像、但做不成"：手的位置差几厘米，箱子就掉了；没有考虑物体重量，机器人就失去平衡。

### 4.2 方法：两阶段"预训练-后训练"残差学习

```
阶段一（预训练）：通用动作追踪策略 π_GMT
   · 在大规模人类动作数据上训练，只关心"像人一样动"
   · 不考虑物体
          │
          ▼
阶段二（后训练）：残差策略 π_res
   · 输入：机器人状态 + 物体状态（位置、姿态）+ 参考动作
   · 输出：对 π_GMT 动作的"修正量" Δa
   · 最终动作：a = a_GMT + Δa
   · 奖励：物体追踪（物体是否按参考轨迹移动）+ 接触奖励 + 课程学习
```

这一设计借鉴了 LLM 的"预训练-后训练"范式：通用追踪器提供"像人的全身运动先验"，残差策略只需学习"为完成具体物体交互所需的小幅修正"，大大降低了学习难度。

### 4.3 结果

在仿真与宇树 G1 真机上，ResMimic 完成了需要全身参与的移动操作任务（如搬运较重物体、用身体多部位接触物体），在任务成功率与动作精度上明显优于直接用通用追踪器或从零训练的基线。

### 4.4 Holosoma：全栈 sim2real 工具链

2025 年 12 月，Abbeel 宣布 Amazon FAR 开源 **Holosoma**——一个完整的人形 sim2real 代码库，覆盖：
- **多种仿真后端**；
- **训练**（RL 训练流程）；
- **重定向**（含 OmniRetarget 类方法）；
- **真机推理部署**；
- 为人形设计，也可适配四足。

**意义**：Holosoma 把 FAR 的研究成果打包成可复用的基础设施，类似于 NVIDIA 的 Isaac Lab + SONIC、或 Hugging Face 的 LeRobot。这表明亚马逊在人形领域的策略是"**先做好工具链与小脑能力，再考虑上层大脑**"。

### 4.5 2026 年的延续

2026 年的后续工作（如 OmniXtreme 等引用并延续 OmniRetarget/ResMimic 的研究）显示，这条"高质量重定向 → 通用追踪 → 残差精修"的路线仍在向高动态、高泛化方向推进。

---

## 5. 亚马逊的更大棋局

FAR 的研究不能孤立来看。亚马逊在机器人上的完整布局包括：

| 层级 | 亚马逊的布局 |
|---|---|
| 场景与需求 | 全球最大的履约网络，上百万台机器人在运行 |
| 专用机器人 | Kiva/Proteus（移动机器人）、Sparrow/Robin/Cardinal（拣选）、Vulcan（触觉拣选，2025）|
| 车队智能 | DeepFleet（机器人车队调度基础模型，2025）|
| 人形探索 | 测试 Agility Digit；据报道建设人形机器人送货测试场 |
| 前沿研究 | Amazon FAR（人形全身控制、数据生成、工具链）|
| 基础模型 | RFM-1 授权（Covariant）|

**分析**：亚马逊的独特优势在于**它同时是需求方、数据方与研发方**。当人形机器人的大脑足够成熟时，亚马逊可以在自家仓库里进行全球最大规模的部署与数据采集。FAR 目前专注于"小脑"工具链，可能是为将来自研或采购人形打下"控制层"的基础。

---

## 6. 横向对比与身位

| 维度 | Amazon FAR | NVIDIA（SONIC/GR00T）| CMU LeCAR（见第 19 章）| Figure S0 |
|---|---|---|---|---|
| 关注层级 | 小脑：重定向、追踪、残差移动操作 | 全栈：大脑 + 小脑 + 仿真 | 小脑：追踪、sim2real | 小脑（自用）|
| 开放 | 论文 + 代码（Holosoma）| 开源 | 开源 | 闭源 |
| 数据观 | 高质量重定向即数据生成 | 动捕规模化 | 人类视频 + 动捕 | 1000+ 小时动捕 |

**身位判断**：
- 在"**人形全身移动操作的数据与训练工具链**"上，Amazon FAR 是全球最有影响力的工业研究团队之一；
- 尚未发布面向人形的"上层大脑"（VLA/世界模型），在大脑层的公开存在感弱于 NVIDIA、DeepMind；
- 亚马逊作为全球最大机器人用户的潜在影响力，远大于 FAR 目前的公开成果。

---

## 7. 未来展望与风险

**展望**：
1. **从小脑到大脑**：FAR 是否会发布面向仓储人形的 VLA/世界模型，是 2027 年值得关注的看点；
2. **仓储人形的规模化**：一旦人形在仓库中证明经济性，亚马逊的采购与部署规模将重塑整个人形产业；
3. **RFM 系列的演进**：RFM-1 授权后，Covariant 技术在亚马逊内部的演进路径尚不透明。

**风险**：
1. **"人才收购"模式的监管风险**：相关交易结构受到反垄断关注；
2. **专用机器人 vs 人形**：亚马逊已有极其成熟的专用机器人体系，人形的增量价值需要充分证明；
3. **人才流动**：顶尖研究者在大公司与创业公司之间流动频繁。

---

## 8. 延伸阅读与核验来源

**三篇代表作**
- RFM-1：Covariant 官方发布（2024.03）
- OmniRetarget：arXiv 2509.26633；[Pieter Abbeel 发布说明](https://x.com/pabbeel/status/1973426729952956652)；[ResearchGate](https://www.researchgate.net/publication/396049126_OmniRetarget_Interaction-Preserving_Data_Generation_for_Humanoid_Whole-Body_Loco-Manipulation_and_Scene_Interaction)
- ResMimic：[arXiv 2510.05070](https://arxiv.org/html/2510.05070v1)；[项目页](https://resmimic.github.io/)；[GitHub amazon-far/ResMimic](https://github.com/amazon-far/ResMimic)

**核验来源**
- [SiliconANGLE：亚马逊雇佣 Covariant 创始人](https://siliconangle.com/2024/09/02/amazon-hires-founders-ai-robotics-startup-covariant)
- [The Batch（DeepLearning.AI）：亚马逊与 Covariant 合作](https://www.deeplearning.ai/the-batch/amazon-strengthens-logistics-and-robotics-with-new-ai-partnership/)
- [Bloomberg：亚马逊将雇佣 Covariant 创始人](https://www.bloomberg.com/news/articles/2024-08-31/amazon-to-hire-covariant-founders-to-expand-warehouse-automation)
- OmniXtreme（arXiv 2602.23843）

> **核验说明**：RFM-1"约 80 亿参数、any-to-any 多模态、视频预测、请求澄清"来自 Covariant 2024 年官方发布与媒体报道（参数量为 Covariant 自述）；交易为"雇佣创始人 + 非独家授权"、金额未披露；OmniRetarget 作者与"约 5 项奖励、4 项域随机化"来自论文相关介绍；Holosoma 来自 Abbeel 公开发布；关于 Covariant"空壳公司"的指控未经证实，本章仅作提及。
