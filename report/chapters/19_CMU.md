# 第 19 章 卡内基梅隆大学（CMU）：人形"小脑"的军火库

> **一句话定位**：CMU 机器人研究所（RI）是全球历史最悠久、规模最大的机器人学术机构。在人形机器人大脑领域，CMU 最突出的贡献集中在"**小脑**"——人形全身控制：从 H2O/OmniH2O（2024，用一个普通摄像头实时遥控全尺寸人形）、ASAP（2025，用"差分动作模型"对齐仿真与真实物理，让人形做出 C 罗庆祝、空中转体等敏捷动作），到 BFM-Zero（2025–2026，ICLR 2026，可被提示的人形行为基础模型，真机零样本）。CMU 的另一条线——Deepak Pathak 与 Abhinav Gupta——则直接孵化了估值 140 亿美元的 Skild AI。Guanya Shi 的 LeCAR 实验室与 NVIDIA GEAR 的深度合作，使 CMU 成为人形控制算法的主要"供给方"之一。

---

## 0. 速览卡片

| 维度 | 内容 |
|---|---|
| 核心实验室 | LeCAR Lab（Guanya Shi，学习与控制）、Pathak 组（具身智能、适应）、Gupta 组（大规模数据）、Changliu Liu 组（安全控制）、Kris Kitani 组（人体运动）、Jessica Hodgins（动画与动捕）、Katerina Fragkiadaki（Genesis 引擎相关）|
| 代表工作（人形相关）| **H2O（2024）/ OmniH2O（2024）**、HOVER（2024，与 NVIDIA）、**ASAP（RSS 2025）**、**BFM-Zero（ICLR 2026）**、FALCON、HumanoidVerse；Pathak 组：Extreme Parkour、LEAP Hand、VRB；Gupta 组：大规模自监督 |
| 人才输出 | Skild AI（Pathak、Gupta）、Genesis AI（周衔）、NVIDIA GEAR（大量合作与实习）、OpenAI（ASAP 一作 Tairan He 据报道已加入 OpenAI）|
| 本章三篇代表作 | ① H2O / OmniH2O：从人到人形的实时全身遥操作（2024）② ASAP：对齐仿真与真实物理，学会敏捷全身技能（2025）③ BFM-Zero：可提示的人形行为基础模型（2025–2026）|
| 身位 | 人形全身控制算法的全球核心供给者之一 |

---

## 1. 故事的开端：从 DARPA 挑战赛到"学习型小脑"

CMU 在人形与足式机器人上的传统可以追溯到 Marc Raibert 早年在 CMU 的跳跃机器人研究（后来他去了 MIT，再创办 Boston Dynamics）。2015 年 DARPA 机器人挑战赛中，CMU 的 CHIMP 机器人获得第三名。但真正让 CMU 在人形领域"重回中心"的，是 2023 年后 **Guanya Shi**（加州理工博士，Neural-Lander、Neural-Fly 作者）加入 CMU 创立 LeCAR 实验室，以及与 NVIDIA GEAR 实验室（Jim Fan、Yuke Zhu）的深度合作。

LeCAR 的研究主线非常清晰：**如何让人形机器人像人一样灵活、稳健地运动，并能被人方便地"教"？**

---

## 2. 代表作一：H2O 与 OmniH2O（2024）——一个摄像头遥控全尺寸人形

> 论文：Tairan He, Zhengyi Luo, Wenli Xiao, Chong Zhang, Kris Kitani, Changliu Liu, Guanya Shi，*Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation*（H2O，IROS 2024）；*OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning*（CoRL 2024）

### 2.1 问题

在 2024 年之前，全尺寸人形的全身遥操作需要动捕服、外骨骼等昂贵设备。能否只用一个 **RGB 摄像头**，让操作员的全身动作被人形实时复现？

### 2.2 H2O 的三步法

1. **"仿真到数据"的动作清洗**：人类动捕数据集（AMASS）中有大量人形**物理上做不到**的动作（例如超出关节限位、需要不可能的平衡）。H2O 先用一个**特权模仿策略**在仿真中尝试追踪每一段动作，**剔除追踪失败的动作**——得到一个"人形可行"的动作数据集；
2. **特权教师策略**：在仿真中训练一个能看到完整动作参考与特权状态的追踪策略；
3. **可部署的学生策略**：蒸馏到一个只使用"真实可获得"输入的学生策略——即从 RGB 摄像头估计出的人体关键点位置（稀疏输入）+ 本体感知。

部署时，操作员站在摄像头前，人体姿态估计器（如基于视觉的 3D 姿态估计）实时输出关键点，学生策略驱动宇树 H1 复现动作——走路、踢腿、拳击、挥手等。

### 2.3 OmniH2O：从遥操作到"自主学习"

OmniH2O 在 H2O 的基础上做了三项扩展：
- **通用接口**：同一个控制策略支持多种输入——VR 头显（追踪头与双手）、RGB 摄像头、甚至**语言指令**（通过动作生成模型把文字转成动作）、以及自主策略的输出；
- **灵巧性**：加上灵巧手，支持精细操作；
- **从遥操作到自主**：用 OmniH2O 遥操作采集的示范，训练自主策略（例如基于 GPT-4o 的高层规划 + 模仿学习策略），实现人形的自主全身操作；并发布了 **OmniH2O-6** 人形全身操作数据集。

### 2.4 意义

H2O/OmniH2O 确立了"**动捕数据清洗 → 特权教师 → 稀疏输入学生**"的人形全身追踪训练范式，并提出"**通用控制接口**"的思想：底层追踪控制器成为一个标准接口，上层可以是人、VR、语言模型或自主策略。这正是 2025–2026 年"三层大脑"中 S0 层的设计思想。

---

## 3. 代表作二：ASAP（RSS 2025）——对齐仿真与真实物理

> 论文：Tairan He, Jiawei Gao, Wenli Xiao, ..., Changliu Liu, Jessica Hodgins, Linxi "Jim" Fan, Yuke Zhu, Guanya Shi 等，*ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills*（RSS 2025，arXiv 2502.01143）——CMU 与 NVIDIA 合作。

### 3.1 问题：sim2real 的"保守困境"

让人形做敏捷动作（跳跃、转体、单腿平衡）时，sim2real 鸿沟会被放大。传统两种方法都有问题：
- **系统辨识（SysID）**：需要大量手工调参，且难以覆盖所有未建模的动力学；
- **域随机化（DR）**：为了在所有随机参数下都稳定，策略变得**过于保守**——牺牲了敏捷性。

### 3.2 方法：差分动作模型（Delta Action Model）

```
阶段一（预训练）：在仿真中训练动作追踪策略（基于重定向的人类动作，例如 C 罗、科比、勒布朗的标志性动作）
        │
        ▼
阶段二（真机采集）：部署到真实 G1，记录真实轨迹
        │
        ▼
阶段三（学习差分动作模型）：训练一个模型 Δa = f(s, a)，
        使得"仿真中执行 a + Δa"的结果 ≈ "真实中执行 a"的结果
        → 它学到的是"真实物理与仿真物理的差异"，以动作修正的形式表达
        │
        ▼
阶段四（在"修正后的仿真"中微调）：把冻结的差分动作模型嵌入仿真器，
        在这个"更像真实"的仿真中继续微调策略
        │
        ▼
部署：最终策略直接在真机上运行（不再需要差分模型）
```

### 3.3 结果

- 在 IsaacGym → IsaacSim、IsaacGym → Genesis、IsaacGym → 真实 G1 三种迁移设置中，ASAP 都显著降低了动作追踪误差，优于 SysID、DR 与差分动力学学习等基线；
- 真机上展示了**空中 180 度转体跳**、**单腿平衡**、模仿 C 罗"Siu"庆祝动作、科比后仰跳投姿态等高敏捷动作。

### 3.4 意义

ASAP 证明了"**用少量真机数据学习'物理差异'，再回到仿真中训练**"是突破 sim2real 保守困境的有效路径。它与 RAI 研究所的 Atlas RL、NVIDIA SONIC、宇树的敏捷运动展示一起，构成了 2025 年"人形敏捷运动爆发"的技术基础。

---

## 4. 代表作三：BFM-Zero（ICLR 2026）——可提示的人形行为基础模型

> 论文：Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Anssi Kanervisto, Andrea Tirinzoni, Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro Lazaric, Matteo Pirotta, Guanya Shi，*BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning*（arXiv 2511.04131，ICLR 2026）——CMU 与 Meta FAIR 合作。

### 4.1 问题：从"追踪器"到"行为基础模型"

H2O、ASAP、SONIC 等都是"**动作追踪器**"——给它一段参考动作，它就复现。但如果任务不是"模仿某个动作"，而是"走到那个点""用一条腿站着""最大化某个奖励"呢？每种任务都要重新训练吗？

BFM-Zero 的目标是：**一个策略、一个共享潜空间，通过"提示"完成多种任务——无需重新训练。**

### 4.2 方法：把 Meta Motivo 的 FB-CPR 搬上真机

- **理论基础**：前向-后向（FB）表征 + Meta 的 FB-CPR 算法（见第 12 章）——学习一个共享潜空间，把**动作、目标与奖励**都嵌入同一空间；
- **预训练数据**：无标签的人类动作数据（LAFAN1）+ **无奖励**的仿真交互——这是一个**无监督强化学习**问题；
- **sim2real 关键设计**：奖励塑形、域随机化、**依赖历史的非对称学习**（让策略从历史中推断真实动力学）；
- **零样本提示**：动作追踪、目标到达、奖励优化；
- **少样本适应**：在潜空间中用交叉熵方法（CEM）等搜索，快速优化新行为（例如单腿站立）。

### 4.3 结果

在 IsaacLab 仿真与**真实宇树 G1** 上：
- 零样本跟随跳舞、行走、格斗等动作；
- 从大幅推搡与跌倒中恢复；
- 通过提示完成目标姿态到达与奖励优化。

这是**首批在真实人形上部署的"可提示行为基础模型"**之一（此前 Meta Motivo 只在仿真中）。

### 4.4 意义

BFM-Zero 把人形小脑从"**执行器**"提升为"**基础模型**"：上层大脑不必只给"参考动作"，还可以给"目标"或"奖励"，小脑自己找到合适的全身行为。这为 VLA/世界模型与全身控制之间提供了一个更灵活、更抽象的接口。

---

## 5. Pathak / Gupta 一脉：Skild 的学术源头

CMU 的另一条重要线索是 Deepak Pathak 与 Abhinav Gupta（详见第 4 章）：
- **Extreme Parkour（2023）**：四足机器人仅凭单目深度相机完成跳跃、攀爬、跨越间隙；
- **LEAP Hand（2023）**：低成本、开源、拟人灵巧手；
- **VRB（2023）**：从人类视频中学习"可供性（affordance）"——哪里可以抓、该怎么动；
- 大规模自监督机器人学习（Gupta）。

这些工作共同构成了 Skild "全身体 + 适应 + 人类视频"路线的学术基础。

---

## 6. 横向对比与身位

| 人形小脑工作 | 机构 | 核心思想 |
|---|---|---|
| H2O / OmniH2O | CMU | 动作清洗 + 教师-学生 + 通用接口 |
| HOVER | NVIDIA + CMU 等 | 多模式统一（掩码切换控制模式）|
| ASAP | CMU + NVIDIA | 差分动作模型对齐 sim/real |
| BFM-Zero | CMU + Meta | 可提示行为基础模型（FB 表征）|
| SONIC | NVIDIA | 动作追踪规模化（42M 参数、700 小时）|
| OmniRetarget / ResMimic | Amazon FAR | 保留交互的重定向 + 残差移动操作 |
| Helix 02 S0 | Figure | 1000+ 小时动捕 + RL，1kHz |
| Agility WBC | Agility | <1M 参数，仿真 RL，商用 |

**身位**：CMU（特别是 LeCAR）是人形全身控制算法**最高产、最有影响力的学术来源之一**，其学生大量进入 NVIDIA、OpenAI、各类人形公司。

---

## 7. 未来展望

1. **小脑与大脑的接口标准化**：BFM-Zero 式的"提示接口"可能成为 VLA 调用全身控制器的标准方式；
2. **全身移动操作**：从"模仿动作"走向"与物体、环境的物理交互"（如 HDMI、ResMimic 一类工作）；
3. **人才流动**：CMU 人形团队的核心成员持续流向产业（NVIDIA、OpenAI、创业公司），学术前沿与产业前沿高度重合。

---

## 8. 延伸阅读与核验来源

- H2O：arXiv 2403.04436；OmniH2O：arXiv 2406.08858
- ASAP：[arXiv 2502.01143](https://arxiv.org/html/2502.01143v3)；[CMU RI 项目页](https://www.ri.cmu.edu/project/asap/)；[RSS 2025](https://roboticsproceedings.org/rss21/p066.html)
- BFM-Zero：[arXiv 2511.04131](https://arxiv.org/html/2511.04131v1)；[ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/80b677bb4d17f77220be88e1d1fdebde-Abstract-Conference.html)；[项目页](https://lecar-lab.github.io/BFM-Zero)
- HOVER：arXiv 2410.21229

> **核验说明**：H2O/OmniH2O 方法细节来自原论文；ASAP 的作者、方法与迁移实验来自论文与 RSS 2025 收录信息（Tairan He 加入 OpenAI 为个人主页信息的转述）；BFM-Zero 的作者、方法与真机结果来自 arXiv 与 ICLR 2026 页面。
