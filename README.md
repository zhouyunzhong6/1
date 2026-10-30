# 人形机器人"大脑"全景深度报告（2026.10）

**全球 50 家核心公司与机构 · 150 项代表作完整拆解 · 技术路线、身位与未来**

> 信息截止：2026 年 10 月 30 日 · 全文约 80 余万字 · 中文

---

## 这份报告是什么

一份面向"**彻底读懂人形机器人大脑**"的长篇研究报告：

- 选取全球 **50 家**在机器人大脑上做得最好或声量最大的公司与机构（美国 20 家、中国 30 家，涵盖前沿大脑公司、人形整机、科技巨头、基础设施与学术机构）；
- 每家选取 **3 篇最重要的代表作**（论文、技术报告或开源项目），以讲故事的方式完整拆解：问题 → 核心思想 → 架构 → 数据 → 训练 → 部署 → 结果 → 局限 → 意义；
- 每章附带**横向对比、身位判断、未来展望与风险**，以及逐条的**核验来源与可靠性说明**；
- 前置**行业全景**（十年演进史、技术构件、数据、评测、算力、产业格局），后置**50 家身位矩阵、趋势预测与附录**（大事年表、150 项代表作索引、术语表）。

---

## 十个关键结论（详见正文）

1. **三层大脑已成共识**：推理（S2）/ 视觉运动（S1）/ 全身控制（S0）——Figure Helix 02、逐际 COSA、NVIDIA、DeepMind、银河、智平方等不约而同。
2. **VLA 与世界模型正在融合**：π0.7 用世界模型生成子目标；NVIDIA 以 DreamZero 为 GR00T N2 主干；智元 GO-3（WRAM）、宇树 WLA、自变量 WALL-B、它石 AWE 把"预测未来"变成大脑的原生能力。
3. **数据从遥操作走向人类与无本体**：两年内数据规模从万小时级跃升到十万–百万小时级（Generalist 50 万小时+、小米 10 万小时 UMI、Being-H0.7 20 万小时、它石百万小时宣称）。
4. **缩放定律在机器人领域被多维验证**（数据多样性、真实数据规模、世界模型规模、运动控制），驱动了 2026 年的资本洪流。
5. **RL 进入大模型后训练**：π\*0.6 RECAP、GR-RL、DPPO、RAMP、模型内 RL、车队级 RL。
6. **上下文学习出现**：Skild S1（一段视频学会 10 分钟任务）、GEN-1.5（3–12 秒示范一次学会）。
7. **家庭零样本与全身灵巧取得突破，但离可靠仍远**：Helix 2.5 在 30 户陌生家庭整任务成功率 56%；Gemini Robotics 2 的五指装配类任务仅 32%–44%；BEHAVIOR 2025 冠军完整成功率约 12%。
8. **从演示态走向部署态**：智元 2 万台下线、宇树 IPO、特斯拉早期生产、Figure 03 进驻宝马、Agility 付费部署、丰田 40 万台计划。
9. **评测进入榜单时代并遭遇信任危机**：RoboChallenge、RoboArena、BEHAVIOR……以及 2026 年 6 月的 RoboArena 操纵事件。
10. **中美格局**：美国领先于前沿范式与单点深度（PI、DeepMind、NVIDIA、Figure、Generalist）；中国领先于出货、开源与迭代速度（智元、宇树、阿里 Qwen 底座、大量开源模型）。

---

## 目录

### 开篇
- [导读与方法论](report/00_导读与方法论.md)（选择标准、50 家名单、章节结构、核验方法与信息分级、阅读路径）
- [第一部分 行业全景](report/01_行业全景.md)（十年七个时代、技术构件拆解、数据、评测、算力、产业格局）

### 第二部分 50 家公司与机构

**第一篇 全球前沿大脑**
1. [Google DeepMind](report/chapters/01_Google_DeepMind.md) — RT-2 / Gemini Robotics 1.0→1.5 / Gemini Robotics 2
2. [Physical Intelligence（π）](report/chapters/02_Physical_Intelligence.md) — π0 / π0.5 / π0.7（含 π\*0.6 RECAP、MEM）
3. [NVIDIA](report/chapters/03_NVIDIA.md) — GR00T N1→N1.7 / Cosmos + DreamGen / DreamZero→GR00T N2
4. [Skild AI](report/chapters/04_Skild_AI.md) — Skild Brain / 跨身体上下文适应 / S1
5. [Generalist AI](report/chapters/05_Generalist_AI.md) — GEN-0 / GEN-1 / GEN-1.5

**第二篇 美国人形整机**
6. [特斯拉 Optimus](report/chapters/06_Tesla_Optimus.md) — 端到端 + 神经世界模拟器 / 人类视频学习 / Gen 3 + AI5 + 产线
7. [Figure AI](report/chapters/07_Figure_AI.md) — Helix / Helix 02 / Helix 2.5 + Index
8. [1X Technologies](report/chapters/08_1X_Technologies.md) — 1X World Model / Redwood / 1XWM
9. [Boston Dynamics](report/chapters/09_Boston_Dynamics.md) — Atlas LBM / 动捕 RL / 量产 Atlas + Gemini
10. [Agility Robotics](report/chapters/10_Agility_Robotics.md) — 双足 RL / 全身控制基础模型 / Digit 5
11. [Sunday Robotics](report/chapters/11_Sunday_Robotics.md) — ALOHA·ACT / UMI / ACT-1

**第三篇 科技巨头与基础设施**
12. [Meta FAIR](report/chapters/12_Meta_FAIR.md) — Meta Motivo / V-JEPA 2 / RoboJEPA
13. [Amazon FAR](report/chapters/13_Amazon_FAR.md) — RFM-1 / OmniRetarget / ResMimic + Holosoma
14. [Hugging Face](report/chapters/14_Hugging_Face_LeRobot.md) — LeRobot / SmolVLA / LeRobot v0.6
15. [丰田研究院 TRI](report/chapters/15_Toyota_Research_Institute.md) — Diffusion Policy / LBM 严谨评测 / Atlas·Walden·丰田
16. [Genesis AI](report/chapters/16_Genesis_AI.md) — Genesis 引擎 / GENE-26.5 / Eno

**第四篇 学术策源地**
17. [斯坦福](report/chapters/17_Stanford.md) — OpenVLA / HumanPlus / BEHAVIOR
18. [UC Berkeley](report/chapters/18_UC_Berkeley.md) — Octo / HIL-SERL / 人形 RL
19. [CMU](report/chapters/19_CMU.md) — H2O·OmniH2O / ASAP / BFM-Zero
20. [UCSD](report/chapters/20_UCSD.md) — ExBody / Open-TeleVision / Humanoid Policy ~ Human Policy
21. [清华大学](report/chapters/21_Tsinghua.md) — RDT-1B / Vidar / RDT2
22. [香港大学](report/chapters/22_HKU.md) — UniVLA / RoboTwin 2.0 / WholeBodyVLA
23. [上海人工智能实验室](report/chapters/23_Shanghai_AI_Lab.md) — GRUtopia / InternVLA-M1 / InternVLA-A1
24. [北京智源研究院](report/chapters/24_BAAI.md) — RoboBrain 1.0 + RoboOS / 2.0 / 2.5

**第五篇 中国科技大厂**
25. [字节跳动 Seed](report/chapters/25_ByteDance_Seed.md) — GR-1·GR-2 / GR-3 / GR-RL + Robix
26. [阿里巴巴](report/chapters/26_Alibaba_DAMO.md) — WorldVLA·RynnVLA / RynnEC·RynnRCP / RynnBrain
27. [蚂蚁灵波](report/chapters/27_Ant_Robbyant.md) — LingBot-VLA / LingBot-World / LingBot-Depth
28. [腾讯](report/chapters/28_Tencent.md) — Tairos / HY-Embodied-0.5 / RxBrain
29. [小米](report/chapters/29_Xiaomi.md) — MiMo-Embodied / Xiaomi-Robotics-0 / Xiaomi-Robotics-1
30. [小鹏汽车](report/chapters/30_XPeng.md) — 世界基座模型 / VLA 2.0 / IRON

**第六篇 中国人形整机与国家平台**
31. [宇树科技](report/chapters/31_Unitree.md) — 开源运控生态 / UnifoLM-WMA-0 / UnifoLM-WLA-1.0
32. [智元机器人](report/chapters/32_AgiBot.md) — AgiBot World + GO-1 / Genie Envisioner / GO-2 → GO-3 Preview
33. [优必选](report/chapters/33_UBTech.md) — 群脑网络 / Co-Agent / Thinker
34. [美的集团](report/chapters/34_Midea.md) — DexVLA / ChatVLA / PointVLA
35. [逐际动力](report/chapters/35_LimX_Dynamics.md) — System 0 / System 1 / COSA
36. [北京人形机器人创新中心](report/chapters/36_X_Humanoid.md) — RoboMIND / Pelican-VL / XR-1
37. [银河通用](report/chapters/37_Galbot.md) — GraspVLA / NavFoM / 银河星脑
38. [星动纪元](report/chapters/38_Robot_Era.md) — VPP / ERA-42 / Ctrl-World

**第七篇 中国具身大脑创业公司**
39. [星海图](report/chapters/39_Galaxea.md) — G0 + GOD / G0 Plus / G0.5 + Fast-WAM + G-Fleet
40. [自变量机器人](report/chapters/40_X_Square_Robot.md) — WALL-OSS / WALL-B / WALL-WM + WALL-SS
41. [千寻智能](report/chapters/41_Spirit_AI.md) — 数据缩放定律 / Spirit v1.5 / Spirit v1.6 与 RoboArena
42. [智平方](report/chapters/42_AI2_Robotics.md) — GOVLA + FiS-VLA / Video2Act / NeuroVLA
43. [它石智航](report/chapters/43_TARS.md) — WIYH / AWE 3.0 / AWE 3.5
44. [极佳视界](report/chapters/44_GigaAI.md) — GigaWorld-0 / GigaBrain-0 / GigaBrain-0.5M\* + GigaWorld-1
45. [原力灵机](report/chapters/45_Dexmal.md) — RoboChallenge / Dexbotic / DM0
46. [灵初智能](report/chapters/46_Psi_Robot.md) — DexGraspVLA / Psi-R1 / Psi-R2 + Psi-W0
47. [智在无界](report/chapters/47_BeingBeyond.md) — Being-H0 / H0.5 / H0.7
48. [穹彻智能](report/chapters/48_Noematrix.md) — AnyGrasp + RH20T / Noematrix Brain / Noe-0
49. [地平线 / 地瓜机器人](report/chapters/49_Horizon_Robotics.md) — H-RDT / HoloMotion / HoloBrain-0
50. [星尘智能](report/chapters/50_Astribot.md) — Astribot S1 / Astribot Suite / Lumo + DuoCore

### 第三部分 综合研判
- [横向对比与身位矩阵](report/90_横向对比与身位矩阵.md)（50 家一表、梯队、分维度横评、六组关键对决、路线谱系图）
- [趋势、预测与建议](report/91_趋势与未来展望.md)（2026 十大趋势、2027–2029 预测与置信度、关键不确定性、给不同读者的建议）

### 附录
- [附录](report/99_附录.md)（大事年表 2016–2026.10、150 项代表作索引、术语表、研究方法与局限）

### 合并版
- [全文合并版（单文件）](report/全文合并版.md)

---

## 关于准确性

- 每章末尾的"**核验说明**"逐条标注了信息来源与可靠性：论文事实 / 官方发布 / 公司口径 / 单一来源 / 本报告分析；
- 2026 年大量"全球第一""登顶"宣称来自公司自评或自有榜单，报告中均已降级标注；
- 研究环境无法直接打开部分网站原文，核验主要依靠检索引擎提供的原文摘录与多源比对，未能逐字核实的数字已注明；
- 评级与预测（第 90、91 章）为分析判断，不构成投资建议。
