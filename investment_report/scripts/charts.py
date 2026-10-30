"""静态图表（用于 Word 报告）。配色采用 dataviz 参考调色板（浅色模式），单系列不设图例，多系列 ≤3 并直接标注。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "charts")
GEN = os.path.join(HERE, "..", "generated")
os.makedirs(OUT, exist_ok=True)

for p in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MUTED = "#a8a7a2"


def base(figsize=(8, 4.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def finish(fig, ax, title, subtitle, source, name):
    fig.text(0.01, 0.97, title, fontsize=12.5, color=INK, weight="bold", va="top")
    fig.text(0.01, 0.905, subtitle, fontsize=9, color=INK2, va="top")
    fig.text(0.01, 0.015, source, fontsize=7.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.86))
    fig.savefig(os.path.join(OUT, name), facecolor=SURFACE)
    plt.close(fig)


# 1. 资本 vs 收入：中国具身融资额与全球人形销售额
fig, ax = base()
labels = ["2024\n中国具身融资\n（量子位）", "2025\n中国具身融资\n（量子位）", "2026H1\n中国具身融资\n（IT桔子）", "2025\n全球人形\n销售额（IDC）"]
vals = [137, 554, 935, 4.4 * 7.1]
colors = [S1, S1, S1, S2]
bars = ax.bar(labels, vals, color=colors, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 15, f"{v:,.0f} 亿元", ha="center", fontsize=9, color=INK)
ax.set_ylabel("亿元人民币", color=INK2, fontsize=9)
finish(fig, ax, "资本流入远超产业收入：半年融资额约为全球人形年销售额的 30 倍",
       "中国具身智能融资额（蓝）与 2025 年全球人形机器人销售额（橙，约 4.4 亿美元，按 7.1 汇率折算）",
       "来源：量子位智库、IT 桔子、IDC（经媒体转引）；口径差异见附录 5。", "c01_capital_vs_revenue.png")

# 2. 全球出货
fig, ax = base()
labels = ["2025 全年\n（IDC）", "2026H1\n（Counterpoint）", "2026E 全年\n（Counterpoint 预测）"]
vals = [1.8, 2.2, 5.0]
bars = ax.bar(labels, vals, color=[S1, S1, MUTED], width=0.5)
for b, v, t in zip(bars, vals, ["约 1.8 万台", ">2.2 万台", ">5 万台"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.08, t, ha="center", fontsize=9, color=INK)
ax.set_ylabel("万台", color=INK2, fontsize=9)
finish(fig, ax, "全球人形机器人出货：从万台级走向五万台级",
       "出货口径；工信部口径的中国产量（2025 约 2 万台、2026H1 超 4 万台）显著高于出货，产量≠出货",
       "来源：IDC、Counterpoint Research（经媒体转引）、工信部讲话（转引）。", "c02_shipments.png")

# 3. 2026H1 厂商份额（横向条形）
fig, ax = base((8, 3.8))
names = ["智元", "宇树", "银河通用", "优必选", "乐聚", "其他"]
vals = [9700, 7000, 1100, 968, 650, 22000 - 9700 - 7000 - 1100 - 968 - 650]
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=GRID, linewidth=0.6)
bars = ax.barh(names[::-1], vals[::-1], color=[MUTED] + [S1] * 5, height=0.55)
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 150, b.get_y() + b.get_height() / 2, f"{v:,} 台（{v/22000:.0%}）", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, 12500)
ax.set_xlabel("台", color=INK2, fontsize=9)
finish(fig, ax, "2026 年上半年全球人形出货高度集中：前五均为中国厂商，合计约 86%",
       "以全球约 2.2 万台为分母；优必选按 4.4% 份额推算；'其他'为余数",
       "来源：Counterpoint Research（2026-08，经媒体转引），B 级。", "c03_vendor_share.png")

# 4. 采购结构
fig, ax = base((8, 3.4))
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=GRID, linewidth=0.6)
cats = ["教育科研机构", "政府及国资平台", "工业企业", "其他"]
v = [55.5, 20.6, 21.1, 2.8]
bars = ax.barh(cats[::-1], v[::-1], color=[MUTED, S2, S1, S1][::-1][::-1], height=0.5)
for b, x in zip(bars, v[::-1]):
    ax.text(x + 0.8, b.get_y() + b.get_height() / 2, f"{x:.1f}%", va="center", fontsize=9, color=INK)
ax.set_xlim(0, 65)
ax.set_xlabel("占公开中标金额比例（%）", color=INK2, fontsize=9)
finish(fig, ax, "谁在买人形机器人：非 ROI 型买方占约八成",
       "2026H1 国内公开披露的人形/具身机器人中标项目（218 个、17.23 亿元）按采购方类型",
       "来源：中国人形机器人场景应用联盟统计（经蓝鲸新闻转引），C 级。", "c04_buyers.png")

# 5. 估值/收入倍数（对数轴）
with open(os.path.join(GEN, "model_results.json"), encoding="utf-8") as f:
    R = json.load(f)
mult = [(m["name"], m["ev_s"]) for m in R["multiples"] if m["ev_s"]]
mult.sort(key=lambda x: x[1])
fig, ax = base((8, 4.4))
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=GRID, linewidth=0.6)
names = [m[0] for m in mult]
vals = [m[1] for m in mult]
bars = ax.barh(names, vals, color=S1, height=0.55)
ax.set_xscale("log")
for b, x in zip(bars, vals):
    ax.text(x * 1.08, b.get_y() + b.get_height() / 2, f"{x:,.0f}x", va="center", fontsize=8.5, color=INK)
ax.set_xlim(5, 4000)
from matplotlib.ticker import FuncFormatter, FixedLocator
ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300, 1000, 3000]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}x"))
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.set_xlabel("估值或市值 / 最近完整年度收入（对数轴）", color=INK2, fontsize=9)
finish(fig, ax, "估值与收入严重脱钩：同一家公司在不同时点相差一个数量级",
       "估值口径各异（发行价、开盘价、收盘价、私募轮次、SPAC 协议），仅作量级参考",
       "来源：交易所/招股书解读、公司公告与媒体报道；本报告推算（附录 5、数据底表 14）。", "c05_multiples.png")

# 6. TCO：每 FTE 年化成本
tco = R["tco_factory"]
fig, ax = base((8, 4.0))
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=GRID, linewidth=0.6)
names = [t["方案"] for t in tco][::-1]
vals = [t["单位人力当量年成本"] for t in tco][::-1]
cols = [S3 if "人形" not in n else S1 for n in names]
bars = ax.barh(names, vals, color=cols, height=0.55)
ax.axvline(10, color=S2, linewidth=1.5)
ax.text(10.3, len(names) - 0.45, "人工：10 万元/人·年", color=INK, fontsize=8.5)
for b, x in zip(bars, vals):
    ax.text(x + 0.4, b.get_y() + b.get_height() / 2, f"{x:.1f}", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, 32)
ax.set_xlabel("每替代 1 个人力当量的年化总成本（万元，含资本成本 8%）", color=INK2, fontsize=9)
finish(fig, ax, "工厂物料搬运：2026 年人形的单位人力成本约为人工的 2.8 倍",
       "蓝：人形（2026/2028/2030 参数）；绿：非人形替代方案（2026）；橙线：一线工人全口径成本",
       "来源：本报告 TCO 模型（D 级假设，参数见第六篇与数据底表 11）。", "c06_tco.png")

# 7. 三情景出货
fig, ax = base((8, 4.2))
years = [e["year"] for e in R["scenarios"]["基准"]]
for name, col in (("乐观", S2), ("基准", S1), ("悲观", S3)):
    ys = [e["units_wan"] for e in R["scenarios"][name]]
    ax.plot(years, ys, color=col, linewidth=2)
    ax.text(years[-1] + 0.15, ys[-1], f"{name} {ys[-1]:g} 万台", color=INK, fontsize=8.5, va="center")
ax.set_yscale("log")
from matplotlib.ticker import FuncFormatter, FixedLocator
ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300, 1000]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.yaxis.set_minor_locator(FixedLocator([]))
ax.set_xlim(2025, 2037.2)
ax.set_ylabel("全球年出货（万台，对数轴）", color=INK2, fontsize=9)
finish(fig, ax, "2025–2035 全球人形出货三情景：差距达 20 倍",
       "2025 年为 IDC 实际值；对照：高盛 2026-09 将 2035 年预测上调至约 650 万台",
       "来源：本报告情景模型（D 级）；IDC；高盛（经媒体转引）。", "c07_scenarios.png")

# 8. 宇树与优必选收入
fig, ax = base((8, 3.8))
labels = ["2023", "2024", "2025", "2026H1"]
ub = [10.56, 13.05, 20.01, 12.7]
ut = [None, 3.93, 17.08, 11.52]
import numpy as np
x = np.arange(len(labels))
w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, [v or 0 for v in ut], w, color=S1)
b2 = ax.bar(x + w / 2 + 0.01, ub, w, color=S2)
for bars, vals in ((b1, ut), (b2, ub)):
    for b, v in zip(bars, vals):
        if v:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x, labels)
ax.set_ylabel("营业收入（亿元）", color=INK2, fontsize=9)
ax.legend([b1, b2], ["宇树科技（2025 毛利率约 60%，盈利）", "优必选（2025 毛利率 37.7%，亏损）"], frameon=False, fontsize=8.5, loc="upper left")
finish(fig, ax, "两种商业模式：卖'研究工具'的宇树盈利，卖'工业+政府项目'的优必选仍亏损",
       "营业收入；宇树 2023 年未纳入（口径未核实），2024 年为媒体转述招股书数据",
       "来源：宇树招股书解读与上市公告、优必选年报/中报（经媒体转引）；优必选 2023–2024 年为年报口径近似值。", "c08_revenue.png")
print(sorted(os.listdir(OUT)))
