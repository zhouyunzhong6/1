"""具身智能投资研究报告 · 量化模型

所有输出均写入 ../generated/ (Markdown 表格片段 + JSON)，供正文与数据底表引用。
数值分三类：
  [公开] 来自公开披露（证据等级见 sources.py）
  [锚定] 以公开数据为锚、经推算得到
  [假设] 示例性研究假设（D 级），不代表任何企业真实财务信息
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "generated")
os.makedirs(OUT, exist_ok=True)

RESULTS = {}


def fmt(x, nd=1):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:,.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def save(name, text):
    with open(os.path.join(OUT, name + ".md"), "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


# ---------------------------------------------------------------------------
# 模型 1：BOM 拆解（全尺寸工业人形，中国供应链，2026 年千台级小批量）
# 单位：元/台。低/基准/高三档。
# 校验锚点：①优必选 Walker S3 BOM 目标 25–28 万元（C 级）；
#          ②摩根士丹利估算 Optimus Gen2 中国供应链 BOM 约 4.6 万美元（B/C 级）；
#          ③券商口径"线性执行器+旋转执行器+灵巧手三大件约占整机成本 73%"（C 级）。
# ---------------------------------------------------------------------------
BOM_FULL = [
    # 名称, 数量, 单价低, 单价基准, 单价高, 说明
    ("旋转关节模组（无框力矩电机+谐波/行星减速器+双编码器+驱动器）", 20, 1800, 3800, 6000, "肩肘腕髋膝踝及腰颈；国产化率高，价格年降 10–20%"),
    ("线性关节模组（行星滚柱丝杠+电机+驱动）", 8, 2500, 5000, 9000, "腿部/手臂直线驱动；行星滚柱丝杠国产化率约 20%、良率偏低"),
    ("灵巧手（只）", 2, 6000, 30000, 70000, "6 自由度连杆手≈数千元；20+ 自由度带触觉手≈数万至十余万元"),
    ("六维力/关节力矩传感器", 4, 2000, 5000, 12000, "腕/踝；高精度六维力仍以海外为主"),
    ("视觉与环境感知（RGB-D×3–4、激光雷达、IMU）", 1, 3000, 8000, 15000, "国产深度相机/激光雷达已具成本优势"),
    ("主控与端侧 AI 计算", 1, 4000, 15000, 30000, "国产 SoC 方案≈数千元；Jetson Thor 模组千片价 2,999 美元"),
    ("电池包与电源管理（1–2 kWh）", 1, 2000, 5000, 8000, "可换电设计增加结构与 BMS 成本"),
    ("结构件、外壳、线束、散热", 1, 8000, 18000, 28000, "铝镁合金/工程塑料；小批量模具摊销高"),
]
ASSEMBLY = {"低": 0.08, "基准": 0.12, "高": 0.18}  # 装配、测试、良率损失（占物料比例）


def bom_calc(items, assembly):
    res = {}
    for k, idx in (("低", 2), ("基准", 3), ("高", 4)):
        mat = sum(q * it[idx] for it in items for q in [it[1]])
        res[k] = {"物料": mat, "装配测试良率": mat * assembly[k], "合计": mat * (1 + assembly[k])}
    return res


bom = bom_calc(BOM_FULL, ASSEMBLY)
base_mat = bom["基准"]["物料"]
rows = []
for name, q, lo, ba, hi, note in BOM_FULL:
    rows.append([name, q, f"{lo*q/1e4:.1f}", f"{ba*q/1e4:.1f}", f"{hi*q/1e4:.1f}", f"{ba*q/base_mat*100:.0f}%", note])
for k in ("物料", "装配测试良率", "合计"):
    rows.append([f"**{k}**", "", f"{bom['低'][k]/1e4:.1f}", f"{bom['基准'][k]/1e4:.1f}", f"{bom['高'][k]/1e4:.1f}",
                 "100%" if k == "物料" else "", ""])
save("t_bom_full", md_table(["部件", "数量", "低（万元）", "基准（万元）", "高（万元）", "基准占物料比", "说明"], rows))

big3 = sum(it[1] * it[3] for it in BOM_FULL[:3]) / base_mat
RESULTS["bom_full"] = {k: round(v["合计"]) for k, v in bom.items()}
RESULTS["bom_big3_share"] = round(big3, 3)

# 小尺寸教育科研人形（G1 类，约 23–29 自由度，无高阶灵巧手）
BOM_SMALL = [
    ("旋转关节模组（小功率）", 23, 600, 900, 1300, ""),
    ("计算（主控+运控）", 1, 1500, 3000, 6000, ""),
    ("感知（深度相机、雷达、IMU）", 1, 1500, 3000, 5000, ""),
    ("电池", 1, 800, 1500, 2500, ""),
    ("结构件、外壳、线束", 1, 2500, 4000, 6000, ""),
]
bom_s = bom_calc(BOM_SMALL, {"低": 0.06, "基准": 0.10, "高": 0.15})
RESULTS["bom_small"] = {k: round(v["合计"]) for k, v in bom_s.items()}
price_small = 99000
RESULTS["bom_small_gm_at_99k"] = round(1 - bom_s["基准"]["合计"] / price_small, 3)

# 规模化降本：经验曲线（累计产量每翻一倍成本下降 LR），并设物理/材料成本地板
def learning(cost0, cum0, cum1, lr, floor_share):
    doublings = math.log2(cum1 / cum0)
    raw = cost0 * (1 - lr) ** doublings
    floor = cost0 * floor_share
    return max(raw, floor), doublings


lc_rows = []
for lr in (0.10, 0.15, 0.20):
    for cum1 in (50_000, 200_000, 1_000_000):
        c, d = learning(bom["基准"]["合计"], 5_000, cum1, lr, 0.28)
        lc_rows.append([f"{int(lr*100)}%", f"{cum1/1e4:.0f} 万台", f"{d:.1f}", f"{c/1e4:.1f}", f"{c/7.1/1e3:.1f}"])
save("t_bom_learning", md_table(["学习率（每翻倍降幅）", "单一企业累计产量", "翻倍次数", "单台 BOM（万元）", "≈千美元"], lc_rows))

# ---------------------------------------------------------------------------
# 模型 2：场景 TCO / ROI —— 中国汽车/3C 工厂物料搬运与上下料（双班）
# ---------------------------------------------------------------------------
HOURS = 16 * 300  # 双班、年 300 天
WAGE = 10.0  # 万元/人·年，一线工人全口径用工成本（工资+社保公积金+管理）[锚定：统计局规模以上企业平均工资 10.6 万元，私营 7.2 万元]


def annuity(pv, rate, years):
    return pv * rate / (1 - (1 + rate) ** -years)


def tco(name, price, integ, life, maint, sup_ratio, sup_cost, energy_kw, soft, throughput, uptime, rate=0.08, wage=WAGE):
    capex = price + integ
    cap_annual = annuity(capex, rate, life)
    opex = price * maint + sup_cost / sup_ratio + energy_kw * HOURS * 0.8 / 1e4 + soft
    fte = 2 * throughput * uptime  # 替代的人力当量（双班）
    saving = fte * wage
    net = saving - opex
    payback = capex / net if net > 0 else None
    annual_cost = cap_annual + opex
    cost_per_fte = annual_cost / fte
    return {
        "方案": name, "初始投入": capex, "年化资本成本": cap_annual, "年运营成本": opex,
        "替代人力当量": fte, "年节省人工": saving, "年净现金流(未含资本)": net,
        "静态回本(年)": payback, "单位人力当量年成本": cost_per_fte,
    }


SCEN_FACTORY = [
    # name, price, integ, life, maint, sup_ratio(台/人), sup_cost, kW, soft, throughput, uptime
    ("人形·2026 现状（自主+远程兜底）", 40, 8, 5, 0.10, 5, 25, 0.6, 1.0, 0.50, 0.80),
    ("人形·2028 改进", 25, 5, 5, 0.08, 10, 25, 0.6, 1.0, 0.70, 0.88),
    ("人形·2030 规模化", 15, 3, 6, 0.06, 20, 25, 0.5, 1.0, 0.85, 0.92),
    ("轮式移动操作（复合机器人）·2026", 22, 8, 6, 0.06, 15, 25, 0.5, 0.5, 0.80, 0.92),
    ("固定式机械臂+工装/输送·2026", 18, 12, 8, 0.05, 25, 25, 0.8, 0.2, 1.10, 0.96),
]
tco_rows = []
tco_json = []
for s in SCEN_FACTORY:
    r = tco(*s)
    tco_json.append(r)
    tco_rows.append([r["方案"], fmt(r["初始投入"]), fmt(r["年运营成本"]), fmt(r["替代人力当量"], 2), fmt(r["年节省人工"]),
                     fmt(r["年净现金流(未含资本)"]), "不回本" if r["静态回本(年)"] is None else f"{r['静态回本(年)']:.1f}",
                     fmt(r["单位人力当量年成本"])])
human_row = ["人工（双班 2 人）", "—", f"{2*WAGE:.1f}", "2.00", "—", "—", "—", f"{WAGE:.1f}"]
tco_rows.append(human_row)
save("t_tco_factory", md_table(["方案", "初始投入（万元）", "年运营成本（万元）", "替代人力当量（FTE）", "年节省人工（万元）",
                                "年净现金流（万元）", "静态回本（年）", "每 FTE 年化成本（万元）"], tco_rows))
RESULTS["tco_factory"] = tco_json

# 敏感性：人形 2028 参数下，整机价格 × 吞吐率 → 静态回本年数
prices = [50, 40, 30, 20, 15, 10]
thrus = [0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
sens = []
for p in prices:
    row = [f"{p} 万元"]
    for t in thrus:
        r = tco("x", p, max(3, p * 0.2), 5, 0.08, 10, 25, 0.6, 1.0, t, 0.88)
        row.append("×" if r["静态回本(年)"] is None or r["静态回本(年)"] > 8 else f"{r['静态回本(年)']:.1f}")
    sens.append(row)
save("t_sens_payback", md_table(["整机价格 \\ 吞吐率（相对熟练工）"] + [f"{int(t*100)}%" for t in thrus], sens))

# 敏感性：远程/现场人员配比（台/人）× 可用率，在 2028 价格下
sup_ratios = [3, 5, 10, 20, 50]
uptimes = [0.7, 0.8, 0.88, 0.95]
sens2 = []
for sr in sup_ratios:
    row = [f"1 人看护 {sr} 台"]
    for u in uptimes:
        r = tco("x", 25, 5, 5, 0.08, sr, 25, 0.6, 1.0, 0.7, u)
        row.append("×" if r["静态回本(年)"] is None or r["静态回本(年)"] > 8 else f"{r['静态回本(年)']:.1f}")
    sens2.append(row)
save("t_sens_supervision", md_table(["人员配比 \\ 可用率"] + [f"{int(u*100)}%" for u in uptimes], sens2))

# ---------------------------------------------------------------------------
# 模型 3：美国仓储 RaaS（以 Agility SEC 文件中的示意性数字为锚）
# 公开锚点：示意性 RaaS 约 8,500 美元/月；宣称替代约 20 万美元/年全口径人工（C 级，S-4/投资者日材料）
# ---------------------------------------------------------------------------
raas_month = 8500
raas_year = raas_month * 12
claimed_labor = 200_000
rows = []
for real in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
    saving = claimed_labor * real
    rows.append([f"{int(real*100)}%", f"{saving/1e3:.0f}", f"{raas_year/1e3:.0f}", f"{(saving-raas_year)/1e3:+.0f}",
                 "是" if saving > raas_year else "否"])
save("t_raas", md_table(["实际达成的宣称替代量", "客户年节省人工（千美元）", "RaaS 年费（千美元）", "客户年净收益（千美元）", "客户是否获益"], rows))
RESULTS["raas_breakeven_share"] = round(raas_year / claimed_labor, 3)

# 供应商侧：RaaS 单台回收期（假设）
vendor_rows = []
for hw_cost in (150_000, 100_000, 60_000, 35_000):
    for svc in (0.25, 0.15):
        gp = raas_year * (1 - svc)
        vendor_rows.append([f"{hw_cost/1e3:.0f}", f"{int(svc*100)}%", f"{gp/1e3:.0f}", f"{hw_cost/gp:.1f}"])
save("t_raas_vendor", md_table(["单台硬件成本（千美元）", "服务与运维占收入", "单台年毛利（千美元）", "硬件回收期（年）"], vendor_rows))

# ---------------------------------------------------------------------------
# 模型 4：家庭场景（以 1X NEO 公开价格为锚：2 万美元或 499 美元/月）
# ---------------------------------------------------------------------------
home_rows = []
for hrs in (0.5, 1, 2, 3):
    for q in (0.5, 0.8):
        value = hrs * 365 * 25 * q
        home_rows.append([hrs, f"{int(q*100)}%", f"{value:,.0f}", f"{499*12:,.0f}", "是" if value > 499 * 12 else "否"])
save("t_home", md_table(["日均有效工作（小时）", "质量折算（相对家政服务）", "年化价值（美元，按 25 美元/小时）", "订阅年费（美元）", "经济上成立"], home_rows))

# ---------------------------------------------------------------------------
# 模型 5：自下而上市场空间（中国制造业"人形可替代岗位"）—— D 级研究假设
# ---------------------------------------------------------------------------
sam = [
    ("制造业一线操作岗位（人）", "6,000–8,000 万", "统计口径推断：制造业城镇就业约 1 亿量级，其中一线操作约六至八成"),
    ("其中物料搬运、上下料、分拣、包装、巡检类", "15%–20% → 900–1,600 万", "按典型工厂岗位结构假设"),
    ("其中难以被固定自动化/AGV 经济替代的（多品种、短周期、空间受限）", "25%–35% → 220–560 万", "核心假设：这是人形/移动操作的真正可服务空间"),
    ("每个双班岗位所需机器人台数（成熟期）", "0.6–0.8 台", "取决于吞吐率与可用率（见 TCO 模型 2030 列）"),
    ("长期可服务存量（台）", "130–450 万台", "= 岗位 × 台数"),
    ("其中人形形态占比（相对轮式移动操作）", "30%–60%", "取决于台阶/狭窄空间与双臂作业需求"),
    ("中国制造业人形长期存量空间（台）", "40–270 万台", "约 10 年渗透期 → 年需求 4–27 万台（成熟期）"),
    ("对应年市场规模（按 10–15 万元/台）", "约 40–400 亿元/年", "对比：2025 全球人形销售额约 4.4 亿美元（IDC）"),
]
save("t_sam", md_table(["推导步骤", "数值区间", "依据/说明"], sam))

# ---------------------------------------------------------------------------
# 模型 6：三情景（全球人形机器人出货量、均价、市场规模）
# 2025 实际：约 1.8 万台、约 4.4 亿美元（IDC，B 级）；2026H1 约 2.2 万台（Counterpoint，B 级）
# ---------------------------------------------------------------------------
years = list(range(2025, 2036))
SCEN = {
    "悲观": {"units": {2025: 1.8, 2026: 4.0, 2027: 5.5, 2028: 7.0, 2029: 9.0, 2030: 12, 2031: 15, 2032: 19, 2033: 24, 2034: 31, 2035: 40},
             "asp": {2025: 24.4, 2026: 22, 2030: 18, 2035: 16}},
    "基准": {"units": {2025: 1.8, 2026: 5.0, 2027: 9.0, 2028: 15, 2029: 24, 2030: 38, 2031: 52, 2032: 70, 2033: 95, 2034: 128, 2035: 170},
             "asp": {2025: 24.4, 2026: 22, 2030: 15, 2035: 12}},
    "乐观": {"units": {2025: 1.8, 2026: 5.5, 2027: 14, 2028: 32, 2029: 65, 2030: 120, 2031: 200, 2032: 320, 2033: 470, 2034: 630, 2035: 800},
             "asp": {2025: 24.4, 2026: 22, 2030: 14, 2035: 10}},
}


def interp(d, y):
    ks = sorted(d)
    if y in d:
        return d[y]
    for a, b in zip(ks, ks[1:]):
        if a < y < b:
            return d[a] + (d[b] - d[a]) * (y - a) / (b - a)
    return d[ks[-1]]


scen_out = {}
rows = []
for name, s in SCEN.items():
    scen_out[name] = []
    for y in years:
        u = s["units"][y]
        asp = interp(s["asp"], y)
        m = u * asp / 10  # 万台 × 千美元 / 10 = 亿美元
        scen_out[name].append({"year": y, "units_wan": u, "asp_k_usd": round(asp, 1), "market_100m_usd": round(m, 1)})
for y in (2025, 2026, 2027, 2028, 2030, 2033, 2035):
    row = [y]
    for name in ("悲观", "基准", "乐观"):
        e = next(x for x in scen_out[name] if x["year"] == y)
        row.append(f"{e['units_wan']:g} 万台 / {e['asp_k_usd']:g}k / {e['market_100m_usd']:.0f} 亿美元")
    rows.append(row)
save("t_scenarios", md_table(["年份", "悲观（出货/均价/规模）", "基准", "乐观"], rows))
RESULTS["scenarios"] = scen_out

# ---------------------------------------------------------------------------
# 模型 7：VC 情景回报模型（三类典型项目，示例性假设，D 级）
# ---------------------------------------------------------------------------
def vc_model(name, post, invest, dilutions, years, outcomes, pref=1.0):
    own0 = invest / post
    keep = 1.0
    for d in dilutions:
        keep *= (1 - d)
    own = own0 * keep
    rows, ev = [], 0.0
    for label, p, exit_val, kind in outcomes:
        # IPO 时优先股转普通股，按持股比例；并购/清算时近似 1x 不参与分配优先清算权（假设与后续轮次同级）
        if exit_val <= 0:
            proceeds = 0.0
        elif kind == "ma":
            proceeds = max(exit_val * own, min(exit_val, invest * pref))
        else:
            proceeds = exit_val * own
        moic = proceeds / invest
        ev += p * moic
        rows.append([label, f"{int(p*100)}%", fmt(exit_val), f"{proceeds:.2f}", f"{moic:.2f}x"])
    irr = ev ** (1 / years) - 1 if ev > 0 else -1
    return {"name": name, "post": post, "invest": invest, "own0": own0, "own_exit": own, "rows": rows, "E_moic": ev, "irr": irr, "years": years}


VC = [
    vc_model("A. 机器人基础模型初创（A 轮）", post=40, invest=2, dilutions=[0.15, 0.15, 0.12], years=6, outcomes=[
        ("成为平台型公司并上市：2031 年收入 30 亿元，EV/S 15x", 0.10, 450, "ipo"),
        ("被大厂/整机企业战略收购", 0.25, 80, "ma"),
        ("技术团队收购/平价退出", 0.30, 30, "ma"),
        ("失败清算", 0.35, 0, "fail"),
    ]),
    vc_model("B. 人形整机初创（B 轮）", post=80, invest=3, dilutions=[0.15, 0.15, 0.12, 0.10], years=5, outcomes=[
        ("头部上市：收入 40 亿元、毛利率 30%，EV/S 6x", 0.20, 240, "ipo"),
        ("中等上市：收入 15 亿元，EV/S 5x", 0.25, 75, "ipo"),
        ("行业整合中被并购", 0.25, 40, "ma"),
        ("失败/僵尸化", 0.30, 0, "fail"),
    ]),
    vc_model("C. 已有真实客户的工业场景机器人公司（B 轮）", post=15, invest=1, dilutions=[0.12, 0.10], years=5, outcomes=[
        ("上市：收入 10 亿元、毛利率 40%，EV/S 5x", 0.35, 50, "ipo"),
        ("增长放缓上市：收入 5 亿元，EV/S 4x", 0.30, 20, "ipo"),
        ("产业并购", 0.20, 12, "ma"),
        ("失败", 0.15, 0, "fail"),
    ]),
]
vc_md = []
for m in VC:
    vc_md.append(f"**{m['name']}**：投后估值 {m['post']} 亿元，投资 {m['invest']} 亿元，初始持股 {m['own0']*100:.2f}%，"
                 f"经后续稀释退出时持股 {m['own_exit']*100:.2f}%。期望 MOIC **{m['E_moic']:.2f}x**，"
                 f"{m['years']} 年对应 IRR 约 **{m['irr']*100:.0f}%**（并购情景按 1x 优先清算权近似、IPO 情景按持股比例）。\n")
    vc_md.append(md_table(["退出情景", "概率", "退出估值（亿元）", "回收（亿元）", "MOIC"], m["rows"]))
    vc_md.append("")
save("t_vc_models", "\n".join(vc_md))
RESULTS["vc"] = [{k: v for k, v in m.items() if k != "rows"} for m in VC]

# 敏感性：A 类项目——入场估值 × 平台化概率 → 期望 MOIC
posts = [20, 30, 40, 60, 100, 150]
probs = [0.05, 0.10, 0.20, 0.30]
rows = []
for p_ in posts:
    row = [f"{p_} 亿元"]
    for pr in probs:
        rest = 1 - pr
        m = vc_model("x", p_, 2, [0.15, 0.15, 0.12], 6, [
            ("平台", pr, 450, "ipo"), ("并购", rest * 0.28, 80, "ma"), ("平价", rest * 0.33, 30, "ma"), ("失败", rest * 0.39, 0, "fail")])
        row.append(f"{m['E_moic']:.2f}x")
    rows.append(row)
save("t_vc_sens_a", md_table(["入场投后估值 \\ 平台化概率"] + [f"{int(p*100)}%" for p in probs], rows))

# 敏感性：B 类项目——入场估值 × 头部上市退出估值
posts = [30, 50, 80, 120, 200]
exits = [120, 240, 400, 800]
rows = []
for p_ in posts:
    row = [f"{p_} 亿元"]
    for ex in exits:
        m = vc_model("x", p_, 3, [0.15, 0.15, 0.12, 0.10], 5, [
            ("头部", 0.20, ex, "ipo"), ("中等", 0.25, 75, "ipo"), ("并购", 0.25, 40, "ma"), ("失败", 0.30, 0, "fail")])
        row.append(f"{m['E_moic']:.2f}x")
    rows.append(row)
save("t_vc_sens_b", md_table(["入场投后估值 \\ 头部情景退出估值（亿元）"] + [str(e) for e in exits], rows))

# ---------------------------------------------------------------------------
# 模型 8：估值倍数横截面（公开数据推算，B/C 级）
# ---------------------------------------------------------------------------
MULT = [
    # 公司, 估值/市值（亿元）, 口径, 收入（亿元）, 收入口径, 备注
    ("宇树科技（IPO 发行价）", 609.93, "发行价对应总市值，2026-08", 17.08, "2025 年营收", "发行市盈率 219 倍"),
    ("宇树科技（首日开盘）", 4449, "开盘价 1,100 元，2026-08-19", 17.08, "2025 年营收", "峰值情绪"),
    ("宇树科技（9 月 18 日收盘）", 2083, "514.98 元 × 4.04 亿股（推算）", 17.08, "2025 年营收", "较开盘高点回撤约 53%"),
    ("云深处（IPO 前估值）", 139, "按募资额与发行比例推算，2026-05", 3.37, "2025 年营收", "2025 年首次盈利"),
    ("乐聚智能（最近一轮）", 43.27, "2025-10 投后估值", 2.58, "2025 年营收", "创业板第四套标准首单"),
    ("智元（传闻新一轮）", 500, "媒体报道约 500 亿元，2026-10，未获公司确认", 10.5, "2025 年营收（公司口径）", "对 2026E 45 亿元收入约 11x"),
    ("Agility（SPAC）", 2.5 * 71, "25 亿美元，2026-06 协议", 0.018 * 7.1, "2025 年营收 180 万美元", "订单储备 3 亿美元以上（里程碑式）"),
    ("Skild AI（C 轮）", 140 * 7.1, "140 亿美元+，2026-01", 0.30 * 7.1, "2025 年未审计收入约 3,000 万美元", "公司称年化收入运行率已过 1 亿美元"),
    ("极智嘉（对照：AMR）", None, "港股上市公司（市值以实时行情为准）", 31.71, "2025 年营收", "毛利率 35.5%，经调整净利润转正"),
]
rows = []
mult_json = []
for name, val, vnote, rev, rnote, note in MULT:
    m = val / rev if (val and rev) else None
    mult_json.append({"name": name, "valuation_100m_cny": val, "revenue_100m_cny": rev, "ev_s": m})
    rows.append([name, fmt(val) if val else "—", vnote, fmt(rev, 2), rnote, f"{m:,.0f}x" if m else "—", note])
save("t_multiples", md_table(["公司", "估值/市值（亿元）", "估值口径", "收入（亿元）", "收入口径", "估值/收入", "备注"], rows))
RESULTS["multiples"] = mult_json

with open(os.path.join(OUT, "model_results.json"), "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=str)

print(json.dumps({"bom_full": RESULTS["bom_full"], "big3": RESULTS["bom_big3_share"], "bom_small": RESULTS["bom_small"],
                  "small_gm": RESULTS["bom_small_gm_at_99k"], "raas_be": RESULTS["raas_breakeven_share"],
                  "vc": [(v["name"], round(v["E_moic"], 2), round(v["irr"], 3)) for v in RESULTS["vc"]]}, ensure_ascii=False, indent=1))
for r in tco_json:
    print(r["方案"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "方案"})
