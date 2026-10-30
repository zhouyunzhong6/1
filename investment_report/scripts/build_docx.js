// 将 md/ 下的章节（含 {{T:...}} 表格占位与图表）组装为正式 Word 报告，并输出全文合并版 Markdown。
// 用法：node build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, ImageRun, Header, Footer, PageNumber, LevelFormat, BorderStyle,
  InternalHyperlink, Bookmark, PageOrientation, TableLayoutType, VerticalAlign, LineRuleType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const MD = path.join(ROOT, "md");
const GEN = path.join(ROOT, "generated");
const OUTDIR = path.join(ROOT, "exports");
const TITLE = "具身智能全产业链深度研究与投资决策报告";
const SUB = "全球产业格局 · 技术成熟度 · 商业化 · 资本周期 · 未来投资机会";

const FONT = { ascii: "Calibri", hAnsi: "Calibri", eastAsia: "Microsoft YaHei", cs: "Calibri" };
const MONO = { ascii: "Consolas", hAnsi: "Consolas", eastAsia: "SimSun", cs: "Consolas" };
const NAVY = "1F3864";
const INK2 = "52514E";

// A4（DXA）：纵向正文宽 11906-2*1134；横向附录宽 16838-2*1134
const PORTRAIT_W = 11906 - 2 * 1134;
const LANDSCAPE_W = 16838 - 2 * 1134;

// ---------- 读取与展开 ----------
const files = fs.readdirSync(MD).filter((f) => f.endsWith(".md")).sort();
function expand(text) {
  return text.replace(/^\{\{T:([\w]+)\}\}\s*$/gm, (_, name) => {
    const p = path.join(GEN, name + ".md");
    if (!fs.existsSync(p)) throw new Error("缺少生成表格：" + name);
    return fs.readFileSync(p, "utf8").trim();
  });
}
const chapters = files.map((f) => ({ file: f, text: expand(fs.readFileSync(path.join(MD, f), "utf8")) }));

// 全文合并版 Markdown（图片路径改为相对 investment_report/ 根目录）
fs.mkdirSync(OUTDIR, { recursive: true });
const merged = chapters.map((c) => c.text.replace(/\]\(\.\.\/charts\//g, "](charts/")).join("\n\n---\n\n");
fs.writeFileSync(path.join(ROOT, "全文合并版.md"), `# ${TITLE}\n\n> ${SUB}\n\n` + merged + "\n");

// ---------- 行内格式 ----------
function displayLen(s) {
  let n = 0;
  for (const ch of s) n += /[⺀-鿿＀-￯　-〿]/.test(ch) ? 2 : 1;
  return n;
}
function runs(text, opts = {}) {
  const out = [];
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter((s) => s !== "");
  for (const p of parts) {
    if (p.startsWith("**") && p.endsWith("**")) out.push(new TextRun({ text: p.slice(2, -2), bold: true, ...opts }));
    else if (p.startsWith("`") && p.endsWith("`")) out.push(new TextRun({ text: p.slice(1, -1), font: MONO, ...opts, color: "8A3B12" }));
    else out.push(new TextRun({ text: p, ...opts }));
  }
  return out;
}

// ---------- 表格 ----------
function splitRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((c) => c.trim());
}
function buildTable(lines, totalW) {
  const rows = lines.filter((l, i) => !(i === 1 && /^\|?\s*:?-{2,}/.test(l.trim()))).map(splitRow);
  const ncol = Math.max(...rows.map((r) => r.length));
  rows.forEach((r) => { while (r.length < ncol) r.push(""); });
  const cap = ncol >= 7 ? 34 : 44;
  const weights = [];
  for (let j = 0; j < ncol; j++) {
    let m = 4;
    for (const r of rows) m = Math.max(m, Math.min(cap, displayLen(r[j].replace(/\*\*/g, ""))));
    weights.push(Math.max(6, m));
  }
  const sum = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((w / sum) * totalW));
  widths[widths.length - 1] += totalW - widths.reduce((a, b) => a + b, 0);
  const size = ncol >= 9 ? 13 : ncol >= 7 ? 14 : ncol >= 5 ? 15 : 16;
  const border = { style: BorderStyle.SINGLE, size: 2, color: "D0D3D9" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const trs = rows.map((r, i) => new TableRow({
    tableHeader: i === 0,
    cantSplit: false,
    children: r.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      borders,
      verticalAlign: VerticalAlign.TOP,
      margins: { top: 40, bottom: 40, left: 70, right: 70 },
      shading: i === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: NAVY } : (i % 2 === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "F3F5F8" } : undefined),
      children: [new Paragraph({
        spacing: { before: 0, after: 0, line: 260, lineRule: LineRuleType.AUTO },
        children: runs(c, i === 0 ? { bold: true, color: "FFFFFF", size } : { size }),
      })],
    })),
  }));
  return new Table({ width: { size: totalW, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED, rows: trs });
}

// ---------- 图片 ----------
let figNo = 0;
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function image(alt, rel, totalW) {
  const p = path.resolve(MD, rel);
  const buf = fs.readFileSync(p);
  const { w, h } = pngSize(buf);
  const maxPx = Math.round((totalW / 1440) * 96 * 0.98);
  const width = Math.min(maxPx, 640);
  figNo += 1;
  const caption = alt.replace(/^图\s*\d+\s*/, "");
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width, height: Math.round((width * h) / w) }, altText: { title: caption, description: caption, name: caption } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: `图 ${figNo}　${caption}`, size: 18, color: INK2 })] }),
  ];
}

// ---------- 解析 Markdown ----------
let bm = 0;
const toc = [];
function parse(text, totalW, isFirstChapter) {
  const out = [];
  const lines = text.split("\n");
  let i = 0;
  let firstH1 = true;
  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();
    if (t === "" || t === "---") { i++; continue; }
    if (t.startsWith("```")) {
      i++;
      const code = [];
      while (i < lines.length && !lines[i].trim().startsWith("```")) code.push(lines[i++]);
      i++;
      code.forEach((c, k) => out.push(new Paragraph({
        spacing: { before: k === 0 ? 80 : 0, after: k === code.length - 1 ? 160 : 0, line: 240, lineRule: LineRuleType.AUTO },
        shading: { type: ShadingType.CLEAR, color: "auto", fill: "F4F4F2" },
        children: [new TextRun({ text: c.replace(/ /g, " ") || " ", font: MONO, size: 15, color: "333333" })],
      })));
      continue;
    }
    if (t.startsWith("|")) {
      const tl = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) tl.push(lines[i++]);
      out.push(buildTable(tl, totalW));
      out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      continue;
    }
    const img = t.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (img) { out.push(...image(img[1], img[2], totalW)); i++; continue; }
    const h = t.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      const txt = h[2].trim();
      const heading = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4][level - 1];
      let children = runs(txt);
      if (level <= 2) {
        const id = "bm" + (++bm);
        toc.push({ level, txt: txt.replace(/\*\*/g, ""), id });
        children = [new Bookmark({ id, children: [new TextRun(txt.replace(/\*\*/g, ""))] })];
      }
      out.push(new Paragraph({ heading, pageBreakBefore: level === 1 && !(isFirstChapter && firstH1), children }));
      if (level === 1) firstH1 = false;
      i++;
      continue;
    }
    if (t.startsWith(">")) {
      const q = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) q.push(lines[i++].trim().replace(/^>\s?/, ""));
      q.filter((s) => s !== "").forEach((s) => out.push(new Paragraph({
        spacing: { before: 40, after: 60, line: 300, lineRule: LineRuleType.AUTO },
        indent: { left: 240 },
        border: { left: { style: BorderStyle.SINGLE, size: 12, color: "9AA8C2", space: 8 } },
        shading: { type: ShadingType.CLEAR, color: "auto", fill: "F5F7FA" },
        children: runs(s, { size: 18, color: "3A3F4A" }),
      })));
      continue;
    }
    const bullet = line.match(/^(\s*)[-*]\s+(.*)$/);
    if (bullet) {
      const lvl = Math.min(1, Math.floor(bullet[1].length / 4));
      out.push(new Paragraph({ numbering: { reference: "bullets", level: lvl }, spacing: { after: 60, line: 312, lineRule: LineRuleType.AUTO }, children: runs(bullet[2]) }));
      i++;
      continue;
    }
    const num = line.match(/^(\s*)(\d+)\.\s+(.*)$/);
    if (num) {
      const lvl = Math.min(1, Math.floor(num[1].length / 4));
      out.push(new Paragraph({ spacing: { after: 60, line: 312, lineRule: LineRuleType.AUTO }, indent: { left: 420 + lvl * 420, hanging: 300 }, children: runs(`${num[2]}. ${num[3]}`) }));
      i++;
      continue;
    }
    // 普通段落（合并连续非空行）
    const para = [t];
    i++;
    while (i < lines.length && lines[i].trim() !== "" && !/^(#|\||>|```|!\[|\s*[-*]\s|\s*\d+\.\s)/.test(lines[i])) para.push(lines[i++].trim());
    out.push(new Paragraph({ spacing: { after: 120, line: 324, lineRule: LineRuleType.AUTO }, alignment: AlignmentType.JUSTIFIED, children: runs(para.join("")) }));
  }
  return out;
}

// ---------- 正文与附录 ----------
const bodyCh = chapters.filter((c) => !c.file.startsWith("18_"));
const appCh = chapters.filter((c) => c.file.startsWith("18_"));
const body = [];
bodyCh.forEach((c) => body.push(...parse(c.text, PORTRAIT_W, false)));
const appendix = [];
appCh.forEach((c) => appendix.push(...parse(c.text, LANDSCAPE_W, true)));

// ---------- 封面与目录 ----------
const cover = [
  new Paragraph({ spacing: { before: 2600 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.LEFT, border: { left: { style: BorderStyle.SINGLE, size: 36, color: NAVY, space: 14 } }, spacing: { after: 240, line: 400, lineRule: LineRuleType.AUTO },
    children: [new TextRun({ text: "具身智能全产业链", bold: true, size: 52, color: NAVY }), new TextRun({ text: "深度研究与投资决策报告", bold: true, size: 52, color: NAVY, break: 1 })] }),
  new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: SUB, size: 26, color: INK2 })] }),
  ...[
    ["信息截止", "2026 年 10 月 21 日（少数关键事件更新至 10 月 30 日，均注明日期）"],
    ["报告版本", "v1.0 · 2026 年 10 月"],
    ["用途", "一级市场投资决策：赛道优先级、项目筛选、估值、尽调与投委会材料"],
    ["结构", "投资决策摘要 + 十七篇正文 + 十个附录；配套 17 张数据底表与可复算模型"],
    ["证据等级", "A 一手/独立验证 · B 可靠二手交叉佐证 · C 单方或单一来源 · D 研究推断"],
  ].map(([k, v]) => new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: k + "　", bold: true, size: 21, color: NAVY }), new TextRun({ text: v, size: 21 })] })),
  new Paragraph({ spacing: { before: 1800 }, children: [new TextRun({ text: "重要声明：本报告基于截至信息截止日可获取的公开资料。所有估值、TCO 与回报模型参数均为示例性研究假设（D 级），不代表任何企业的真实财务信息；企业单方披露信息已标注为 C 级。本报告不构成任何投资建议。", size: 17, color: INK2 })] }),
];
const tocParas = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("目　录")] }),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "点击条目可跳转；在 Word 中也可通过「视图 → 导航窗格」浏览全部标题。", size: 18, color: INK2 })] }),
  ...toc.map((e) => new Paragraph({
    spacing: { after: e.level === 1 ? 60 : 20, before: e.level === 1 ? 100 : 0 },
    indent: { left: e.level === 1 ? 0 : 420 },
    children: [new InternalHyperlink({ anchor: e.id, children: [new TextRun({ text: e.txt, bold: e.level === 1, size: e.level === 1 ? 21 : 19, color: e.level === 1 ? NAVY : "333333" })] })],
  })),
];

const header = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9CED8", space: 4 } },
  children: [new TextRun({ text: TITLE + " · 2026-10", size: 16, color: "8A8F99" })] })] });
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["第 ", PageNumber.CURRENT, " 页"], size: 17, color: "8A8F99" })] })] });
const margin = { top: 1134, bottom: 1134, left: 1134, right: 1134, header: 600, footer: 600 };

const doc = new Document({
  creator: "联合研究团队",
  title: TITLE,
  description: SUB,
  styles: {
    default: { document: { run: { font: FONT, size: 21 }, paragraph: { spacing: { line: 324, lineRule: LineRuleType.AUTO } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 36, bold: true, color: NAVY }, paragraph: { spacing: { before: 240, after: 280 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: NAVY }, paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "2F4A7A" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 } },
      { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 22, bold: true, color: "2F4A7A" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 3 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 420, hanging: 260 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 840, hanging: 260 } } } },
  ] }] },
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 }, margin } }, children: cover },
    { properties: { page: { size: { width: 11906, height: 16838 }, margin, pageNumbers: { start: 1 } } }, headers: { default: header }, footers: { default: footer }, children: [...tocParas, ...body] },
    { properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin } }, headers: { default: header }, footers: { default: footer }, children: appendix },
  ],
});

const outPath = path.join(OUTDIR, "具身智能全产业链深度研究与投资决策报告_2026-10.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  require("child_process").execFileSync("python3", [path.join(__dirname, "fix_bookmarks.py"), outPath], { stdio: "inherit" });
  const chars = chapters.map((c) => c.text).join("").replace(/[\s|\-#*>`{}]/g, "");
  const cjk = (chapters.filter((c) => !c.file.startsWith("18_")).map((c) => c.text).join("").match(/[一-鿿]/g) || []).length;
  console.log("written", outPath, "figures:", figNo, "toc entries:", toc.length, "body CJK chars:", cjk, "all chars:", chars.length);
});
