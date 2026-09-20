"""Render paper/driftlab.md to an arXiv-styled PDF via HTML + headless chromium.

    uv run --with markdown --with pymdown-extensions python paper/render.py

Pipeline: python-markdown (with arithmatex protecting LaTeX math) -> HTML with
Computer Modern webfonts and KaTeX -> chromium print-to-pdf. Post-processing
turns [n] citations into anchors to the reference list, linkifies arXiv ids
and DOIs in the references, and links named benchmark mentions to their
sources. Page numbers and the repo footer come from CSS @page margin boxes.
"""

import re
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).parent
REPO = "https://github.com/unfoldinglabs/driftlab"
md = (HERE / "driftlab.md").read_text()
body = markdown.markdown(md, extensions=["tables", "smarty", "toc", "pymdownx.arithmatex"],
                         extension_configs={"pymdownx.arithmatex": {"generic": True,
                                            "inline_syntax": ["round"], "block_syntax": ["dollar", "square"]}})

# arXiv-style front matter: center the title, the author block, and the preprint note
body = re.sub(r"(</h1>\s*)<p>(.*?)</p>\s*<p>(<em>Preprint.*?</em>)</p>",
              r'\1<p class="authors">\2</p><p class="note">\3</p>', body, count=1, flags=re.S)

# ---- citation and reference links ------------------------------------------------------
# author-year labels for each numbered bibliography entry, used as the in-text citation
REF_LABELS = {
    1: "Jimenez et al., 2024", 2: "Mialon et al., 2023", 3: "Zhou et al., 2023",
    4: "Liu et al., 2024", 5: "Yao et al., 2024", 6: "Wu et al., 2024",
    7: "Gama et al., 2014", 8: "Khetarpal et al., 2022", 9: "McCloskey and Cohen, 1989",
    10: "Garivier and Moulines, 2011", 11: "Shinn et al., 2023", 12: "Wang et al., 2023",
    13: "Packer et al., 2023", 14: "Park et al., 2023", 15: "Xiong et al., 2024",
    16: "Tian et al., 2023", 17: "Behrens et al., 2007", 18: "Nassar et al., 2010",
    19: "Xu et al., 2026", 20: "Wu et al., 2025", 21: "Chu et al., 2026",
    22: "Iten et al., 2026", 23: "Lopez-Paz and Ranzato, 2017",
    24: "Kirkpatrick et al., 2017", 25: "Cabrera Martin et al., 2025",
}

# split out the references section: citations link inward, identifiers link outward.
# The appendices now follow the references, so the block ends at the next <h2>.
m_ref = re.search(r"<h2[^>]*>References</h2>", body)
head_part, ref_heading, rest = body[:m_ref.start()], m_ref.group(0), body[m_ref.end():]
m_end = re.search(r"<h2[^>]*>", rest)
ref_part, tail_part = (rest[:m_end.start()], rest[m_end.start():]) if m_end else (rest, "")
N_REFS = len(re.findall(r"<p>\[\d+\]", ref_part))

# each reference paragraph gets an anchor id, and drops its now-unused number
ref_part = re.sub(r"<p>\[(\d+)\]\s*", r'<p class="ref" id="ref-\1">', ref_part)
# arXiv ids and DOIs become external links
ref_part = re.sub(r"arXiv:(\d{4}\.\d{4,5})", r'<a href="https://arxiv.org/abs/\1">arXiv:\1</a>', ref_part)
ref_part = re.sub(r"doi:(10\.[^\s<]+?)(?=[.,]?(?:\s|<))", r'<a href="https://doi.org/\1">doi:\1</a>', ref_part)


def _cite(mm):
    nums = [n.strip() for n in mm.group(1).split(",")]
    if any(not n.isdigit() or not (1 <= int(n) <= N_REFS) for n in nums):
        return mm.group(0)   # not a citation (e.g. an interval like [0, 1])
    links = "; ".join(f'<a class="cite" href="#ref-{n}">{REF_LABELS[int(n)]}</a>' for n in nums)
    return f"({links})"


# named mentions link to their sources (applied outside the reference list and outside SVG)
NAME_LINKS = {
    "SWE-bench": "2310.06770", "GAIA": "2311.12983", "WebArena": "2307.13854",
    "AgentBench": "2308.03688", "τ-bench": "2406.12045", "StreamBench": "2406.08747",
    "LongMemEval": "2410.10813", "EvoArena": "2606.13681", "Reflexion": "2303.11366",
    "Voyager": "2305.16291", "MemGPT": "2310.08560", "Generative Agents": "2304.03442",
}


# heading anchors: "3.4" -> its id, "Appendix C" -> its id (built after markdown runs)
SEC_IDS = {}
for hid, htext in re.findall(r'<h[23] id="([^"]+)">([^<]+)</h[23]>', body):
    first = htext.split()[0]
    if re.fullmatch(r"\d+(\.\d+)?", first):
        SEC_IDS[first] = hid
    elif htext.startswith("Appendix "):
        SEC_IDS["Appendix " + htext.split()[1].rstrip(":")] = hid


def _secref(mm):
    num = mm.group(1)
    hid = SEC_IDS.get(num)
    return f'<a class="secref" href="#{hid}">Section {num}</a>' if hid else mm.group(0)


def _apxref(mm):
    key = "Appendix " + mm.group(1)
    hid = SEC_IDS.get(key)
    return f'<a class="secref" href="#{hid}">{key}</a>' if hid else mm.group(0)


def _link_text(seg: str) -> str:
    seg = re.sub(r"§(\d+(?:\.\d+)?)", _secref, seg)
    seg = re.sub(r"\bSection (\d+(?:\.\d+)?)", _secref, seg)
    seg = re.sub(r"\bAppendix ([A-D])\b(?!:)", _apxref, seg)
    seg = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", _cite, seg)
    for name, arx in NAME_LINKS.items():
        seg = re.sub(rf"(?<![\w>]){re.escape(name)}(?![\w<])",
                     f'<a class="named" href="https://arxiv.org/abs/{arx}">{name}</a>', seg)
    return seg


# apply only outside <svg> blocks (figure text must stay plain)
def _link_outside_svg_and_headings(text):
    parts = re.split(r"(<svg.*?</svg>|<h[23][^>]*>.*?</h[23]>)", text, flags=re.S)
    return "".join(p if p.startswith(("<svg", "<h")) else _link_text(p) for p in parts)


head_part = _link_outside_svg_and_headings(head_part)
tail_part = _link_outside_svg_and_headings(tail_part)
body = head_part + ref_heading + ref_part + tail_part

CSS = """
@page {
  size: letter; margin: 0.75in 0.75in 1in 0.75in;
  @bottom-center { content: counter(page); font-size: 9pt; font-family: serif; color: #111; }
}
html { -webkit-print-color-adjust: exact; }
body { font-family: "Computer Modern Serif", "STIX Two Text", "Times New Roman", serif;
       font-size: 9.8pt; line-height: 1.38; color: #111; margin: 0; text-align: justify;
       hyphens: auto; }
.twocol { column-count: 2; column-gap: 0.3in; }
h2, h3 { break-after: avoid; }
h1 { font-size: 17pt; text-align: center; font-weight: 700; line-height: 1.25;
     margin: 0 0 14pt; text-align-last: center; }
.authors { text-align: center; text-align-last: center; font-size: 11pt; margin: 0 0 2pt; }
.authors strong { font-size: 12pt; }
.note { text-align: center; text-align-last: center; font-size: 9.5pt; color: #444; margin: 2pt 0 18pt; }
h2 { font-size: 12.5pt; font-weight: 700; margin: 16pt 0 6pt; text-align: left; text-align-last: left; }
h3 { font-size: 11pt; font-weight: 700; font-style: italic; margin: 12pt 0 4pt; text-align: left; }
p { margin: 0 0 7pt; }
strong { font-weight: 700; }
a { color: #0b3d91; text-decoration: none; }
a.cite { color: #1a6633; }
a.named { color: inherit; }
code { font-family: "Computer Modern Typewriter", "Liberation Mono", monospace; font-size: 9.5pt;
       background: #f4f4f2; padding: 0 2px; border-radius: 2px; overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 8pt 0 10pt;
        line-height: 1.3; }
/* Table 1 has one row per model/world and needs the full page width inside the
   two-column results flow; the appendix tables remain in their local column. */
table:first-of-type { column-span: all; font-size: 8.2pt; }
th { border-top: 1.2pt solid #111; border-bottom: 0.6pt solid #111; padding: 3pt 5px;
     text-align: left; font-weight: 700; }
td { padding: 3pt 5px; vertical-align: top; border-bottom: 0.3pt solid #ccc; }
tr:last-child td { border-bottom: 1.2pt solid #111; }
table code { white-space: nowrap; overflow-wrap: normal; }
ol, ul { margin: 0 0 7pt; padding-left: 20pt; }
li { margin-bottom: 3pt; }
p.ref { font-size: 9pt; text-align: left; margin-bottom: 4.5pt; }
em { font-style: italic; }
.arithmatex { font-size: 0.84em; }
div.arithmatex { margin: 6pt 0 9pt; text-align: center; font-size: 0.92em; }
.abs { font-size: 9.3pt; }
.abs p { text-align: justify; }
.fig { margin: 10pt 0; break-inside: avoid; }
.fig2 { break-inside: auto; }
.brk { break-before: page; }
a.secref { color: #0b3d91; }
.fig svg { width: 100%; display: block; margin: 0 auto; }
.figcap { font-size: 8.6pt; color: #222; margin: 2pt auto 0; text-align: justify; }
"""

# The two-column flow starts at the abstract, which opens the first column.
# Headings carry toc-generated ids, so match them by regex, not literal text.
body = re.sub(r'<h2[^>]*>Abstract</h2>',
              '<div class="twocol"><h2 style="font-size:11.5pt">Abstract</h2><div class="abs">', body)
body = re.sub(r'(<h2[^>]*>1 Introduction</h2>)', r'</div>\1', body) + "</div>"
body = re.sub(r'(<h2[^>]*>Appendix A:)', r'</div><div class="twocol brk">\1', body)

html = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/aaaakshat/cm-web-fonts@latest/fonts.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", () => renderMathInElement(document.body, {{
  delimiters: [{{left: "\\\\(", right: "\\\\)", display: false}}, {{left: "\\\\[", right: "\\\\]", display: true}}],
  throwOnError: false,
}}));
</script>
<style>{CSS}</style></head><body>{body}</body></html>"""
(HERE / "driftlab.html").write_text(html)

subprocess.run(["chromium", "--headless", "--disable-gpu", "--no-pdf-header-footer",
                "--virtual-time-budget=25000", "--run-all-compositor-stages-before-draw",
                f"--print-to-pdf={HERE / 'driftlab.pdf'}", str(HERE / "driftlab.html")], check=True)
print("wrote", HERE / "driftlab.pdf")
