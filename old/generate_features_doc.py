"""
Generate FEATURES.txt and FEATURES.pdf from FEATURES.md
"""
import re
from fpdf import FPDF

# ── Read source ──────────────────────────────────────────────────────────────
with open("FEATURES.md", "r") as f:
    content = f.read()

# ── Plain TXT ────────────────────────────────────────────────────────────────
txt = content
txt = re.sub(r'^#{1,6}\s*', '', txt, flags=re.MULTILINE)
txt = re.sub(r'\*\*(.*?)\*\*', r'\1', txt)
txt = re.sub(r'\*(.*?)\*', r'\1', txt)
txt = re.sub(r'`(.*?)`', r'\1', txt)
txt = re.sub(r'\$\$.*?\$\$', lambda m: m.group(0).strip('$'), txt, flags=re.DOTALL)
txt = re.sub(r'\$(.*?)\$', r'\1', txt)
txt = re.sub(r'^\|[-| :]+\|$', '', txt, flags=re.MULTILINE)  # separator rows
# Clean table rows into readable lines
def fmt_table_row(m):
    cells = [c.strip() for c in m.group(0).strip().strip('|').split('|')]
    if len(cells) >= 2:
        return f"  {cells[0]:<35} {cells[1]}"
    return m.group(0)
txt = re.sub(r'^\|.+\|$', fmt_table_row, txt, flags=re.MULTILINE)
txt = re.sub(r'^---+$', '-' * 72, txt, flags=re.MULTILINE)
txt = re.sub(r'\n{3,}', '\n\n', txt)

with open("FEATURES.txt", "w") as f:
    f.write(txt)
print("TXT saved → FEATURES.txt")


# ── PDF ──────────────────────────────────────────────────────────────────────
UNICODE_MAP = {
    '\u2014': '--',   # em dash
    '\u2013': '-',    # en dash
    '\u2018': "'",    # left single quote
    '\u2019': "'",    # right single quote
    '\u201c': '"',    # left double quote
    '\u201d': '"',    # right double quote
    '\u2022': '*',    # bullet
    '\u2192': '->',   # arrow
    '\u2190': '<-',
    '\u03c1': 'rho',
    '\u03bc': 'mu',
    '\u03c3': 'sigma',
    '\u03c6': 'phi',
    '\u00b1': '+/-',
    '\u2265': '>=',
    '\u2264': '<=',
    '\u2260': '!=',
    '\u221e': 'inf',
    '\u2248': '~=',
}

def to_latin(s):
    for char, repl in UNICODE_MAP.items():
        s = s.replace(char, repl)
    return s.encode('latin-1', errors='replace').decode('latin-1')

def strip_md(s):
    s = re.sub(r'\*\*(.*?)\*\*', r'\1', s)
    s = re.sub(r'`(.*?)`', r'\1', s)
    s = re.sub(r'\$(.*?)\$', r'\1', s)
    return to_latin(s.strip())


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 6, "IntelliCage Feature Dictionary", align="R")
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def render_table(pdf, rows):
    if not rows:
        return
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    col1 = 50
    col2 = page_w - col1
    row_fill = [False, True]
    fill_idx = 0
    for row in rows:
        parts = [p.strip() for p in row.strip().strip("|").split("|")]
        if len(parts) < 2:
            continue
        feat = strip_md(parts[0])
        desc = strip_md(parts[1])
        # measure height
        pdf.set_font("Helvetica", "B", 8)
        feat_lines = pdf.multi_cell(col1 - 3, 4.5, feat, dry_run=True, output="LINES")
        pdf.set_font("Helvetica", size=8)
        desc_lines = pdf.multi_cell(col2 - 3, 4.5, desc, dry_run=True, output="LINES")
        h = max(len(feat_lines), len(desc_lines)) * 4.5 + 3
        if pdf.get_y() + h > pdf.h - pdf.b_margin:
            pdf.add_page()
        x = pdf.l_margin
        y = pdf.get_y()
        # alternating row fill
        if row_fill[fill_idx % 2]:
            pdf.set_fill_color(242, 245, 252)
            pdf.rect(x, y, page_w, h, style="F")
        fill_idx += 1
        # feature name cell
        pdf.set_font("Courier", "B", 8)
        pdf.set_text_color(20, 60, 140)
        pdf.set_xy(x + 1.5, y + 1.5)
        pdf.multi_cell(col1 - 3, 4.5, feat)
        # description cell
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(40, 40, 40)
        pdf.set_xy(x + col1 + 1.5, y + 1.5)
        pdf.multi_cell(col2 - 3, 4.5, desc)
        # row border
        pdf.set_draw_color(215, 220, 235)
        pdf.line(x, y + h, x + page_w, y + h)
        pdf.set_y(y + h)
    # vertical divider
    pdf.ln(3)


pdf = PDF()
pdf.set_margins(18, 18, 18)
pdf.set_auto_page_break(auto=True, margin=16)
pdf.add_page()

lines = content.split("\n")
in_table = False
table_rows = []
i = 0

while i < len(lines):
    line = lines[i]

    # ── Heading 1
    if re.match(r'^# [^#]', line):
        if in_table:
            render_table(pdf, table_rows); table_rows = []; in_table = False
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 17)
        pdf.set_text_color(15, 50, 110)
        pdf.multi_cell(0, 10, strip_md(line[2:]))
        pdf.set_draw_color(15, 50, 110)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(5)

    # ── Heading 2
    elif line.startswith("## "):
        if in_table:
            render_table(pdf, table_rows); table_rows = []; in_table = False
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 80, 160)
        pdf.multi_cell(0, 8, strip_md(line[3:]))
        pdf.set_draw_color(160, 190, 230)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(3)

    # ── Heading 3
    elif line.startswith("### "):
        if in_table:
            render_table(pdf, table_rows); table_rows = []; in_table = False
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(50, 100, 180)
        pdf.multi_cell(0, 6, strip_md(line[4:]))
        pdf.ln(1)

    # ── Table separator — skip
    elif re.match(r'^\|[-| :]+\|$', line.strip()):
        i += 1
        continue

    # ── Table header row (peek: next line is separator)
    elif line.startswith("|"):
        if not in_table:
            if i + 1 < len(lines) and re.match(r'^\|[-| :]+\|$', lines[i + 1].strip()):
                # draw column header bar
                page_w = pdf.w - pdf.l_margin - pdf.r_margin
                col1 = 50
                pdf.set_fill_color(30, 70, 150)
                pdf.rect(pdf.l_margin, pdf.get_y(), page_w, 6, style="F")
                parts = [p.strip() for p in line.strip().strip("|").split("|")]
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(255, 255, 255)
                pdf.set_xy(pdf.l_margin + 1.5, pdf.get_y() + 1)
                pdf.cell(col1 - 3, 4, parts[0] if parts else "Feature")
                if len(parts) > 1:
                    pdf.set_xy(pdf.l_margin + col1 + 1.5, pdf.get_y())
                    pdf.cell(page_w - col1 - 3, 4, parts[1])
                pdf.ln(6)
                i += 2  # skip separator
                in_table = True
                continue
            else:
                in_table = True
        table_rows.append(line)

    # ── End of table
    elif in_table and not line.startswith("|"):
        render_table(pdf, table_rows); table_rows = []; in_table = False
        # fall through to handle this line normally

    # ── Horizontal rule
    if not in_table and line.strip() == "---":
        pdf.set_draw_color(200, 200, 200)
        pdf.ln(2)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(3)

    # ── Bullet
    elif not in_table and line.startswith("- "):
        txt_line = strip_md(line[2:])
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(40, 40, 40)
        pdf.set_x(pdf.l_margin + 4)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 4, 5, "*  " + txt_line)

    # ── Normal text
    elif not in_table and not line.startswith("|") and line.strip() and not line.startswith("#"):
        txt_line = strip_md(line)
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, txt_line)
        pdf.ln(1)

    # ── Blank line
    elif not in_table and not line.strip():
        pdf.ln(2)

    i += 1

if in_table:
    render_table(pdf, table_rows)

pdf.output("FEATURES.pdf")
print("PDF saved → FEATURES.pdf")
