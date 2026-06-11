"""Minimal, dependency-free renderer for the constrained markdown subset used
by results/stats_summary.md and results/comparison_summary.md: headers,
paragraphs, single-level bullet lists, and pipe-delimited GFM tables.

Not a general-purpose markdown engine — no ordered lists, nested lists,
blockquotes, or images.
"""
import html
import re

_HEADER_RE = re.compile(r'^(#{1,6})\s+(.*)$')
_LIST_ITEM_RE = re.compile(r'^(\s*)-\s+(.*)$')
_TABLE_SEPARATOR_RE = re.compile(r'^\|?[\s:|-]+\|?$')
_INLINE_CODE_RE = re.compile(r'`([^`]+)`')
_BOLD_RE = re.compile(r'\*\*([^*]+)\*\*')
_ITALIC_RE = re.compile(r'(?<!\*)\*([^*]+)\*(?!\*)')


def _inline(text):
    text = html.escape(text, quote=False)
    text = _INLINE_CODE_RE.sub(r'<code>\1</code>', text)
    text = _BOLD_RE.sub(r'<strong>\1</strong>', text)
    text = _ITALIC_RE.sub(r'<em>\1</em>', text)
    return text


def _split_row(line):
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def _cell_alignment(marker):
    left = marker.startswith(':')
    right = marker.endswith(':')
    if left and right:
        return 'center'
    if right:
        return 'right'
    if left:
        return 'left'
    return None


def _render_table(lines):
    header_cells = _split_row(lines[0])
    aligns = [_cell_alignment(cell) for cell in _split_row(lines[1])]

    def style_for(index):
        if index < len(aligns) and aligns[index]:
            return f' style="text-align:{aligns[index]}"'
        return ''

    out = ['<table>', '<thead><tr>']
    for i, cell in enumerate(header_cells):
        out.append(f'<th{style_for(i)}>{_inline(cell)}</th>')
    out.append('</tr></thead>')
    out.append('<tbody>')
    for row in lines[2:]:
        out.append('<tr>')
        for i, cell in enumerate(_split_row(row)):
            out.append(f'<td{style_for(i)}>{_inline(cell)}</td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def render_markdown(text):
    """Convert the constrained markdown subset above into an HTML string."""
    lines = text.split('\n')
    n = len(lines)
    out = []
    paragraph_buf = []
    in_list = False

    def flush_paragraph():
        if paragraph_buf:
            out.append('<p>' + _inline(' '.join(paragraph_buf)) + '</p>')
            paragraph_buf.clear()

    def close_list():
        nonlocal in_list
        if in_list:
            out.append('</ul>')
            in_list = False

    i = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            close_list()
            i += 1
            continue

        header_match = _HEADER_RE.match(stripped)
        if header_match:
            flush_paragraph()
            close_list()
            level = len(header_match.group(1))
            out.append(f'<h{level}>{_inline(header_match.group(2))}</h{level}>')
            i += 1
            continue

        is_table_start = (
            stripped.startswith('|')
            and i + 1 < n
            and _TABLE_SEPARATOR_RE.match(lines[i + 1].strip())
            and '-' in lines[i + 1]
        )
        if is_table_start:
            flush_paragraph()
            close_list()
            table_lines = [stripped, lines[i + 1].strip()]
            j = i + 2
            while j < n and lines[j].strip().startswith('|'):
                table_lines.append(lines[j].strip())
                j += 1
            out.append(_render_table(table_lines))
            i = j
            continue

        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            flush_paragraph()
            if not in_list:
                out.append('<ul class="md-list">')
                in_list = True
            indent = len(list_match.group(1))
            css_class = ' class="nested"' if indent > 0 else ''
            out.append(f'<li{css_class}>{_inline(list_match.group(2))}</li>')
            i += 1
            continue

        close_list()
        paragraph_buf.append(stripped)
        i += 1

    flush_paragraph()
    close_list()
    return '\n'.join(out)
