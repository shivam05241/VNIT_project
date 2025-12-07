#!/usr/bin/env python3
"""
Fix RST title adornment mismatches across docs/ by:
- Removing surrounding bold markup from title lines like "**Title**"
- Ensuring underline/overline lines are the same length as the title line

This script modifies files in-place. Run from repository root.
"""
from pathlib import Path
import re

RST_DIR = Path('docs')

# punctuation-only line regex: any line containing no word characters (letters/digits/underscore)
# this is a simpler heuristic to detect adornment lines like === --- ***
punct_re = re.compile(r'^[^\w\n]+$')

def fix_file(p: Path):
    text = p.read_text(encoding='utf-8')
    lines = text.splitlines()
    changed = False
    out = []
    i = 0
    in_directive_block = False
    directive_indent = None
    while i < len(lines):
        line = lines[i]

        # detect start of a directive (e.g. .. code-block:: python) or literal block '::'
        stripped = line.lstrip()
        if stripped.startswith('.. '):
            # enter directive mode; subsequent indented lines are part of the directive
            in_directive_block = True
            directive_indent = None
            out.append(line)
            i += 1
            continue

        # if previous line ended with '::' and current line is indented, treat as literal block
        if i > 0 and lines[i-1].rstrip().endswith('::') and (len(line) - len(stripped) > 0):
            in_directive_block = True
            directive_indent = len(line) - len(stripped)
            out.append(line)
            i += 1
            continue

        # if currently in directive block, keep consuming indented lines
        if in_directive_block:
            if line.strip() == '':
                # blank line may end directive block, but we must check next line's indentation
                out.append(line)
                # look ahead to see if next non-empty line has same indent
                j = i+1
                ended = True
                while j < len(lines):
                    if lines[j].strip() == '':
                        j += 1
                        continue
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    if directive_indent is None:
                        # if not set, treat first indented line as block content
                        if indent > 0:
                            ended = False
                    else:
                        if indent >= directive_indent:
                            ended = False
                    break
                if ended:
                    in_directive_block = False
                    directive_indent = None
                i += 1
                continue
            else:
                indent = len(line) - len(line.lstrip())
                if directive_indent is None and indent == 0:
                    # non-indented line likely ends the block
                    in_directive_block = False
                    directive_indent = None
                elif directive_indent is not None and indent < directive_indent:
                    in_directive_block = False
                    directive_indent = None
                out.append(line)
                i += 1
                continue

        # remove bold wrappers around entire title lines: **Title** -> Title
        m = re.match(r"^\*\*(.+?)\*\*$", line.strip())
        if m:
            new_title = m.group(1).strip()
            if new_title != line:
                line = new_title
                changed = True

        # check for overline/underline pattern (punctuation, title, punctuation)
        if i+2 < len(lines) and punct_re.match(lines[i]) and lines[i+1].strip() != '' and punct_re.match(lines[i+2]):
            # ensure we are not inside a code block (indented title would be unlikely)
            if (len(lines[i+1]) - len(lines[i+1].lstrip())) == 0:
                title = lines[i+1].rstrip('\n')
                char = lines[i].strip()[0]
                target = char * len(title)
                if lines[i] != target:
                    lines[i] = target
                    changed = True
                if lines[i+2] != target:
                    lines[i+2] = target
                    changed = True
                out.append(lines[i])
                out.append(lines[i+1])
                out.append(lines[i+2])
                i += 3
                continue

        # check for simple title underline (title then punctuation line)
        if i+1 < len(lines) and lines[i].strip() != '' and punct_re.match(lines[i+1]):
            # avoid touching lines where the title is indented (likely code)
            if (len(lines[i]) - len(lines[i].lstrip())) == 0:
                title = lines[i].rstrip('\n')
                underline = lines[i+1]
                ch = underline.strip()[0] if underline.strip() else '='
                target = ch * len(title)
                if underline != target:
                    lines[i+1] = target
                    changed = True

        out.append(line)
        i += 1

    if changed:
        p.write_text('\n'.join(out) + '\n', encoding='utf-8')
        print(f'Fixed: {p}')

def main():
    files = list(RST_DIR.rglob('*.rst'))
    for f in files:
        fix_file(f)

if __name__ == '__main__':
    main()
