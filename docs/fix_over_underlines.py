#!/usr/bin/env python3
"""
Ensure RST files have matching overline/underline lengths for titles.

This script adjusts lines where a punctuation-only line appears before and/or
after a title and sets them to the exact length of the title text.
"""
from pathlib import Path
import re

punct_re = re.compile(r'^[^\w\n]+$')

def fix_file(p: Path):
    changed = False
    lines = p.read_text(encoding='utf-8').splitlines()
    out = []
    i = 0
    while i < len(lines):
        if i+2 < len(lines) and punct_re.match(lines[i]) and lines[i+1].strip() and punct_re.match(lines[i+2]):
            title = lines[i+1].strip()
            ch = lines[i].strip()[0]
            target = ch * len(title)
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
        out.append(lines[i])
        i += 1
    if changed:
        p.write_text('\n'.join(out) + '\n', encoding='utf-8')
        print('Fixed over/underline in', p)

def main():
    paths = list(Path('docs/modules').rglob('*.rst')) + list(Path('docs/guides').rglob('*.rst'))
    for p in paths:
        fix_file(p)

if __name__ == '__main__':
    main()
