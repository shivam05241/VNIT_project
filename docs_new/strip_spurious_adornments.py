#!/usr/bin/env python3
"""
Remove punctuation-only lines that are likely spurious adornments
inserted inside code blocks or adjacent to indented code in RST files.

This is a heuristic cleanup to repair files after automated edits.
"""
from pathlib import Path
import re

punct_re = re.compile(r'^[^\w\n]+$')

def clean_file(p: Path):
    lines = p.read_text(encoding='utf-8').splitlines()
    out = []
    i = 0
    changed = False
    while i < len(lines):
        line = lines[i]
        if punct_re.match(line):
            # check surrounding lines
            prev = lines[i-1] if i-1 >= 0 else ''
            nxt = lines[i+1] if i+1 < len(lines) else ''
            # if next or prev is an indented code line, drop this punctuation line
            if (prev.startswith('    ') or nxt.startswith('    ')):
                changed = True
                i += 1
                continue
        out.append(line)
        i += 1
    if changed:
        p.write_text('\n'.join(out)+"\n", encoding='utf-8')
        print('Cleaned:', p)

def main():
    for f in Path('docs').rglob('*.rst'):
        clean_file(f)

if __name__ == '__main__':
    main()
