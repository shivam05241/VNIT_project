#!/usr/bin/env python3
"""
Scan all RST files under docs/ and replace headings written as ``**Title**``
followed by an adornment line (e.g. === or ---) with unbolded title and
correctly sized adornment.
"""
from pathlib import Path
import re

FILES = list(Path('docs').rglob('*.rst'))
bold_re = re.compile(r"^\*\*(.+?)\*\*$")
punct_re = re.compile(r'^[^\w\n]+$')

def fix_file(p: Path):
    lines = p.read_text(encoding='utf-8').splitlines()
    changed = False
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = bold_re.match(line.strip())
        if m and i+1 < len(lines) and punct_re.match(lines[i+1]):
            title = m.group(1).strip()
            ch = lines[i+1].strip()[0]
            underline = ch * len(title)
            out.append(title)
            out.append(underline)
            changed = True
            i += 2
            continue
        out.append(line)
        i += 1
    if changed:
        p.write_text('\n'.join(out) + '\n', encoding='utf-8')
        print('Fixed bold headings in', p)

def main():
    for f in FILES:
        fix_file(f)

if __name__ == '__main__':
    main()
