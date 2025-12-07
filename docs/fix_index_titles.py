#!/usr/bin/env python3
"""
Fix bolded section titles (e.g. **Title**) and ensure underline/adornment
lines match the title length for specified RST files.

Run from repository root.
"""
from pathlib import Path
import re

FILES = [Path('docs/index.rst'), Path('docs/index_new.rst')]

bold_re = re.compile(r"^\*\*(.+?)\*\*$")
punct_re = re.compile(r"^[^\w\n]+$")

def fix_file(p: Path):
    if not p.exists():
        return
    lines = p.read_text(encoding='utf-8').splitlines()
    changed = False
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = bold_re.match(line.strip())
        if m and i+1 < len(lines) and punct_re.match(lines[i+1]):
            title = m.group(1).strip()
            underline_char = lines[i+1].strip()[0] if lines[i+1].strip() else '='
            underline = underline_char * len(title)
            out.append(title)
            out.append(underline)
            changed = True
            i += 2
            continue
        out.append(line)
        i += 1
    if changed:
        p.write_text('\n'.join(out) + '\n', encoding='utf-8')
        print('Patched:', p)

def main():
    for f in FILES:
        fix_file(f)

if __name__ == '__main__':
    main()
