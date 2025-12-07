#!/usr/bin/env python3
"""
PDF Generator for VNIT Project Documentation
Converts HTML documentation to PDF using weasyprint
"""

import os
import sys
#!/usr/bin/env python3
"""
PDF Generator for VNIT Project Documentation
Converts HTML documentation to a merged PDF using WeasyPrint and pypdf.

Behavior:
- Parses `docs/_build/html/index.html` to extract the table-of-contents links
  (the same order shown in the HTML site).
- Converts each referenced HTML page to a per-page PDF using WeasyPrint.
- Merges the per-page PDFs (in order) into a single PDF at
  `docs/_build/pdf/VNIT_Housing_Prediction_full.pdf`.

If a dependency is missing, the script will attempt to install it.
"""

import sys
import subprocess
from pathlib import Path

def ensure_package(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
        return True
    except ImportError:
        print(f"Installing {pkg_name}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg_name], check=True)
        try:
            __import__(import_name or pkg_name)
            return True
        except ImportError:
            return False

def extract_toc_html_order(html_index_path):
    """Return a list of relative HTML hrefs in the order shown on index.html TOC."""
    from bs4 import BeautifulSoup

    with open(html_index_path, 'r', encoding='utf-8') as fh:
        soup = BeautifulSoup(fh, 'html.parser')

    from collections import OrderedDict

    ordered = OrderedDict()
    # Sphinx places toctrees inside <div class="toctree-wrapper">. Find all anchors.
    for wrapper in soup.select('.toctree-wrapper'):
        for a in wrapper.select('a[href]'):
            href = a.get('href')
            # skip anchors and external links
            if not href or href.startswith('http') or href.startswith('#'):
                continue
            # normalize: remove fragment identifiers and leading './'
            clean = href.split('#')[0]
            if clean.startswith('./'):
                clean = clean[2:]
            # collapse repeated slashes
            clean = clean.replace('//', '/')
            # If it doesn't end with .html, Sphinx sometimes links directories; append .html
            if not clean.endswith('.html'):
                clean = clean + '.html'
            if clean not in ordered:
                ordered[clean] = True
    return list(ordered.keys())

def html_to_pdf(src_html, target_pdf, html_dir):
    from weasyprint import HTML, CSS
    HTML(filename=str(src_html), base_url=str(html_dir)).write_pdf(
        str(target_pdf),
        stylesheets=[CSS(string="@page { size: A4; margin: 2cm; }")]
    )

def merge_pdfs(pdf_paths, out_path):
    try:
        from pypdf import PdfMerger
        merger = PdfMerger()
    except Exception:
        # fallback for older/newer pypdf variants
        try:
            from PyPDF2 import PdfMerger as PyPdfMerger
            merger = PyPdfMerger()
        except Exception:
            from pypdf import PdfWriter
            writer = PdfWriter()
            for p in pdf_paths:
                from pypdf import PdfReader
                reader = PdfReader(str(p))
                for page in reader.pages:
                    writer.add_page(page)
            with open(out_path, 'wb') as fh:
                writer.write(fh)
            return

    for p in pdf_paths:
        merger.append(str(p))
    merger.write(str(out_path))
    merger.close()

def generate_pdf():
    # Ensure dependencies
    if not ensure_package('weasyprint'):
        print('Failed to install weasyprint. Aborting.')
        return False
    if not ensure_package('beautifulsoup4', 'bs4'):
        print('Failed to install beautifulsoup4. Aborting.')
        return False
    if not ensure_package('pypdf'):
        print('Failed to install pypdf. Aborting.')
        return False

    html_dir = Path(__file__).parent / '_build' / 'html'
    output_dir = Path(__file__).parent / '_build' / 'pdf'
    output_dir.mkdir(parents=True, exist_ok=True)

    index_html = html_dir / 'index.html'
    if not index_html.exists():
        print(f'Error: {index_html} not found. Run `sphinx-build` first.')
        return False

    print('Extracting TOC order from index.html...')
    hrefs = extract_toc_html_order(index_html)
    if not hrefs:
        print('No TOC links found; falling back to index.html only.')
        hrefs = ['index.html']

    # Build list of HTML files in order (dedupe and normalize)
    html_pages = []
    for h in hrefs:
        path = (html_dir / h).resolve()
        if path.exists():
            html_pages.append(path)
        else:
            # sometimes links are like "modules/preprocessing.html#module..." — strip fragment
            clean = h.split('#')[0]
            candidate = (html_dir / clean).resolve()
            if candidate.exists():
                html_pages.append(candidate)

    # Always start with index.html
    if html_dir.joinpath('index.html').resolve() not in html_pages:
        html_pages.insert(0, html_dir.joinpath('index.html').resolve())

    print(f'Converting {len(html_pages)} HTML pages to PDF...')

    tmp_dir = output_dir / 'tmp_pages'
    tmp_dir.mkdir(exist_ok=True)
    pdf_paths = []
    for i, hpath in enumerate(html_pages, start=1):
        out_pdf = tmp_dir / f'{i:03d}_{hpath.stem}.pdf'
        print(f'  [{i}/{len(html_pages)}] {hpath.name} -> {out_pdf.name}')
        try:
            html_to_pdf(hpath, out_pdf, html_dir)
            pdf_paths.append(out_pdf)
        except Exception as e:
            print(f'    Error converting {hpath}: {e}')

    if not pdf_paths:
        print('No PDFs were generated. Aborting.')
        return False

    final_pdf = output_dir / 'VNIT_Housing_Prediction_full.pdf'
    print(f'Merging {len(pdf_paths)} PDFs into {final_pdf}...')
    merge_pdfs(pdf_paths, final_pdf)

    # Clean up temporary files
    for p in pdf_paths:
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    if final_pdf.exists():
        size_mb = final_pdf.stat().st_size / (1024 * 1024)
        print(f'✓ Merged PDF created: {final_pdf} ({size_mb:.2f} MB)')
        return True
    else:
        print('❌ Failed to create merged PDF')
        return False

if __name__ == '__main__':
    ok = generate_pdf()
    sys.exit(0 if ok else 1)
