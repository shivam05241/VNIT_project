#!/usr/bin/env python3
"""
PDF Generator for VNIT Project Documentation
Converts HTML documentation to PDF using weasyprint
"""

import os
import sys
from pathlib import Path

def generate_pdf():
    """Generate PDF from HTML documentation."""
    try:
        from weasyprint import HTML, CSS
        print("✓ weasyprint found, generating PDF...")
    except ImportError:
        print("Installing weasyprint...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "weasyprint"], check=True)
        from weasyprint import HTML, CSS
    
    # Paths
    html_dir = Path(__file__).parent / "_build" / "html"
    output_dir = Path(__file__).parent / "_build" / "pdf"
    output_dir.mkdir(exist_ok=True)
    
    index_html = html_dir / "index.html"
    output_pdf = output_dir / "VNIT_Housing_Prediction.pdf"
    
    if not index_html.exists():
        print(f"❌ Error: {index_html} not found")
        return False
    
    print(f"Converting {index_html} to {output_pdf}...")
    
    try:
        # Generate PDF
        HTML(string=open(index_html).read(), base_url=str(html_dir)).write_pdf(
            str(output_pdf),
            stylesheets=[CSS(string="@page { size: A4; margin: 2cm; }")]
        )
        
        # Verify
        if output_pdf.exists():
            size_mb = output_pdf.stat().st_size / (1024 * 1024)
            print(f"✓ PDF generated successfully: {output_pdf} ({size_mb:.2f} MB)")
            return True
        else:
            print("❌ PDF generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

if __name__ == "__main__":
    success = generate_pdf()
    sys.exit(0 if success else 1)
