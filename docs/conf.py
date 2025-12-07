# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project source to path
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../src/preprocessing'))
sys.path.insert(0, os.path.abspath('../src/modeling'))

# Project information
project = 'VNIT Housing Price Prediction'
copyright = '2025, Shivam Awasthi & Abhishiek Bhadauria'
author = 'Shivam Awasthi, Abhishiek Bhadauria'
release = '1.0.0'

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'rst2pdf.pdfbuilder'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_method = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# HTML theme
html_theme = 'sphinx_rtd_theme'
html_static_path = []
html_theme_options = {
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
}

# LaTeX / PDF settings
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '12pt',
    'preamble': r'''
\usepackage{xcolor}
\definecolor{vnit-blue}{HTML}{2980B9}
''',
    'fncychap': '\\usepackage[Bjornstrup]{fncychap}',
}

latex_documents = [
    ('index', 'VNIT_Housing_Prediction.tex', 'VNIT Housing Price Prediction Documentation',
     'Shivam Awasthi, Abhishiek Bhadauria', 'manual'),
]

# Source files
source_suffix = '.rst'
master_doc = 'index'
language = 'en'
exclude_patterns = ['_build']
pygments_style = 'sphinx'
todo_include_todos = True

# rst2pdf options (improve Sphinx cross-reference support)
rst2pdf = {
    'use_sphinx': True,
}
