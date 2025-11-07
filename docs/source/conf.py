"""
Configuration file for the Sphinx documentation builder.
"""

import os
import sys

# Add the package to the path
sys.path.insert(0, os.path.abspath('../../src'))

# Import version
try:
    from pathsim_chem import __version__
except ImportError:
    __version__ = "unknown"

# -- Project information -----------------------------------------------------

project = 'pathsim-chem'
copyright = '2025, PathSim Contributors'
author = 'PathSim Contributors'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static', 'logos']
html_css_files = ['custom.css']
html_logo = 'logos/pathsim_logo.png'
html_title = "PathSim-Chem Documentation"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#377eb8",
        "color-brand-content": "#377eb8",
        "color-api-keyword": "#377eb8",
        "color-highlight-on-target": "#fff3cd",
    },
    "dark_css_variables": {
        "color-brand-primary": "#377eb8",
        "color-brand-content": "#377eb8",
        "color-api-keyword": "#377eb8",
        "color-highlight-on-target": "#fff3cd",
    },
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/pathsim/pathsim-chem",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping to link to core pathsim documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pathsim': ('https://pathsim.readthedocs.io/en/latest/', None),
}

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
