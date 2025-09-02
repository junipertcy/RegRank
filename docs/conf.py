# docs/conf.py

import os
import sys

# -- Path setup --------------------------------------------------------------
# Make the 'regrank' package available to Sphinx
sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------
project = "regrank"
copyright = "2023-2024, Tzu-Chi Yen"
author = "Tzu-Chi Yen"
release = "0.1.1"  # The full version, including alpha/beta/rc tags
version = "0.1"  # The short X.Y version


# -- General configuration ---------------------------------------------------
# A list of Sphinx extension module names
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    # Third-party extensions
    "sphinx_rtd_theme",  # The "Read the Docs" theme
    "autoapi.extension",  # Modern API documentation generator
    "myst_parser",  # For parsing Markdown files
    "sphinx.ext.napoleon",  # To understand Google and NumPy style docstrings
]

# The master toctree document.
master_doc = "index"

# List of patterns to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of file extensions to consider as source files.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# -- AutoAPI configuration ---------------------------------------------------
# Configuration for the sphinx-autoapi extension
autoapi_type = "python"
autoapi_dirs = ["../src/regrank"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_keep_files = True


# -- Intersphinx configuration ---------------------------------------------
# Allows linking to the documentation of other projects.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "graph_tool": ("https://graph-tool.skewed.de/static/doc/", None),
    "omegaconf": ("https://omegaconf.readthedocs.io/en/latest/", None),
}


# -- HTML output options -----------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]  # For custom CSS or JS
html_logo = "assets/regrank-logo.png"
html_favicon = "assets/regrank-favicon.png"

# Theme-specific options
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#0074D9",
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Context to pass to the HTML templates, used for the GitHub links
html_context = {
    "display_github": True,
    "github_user": "junipertcy",
    "github_repo": "regrank",
    "github_version": "main/docs/",
}


# -- Options for other output formats (LaTeX, man pages, etc.) ---------------
# These are generally fine to leave as is unless you specifically need to
# customize the output for these formats.

latex_documents = [
    (master_doc, "regrank.tex", "regrank Documentation", author, "manual"),
]

man_pages = [(master_doc, "regrank", "regrank Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "regrank",
        "regrank Documentation",
        author,
        "regrank",
        "Regularized methods for efficient ranking in networks.",
        "Miscellaneous",
    ),
]

epub_title = project
epub_exclude_files = ["search.html"]
