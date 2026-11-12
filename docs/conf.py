# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TabICL'
copyright = '2026, TabICL authors'
author = 'TabICL authors'
release = ''

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Local Sphinx extensions (see docs/_ext/): the gallery figures of the
# "compute time and memory footprint" page are regenerated from the results
# table in docs/_data/ at the start of every build. Making the folder
# importable this way is what the Sphinx tutorial recommends for extensions
# that are not installed as a package:
# https://www.sphinx-doc.org/en/master/development/tutorials/extending_syntax.html#using-the-extension
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

extensions = [
    "sphinx_gallery.gen_gallery",
    "compute_tradeoffs_figures",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

# opengraph, to generate social-media thumbnails
try:
    import sphinxext.opengraph  # noqa

    extensions.append("sphinxext.opengraph")
except ImportError:
    print("ERROR: sphinxext.opengraph import failed")


sphinx_gallery_conf = {
    "examples_dirs": ["../tutorials"],
    "gallery_dirs": ["tutorials"],
    "filename_pattern": r"\.py$",
}

# autodoc settings
autodoc_default_options = {
    "inherited-members": False,
}
autodoc_typehints = "none"
autodoc_mock_imports = ["shap", "shapiq"]
# Keep default arguments as their source-code names (e.g. `DEFAULT_FIXED_HP`)
# instead of evaluating them into the full dict/list repr, which bloats
# signatures like `PriorDataset(..., scm_fixed_hp={<huge dict>}, ...)`.
autodoc_preserve_defaults = True

# intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "skrub": ("https://skrub-data.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_title = "TabICL" # A simpler title in the landing

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "TabICL.ico"

html_static_path = ['_static']
html_css_files = ["css/custom.css"]
# Resize plotly figures once the DOM/CSS layout is loaded; works around a known
# crop bug between plotly and the pydata-sphinx-theme secondary sidebar.
html_js_files = ["js/sg_plotly_resize.js"]

# -- Theme Options -----------------------------------------------------------

html_theme_options = {
    "logo": {
        "image_light": "TabICL_logo.svg",
        "image_dark": "TabICL_logo_dark.svg",
    },
}
