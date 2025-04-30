# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
# allow sphinx to import your package
sys.path.insert(0, os.path.abspath("../.."))

project = "canml"
author = "Cosmin B. Memetea"
release = "0.1.12-alpha"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# use the Read-the-Docs theme locally
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# conf.py
html_logo = "_static/canml-icon.svg"
html_favicon = "_static/canml-icon.svg"  # optional: browser tab icon
