# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../../zeus/'))

import zeus

import sphinx_bootstrap_theme


# -- Project information -----------------------------------------------------

project = 'zeus'
copyright = '2019-2022, Minas Karamanis'
author = 'Minas Karamanis'

# The full version, including alpha/beta/rc tags
release = zeus.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    #'numpydoc',
    'nbsphinx',
    'sphinx.ext.coverage',
    'IPython.sphinxext.ipython_console_highlighting',
]

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ".rst"
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
#exclude_patterns = ['_build']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'
html_favicon = "_static/favicon.png"


# (Optional) Logo. Should be small enough to fit the navbar (ideally 24x24).
# Path should be relative to the ``_static`` files directory.
#html_logo = "my_logo.png"

# Theme options are theme-specific and customize the look and feel of a
# theme further.
html_theme_options = {
    'navbar_title': "zeus",
    'navbar_site_name': "Contents",

    'navbar_links': [

    ("Cookbook", "cookbook"),
    ("FAQ", "faq"),
    ("API", "api"),

    ],

    'navbar_sidebarrel': False, # Render the next and previous page links in navbar. (Default: true)

    'navbar_pagenav': False, # Render the current pages TOC in the navbar. (Default: true)

    'navbar_pagenav_name': "Page", # Tab name for the current pages TOC. (Default: "Page")

    # Global TOC depth for "site" navbar tab. (Default: 1)
    # Switching to -1 shows all levels.
    'globaltoc_depth': 2,

    # Include hidden TOCs in Site navbar?
    #
    # Note: If this is "false", you cannot have mixed ``:hidden:`` and
    # non-hidden ``toctree`` directives in the same page, or else the build
    # will break.
    #
    # Values: "true" (default) or "false"
    'globaltoc_includehidden': "true",

    # HTML navbar class (Default: "navbar") to attach to <div> element.
    # For black navbar, do "navbar navbar-inverse"
    #'navbar_class': "navbar navbar-inverse",

    # Fix navigation bar to top of page?
    # Values: "true" (default) or "false"
    'navbar_fixed_top': "true",

    #'bootswatch_theme': "united",
    #'bootswatch_theme': "paper",
    #'bootswatch_theme': "cosmo",
    'bootswatch_theme': "readable",
    #'bootswatch_theme': "flatly",
    #'bootswatch_theme': "Yeti",

    'bootstrap_version': "3",

    'body_max_width' : '100%',
    #'body_min_width' : '70%',

}

html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# Add the 'copybutton' javascript, to hide/show the prompt in code
# examples, originally taken from scikit-learn's doc/conf.py
#def setup(app):
#    app.add_javascript('copybutton.js')
#    app.add_stylesheet('default.css')

#html_css_files = ['_static',]

#html_context = {'css_files': ['_static/default.css',  # override wide tables in RTD theme
#],}

#autodoc_default_options = {
#    'exclude-members': '__init__'
#}

#autoclass_content = ["class"]