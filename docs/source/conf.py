import os
import sys
from unittest.mock import MagicMock

# Force a non-Qt matplotlib backend BEFORE any ant/pyqtgraph import so that
# matplotlib does not try to version-check the mocked PyQt6.
os.environ.setdefault("MPLBACKEND", "Agg")

# Insert the package source so autodoc can import ant regardless of CWD
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))

# ---------------------------------------------------------------------------
# Mock heavy GUI / hardware dependencies so autodoc works without them.
# These are installed into sys.modules HERE (at conf.py load time) so that
# autosummary's module-import phase finds them before autodoc's own mocking.
# ---------------------------------------------------------------------------
_MOCK_MODULES = [
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "pyqtgraph",
    "pyqtgraph.Qt",
    "pyqtgraph.exporters",
    "pyvista",
    "pyvistaqt",
    "pylsl",
    "mne_lsl",
    "mne_lsl.lsl",
    "mne_lsl.player",
    "mne_lsl.stream",
    "pactools",
    "pyunlocbox",
    "python_osc",
    "python_osc.udp_client",
    "python_osc.dispatcher",
    "python_osc.osc_server",
    "nibabel",
    "nibabel.freesurfer",
    "mne_icalabel",
    # matplotlib Qt backends (topo_plot.py imports FigureCanvasQTAgg directly)
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.qt_compat",
]
for _mod in _MOCK_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

# Also tell autodoc to mock them (belt-and-suspenders)
autodoc_mock_imports = _MOCK_MODULES

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = 'Advanced Neurofeedback Toolbox (ANT)'
copyright = '2025, Payam S. Shabestari'
author = 'Payam S. Shabestari'
release = '1.0.0'

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "sphinx_tabs.tabs",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

suppress_warnings = [
    "ref.python",            # unresolved Python cross-refs
    "app.add_node",          # sphinx-tabs node duplication
    "autosummary",           # autosummary miscellaneous
    "autoapi.python_import_resolution",
]

# autodoc / autosummary
autosummary_generate = True
autosummary_generate_overwrite = False   # keep hand-written RSTs intact
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# numpydoc
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True       # makes types clickable cross-references
numpydoc_xref_ignore = {"optional", "default", "or"}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "ANT"
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    "logo": {
        "image_light": "_static/ANT_Logo_Horizontal.svg",
        "image_dark":  "_static/ANT_Logo_Horizontal.svg",
    },
    "github_url": "https://github.com/payamsash/ANT",
    "navbar_end": ["navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_toc_level": 2,
}

html_sidebars = {
    "index": [],
}

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy":  ("https://numpy.org/doc/stable/", None),
    "scipy":  ("https://docs.scipy.org/doc/scipy/", None),
    "mne":    ("https://mne.tools/stable/", None),
}

# ---------------------------------------------------------------------------
# Bibliography
# ---------------------------------------------------------------------------

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'

# ---------------------------------------------------------------------------
# Sphinx Gallery
# ---------------------------------------------------------------------------

sphinx_gallery_conf = {
    'examples_dirs':        os.path.join(_here, '..', '..', 'examples'),
    'gallery_dirs':         'auto_examples',
    'filename_pattern':     r'plot_.*\.py',
    'run_stale_examples':   False,   # only re-run when source md5 changes
    'plot_gallery':         True,
    'download_all_examples': False,
    'show_memory':          False,
    'backreferences_dir':   os.path.join('generated', 'backreferences'),
    'doc_module':           ('ant',),
    'reference_url':        {'ant': None},
    'first_notebook_cell':  "import matplotlib\nmatplotlib.use('Agg')\n",
    # Show source code even if execution fails
    'abort_on_example_error': False,
}
