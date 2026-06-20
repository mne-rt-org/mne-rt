import os
import sys
import warnings
from unittest.mock import MagicMock

# Suppress third-party warnings that are irrelevant to our docs build:
#
# 1. numpydoc uses Sphinx's deprecated dict interface on Options objects.
#    RemovedInSphinx11Warning (a PendingDeprecationWarning subclass) fires
#    on every documented member with Sphinx 9.x — nothing actionable here.
warnings.filterwarnings(
    "ignore",
    message=".*mapping interface for autodoc options.*",
)
try:
    from sphinx.deprecation import RemovedInSphinx11Warning
    warnings.filterwarnings("ignore", category=RemovedInSphinx11Warning)
except Exception:
    pass

# 2. numpydoc UserWarnings about custom sections in existing docstrings
#    (OperantProtocol, ModalityMixin) — cosmetic, not a build failure.
warnings.filterwarnings(
    "ignore",
    message=".*Unknown section.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*potentially wrong underline length.*",
    category=UserWarning,
)

# Force a non-Qt matplotlib backend BEFORE any ant/pyqtgraph import so that
# matplotlib does not try to version-check the mocked Qt modules.
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
    "qtpy",
    "qtpy.QtCore",
    "qtpy.QtGui",
    "qtpy.QtWidgets",
    "pyqtgraph",
    "pyqtgraph.Qt",
    "pyqtgraph.exporters",
    "pyvista",
    "pyvistaqt",
    "pylsl",          # not installed standalone; mne_lsl bundles its own bindings
    "pactools",
    "pyunlocbox",
    "python_osc",
    "python_osc.udp_client",
    "python_osc.dispatcher",
    "python_osc.osc_server",
    "mne_icalabel",
    # matplotlib Qt backends (topo_plot.py imports FigureCanvasQTAgg directly)
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.qt_compat",
    # optional heavy deps
    "pyriemann",
    "pyriemann.clustering",
    "pyriemann.estimation",
]
for _mod in _MOCK_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

# SignalPlot and TopoPlot inherit from QMainWindow.  When QMainWindow is a
# MagicMock instance, autodoc resolves those classes as "alias of <MagicMock>"
# and drops the docstring entirely.  Replacing QMainWindow with a real (dummy)
# Python class lets autodoc see the proper class hierarchy and renders the docs.
class _MockQMainWindow:
    """Placeholder used during Sphinx docs build for qtpy.QtWidgets.QMainWindow."""
    def __init__(self, *args, **kwargs): pass

sys.modules["qtpy.QtWidgets"].QMainWindow = _MockQMainWindow

# Also tell autodoc to mock them (belt-and-suspenders)
autodoc_mock_imports = _MOCK_MODULES

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = 'MNE-RT'
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
html_title = "MNE-RT"
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    "logo": {
        "image_light": "_static/mne_rt_logo.svg",
        "image_dark":  "_static/mne_rt_logo.svg",
    },
    "github_url": "https://github.com/payamsash/mne-rt",
    "navbar_end": ["navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
}

html_sidebars = {
    "index": [],
    "**": ["sidebar-nav-bs"],
}

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python":  ("https://docs.python.org/3/", None),
    "numpy":   ("https://numpy.org/doc/stable/", None),
    "scipy":   ("https://docs.scipy.org/doc/scipy/", None),
    "mne":     ("https://mne.tools/stable/", None),
    "mne_lsl": ("https://mne.tools/mne-lsl/stable/", None),
}

# ---------------------------------------------------------------------------
# Bibliography
# ---------------------------------------------------------------------------

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'

# ---------------------------------------------------------------------------
# Sphinx Gallery
# ---------------------------------------------------------------------------

_examples_root = os.path.abspath(os.path.join(_here, '..', '..', 'examples'))

sphinx_gallery_conf = {
    'examples_dirs':        _examples_root,
    'gallery_dirs':         'auto_examples',
    'filename_pattern':     r'ex_.*\.py',
    'run_stale_examples':   False,   # only re-run when source md5 changes
    'plot_gallery':         True,
    'download_all_examples': False,
    'show_memory':          False,
    'backreferences_dir':   None,
    'first_notebook_cell':  "import matplotlib\nmatplotlib.use('Agg')\n",
    'abort_on_example_error': False,
}
