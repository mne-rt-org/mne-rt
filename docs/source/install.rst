.. _install:

Installation
============

Requirements
------------

- Python ≥ 3.11
- A running `Lab Streaming Layer (LSL) <https://labstreaminglayer.org>`_ stream
  **or** any MNE-readable file for mock mode
  (``.fif``, ``.vhdr``, ``.edf``, ``.bdf``, ``.set``, …)
- FreeSurfer subjects directory *(only for source-localisation and brain-plot features)*

pip (recommended)
-----------------

.. code-block:: bash

    # Latest release, with a Qt binding for the live windows
    pip install "mne-rt[pyqt6]"

    # All optional extras (viz, dev, docs)
    pip install "mne-rt[full,pyqt6]"

.. note::

   The live windows need a Qt binding: PyQt6 (``mne-rt[pyqt6]``) or PySide6
   (``mne-rt[pyside6]``). ``pip install mne-rt`` alone installs none, which
   suits a session without windows, analysis and the CLI; opening a window
   then says which to install.

Editable / development install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/mne-rt-org/mne-rt.git
    cd mne-rt
    pip install -e ".[dev,pyqt6]"

uv — recommended fast installer
---------------------------------

`uv <https://docs.astral.sh/uv/>`_ resolves and installs packages in Rust —
typically **10–20× faster** than plain ``pip``.  It reads ``pyproject.toml``
directly and handles all extras.

.. code-block:: bash

    # Install uv once
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install mne-rt into an active environment
    uv pip install "mne-rt[pyqt6]"
    uv pip install "mne-rt[full,pyqt6]"   # all extras (viz, dev, docs)

    # Editable install from source (recommended for development)
    git clone https://github.com/mne-rt-org/mne-rt.git
    cd mne-rt
    uv pip install -e ".[dev,pyqt6]"

.. note::

   ``uv add mne-rt`` is for adding mne-rt as a dependency *of another project*.
   Inside the mne-rt source tree use ``uv pip install -e .`` instead.

conda / mamba
-------------

MNE-RT is published on `conda-forge <https://anaconda.org/conda-forge/mne-rt>`_.
**mamba** (or micromamba) is strongly recommended over plain ``conda``
because it uses a faster C++ dependency solver.

.. code-block:: bash

    # Install mamba into base (once)
    conda install -n base -c conda-forge mamba

    # Install the latest release, with a Qt binding for the live windows
    mamba install -c conda-forge mne-rt pyqt6

    # Or with plain conda (slower)
    conda install -c conda-forge mne-rt pyqt6

Development environment
^^^^^^^^^^^^^^^^^^^^^^^^

If you're developing MNE-RT itself (rather than just using it), the
provided ``environment.yml`` creates a complete conda environment with
an editable install from source.

.. code-block:: bash

    # Create the mne-rt environment
    mamba env create -f environment.yml   # ~2 min

    # Or with plain conda (slower)
    conda env create -f environment.yml

    # Activate
    conda activate mne-rt

    # Update after pulling new changes
    mamba env update -f environment.yml --prune

Verifying
---------

.. code-block:: bash

    mne-rt info     # prints MNE-RT and dependency versions
    mne-rt demo     # runs a 120-second mock real-time session

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 10 40 30

   * - Extra
     - What it adds
     - Install command
   * - ``viz``
     - 3D brain visualisation (pyvista, pyvistaqt) — needed for BrainPlot
     - ``pip install "mne-rt[viz]"``
   * - ``dev``
     - Testing only (pytest, pytest-cov, pytest-timeout)
     - ``pip install "mne-rt[dev]"``
   * - ``lint``
     - Linting and formatting (ruff, mypy, pre-commit)
     - ``pip install "mne-rt[lint]"``
   * - ``docs``
     - Documentation build tools (Sphinx, sphinx-gallery, …)
     - ``pip install "mne-rt[docs]"``
   * - ``full``
     - All of the above
     - ``pip install "mne-rt[full]"``
