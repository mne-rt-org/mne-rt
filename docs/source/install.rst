.. _install:

Installation
============

Requirements
------------

- Python ≥ 3.11
- A running `Lab Streaming Layer (LSL) <https://labstreaminglayer.org>`_ stream
  **or** a BrainVision ``.vhdr`` file for mock mode
- FreeSurfer subjects directory *(only for source-localisation and brain-plot features)*

pip (recommended)
-----------------

.. code-block:: bash

    # Latest release (OSC output included)
    pip install ANT

    # All optional extras (viz, dev, docs)
    pip install "ANT[full]"

Editable / development install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/payamsash/ANT.git
    cd ANT
    pip install -e ".[dev]"

uv (fast Rust-based installer)
-------------------------------

`uv <https://docs.astral.sh/uv/>`_ is a drop-in replacement for pip and
respects ``pyproject.toml`` fully.

.. code-block:: bash

    # Install uv (once)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install ANT into an active environment
    uv pip install ANT
    uv pip install "ANT[full]"   # all extras (viz, dev, docs)

.. note::

   ``uv add ANT`` is for adding ANT as a dependency *of another project*.
   If you are working inside the ANT source tree, use ``uv pip install`` or
   the editable install below.

.. code-block:: bash

    # Editable install from source
    git clone https://github.com/payamsash/ANT.git
    cd ANT
    uv pip install -e ".[dev]"
    uv pip install -e ".[full]"

conda / mamba
-------------

The provided ``environment.yml`` creates a complete conda environment:

.. code-block:: bash

    # Create environment (first time)
    conda env create -f environment.yml

    # Activate
    conda activate ant

    # Update after pulling new changes
    conda env update -f environment.yml --prune

Verifying
---------

.. code-block:: bash

    ANT info     # prints ANT and dependency versions
    ANT demo     # runs a 60-second mock NF session

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
     - ``pip install "ANT[viz]"``
   * - ``dev``
     - Testing only (pytest, pytest-cov)
     - ``pip install "ANT[dev]"``
   * - ``lint``
     - Linting and formatting (ruff, mypy, pre-commit)
     - ``pip install "ANT[lint]"``
   * - ``docs``
     - Documentation build tools (Sphinx, sphinx-gallery, …)
     - ``pip install "ANT[docs]"``
   * - ``full``
     - All of the above
     - ``pip install "ANT[full]"``
