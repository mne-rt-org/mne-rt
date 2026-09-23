# Contributing to MNE-RT

Thanks for your interest in MNE-RT. Bug reports, documentation fixes, new
neurofeedback modalities, protocols and visualisation windows are all welcome.

If you only want to **report a problem or ask a question**, see
[SUPPORT.md](SUPPORT.md) instead — you do not need to read this file.

By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

MNE-RT needs Python ≥ 3.11 and a Qt binding (`mne_rt` imports `pyqtgraph` at
module scope, so `import mne_rt` fails without one, even headless).

```bash
git clone https://github.com/mne-rt-org/mne-rt.git
cd mne-rt
pip install -e ".[dev,lint,pyqt6]"     # or pyside6 instead of pyqt6
```

Check the install:

```bash
mne-rt info                 # prints MNE-RT and dependency versions
mne-rt demo --duration 10   # runs the full pipeline, no amplifier needed
```

Some source-space tests need FreeSurfer anatomy (`fetch_fsaverage`, ~700 MB) and
are marked `slow`. Skip them while iterating:

```bash
pytest -m "not slow"
```

## Before you open a pull request

CI runs exactly these, so run them locally first:

```bash
ruff check src tests
ruff format --check src tests
pytest -m "not slow"
```

`ruff format src tests` fixes formatting in place. CI additionally runs the full
suite on Python 3.11, 3.12 and 3.13, and a `minimum` job that installs every
dependency at its declared lower bound — if you add a dependency or raise a
bound, say why in a comment next to it, as the existing bounds in
`pyproject.toml` do.

## Workflow

- Work on a branch off `main` and open a pull request; `main` is not pushed to
  directly.
- Keep one logical change per pull request.
- Add or update tests in `tests/` for anything you change. The suite is large
  (950+ tests across 38 modules) and it is what keeps the real-time paths honest.
- Add a line to `docs/source/whats_new.rst` under the current unreleased version.
- Public functions and methods use NumPy-style docstrings and accept a
  `verbose=` keyword, following MNE-Python's conventions.

## Extending MNE-RT

The package is built so that the parts that vary between studies are small,
self-contained classes. To add one, copy the nearest existing example:

**A neurofeedback modality.** Add a `_<name>(self, data, ...)` method to
`ModalityMixin` in `src/mne_rt/modalities.py`, returning one float per window,
plus an optional `_<name>_prep(self)` for anything that can be computed once per
session. Register its defaults under `NF_modality` in
`src/mne_rt/config_methods.yml` and give it a display scale in
`rt_stream.py`. Modalities are resolved by name, so the method name must match
the config key.

**A protocol.** Add a class in `src/mne_rt/protocols/` implementing
`evaluate(value) -> (crossed, magnitude)`, `reset()`, and — if it has one — a
`current_threshold` property, which `NFPlot` draws as a live threshold line.
Export it from `src/mne_rt/protocols/__init__.py`.

**An artifact-correction method.** Add a class in `src/mne_rt/tools/` with
`fit()` / `transform()`, then wire it into `_correct` and
`_VALID_ARTIFACT_METHODS` in `src/mne_rt/rt_stream.py`.

**A visualisation window.** Add a class in `src/mne_rt/viz/`. Anything drawn on
the Qt main thread must not block acquisition; follow the existing
`NFPlot`/`RawPlot` pattern of a timer-driven refresh reading from a queue.

Real-time code has one extra rule: **anything that can be precomputed must be**.
The online step should be a bounded per-window cost. Where a computation cannot
meet that bar exactly, prefer refusing it to returning an approximation — see
`SourceModel.supports_kernel` in `src/mne_rt/source.py` for the pattern.

## Documentation

```bash
pip install -e ".[docs]"
cd docs && make html      # output in docs/_build/html
```

Examples in `examples/` are executed by sphinx-gallery, so they must run
end-to-end without hardware.
