# JOSS submission checklist — MNE-RT

Working notes for submitting MNE-RT to the [Journal of Open Source
Software](https://joss.theoj.org). Requirements below follow the JOSS author
guidelines and reviewer checklist as of September 2026.

---

## Blocking — you must do these by hand

| # | Task | Where |
|---|---|---|
| 1 | **Tag and release `v1.2.0`** — `release.yml` then publishes to PyPI automatically | GitHub |
| 2 | **Archive the release and get a DOI** — enable the GitHub–Zenodo integration, then re-publish the release | [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github) |
| 3 | **Record the DOI** — uncomment the `doi:` line in `CITATION.cff` and add a README badge | repo root |
| 4 | **Submit** — repo URL, version `v1.2.0`, and the Zenodo DOI | [joss.theoj.org/papers/new](https://joss.theoj.org/papers/new) |

ORCID `0000-0002-2647-4891` and all three affiliations are filled in.

---

## Done in this pass

### Paper
- [x] Author block complete — ORCID, three affiliations, `corresponding: true`,
      ROR `01462r250` for University Hospital Zurich (ZNZ and FCBG have no
      unambiguous ROR record; `ror` is optional)
- [x] `paper/paper.md` — 1,396 words (JOSS asks for 750–1750)
- [x] All six mandatory sections present: Summary · Statement of need ·
      State of the field · Software design · Research impact statement ·
      AI usage disclosure
- [x] `paper/paper.bib` — 20 entries, **every DOI verified against Crossref**
- [x] `paper/figure1_architecture.png` + `paper/make_figure.py` (regenerable)
- [x] `.github/workflows/draft-paper.yml` — renders `paper.pdf` with the
      official JOSS action, on demand or on any PR touching `paper/`

### Repository requirements
- [x] OSI-approved licence in a plain-text `LICENSE` (BSD 3-Clause), now
      **consistent** with `pyproject.toml`, the README badge and the README footer
- [x] `CONTRIBUTING.md` — how to contribute
- [x] `CODE_OF_CONDUCT.md` — Contributor Covenant 2.1
- [x] `SUPPORT.md` — how to report issues and seek help
- [x] `CITATION.cff` — GitHub renders a "Cite this repository" button from it
- [x] `.github/ISSUE_TEMPLATE/` (bug report, feature request, config) and
      `.github/PULL_REQUEST_TEMPLATE.md`
- [x] README: CI and Codecov badges, and a "Getting help" section

### Already satisfied before this pass
- [x] **Development history** — public since May 2024 (> 2 years, 512 commits,
      32 merged PRs). JOSS requires ≥ 6 months of open development.
- [x] **Automated tests** — 952 passing, 14 skipped, in 38 modules; CI on
      Python 3.11/3.12/3.13
      plus a lowest-direct-dependency resolution job
- [x] **Documentation** — statement of need, installation, dependency list,
      tutorials, API reference, and 8 executable gallery examples
- [x] **Packaging** — PyPI and conda-forge, standard `pyproject.toml`
- [x] **Issue tracker** — public, no registration needed to browse

---

## Verify before you submit

```bash
# 1. Word count must land in 750–1750
python -c "import re;t=open('paper/paper.md').read().split('---',2)[2];\
t=re.sub(r'!\[[^]]*\]\([^)]*\)','',t,flags=re.S);print(len(re.findall(r'\S+',t)))"

# 2. Render the paper with the same pipeline the JOSS editor uses.
#    Easiest route: run the "Draft JOSS paper" workflow from the Actions tab
#    (.github/workflows/draft-paper.yml) and download the `paper` artifact.
#    Locally, if you have Docker:
docker run --rm -v "$PWD":/data -u $(id -u):$(id -g) \
  openjournals/inara -o pdf,crossref paper/paper.md
#    → check paper/paper.pdf: no unresolved "?" citations, figure embedded,
#      all six section headings present

# 3. Repository still clean
ruff check src tests && ruff format --check src tests
pytest -m "not slow"
python -m build          # sdist must still prune paper/ (MANIFEST.in)

# 4. The claims a reviewer will actually test
mne-rt info
mne-rt demo --duration 10
```

---

## Recommended, not blocking

These will not stop a submission, but reviewers commonly raise them.

- **macOS / Windows CI runners.** `pyproject.toml` claims
  `Operating System :: OS Independent`, but `ci.yml` only runs `ubuntu-latest`.
- **Stale config entry.** `src/mne_rt/config_methods.yml` lists a 21st modality,
  `instantaneous_phase`, but no `_instantaneous_phase` method exists, so
  requesting it raises `NotImplementedError` (`rt_stream.py:1923`). Either
  implement it — `compute_instantaneous_phase` already exists in
  `tools/tools.py:396` — or drop the entry. The paper deliberately says **20**
  modalities, which is the number that actually works.
- **Gallery is partially stale.** Only 4 of the 8 `examples/*.py` have rendered
  output under `docs/source/auto_examples/`; `ex_maxwell_realtime`,
  `ex_motor_imagery_decode` and `ex_motor_imagery_nf` are missing entirely.
- **`.gitattributes` is a copied Unity/USD template.** It routes `*.json` and
  `*.ipynb` through git-LFS, which can bite the committed gallery notebooks, and
  carries irrelevant `*.usd`/`*.shader`/`*.cginc` rules.
- **Typos in `docs/source/_static/code_design.svg`** — "Fearure extraction",
  "Qt dispalys", "connectiity", "Psychopy". The paper uses its own figure, but
  this one is on the documentation site.
- **Docstring overstates a tolerance.** `src/mne_rt/source.py:19` says the ROI
  kernel is "verified to ~1e-15 relative error in the test suite", but
  `tests/test_source.py:439` asserts `< 1e-10`. The paper quotes 1e-10, the
  number that is actually asserted; either tighten the test or soften the
  docstring so the two agree.
- **The dev virtualenv is stale.** `mne-rt info` in `.venv` reports 1.0.2 while
  `pyproject.toml` says 1.2.0 — re-run the editable install before measuring
  anything for the submission.
- **`setuptools-scm` is declared but unused.** It is in `build-system.requires`
  with no `[tool.setuptools_scm]` section, while `version` is hardcoded.
