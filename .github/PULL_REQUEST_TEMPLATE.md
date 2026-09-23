## What does this change?

<!-- One or two sentences. Link the issue it closes, if there is one. -->

Closes #

## Why?

<!-- What was wrong, or what does this make possible? -->

## Checklist

- [ ] `ruff check src tests` and `ruff format --check src tests` pass
- [ ] `pytest -m "not slow"` passes
- [ ] Tests added or updated for this change
- [ ] Docstrings are NumPy-style; public methods take `verbose=`
- [ ] Entry added to `docs/source/whats_new.rst`
- [ ] If this runs inside the online loop: anything that could be precomputed is
      done in the baseline phase, and the per-window cost is bounded
