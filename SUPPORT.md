# Getting help with MNE-RT

## Start with the documentation

Full documentation, including installation notes, a tutorial series, the API
reference and an example gallery, is at
**<https://mne-rt-org.github.io/mne-rt/>**.

Two commands answer most setup questions:

```bash
mne-rt info                 # MNE-RT and all dependency versions
mne-rt demo --duration 10   # runs the whole pipeline without an amplifier
```

If `mne-rt demo` works, your installation is sound and the problem is likely in
the stream configuration rather than in MNE-RT.

## Reporting a bug

Open an issue: <https://github.com/mne-rt-org/mne-rt/issues>

Please include:

- the output of `mne-rt info`;
- your operating system and Qt binding (PyQt6 or PySide6);
- a minimal script that reproduces the problem — ideally using
  `connect_to_lsl(mock_lsl=True)` or `connect_to_array(...)` so it runs without
  your hardware;
- the full traceback, and the log output with `mne_rt.set_log_level("DEBUG")`.

For real-time problems that are not crashes — dropped windows, latency, a
feature that never moves — attach the session JSON written by `save()`. It
records per-window latencies, artifact rate and SNR, which is usually enough to
tell an acquisition problem from a processing one.

## Asking a question

For usage questions, experiment design, or "is this the right modality for my
study", open a
[GitHub Discussion](https://github.com/mne-rt-org/mne-rt/discussions) if the
category is enabled, otherwise an issue labelled `question`.

Questions about the underlying M/EEG analysis rather than MNE-RT itself are
often better answered by the [MNE-Python forum](https://mne.discourse.group).

## Contributing a fix

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Please do not report security issues in a public issue. Email the maintainer at
<payam.sadeghi74@gmail.com>.
