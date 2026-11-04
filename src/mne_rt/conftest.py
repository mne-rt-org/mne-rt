"""Fixtures shared by every test in mne_rt."""

import pytest


@pytest.fixture(scope="module")
def meg_info():
    """Return MEG info from MNE sample data, or skip if unavailable."""
    pytest.importorskip("mne")
    try:
        import mne

        data_path = mne.datasets.sample.data_path(download=False, verbose=False)
        fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
        if not fname.exists():
            pytest.skip("MNE sample data not downloaded")
        raw = mne.io.read_raw_fif(fname, preload=False, verbose=False)
        return raw.info
    except Exception:
        pytest.skip("MNE sample data not available")
