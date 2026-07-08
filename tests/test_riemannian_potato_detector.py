"""Tests for RiemannianPotatoDetector."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pyriemann_available = importlib.util.find_spec("pyriemann") is not None
requires_pyriemann = pytest.mark.skipif(
    not pyriemann_available,
    reason="pyriemann not installed (install with: pip install 'mne-rt[riemann]')",
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N_CH = 8
N_SAMP = 200  # samples per window
N_WIN = 30  # calibration windows


def _clean_windows(n_win=N_WIN, n_ch=N_CH, n_samp=N_SAMP, rng=None):
    """Return low-amplitude, well-conditioned random windows."""
    rng = rng or RNG
    return rng.standard_normal((n_win, n_ch, n_samp)).astype(np.float64)


def _artifact_windows(n_win=5, n_ch=N_CH, n_samp=N_SAMP, scale=100.0):
    """Return windows with extremely high amplitude (obvious artifacts)."""
    return _clean_windows(n_win, n_ch, n_samp) * scale


@pytest.fixture
def detector():
    if not pyriemann_available:
        pytest.skip("pyriemann not installed (install with: pip install 'mne-rt[riemann]')")
    from mne_rt.tools import RiemannianPotatoDetector

    return RiemannianPotatoDetector(threshold=3.0)


@pytest.fixture
def fitted_detector():
    if not pyriemann_available:
        pytest.skip("pyriemann not installed (install with: pip install 'mne-rt[riemann]')")
    from mne_rt.tools import RiemannianPotatoDetector

    det = RiemannianPotatoDetector(threshold=3.0)
    det.fit(_clean_windows())
    return det


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@requires_pyriemann
def test_default_params():
    from mne_rt.tools import RiemannianPotatoDetector

    det = RiemannianPotatoDetector()
    assert det.threshold == 3.0
    assert det.estimator == "oas"
    assert det.metric == "riemann"


@requires_pyriemann
def test_custom_params():
    from mne_rt.tools import RiemannianPotatoDetector

    det = RiemannianPotatoDetector(threshold=2.5, estimator="scm", metric="logeuclid")
    assert det.threshold == 2.5
    assert det.estimator == "scm"
    assert det.metric == "logeuclid"


@requires_pyriemann
def test_invalid_threshold():
    from mne_rt.tools import RiemannianPotatoDetector

    with pytest.raises(ValueError, match="threshold must be > 0"):
        RiemannianPotatoDetector(threshold=0.0)
    with pytest.raises(ValueError, match="threshold must be > 0"):
        RiemannianPotatoDetector(threshold=-1.0)


def test_importerror_when_pyriemann_missing():
    """Constructor must raise ImportError if pyriemann is not available."""
    with patch.dict(sys.modules, {"pyriemann": None}):
        from mne_rt.tools import riemannian_potato as mod

        with pytest.raises(ImportError, match="pyriemann"):
            mod.RiemannianPotatoDetector._check_pyriemann()


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_sets_attributes(detector):
    detector.fit(_clean_windows(n_win=20))
    assert detector.is_fitted_ is True
    assert detector.n_channels_ == N_CH
    assert detector.n_calibration_windows_ == 20


def test_fit_returns_self(detector):
    ret = detector.fit(_clean_windows())
    assert ret is detector


def test_fit_wrong_ndim(detector):
    with pytest.raises(ValueError, match="3-D"):
        detector.fit(np.zeros((N_WIN, N_CH)))


def test_fit_too_few_windows(detector):
    with pytest.raises(ValueError, match="At least 2"):
        detector.fit(_clean_windows(n_win=1))


def test_fit_minimum_windows(detector):
    detector.fit(_clean_windows(n_win=2))
    assert detector.n_calibration_windows_ == 2


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


def test_detect_before_fit_raises(detector):
    window = _clean_windows(n_win=1)[0]
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        detector.detect(window)


def test_detect_wrong_n_channels(fitted_detector):
    wrong_window = np.zeros((N_CH + 2, N_SAMP))
    with pytest.raises(ValueError, match="Expected window shape"):
        fitted_detector.detect(wrong_window)


def test_detect_wrong_ndim(fitted_detector):
    with pytest.raises(ValueError, match="Expected window shape"):
        fitted_detector.detect(np.zeros((N_CH, N_SAMP, 1)))


def test_detect_clean_data_is_clean(fitted_detector):
    """Windows from the same distribution as calibration should be accepted."""
    accepted = 0
    for _ in range(20):
        window = _clean_windows(n_win=1)[0]
        is_clean, z_score = fitted_detector.detect(window)
        assert isinstance(is_clean, bool)
        assert isinstance(z_score, float)
        if is_clean:
            accepted += 1
    # At least half of clean windows should be accepted
    assert accepted >= 10, f"Only {accepted}/20 clean windows accepted"


def test_detect_artifact_high_zscore(fitted_detector):
    """High-amplitude artifacts should produce elevated z-scores."""
    z_scores = []
    for _ in range(10):
        window = _artifact_windows(n_win=1)[0]
        _, z_score = fitted_detector.detect(window)
        z_scores.append(z_score)
    assert np.mean(z_scores) > fitted_detector.threshold


def test_detect_returns_tuple(fitted_detector):
    window = _clean_windows(n_win=1)[0]
    result = fitted_detector.detect(window)
    assert len(result) == 2
    is_clean, z_score = result
    assert isinstance(is_clean, bool)
    assert z_score >= 0.0


# ---------------------------------------------------------------------------
# Attributes after fit
# ---------------------------------------------------------------------------


def test_not_fitted_initially(detector):
    assert detector.is_fitted_ is False
    assert detector.n_channels_ == 0
    assert detector.n_calibration_windows_ == 0


def test_refit_updates_attributes(detector):
    detector.fit(_clean_windows(n_win=20))
    assert detector.n_calibration_windows_ == 20
    detector.fit(_clean_windows(n_win=40))
    assert detector.n_calibration_windows_ == 40


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_before_fit(detector):
    r = repr(detector)
    assert "RiemannianPotatoDetector" in r
    assert "not fitted" in r
    assert "threshold=3.0" in r


def test_repr_after_fit(fitted_detector):
    r = repr(fitted_detector)
    assert "fitted on" in r
    assert str(N_WIN) in r
    assert str(N_CH) in r
