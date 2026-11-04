"""Tests for mne_rt.decoding.RTDecode."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from mne_rt.decoding import RTDecode

SFREQ = 256.0
N_CHANNELS = 4
N_TIMES = 256
N_EPOCHS = 20


@pytest.fixture()
def info():
    return mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(N_CHANNELS)],
        sfreq=SFREQ,
        ch_types="eeg",
    )


def _make_calibration_data(n_epochs=N_EPOCHS, n_channels=N_CHANNELS, n_times=N_TIMES):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_epochs, n_channels, n_times))
    y = np.array([0, 1] * (n_epochs // 2))
    return X, y


def _make_window(n_channels=N_CHANNELS, n_times=N_TIMES, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_times))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRTDecodeConstruction:
    def test_default_is_csp_logistic_regression(self, info):
        from sklearn.linear_model import LogisticRegression

        d = RTDecode(info=info)
        assert d.spatial_filter == "csp"
        assert isinstance(d.estimator, LogisticRegression)
        assert not d.fitted

    def test_invalid_spatial_filter_raises(self, info):
        with pytest.raises(ValueError, match="spatial_filter"):
            RTDecode(info=info, spatial_filter="bogus")

    def test_scaler_without_info_raises(self):
        with pytest.raises(ValueError, match="info"):
            RTDecode(spatial_filter="scaler")

    def test_repr_reflects_fitted_state(self, info):
        X, y = _make_calibration_data()
        d = RTDecode(info=info, n_components=2)
        assert "not fitted" in repr(d)
        d.fit(X, y, verbose=False)
        assert "not fitted" not in repr(d)
        assert "fitted" in repr(d)


# ---------------------------------------------------------------------------
# fit() validation
# ---------------------------------------------------------------------------


class TestRTDecodeFit:
    def test_fit_returns_self(self, info):
        X, y = _make_calibration_data()
        d = RTDecode(info=info, n_components=2)
        assert d.fit(X, y, verbose=False) is d

    def test_fit_sets_classes(self, info):
        X, y = _make_calibration_data()
        d = RTDecode(info=info, n_components=2).fit(X, y, verbose=False)
        np.testing.assert_array_equal(d.classes_, [0, 1])

    def test_fit_sets_n_channels(self, info):
        X, y = _make_calibration_data()
        assert RTDecode(info=info, n_components=2).n_channels_ is None
        d = RTDecode(info=info, n_components=2).fit(X, y, verbose=False)
        assert d.n_channels_ == N_CHANNELS

    def test_fit_rejects_non_3d_input(self, info):
        d = RTDecode(info=info, n_components=2)
        with pytest.raises(ValueError, match="3D"):
            d.fit(np.zeros((N_EPOCHS, N_CHANNELS)), np.zeros(N_EPOCHS))

    def test_fit_rejects_mismatched_labels(self, info):
        X, _ = _make_calibration_data()
        d = RTDecode(info=info, n_components=2)
        with pytest.raises(ValueError, match="epochs"):
            d.fit(X, np.zeros(5))


# ---------------------------------------------------------------------------
# predict() / predict_proba() — csp path
# ---------------------------------------------------------------------------


class TestRTDecodeCSP:
    def test_predict_before_fit_raises(self, info):
        d = RTDecode(info=info, n_components=2)
        with pytest.raises(RuntimeError, match="fit"):
            d.predict(_make_window())

    def test_predict_proba_before_fit_raises(self, info):
        d = RTDecode(info=info, n_components=2)
        with pytest.raises(RuntimeError, match="fit"):
            d.predict_proba(_make_window())

    def test_predict_returns_known_class(self, info):
        X, y = _make_calibration_data()
        d = RTDecode(info=info, n_components=2).fit(X, y, verbose=False)
        label = d.predict(_make_window())
        assert label in (0, 1)

    def test_predict_proba_sums_to_one(self, info):
        X, y = _make_calibration_data()
        d = RTDecode(info=info, n_components=2).fit(X, y, verbose=False)
        proba = d.predict_proba(_make_window())
        assert proba.shape == (2,)
        assert np.sum(proba) == pytest.approx(1.0, abs=1e-6)

    def test_predict_rejects_non_2d_window(self, info):
        X, y = _make_calibration_data()
        d = RTDecode(info=info, n_components=2).fit(X, y, verbose=False)
        with pytest.raises(ValueError, match="2D"):
            d.predict(np.zeros((1, N_CHANNELS, N_TIMES)))


# ---------------------------------------------------------------------------
# predict() / predict_proba() — scaler + vectorizer path
# ---------------------------------------------------------------------------


class TestRTDecodeScaler:
    def test_scaler_pipeline_fits_and_predicts(self, info):
        X, y = _make_calibration_data()
        d = RTDecode(info=info, spatial_filter="scaler").fit(X, y, verbose=False)
        assert d.predict(_make_window()) in (0, 1)
        proba = d.predict_proba(_make_window())
        assert proba.shape == (2,)
        assert np.sum(proba) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# predict_proba() without a probabilistic estimator
# ---------------------------------------------------------------------------


class TestRTDecodeNonProbabilisticEstimator:
    def test_predict_proba_raises_without_support(self, info):
        from sklearn.svm import LinearSVC

        X, y = _make_calibration_data()
        d = RTDecode(info=info, estimator=LinearSVC(), n_components=2)
        d.fit(X, y, verbose=False)
        with pytest.raises(AttributeError, match="predict_proba"):
            d.predict_proba(_make_window())
        # predict() still works since LinearSVC supports it natively.
        assert d.predict(_make_window()) in (0, 1)
