"""Tests for real-time EEG microstate backfitting and the ``microstates`` modality."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mne_rt.tools.microstates import (
    METRICS,
    backfit,
    microstate_parameters,
    reject_edge_segments,
    resolve_templates,
    zscore_templates,
)

CONFIG = Path(__file__).parent.parent / "src" / "mne_rt" / "config_methods.yml"
SFREQ = 256.0
CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "T7", "T8"]


def _maps(n_states=4, n_channels=len(CH), seed=0):
    return np.random.default_rng(seed).standard_normal((n_states, n_channels))


def _states(n, rng, n_states=4, min_s=0.05, max_s=0.15):
    """A state sequence of random segments lasting min_s..max_s seconds."""
    states = np.empty(n, int)
    i = 0
    while i < n:
        length = int(rng.uniform(min_s, max_s) * SFREQ)
        states[i : i + length] = rng.integers(n_states)
        i += length
    return states


def _oscillating(maps, states, rng, noise=0.02, freq=10.0):
    """Microstate-like data: each segment's map, modulated at ``freq`` Hz.

    The modulation keeps the signal inside the modality's default band-pass;
    flipping polarity every sample would put it far above 20 Hz.
    """
    t = np.arange(states.size) / SFREQ
    x = maps[states].T * np.sin(2 * np.pi * freq * t)
    return (x + noise * rng.standard_normal(x.shape)) * 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Core: backfitting and metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_backfit_recovers_known_states():
    rng = np.random.default_rng(1)
    maps = _maps()
    states = _states(int(4 * SFREQ), rng)
    x = maps[states].T + 0.02 * rng.standard_normal((maps.shape[1], states.size))
    labels, corr, gfp = backfit(x, zscore_templates(maps))
    assert np.mean(labels == states) > 0.99
    assert corr.shape == (4, states.size) and gfp.shape == (states.size,)
    assert np.all(np.abs(corr) <= 1 + 1e-12)


def test_backfit_ignores_polarity():
    rng = np.random.default_rng(2)
    maps = _maps()
    states = _states(int(2 * SFREQ), rng)
    x = maps[states].T
    flipped = x * rng.choice([-1.0, 1.0], states.size)
    assert np.array_equal(
        backfit(x, zscore_templates(maps))[0], backfit(flipped, zscore_templates(maps))[0]
    )


def test_backfit_is_reference_free():
    """Backfitting average-references each sample, so a common offset changes nothing."""
    rng = np.random.default_rng(3)
    maps = _maps()
    states = _states(int(SFREQ), rng)
    x = maps[states].T
    shifted = x + rng.standard_normal(
        states.size
    )  # a different value added to every channel, per sample
    assert np.array_equal(
        backfit(x, zscore_templates(maps))[0], backfit(shifted, zscore_templates(maps))[0]
    )


def test_parameters_match_hand_computed_values():
    # state 1 occupies two runs of 3 and 5 samples out of 16 labelled samples
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 2])
    corr = np.full((3, labels.size), 0.5)
    gfp = np.ones(labels.size)
    p = microstate_parameters(labels, corr, gfp, sfreq=8.0, state=1)
    assert p["timecov"] == pytest.approx(8 / 16)
    assert p["meandurs"] == pytest.approx(np.mean([3, 5]) / 8.0)
    assert p["occurrences"] == pytest.approx(2 / 16 * 8.0)
    assert p["gev"] == pytest.approx(8 * 0.25 / 16)
    assert p["mean_corr"] == pytest.approx(0.5)


def test_parameters_exclude_unlabelled_from_time_but_not_gev():
    labels = np.array([-1, -1, 1, 1, 0, 0, -1, -1])
    corr, gfp = np.ones((2, 8)), np.ones(8)
    p = microstate_parameters(labels, corr, gfp, sfreq=1.0, state=1)
    assert p["timecov"] == pytest.approx(2 / 4)  # of the 4 labelled samples
    assert p["gev"] == pytest.approx(2 / 8)  # of all 8 samples


def test_parameters_of_an_absent_state_are_zero():
    labels = np.zeros(10, int)
    p = microstate_parameters(labels, np.ones((2, 10)), np.ones(10), SFREQ, state=1)
    assert p == dict.fromkeys(METRICS, 0.0)


def test_reject_edge_segments_drops_first_and_last():
    out = reject_edge_segments(np.array([1, 1, 2, 2, 3, 3, 3]))
    assert out.tolist() == [-1, -1, 2, 2, -1, -1, -1]


def test_reject_edge_segments_rejects_a_single_segment_window():
    """pycrostates' rule; the reason the modality defaults to reject_edges=False."""
    assert reject_edge_segments(np.array([2, 2, 2])).tolist() == [-1, -1, -1]


def test_resolve_templates_accepts_either_orientation():
    maps = _maps()
    a = resolve_templates(maps, CH)[0]
    b = resolve_templates(maps.T, CH)[0]
    np.testing.assert_array_equal(a, b)


def test_resolve_templates_reads_a_fitted_model():
    """A fitted pycrostates ModKMeans works by duck typing, without pycrostates."""
    maps = _maps()
    model = SimpleNamespace(
        cluster_centers_=maps, info={"ch_names": CH}, cluster_names=["A", "B", "C", "D"]
    )
    centers, ch_names, names = resolve_templates(model)
    np.testing.assert_array_equal(centers, maps)
    assert ch_names == CH and names == ["A", "B", "C", "D"]


@pytest.mark.parametrize(
    "templates, ch_names, state_names, match",
    [
        (None, CH, None, "needs template maps"),
        (np.ones((4, len(CH))), None, None, "template_ch_names"),
        (np.ones((4, 7)), CH, None, "does not match"),
        (np.ones((4, len(CH))), CH, ["A", "B"], "state_names"),
        (np.ones((4, len(CH))), CH[:-1] + CH[:1], None, "duplicates"),
    ],
)
def test_resolve_templates_rejects_bad_input(templates, ch_names, state_names, match):
    with pytest.raises(ValueError, match=match):
        resolve_templates(templates, ch_names, state_names)


def test_zscore_templates_rejects_a_flat_map():
    with pytest.raises(ValueError, match="constant"):
        zscore_templates(np.vstack([_maps(1), np.ones(len(CH))]))


def test_matches_pycrostates():
    """Labels and all five metrics agree with pycrostates, window by window."""
    pytest.importorskip("pycrostates")
    import mne
    from pycrostates.cluster import ModKMeans

    rng = np.random.default_rng(7)
    info = mne.create_info(CH, SFREQ, "eeg")  # backfitting needs no positions
    states = _states(int(30 * SFREQ), rng)
    raw = mne.io.RawArray(
        _oscillating(_maps(), states, rng, noise=0.1), info, verbose=False
    ).set_eeg_reference("average", verbose=False)
    model = ModKMeans(n_clusters=4, n_init=10, random_state=0)
    model.fit(raw, verbose=False)
    centers, _, names = resolve_templates(model)
    templates_z = zscore_templates(centers)

    for reject in (False, True):
        for start in range(0, raw.n_times - int(SFREQ), int(SFREQ)):
            window = raw.get_data()[:, start : start + int(SFREQ)]
            seg = model.predict(
                mne.io.RawArray(window, raw.info, verbose=False), factor=0, reject_edges=reject
            )
            labels, corr, gfp = backfit(window, templates_z)
            if reject:
                labels = reject_edge_segments(labels)
            np.testing.assert_array_equal(labels, seg.labels)
            ref = seg.compute_parameters()
            for s, name in enumerate(names):
                mine = microstate_parameters(labels, corr, gfp, SFREQ, s)
                for metric in METRICS:
                    assert mine[metric] == pytest.approx(ref[f"{name}_{metric}"], abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# The modality
# ─────────────────────────────────────────────────────────────────────────────


def _session(ch_names, data_type="eeg", bads=()):
    import mne

    from mne_rt.modalities import ModalityMixin

    info = mne.create_info(list(ch_names), SFREQ, "eeg")
    info["bads"] = list(bads)

    class _Session(ModalityMixin):
        pass

    session = _Session()
    session._sfreq = SFREQ
    session.data_type = data_type
    session.rec_info = info
    session.picks = None
    session.config_file = str(CONFIG)
    return session


def _prep(session, **overrides):
    from mne_rt.tools import get_params

    session.params = get_params(session.config_file, "microstates", {"microstates": overrides})
    return session._microstates_prep()


def _value(session, data, **overrides):
    value, delay = session._microstates(data, **_prep(session, **overrides))
    assert delay >= 0
    return value


def test_modality_tracks_the_target_state_coverage():
    rng = np.random.default_rng(4)
    maps = _maps()
    states = _states(int(SFREQ), rng)
    x = _oscillating(maps, states, rng)
    session = _session(CH)
    for target in range(4):
        value = _value(session, x, templates=maps, template_ch_names=CH, target_state=target)
        assert value == pytest.approx(np.mean(states == target), abs=0.08)


def test_modality_aligns_channels_by_name_and_drops_bad_ones():
    """The stream's channel order need not match the templates', and a channel
    marked bad is left out of the fit rather than shifting every position."""
    rng = np.random.default_rng(5)
    maps = _maps()
    states = _states(int(SFREQ), rng)
    x = _oscillating(maps, states, rng)
    reference = _value(_session(CH), x, templates=maps, template_ch_names=CH, target_state=2)

    order = rng.permutation(len(CH))
    stream_ch = [CH[i] for i in order]
    session = _session(stream_ch + ["EXTRA"], bads=[CH[0]])
    acquired = session._acquired_ch_names()  # stream order, without the bad channel
    stream_data = np.vstack(
        [x[CH.index(ch)] if ch in CH else rng.standard_normal(x.shape[1]) for ch in acquired]
    )
    with pytest.warns(UserWarning, match="not in the acquired data"):
        value = _value(session, stream_data, templates=maps, template_ch_names=CH, target_state=2)
    assert value == pytest.approx(reference, abs=0.05)


def test_modality_accepts_a_fitted_model_and_a_state_name():
    rng = np.random.default_rng(6)
    maps = _maps()
    states = _states(int(SFREQ), rng)
    x = _oscillating(maps, states, rng)
    model = SimpleNamespace(
        cluster_centers_=maps, info={"ch_names": CH}, cluster_names=list("ABCD")
    )
    by_name = _value(_session(CH), x, templates=model, target_state="C")
    by_index = _value(_session(CH), x, templates=maps, template_ch_names=CH, target_state=2)
    assert by_name == pytest.approx(by_index)


@pytest.mark.parametrize("metric", METRICS)
def test_modality_returns_every_metric(metric):
    rng = np.random.default_rng(8)
    maps = _maps()
    x = _oscillating(maps, _states(int(SFREQ), rng), rng)
    value = _value(_session(CH), x, templates=maps, template_ch_names=CH, metric=metric)
    assert isinstance(value, float) and np.isfinite(value) and value >= 0


def test_modality_without_band_pass():
    rng = np.random.default_rng(9)
    maps = _maps()
    states = _states(int(SFREQ), rng)
    x = maps[states].T * 1e-5 + 3e-5  # a DC offset per channel, removed when frange is None
    value = _value(
        _session(CH), x, templates=maps, template_ch_names=CH, frange=None, target_state=1
    )
    assert value == pytest.approx(np.mean(states == 1), abs=0.02)


def test_instances_can_train_different_states():
    """``microstates@A`` and ``microstates@C`` keep separate target states."""
    from mne_rt.tools import get_params

    rng = np.random.default_rng(10)
    maps = _maps()
    states = _states(int(SFREQ), rng)
    x = _oscillating(maps, states, rng)
    overrides = {
        "microstates": {"templates": maps, "template_ch_names": CH},
        "microstates@A": {"target_state": 0},
        "microstates@C": {"target_state": 2},
    }
    session = _session(CH)
    values = {}
    for name in ("microstates@A", "microstates@C"):
        session.params = get_params(session.config_file, "microstates", overrides, instance=name)
        values[name] = session._microstates(x, **session._microstates_prep())[0]
    assert values["microstates@A"] == pytest.approx(np.mean(states == 0), abs=0.08)
    assert values["microstates@C"] == pytest.approx(np.mean(states == 2), abs=0.08)


@pytest.mark.parametrize(
    "kwargs, overrides, match",
    [
        ({"data_type": "meg"}, {}, "EEG only"),
        ({}, {"templates": None}, "needs template maps"),
        ({}, {"target_state": "Z"}, "not one of the states"),
        ({}, {"target_state": 9}, "out of range"),
        ({}, {"metric": "duration"}, "metric must be one of"),
        ({"ch_names": ["X1", "X2", "X3"]}, {}, "at least 3"),
    ],
)
def test_modality_rejects_bad_configuration(kwargs, overrides, match):
    ch_names = kwargs.pop("ch_names", CH)
    session = _session(ch_names, **kwargs)
    params = {"templates": _maps(), "template_ch_names": CH} | overrides
    with pytest.raises(ValueError, match=match):
        _prep(session, **params)
