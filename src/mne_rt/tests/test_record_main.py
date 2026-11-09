"""The record_main loop: failures and their recovery, modalities and
correctors in the loop, combiners, instanced modalities and z-scoring.
"""

from __future__ import annotations

import json
import pathlib
import time
from pathlib import Path

import numpy as np
import pytest

from mne_rt.tests._helpers import (
    _HEADLESS,
    _CountingProtocol,
    _failing_after,
    _make_array_data,
    _make_rt_stream,
)

pytestmark = pytest.mark.usefixtures("fast_clock")


def test_connect_to_array_record_main_end_to_end(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1", bandpass_freq=(1.0, 40.0), notch_freq=50.0)
    data = _make_array_data(array_info, duration=20.0)
    try:
        nf.connect_to_array(data, array_info, n_repeat=np.inf)
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf.record_main(
            duration=2.0,
            modality="sensor_power",
            show_raw_signal=False,
            show_nf_signal=False,
        )
        assert len(nf.nf_data["sensor_power"]) > 0
    finally:
        nf.save()


def test_record_main_raises_when_acquisition_fails(tmp_path, array_info, monkeypatch):
    """#49: a failure in the acquisition thread was logged and swallowed, so
    record_main returned as if the run had succeeded."""
    import threading

    thread_hook = []
    monkeypatch.setattr(threading, "excepthook", thread_hook.append)

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf._sensor_power = _failing_after(2, RuntimeError("stream died"))
        with pytest.raises(RuntimeError, match="stream died"):
            nf.record_main(duration=5.0, modality="sensor_power", **_HEADLESS)
        # What was recorded before the failure is kept, and saved as for a
        # successful run ...
        assert len(nf.nf_data["sensor_power"]) == 2
        assert list(tmp_path.rglob("*_beh.json"))
        # ... and the traceback is not printed a second time by the thread.
        assert thread_hook == []
    finally:
        nf.save()


def test_failed_run_is_saved_with_whole_windows_only(tmp_path, array_info):
    """A window that fails part-way is rolled back: the second modality failing
    must not leave the first one, the onsets or the counters a window ahead."""
    import json

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf._band_ratio = _failing_after(3, RuntimeError("second modality died"))
        with pytest.raises(RuntimeError, match="second modality died"):
            nf.record_main(duration=5.0, modality=["sensor_power", "band_ratio"], **_HEADLESS)
        assert {m: len(v) for m, v in nf.nf_data.items()} == {"sensor_power": 3, "band_ratio": 3}
        assert len(nf.window_onsets) == len(nf.window_durations) == 3
        assert nf.n_total_windows == 3
        assert len(nf._raw_chunks) == 3  # the raw FIF ends on the same window
        (saved,) = tmp_path.rglob("*_beh.json")
        meta = json.loads(saved.read_text())["meta"]
        assert meta["n_windows"] == {"sensor_power": 3, "band_ratio": 3}
    finally:
        nf.save()


def test_a_failing_save_does_not_hide_the_acquisition_error(tmp_path, array_info, monkeypatch):
    """The acquisition error is raised, keeping its own cause; the save error
    is attached to it as a note."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        died = RuntimeError("stream died")
        died.__context__ = ValueError("root cause")
        nf._sensor_power = _failing_after(1, died)

        def broken_save(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(nf, "save", broken_save)
        with pytest.raises(RuntimeError, match="stream died") as info:
            nf.record_main(duration=5.0, modality="sensor_power", **_HEADLESS)
        # Neither error is lost, and the save error does not replace the cause.
        assert repr(info.value.__context__) == "ValueError('root cause')"
        assert any("disk full" in note for note in info.value.__notes__)
    finally:
        monkeypatch.undo()  # before saving, or a failure above is hidden by this one
        nf.save()


@pytest.mark.real_time
def test_record_main_raises_when_the_acquisition_thread_hangs(tmp_path, array_info, monkeypatch):
    """A modality that never returns used to hang record_main for good: the
    headless join had no timeout, and the executor shutdown then waited on the
    stuck task. Now the run is saved up to its last whole window, and what was
    saved does not change when the thread wakes up later."""
    import threading

    from mne_rt import rt_stream

    monkeypatch.setattr(rt_stream, "_ACQ_STOP_TIMEOUT", 1.0)
    release = threading.Event()
    calls = {"n": 0}

    def hangs_after_two(data, **_):
        calls["n"] += 1
        if calls["n"] > 2:
            release.wait(30)
        return 1.0, 0.0

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf._sensor_power = hangs_after_two
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="did not stop"):
            nf.record_main(duration=2.0, modality="sensor_power", **_HEADLESS)
        assert time.monotonic() - t0 < 15
        assert len(nf.nf_data["sensor_power"]) == 2
        assert len(nf.window_onsets) == len(nf._raw_chunks) == nf.n_total_windows == 2
        # The thread wakes up and finishes its window: nothing saved changes.
        release.set()
        time.sleep(1.0)
        assert len(nf.nf_data["sensor_power"]) == 2
        assert len(nf.window_onsets) == len(nf._raw_chunks) == 2
    finally:
        release.set()
        nf.save()


@pytest.mark.real_time
def test_ctrl_c_during_a_headless_run_stops_saves_and_reraises(tmp_path, array_info, monkeypatch):
    """Ctrl-C used to leave record_main with the thread still reading the
    stream and nothing saved."""
    import threading

    real_join = threading.Thread.join
    pressed = []

    def interrupted_join(self, timeout=None):
        # The first wait on the acquisition thread only: the one that ends the run.
        if getattr(self._target, "__name__", "") == "_acquire" and not pressed:
            pressed.append(True)
            time.sleep(1.0)  # a few windows in
            raise KeyboardInterrupt
        return real_join(self, timeout)

    monkeypatch.setattr(threading.Thread, "join", interrupted_join)
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        with pytest.raises(KeyboardInterrupt):
            nf.record_main(duration=30.0, modality="sensor_power", **_HEADLESS)
        n = len(nf.nf_data["sensor_power"])
        assert 0 < n == len(nf.window_onsets)
        assert list(tmp_path.rglob("*_beh.json"))
        time.sleep(1.0)  # the thread was stopped, so nothing is added later
        assert len(nf.nf_data["sensor_power"]) == n
    finally:
        monkeypatch.undo()
        nf.save()


def test_record_main_still_returns_normally_when_nothing_fails(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        assert nf.record_main(duration=2.0, modality="sensor_power", **_HEADLESS) is None
        assert len(nf.nf_data["sensor_power"]) > 0
    finally:
        nf.save()


def test_microstates_record_main_save_and_report(tmp_path, array_info):
    """The template array travels through modality_params; a session using it
    must record, save and report without trying to serialise it."""
    rng = np.random.default_rng(0)
    sfreq = array_info["sfreq"]
    n = int(20.0 * sfreq)
    maps = rng.standard_normal((3, len(array_info["ch_names"])))
    states = np.repeat(rng.integers(3, size=n // 25 + 1), 25)[:n]  # ~100 ms segments
    data = maps[states].T * np.sin(2 * np.pi * 10 * np.arange(n) / sfreq) * 1e-5
    data += rng.standard_normal(data.shape) * 1e-7

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(data, array_info, n_repeat=np.inf)
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf.record_main(
            duration=2.0,
            modality="microstates",
            modality_params={
                "microstates": {
                    "templates": maps,
                    "template_ch_names": array_info["ch_names"],
                    "target_state": 1,
                }
            },
            show_raw_signal=False,
            show_nf_signal=False,
        )
        values = np.asarray(nf.nf_data["microstates"])
        assert values.size > 0 and np.all((values >= 0) & (values <= 1))
    finally:
        saved = nf.save()
    assert Path(saved["nf_data"]).is_file()
    assert Path(nf.create_report(open_browser=False)).is_file()


@pytest.mark.real_time
def test_microstates_with_individual_maps_fitted_on_the_baseline(tmp_path, array_info):
    """#44's second approach, with no extra machinery: fit pycrostates on the
    session's own baseline, choose the state to train, pass the fitted model."""
    pytest.importorskip("pycrostates")
    from pycrostates.cluster import ModKMeans

    rng = np.random.default_rng(1)
    sfreq = array_info["sfreq"]
    n = int(30.0 * sfreq)
    maps = rng.standard_normal((3, len(array_info["ch_names"])))
    # state 1 — the one trained — dominates, so training the wrong map would
    # read ~0.2 instead of ~0.6
    states = np.repeat(rng.choice(3, size=n // 25 + 1, p=[0.2, 0.6, 0.2]), 25)[:n]
    data = maps[states].T * np.sin(2 * np.pi * 10 * np.arange(n) / sfreq) * 1e-5
    data += rng.standard_normal(data.shape) * 1e-7

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(data, array_info, n_repeat=np.inf)
        nf.record_baseline(baseline_duration=4.0, winsize=0.5)

        model = ModKMeans(n_clusters=3, n_init=10, random_state=0)
        # the documented recipe: prepare the baseline as each window is prepared
        baseline = (
            nf.raw_baseline.copy()
            .filter(2, 20, verbose=False)
            .set_eeg_reference("average", verbose=False)
        )
        model.fit(baseline, verbose=False)
        # The manual step: pick the fitted map to train — here, the one matching
        # true map 1 — and name it, as one would after model.plot().
        corr = [abs(np.corrcoef(c, maps[1])[0, 1]) for c in model.cluster_centers_]
        names = ["other_a", "other_b"]
        names.insert(int(np.argmax(corr)), "target")
        model.rename_clusters(new_names=names)

        nf.record_main(
            duration=3.0,
            modality="microstates",
            modality_params={"microstates": {"templates": model, "target_state": "target"}},
            show_raw_signal=False,
            show_nf_signal=False,
        )
        values = np.asarray(nf.nf_data["microstates"])
        assert values.size > 0
        assert 0.45 < values.mean() < 0.8  # state 1's share, not another map's ~0.2
    finally:
        nf.save()


# Real time: GEDAI's per-window compute, 10x longer in session time, outran the
# hop on CI runners and tripped its missing-samples check.
@pytest.mark.real_time
def test_causal_gedai_record_main_uses_sensai_and_delayed_windows(tmp_path, array_info):
    pytest.importorskip("gedai")
    from unittest.mock import patch

    from mne_rt.tools.adaptive_gedai import AdaptiveMultibandGEDAIDenoiser

    nf = _make_rt_stream(
        tmp_path,
        montage="easycap-M1",
        artifact_correction="gedai",
        bandpass_freq=(1.0, 40.0),
        notch_freq=50.0,
    )
    data = _make_array_data(array_info, duration=20.0)
    try:
        nf.connect_to_array(data, array_info, n_repeat=np.inf)
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf.gedai = AdaptiveMultibandGEDAIDenoiser(
            data.shape[0],
            array_info["sfreq"],
            broadband_pass=False,
            prefilter_mode="none",
            wavelet_low_cutoff=1,
        ).fit(data[:, :1024], leadfield=np.eye(data.shape[0]))
        nf.blink_template = np.arange(data.shape[0], dtype=float)
        with patch.object(
            nf.gedai, "update_and_denoise", wraps=nf.gedai.update_and_denoise
        ) as update:
            nf.record_main(
                duration=3.0,
                winsize=0.5,
                modality="sensor_power",
                show_raw_signal=False,
                show_nf_signal=False,
            )
        assert len(nf.nf_data["sensor_power"]) > 0
        assert update.call_count > len(nf.nf_data["sensor_power"])
        assert all(len(call.args) == 1 and not call.kwargs for call in update.call_args_list)
        assert max(call.args[0].shape[1] for call in update.call_args_list) <= round(
            0.05 * array_info["sfreq"]
        )
        assert len(nf.window_onsets) == len(nf.nf_data["sensor_power"])
        assert np.all(np.diff(nf.window_onsets) > 0)
    finally:
        nf.save()


def _make_fitted_decoder(info, n_epochs=20):
    from mne_rt import RTDecode

    rng = np.random.default_rng(0)
    n_ch = len(info["ch_names"])
    X = rng.standard_normal((n_epochs, n_ch, int(info["sfreq"])))
    y = np.array([0, 1] * (n_epochs // 2))
    return RTDecode(info=info, n_components=2).fit(X, y, verbose=False)


def test_set_decoder_rejects_wrong_type(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(TypeError, match="RTDecode"):
        nf.set_decoder(object())


def test_set_decoder_rejects_unfitted_decoder(tmp_path, array_info):
    from mne_rt import RTDecode

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(RuntimeError, match="fit"):
        nf.set_decoder(RTDecode(info=array_info))


def test_decode_modality_without_decoder_raises(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    try:
        nf.connect_to_array(data, array_info, n_repeat=np.inf)
        with pytest.raises(RuntimeError, match="set_decoder"):
            nf.record_main(duration=1.0, modality="decode", show_nf_signal=False)
    finally:
        nf.save()


def test_connect_to_array_decode_modality_end_to_end(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    decoder = _make_fitted_decoder(array_info)
    try:
        nf.connect_to_array(data, array_info, n_repeat=np.inf)
        nf.set_decoder(decoder)
        nf.record_main(duration=2.0, modality="decode", show_nf_signal=False)
        assert len(nf.nf_data["decode"]) > 0
        assert all(0.0 <= v <= 1.0 for v in nf.nf_data["decode"])
    finally:
        nf.save()


def _run_with_combiner(tmp_path, array_info, combiner, **kwargs):
    """Run a short two-modality session and return the session object."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=2.0,
            winsize=1.0,
            modality=["sensor_power", "hjorth"],
            combiner=combiner,
            zscore_normalize=True,
            zscore_warmup=2,
            show_raw_signal=False,
            show_nf_signal=False,
            **kwargs,
        )
    finally:
        nf.save()
    return nf


def test_combiner_adds_a_combined_trace(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner

    combiner = WeightedSumCombiner(weights={"sensor_power": 0.6, "hjorth": 0.4})
    nf = _run_with_combiner(tmp_path, array_info, combiner)

    assert "combined" in nf.nf_data
    assert len(nf.nf_data["combined"]) > 0
    # the per-modality traces are kept, not replaced
    assert len(nf.nf_data["sensor_power"]) == len(nf.nf_data["combined"])
    assert len(nf.nf_data["hjorth"]) == len(nf.nf_data["combined"])


def test_combined_value_is_the_combiner_output(tmp_path, array_info):
    """Once z-scoring is warmed up, the combined trace is the combiner's output."""
    from mne_rt import WeightedSumCombiner

    combiner = WeightedSumCombiner(weights={"sensor_power": 0.6, "hjorth": 0.4})
    nf = _run_with_combiner(tmp_path, array_info, combiner)

    warm = 2 - 1  # zscore_warmup=2 in the helper; normalised from that index on
    power = np.asarray(nf.nf_data["sensor_power"])[warm:]
    hjorth = np.asarray(nf.nf_data["hjorth"])[warm:]
    combined = np.asarray(nf.nf_data["combined"])[warm:]
    np.testing.assert_allclose(combined, 0.6 * power + 0.4 * hjorth, rtol=1e-9)


def test_combined_held_at_zero_until_zscoring_warms_up(tmp_path, array_info):
    """Guards against a scale discontinuity at the end of warmup.

    _apply_zscore passes values through in their native units until each
    modality has `zscore_warmup` windows behind it. Combining then would mix
    V²/Hz with a dimensionless index, and the combined trace would jump by
    orders of magnitude the moment normalisation engaged — fitting any attached
    protocol's baseline on meaningless numbers.
    """
    from mne_rt import WeightedSumCombiner

    combiner = WeightedSumCombiner(weights={"sensor_power": 0.6, "hjorth": 0.4})
    nf = _run_with_combiner(tmp_path, array_info, combiner)
    assert nf.nf_data["combined"][0] == 0.0
    assert any(v != 0.0 for v in nf.nf_data["combined"][1:])


def test_geometric_mean_with_zscoring_warns(tmp_path, array_info):
    """z-scores go negative; GeometricMeanCombiner logs them and collapses."""
    from mne_rt import GeometricMeanCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=10.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        with pytest.warns(RuntimeWarning, match="positive inputs"):
            nf.record_main(
                duration=1.0,
                winsize=1.0,
                modality=["sensor_power", "hjorth"],
                combiner=GeometricMeanCombiner(features=["sensor_power", "hjorth"]),
                zscore_normalize=True,
                zscore_warmup=2,
                show_raw_signal=False,
                show_nf_signal=False,
            )
    finally:
        nf.save()


def test_zscored_norm_combiner_does_not_warn(tmp_path, array_info, recwarn):
    """It normalises internally, so advising zscore_normalize would be wrong."""
    from mne_rt import ZScoredNormCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=10.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=1.0,
            winsize=1.0,
            modality=["sensor_power", "hjorth"],
            combiner=ZScoredNormCombiner(features=["sensor_power", "hjorth"], warmup=2),
            zscore_normalize=False,
            show_raw_signal=False,
            show_nf_signal=False,
        )
    finally:
        nf.save()
    assert not [w for w in recwarn if "zscore_normalize" in str(w.message)]


def test_a_failing_combiner_does_not_hang_the_session(tmp_path, array_info):
    """combine() is user code; raising must not skip the loop's shutdown."""

    class _Exploding:
        features = ["sensor_power"]

        def combine(self, values):
            raise RuntimeError("boom")

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=10.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=1.0,
            winsize=1.0,
            modality=["sensor_power"],
            combiner=_Exploding(),
            show_raw_signal=False,
            show_nf_signal=False,
        )
    finally:
        nf.save()
    # the session completed, and the failed windows fell back to 0.0
    assert len(nf.nf_data["combined"]) == len(nf.nf_data["sensor_power"])
    assert all(v == 0.0 for v in nf.nf_data["combined"])


@pytest.mark.parametrize("name", ["snr_db", "entropy", "reward_x"])
def test_reserved_combined_names_rejected(tmp_path, array_info, name):
    """The combined trace shares namespaces with scales, labels and saved columns."""
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=5.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="already in use"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power"],
            combiner=WeightedSumCombiner(weights={"sensor_power": 1.0}),
            combined_name=name,
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_combined_trace_can_drive_a_protocol(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner, ZScoreProtocol

    combiner = WeightedSumCombiner(weights={"sensor_power": 1.0, "hjorth": 1.0})
    nf = _run_with_combiner(
        tmp_path,
        array_info,
        combiner,
        protocol={"combined": ZScoreProtocol(warmup_windows=1)},
    )
    assert "combined" in nf.reward_data
    assert len(nf.reward_data["combined"]) == len(nf.nf_data["combined"])


def test_combined_trace_is_saved(tmp_path, array_info):
    import json

    from mne_rt import WeightedSumCombiner

    combiner = WeightedSumCombiner(weights={"sensor_power": 1.0, "hjorth": 1.0})
    nf = _run_with_combiner(tmp_path, array_info, combiner)
    saved = nf.save()
    payload = json.loads(saved["nf_data"].read_text())
    assert "combined" in payload["data"]


def test_combiner_must_expose_combine(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=5.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(TypeError, match="combine"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power"],
            combiner=object(),
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_combined_name_collision_rejected(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=5.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="collides"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power"],
            combiner=WeightedSumCombiner(weights={"sensor_power": 1.0}),
            combined_name="sensor_power",
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_combiner_requiring_an_inactive_modality_is_rejected(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=5.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="laterality"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power"],
            combiner=WeightedSumCombiner(weights={"sensor_power": 1.0, "laterality": 1.0}),
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_combiner_without_zscore_warns_about_scale(tmp_path, array_info):
    """Mixing raw features of different magnitudes needs normalisation."""
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=10.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        with pytest.warns(RuntimeWarning, match="zscore_normalize"):
            nf.record_main(
                duration=1.0,
                winsize=1.0,
                modality=["sensor_power", "hjorth"],
                combiner=WeightedSumCombiner(weights={"sensor_power": 1.0, "hjorth": 1.0}),
                zscore_normalize=False,
                show_raw_signal=False,
                show_nf_signal=False,
            )
    finally:
        nf.save()


def _run_instanced(tmp_path, array_info, modality, **kwargs):
    """Run a short session and return the session object."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=2.0,
            winsize=1.0,
            modality=modality,
            show_raw_signal=False,
            show_nf_signal=False,
            **kwargs,
        )
    finally:
        nf.save()
    return nf


def test_two_instances_of_one_modality_are_independent(tmp_path, array_info):
    """The load-bearing test: two bands of one measure, two distinct traces.

    If ``precomps`` or ``mod_params_dict`` collapsed on the shared base name,
    both instances would compute the same band and the traces would be equal.
    """
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        modality_params={
            "sensor_power@alpha": {"frange": [8, 13]},
            "sensor_power@theta": {"frange": [4, 8]},
        },
    )

    assert set(nf.nf_data) == {"sensor_power@alpha", "sensor_power@theta"}
    alpha = nf.nf_data["sensor_power@alpha"]
    theta = nf.nf_data["sensor_power@theta"]
    assert len(alpha) == len(theta) > 0
    assert alpha != theta

    assert nf.mod_params_dict["sensor_power@alpha"]["frange"] == [8, 13]
    assert nf.mod_params_dict["sensor_power@theta"]["frange"] == [4, 8]


def test_base_params_are_shared_across_instances(tmp_path, array_info):
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        modality_params={
            "sensor_power": {"method": "multitaper"},
            "sensor_power@alpha": {"frange": [8, 13]},
            "sensor_power@theta": {"frange": [4, 8]},
        },
    )
    for name in ("sensor_power@alpha", "sensor_power@theta"):
        assert nf.mod_params_dict[name]["method"] == "multitaper"


def test_plain_and_instanced_names_coexist(tmp_path, array_info):
    nf = _run_instanced(tmp_path, array_info, ["sensor_power", "sensor_power@theta"])
    assert set(nf.nf_data) == {"sensor_power", "sensor_power@theta"}


def test_duplicate_instance_names_rejected(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="listed more than once"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power@a", "sensor_power@a"],
            show_raw_signal=False,
            show_nf_signal=False,
        )


@pytest.mark.parametrize(
    "name, match",
    [
        ("sensor_power@a@b", "more than one"),
        ("@alpha", "empty base"),
        ("sensor_power@", "empty instance label"),
        ("sensor_power@bad label", "may only contain"),
    ],
)
def test_malformed_instance_names_rejected(tmp_path, array_info, name, match):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match=match):
        nf.record_main(duration=1.0, modality=[name], show_raw_signal=False, show_nf_signal=False)


def test_picks_rejected_for_instanced_source_modality(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="'picks' must be None"):
        nf.record_main(
            duration=1.0,
            modality=["source_connectivity@theta"],
            picks=array_info["ch_names"][:4],
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_picks_allowed_when_only_the_label_says_source(tmp_path, array_info):
    """The sensor/source test must read the base, not the whole name."""
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@source"],
        picks=array_info["ch_names"][:4],
    )
    assert "sensor_power@source" in nf.nf_data


def test_each_instance_drives_its_own_protocol(tmp_path, array_info):
    """One protocol object per instance, each evaluated exactly once a window.

    A shared object would be fired twice per window with two different values,
    silently corrupting any protocol that accumulates state.
    """
    p_alpha, p_theta = _CountingProtocol(), _CountingProtocol()
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        modality_params={
            "sensor_power@alpha": {"frange": [8, 13]},
            "sensor_power@theta": {"frange": [4, 8]},
        },
        protocol={"sensor_power@alpha": p_alpha, "sensor_power@theta": p_theta},
    )

    n_windows = len(nf.nf_data["sensor_power@alpha"])
    assert n_windows > 0
    assert len(p_alpha.values) == len(p_theta.values) == n_windows
    assert p_alpha.values != p_theta.values
    assert len(nf.reward_data["sensor_power@alpha"]) == n_windows


def test_protocol_keyed_by_instanced_base_rejected(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="active instance"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power@alpha", "sensor_power@theta"],
            protocol={"sensor_power": _CountingProtocol()},
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_protocol_validation_happens_before_resources_are_created(tmp_path, array_info):
    """record_main has no teardown path, so it must reject before it allocates.

    Raising after the thread pool and the Qt windows exist would leave both
    orphaned.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="active instance"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power@alpha", "sensor_power@theta"],
            protocol={"sensor_power": _CountingProtocol()},
            show_raw_signal=False,
            show_nf_signal=False,
        )
    assert getattr(nf, "executor", None) is None


def test_protocol_keyed_by_unknown_modality_rejected(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="does not name an active modality"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power@alpha"],
            protocol={"hjorth": _CountingProtocol()},
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_combiner_over_two_instances(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner

    combiner = WeightedSumCombiner(weights={"sensor_power@alpha": 0.5, "sensor_power@theta": 0.5})
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        modality_params={
            "sensor_power@alpha": {"frange": [8, 13]},
            "sensor_power@theta": {"frange": [4, 8]},
        },
        combiner=combiner,
        zscore_normalize=True,
        zscore_warmup=2,
    )
    assert "combined" in nf.nf_data
    assert len(nf.nf_data["combined"]) == len(nf.nf_data["sensor_power@alpha"])


def test_combiner_naming_an_instanced_base_suggests_the_instances(tmp_path, array_info):
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    with pytest.raises(ValueError, match="Did you mean"):
        nf.record_main(
            duration=1.0,
            modality=["sensor_power@alpha", "sensor_power@theta"],
            combiner=WeightedSumCombiner(weights={"sensor_power": 1.0}),
            show_raw_signal=False,
            show_nf_signal=False,
        )


def test_instanced_names_round_trip_through_save(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=2.0,
            winsize=1.0,
            modality=["sensor_power@alpha", "sensor_power@theta"],
            protocol={"sensor_power@alpha": _CountingProtocol(threshold=-1e30)},
            show_raw_signal=False,
            show_nf_signal=False,
        )
    finally:
        nf.save(bids_tsv=True)

    beh = next(pathlib.Path(nf.subject_dir).rglob("*_beh.json"))
    payload = json.loads(beh.read_text())
    assert set(payload["meta"]["modalities"]) == {"sensor_power@alpha", "sensor_power@theta"}
    assert payload["meta"]["modality_bases"] == {
        "sensor_power@alpha": "sensor_power",
        "sensor_power@theta": "sensor_power",
    }
    assert "sensor_power@alpha" in payload["data"]
    assert "reward_sensor_power@alpha" in payload["data"]

    tsv = next(pathlib.Path(nf.subject_dir).rglob("*_beh.tsv"))
    header = tsv.read_text().splitlines()[0].split("\t")
    assert "sensor_power@alpha" in header and "sensor_power@theta" in header


def test_osc_addresses_are_sanitised_and_distinct(tmp_path, array_info):
    """'@' must not reach an OSC address, and the two must stay distinct."""

    class _FakeOSC:
        def __init__(self):
            self.sent = []

        def send_all(self, modalities, values):
            self.sent.append(list(modalities))

    sender = _FakeOSC()
    _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        osc_sender=sender,
    )
    assert sender.sent
    for names in sender.sent:
        assert names == ["sensor_power_alpha", "sensor_power_theta"]
        assert not any("@" in n for n in names)


def test_per_instance_delays_are_recorded(tmp_path, array_info):
    nf = _run_instanced(
        tmp_path,
        array_info,
        ["sensor_power@alpha", "sensor_power@theta"],
        estimate_delays=True,
    )
    assert set(nf.method_delays) == {"sensor_power@alpha", "sensor_power@theta"}
    assert all(v for v in nf.method_delays.values())


def _zscored_session(tmp_path, array_info, **kwargs):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=40.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=12.0,
            winsize=1.0,
            modality="sensor_power",
            zscore_normalize=True,
            show_raw_signal=False,
            show_nf_signal=False,
            **kwargs,
        )
    finally:
        nf.save()
    return nf


def test_zscored_trace_is_on_the_order_of_one(tmp_path, array_info):
    """Band power is ~1e-13, so an absolute std floor swamped it entirely.

    Before this was fixed the "z-scores" came out ~1e-8. The loose bound is
    deliberate — the point is the order of magnitude, not the exact spread.
    """
    nf = _zscored_session(tmp_path, array_info, zscore_warmup=4)
    z = np.asarray(nf.nf_data["sensor_power"][4:])
    assert len(z) > 2
    assert np.max(np.abs(z)) > 0.1  # would be ~1e-8 under the old floor
    assert np.max(np.abs(z)) < 100.0


def test_zscore_state_is_kept_after_the_warmup_buffer_is_dropped(tmp_path, array_info):
    """The combiner warm-gate reads a counter, not the (now freed) buffer."""
    from mne_rt import WeightedSumCombiner

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=40.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=12.0,
            winsize=1.0,
            modality=["sensor_power", "hjorth"],
            combiner=WeightedSumCombiner(weights={"sensor_power": 0.5, "hjorth": 0.5}),
            zscore_normalize=True,
            zscore_warmup=3,
            show_raw_signal=False,
            show_nf_signal=False,
        )
    finally:
        nf.save()
    combined = nf.nf_data["combined"]
    # Held at 0.0 until every feature is normalised, then live.
    assert combined[0] == 0.0
    assert any(v != 0.0 for v in combined[3:])


def test_adaptive_zscore_tracks_variance_not_absolute_deviation(tmp_path, array_info):
    """With zscore_alpha > 0 the running spread must stay a variance.

    Exponentially averaging ``abs(delta)`` instead gives the mean absolute
    deviation (~0.798 sigma for Gaussian input), inflating z by ~25%.
    """
    nf = _zscored_session(tmp_path, array_info, zscore_warmup=4, zscore_alpha=0.05)
    z = np.asarray(nf.nf_data["sensor_power"][4:])
    assert np.all(np.isfinite(z))
    assert np.max(np.abs(z)) > 0.1


def test_display_scales_follow_the_zscored_units(tmp_path, array_info):
    """A z-score is O(1); dividing it by a 1e-12 band-power scale hides it."""
    import mne_rt.rt_stream as rs

    captured = {}
    real_init = rs.NFPlot.__init__

    def _spy(self, modalities, scales_dict, *a, **kw):
        captured["scales"] = dict(scales_dict)
        return real_init(self, modalities, scales_dict, *a, **kw)

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=10.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(rs.NFPlot, "__init__", _spy)
    try:
        nf.record_main(
            duration=2.0,
            winsize=1.0,
            modality="sensor_power",
            zscore_normalize=True,
            zscore_warmup=2,
            show_raw_signal=False,
            show_nf_signal=True,
        )
    finally:
        monkey.undo()
        nf.save()
    assert captured["scales"]["sensor_power"] == 1.0
