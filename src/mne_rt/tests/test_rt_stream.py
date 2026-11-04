"""RTStream construction and validation, corrector fitting, source
configuration, bad channels, the head model, and the brain display's inverse.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import numpy as np
import pytest

from mne_rt.tests._helpers import _make_array_data, _make_baseline_raw, _make_rt_stream


def test_valid_instantiation(subjects_dir):
    from mne_rt import RTStream as RTStream

    nf = RTStream(
        subject_id="sub01",
        session="01",
        subjects_dir=subjects_dir,
        montage="easycap-M1",
        data_type="eeg",
    )
    assert nf.subject_id == "sub01"
    assert nf.session == "01"


def test_invalid_subject_id(subjects_dir):
    from mne_rt import RTStream as RTStream

    with pytest.raises(ValueError, match="subject_id"):
        RTStream(
            subject_id="",
            session="01",
            subjects_dir=subjects_dir,
            montage="easycap-M1",
        )


def test_invalid_session(subjects_dir):
    from mne_rt import RTStream as RTStream

    with pytest.raises(ValueError, match="session"):
        RTStream(
            subject_id="sub01",
            session="",
            subjects_dir=subjects_dir,
            montage="easycap-M1",
        )


def test_invalid_data_type(subjects_dir):
    from mne_rt import RTStream as RTStream

    with pytest.raises(ValueError, match="data_type"):
        RTStream(
            subject_id="sub01",
            session="01",
            subjects_dir=subjects_dir,
            montage="easycap-M1",
            data_type="fnirs",
        )


def test_invalid_artifact_correction(subjects_dir):
    from mne_rt import RTStream as RTStream

    with pytest.raises(ValueError, match="artifact_correction"):
        RTStream(
            subject_id="sub01",
            session="01",
            subjects_dir=subjects_dir,
            montage="easycap-M1",
            artifact_correction="invalid_method",
        )


def test_subject_dir_layout(tmp_path):

    from mne_rt import RTStream as RTStream

    nf = RTStream(
        subject_id="newsub",
        session="pre",
        subjects_dir=str(tmp_path),
        montage="easycap-M1",
    )
    assert Path(nf.subjects_dir) == tmp_path
    assert nf.subject_dir == tmp_path / "sub-newsub" / "ses-pre"


def test_modality_params_setter(subjects_dir):
    from mne_rt import RTStream as RTStream

    nf = RTStream(
        subject_id="sub01",
        session="01",
        subjects_dir=subjects_dir,
        montage="easycap-M1",
    )
    nf.modality_params = {"sensor_power": {"frange": [8, 12]}}
    assert nf.modality_params["sensor_power"]["frange"] == [8, 12]


def test_modality_params_invalid(subjects_dir):
    from mne_rt import RTStream as RTStream

    nf = RTStream(
        subject_id="sub01",
        session="01",
        subjects_dir=subjects_dir,
        montage="easycap-M1",
    )
    with pytest.raises(ValueError):
        nf.modality_params = "not_a_dict"


class TestNFRealtimeNewFeatures:
    def test_replay_method_exists(self):
        from mne_rt import RTStream

        assert hasattr(RTStream, "replay")

    def test_run_blocks_method_exists(self):
        from mne_rt import RTStream

        assert hasattr(RTStream, "run_blocks")

    def test_run_blocks_empty_raises(self, tmp_path):
        from mne_rt import RTStream

        nf = RTStream(
            subject_id="sub01",
            session="01",
            subjects_dir=str(tmp_path),
            montage="easycap-M1",
        )
        with pytest.raises(ValueError, match="blocks"):
            nf.run_blocks(blocks=[])


def test_fit_asr_before_baseline_raises(tmp_path):
    nf = _make_rt_stream(tmp_path)
    with pytest.raises(RuntimeError, match="record_baseline"):
        nf.fit_asr()


def test_fit_asr_happy_path(tmp_path):
    from mne_rt.tools import ASRDenoiser

    nf = _make_rt_stream(tmp_path)
    nf.raw_baseline = _make_baseline_raw()
    nf._sfreq = 256.0
    nf.fit_asr(cutoff=3.0)
    assert isinstance(nf.asr, ASRDenoiser)
    assert nf.asr.thresholds.shape == (8,)


def test_fit_gedai_before_baseline_raises(tmp_path):
    nf = _make_rt_stream(tmp_path)
    with pytest.raises(RuntimeError, match="record_baseline"):
        nf.fit_gedai()


def test_fit_gedai_band_filter_mode(tmp_path):
    from mne_rt.tools import GEDAIDenoiser

    nf = _make_rt_stream(tmp_path)
    nf.raw_baseline = _make_baseline_raw()
    nf.rec_info = nf.raw_baseline.info
    nf._sfreq = 256.0
    nf.fit_gedai(band=(8.0, 13.0), use_leadfield=False)
    assert isinstance(nf.gedai, GEDAIDenoiser)


@pytest.mark.parametrize("multiband", [True, "causal"])
@pytest.mark.parametrize("cycles", [None, 4])
def test_fit_gedai_multiband_options(tmp_path, monkeypatch, multiband, cycles):
    from unittest.mock import Mock

    import mne_rt.rt_stream as module

    model = Mock(n_bands=8, latency=12.78)
    constructor = Mock(return_value=model)
    monkeypatch.setattr(module, "AdaptiveMultibandGEDAIDenoiser", constructor)
    nf = _make_rt_stream(tmp_path)
    nf.raw_baseline = _make_baseline_raw()
    nf.rec_info = nf.raw_baseline.info
    nf._sfreq = 256.0
    nf.fit_gedai(
        use_leadfield=False,
        multiband=multiband,
        cycles_per_wavelet=cycles,
        causal_wavelet_low_cutoff=1.0,
        causal_prefilter_mode="none",
        causal_broadband_pass=False,
    )
    expected_cycles = cycles if cycles is not None else 6
    assert constructor.call_args.kwargs["cycles_per_wavelet"] == expected_cycles
    assert constructor.call_args.kwargs["wavelet_low_cutoff"] == 1.0
    assert nf.gedai is model
    assert constructor.call_args.kwargs["prefilter_mode"] == "none"
    assert constructor.call_args.kwargs["broadband_pass"] is False
    model.fit.assert_called_once()


def test_fit_gedai_multiband_refuses_meg(tmp_path):
    """Multiband SENSAI only implements gedai's EEG objective; refuse MEG rather
    than calibrate it with the wrong model."""
    nf = _make_rt_stream(tmp_path)
    nf.raw_baseline = _make_baseline_raw()
    nf.data_type = "meg"
    with pytest.raises(ValueError, match="EEG only"):
        nf.fit_gedai(use_leadfield=False, multiband="causal")


@pytest.mark.parametrize("multiband", ["windowed", "invalid", 1, None])
def test_fit_gedai_rejects_unsupported_multiband_mode(tmp_path, multiband):
    nf = _make_rt_stream(tmp_path)
    nf.raw_baseline = _make_baseline_raw()
    with pytest.raises(ValueError, match="multiband"):
        nf.fit_gedai(use_leadfield=False, multiband=multiband)


def test_fit_gedai_preserves_positional_verbose(tmp_path, monkeypatch):
    from unittest.mock import Mock

    import mne_rt.rt_stream as module

    model = Mock()
    constructor = Mock(return_value=model)
    monkeypatch.setattr(module, "GEDAIDenoiser", constructor)
    nf = _make_rt_stream(tmp_path)
    nf.raw_baseline = _make_baseline_raw()
    nf.rec_info = nf.raw_baseline.info
    nf._sfreq = 256.0
    nf.fit_gedai((8.0, 13.0), 0.01, False, "ERROR")
    assert nf.gedai is model
    model.fit_from_raw.assert_called_once()


def test_fit_maxwell_before_connect_raises(tmp_path):
    nf = _make_rt_stream(tmp_path, data_type="meg")
    with pytest.raises(RuntimeError, match="connect_to_lsl"):
        nf.fit_maxwell()


def test_fit_maxwell_happy_path(tmp_path, meg_info):
    nf = _make_rt_stream(tmp_path, data_type="meg", montage=None)
    nf.rec_info = meg_info
    nf.fit_maxwell()
    assert nf.maxwell_filter.mode == "sss"


def test_run_orica_sets_instance(tmp_path):
    from mne_rt.tools import ORICA

    nf = _make_rt_stream(tmp_path)
    nf.run_orica(n_channels=8, block_size=64)
    assert isinstance(nf.orica, ORICA)


def test_source_space_defaults_to_surface(tmp_path):
    nf = _make_rt_stream(tmp_path)
    assert nf.source_space == "surface"
    assert nf.source_atlas == "aparc"  # a surface annotation
    assert nf.src is None
    assert nf.data_cov is None


def test_volume_source_space_defaults_to_volumetric_atlas(tmp_path):
    """'aparc' is an annot with no .mgz, so a volume session must not use it."""
    nf = _make_rt_stream(tmp_path, source_space="volume")
    assert nf.source_atlas == "aparc+aseg"


def test_source_atlas_explicit_is_respected(tmp_path):
    nf = _make_rt_stream(tmp_path, source_space="volume", source_atlas="aseg")
    assert nf.source_atlas == "aseg"


def test_invalid_source_space_raises(tmp_path):
    with pytest.raises(ValueError, match="source_space"):
        _make_rt_stream(tmp_path, source_space="mixed")


@pytest.mark.parametrize("pos", [0, -1, "5"])
def test_invalid_source_pos_raises(tmp_path, pos):
    with pytest.raises(ValueError, match="source_pos"):
        _make_rt_stream(tmp_path, source_pos=pos)


def test_source_pos_stored(tmp_path):
    assert _make_rt_stream(tmp_path, source_pos=3.0).source_pos == 3.0


def _bads_info(n_channels=16):
    """Info with digitised positions, no bads yet."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    names = [c for c in montage.ch_names if c not in ("T3", "T4", "T5", "T6")][:n_channels]
    info = mne.create_info(names, 250.0, "eeg")
    info.set_montage(montage)
    return info, names


def _connected(tmp_path, info, names, duration=24.0):
    nf = _make_rt_stream(tmp_path, montage=None)
    data = np.random.default_rng(0).standard_normal((len(names), int(250.0 * duration))) * 1e-6
    nf.connect_to_array(data, info, n_repeat=np.inf)
    return nf


def test_acquired_ch_names_excludes_bads(tmp_path):
    """get_data() drops bads; rec_info keeps them. The two must not be confused."""
    info, names = _bads_info()
    nf = _connected(tmp_path, info, names)
    try:
        nf.rec_info["bads"] = [names[2], names[7]]
        acquired = nf._acquired_ch_names()
        assert names[2] not in acquired and names[7] not in acquired
        assert len(acquired) == len(names) - 2
        # rec_info still lists them, which the forward model and Maxwell need
        assert names[2] in nf.rec_info["ch_names"]
        window = nf.stream.get_data(1.0)[0]
        assert window.shape[0] == len(acquired)
    finally:
        nf.save()


def test_record_baseline_with_a_bad_channel_marked_after_connect(tmp_path):
    """Regression: this raised before a session could start.

    Bads are normally identified *after* connecting — during the baseline, or
    by BadChannelDetector — so excluding them at connect time would not help.
    """
    info, names = _bads_info()
    nf = _connected(tmp_path, info, names)
    try:
        nf.rec_info["bads"] = [names[2]]
        nf.record_baseline(baseline_duration=2.0, winsize=1.0)
        assert len(nf.raw_baseline.ch_names) == len(names) - 1
    finally:
        nf.save()


def test_record_main_with_a_bad_channel(tmp_path):
    info, names = _bads_info()
    nf = _connected(tmp_path, info, names)
    try:
        nf.rec_info["bads"] = [names[2]]
        nf.record_main(
            duration=2.0,
            winsize=1.0,
            modality=["sensor_power"],
            show_raw_signal=False,
            show_nf_signal=False,
        )
        assert len(nf.nf_data["sensor_power"]) > 0
    finally:
        nf.save()


def test_connectivity_indices_follow_the_acquired_channels(tmp_path):
    """A bad channel shifts every position in the acquired array.

    Indices built from rec_info would silently select the wrong pair.
    """
    info, names = _bads_info()
    nf = _connected(tmp_path, info, names)
    try:
        nf.rec_info["bads"] = [names[0]]  # shifts everything after it
        nf.picks = None
        nf.winsize = 1.0
        nf._sfreq = 250.0
        nf.params = {
            "frange": [8, 13],
            "channels": [[names[3], names[5]]],
            "method": "coh",
            "mode": "cwt_morlet",
            "n_cycles": 5,
        }
        prep = nf._sensor_connectivity_prep()
        acquired = nf._acquired_ch_names()
        assert prep["indices"][0][0] == acquired.index(names[3])
        assert prep["indices"][1][0] == acquired.index(names[5])
    finally:
        nf.save()


def test_no_bad_channels_leaves_indexing_unchanged(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=5.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        assert nf._acquired_ch_names() == list(nf.rec_info["ch_names"])
        assert nf._acquired_info() is nf.rec_info
    finally:
        nf.save()


def test_record_baseline_does_not_build_a_head_model(tmp_path, array_info, monkeypatch):
    """A sensor-space baseline must not reach across the network.

    `compute_inv_operator` calls `fetch_fsaverage()`, ~700 MB from osf.io, and
    `record_baseline` used to call it unconditionally -- so four sensor-space
    tests that never look at a source depended on that download, and failed
    whenever OSF was slow.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    calls = []
    monkeypatch.setattr(type(nf), "compute_inv_operator", lambda self, *a, **kw: calls.append(1))
    try:
        nf.connect_to_array(_make_array_data(array_info), array_info)
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)

        assert nf.raw_baseline is not None
        assert calls == []
        assert nf.src is None
        assert list(pathlib.Path(nf.subject_dir).rglob("*_inv.fif")) == []
    finally:
        nf.save()


def test_head_model_is_built_on_first_request(tmp_path, array_info, monkeypatch):
    """Deferred, not dropped: whatever needs it triggers the build."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    calls = []

    def _fake_compute(self, *a, **kw):
        calls.append(1)
        self.src = object()

    monkeypatch.setattr(type(nf), "compute_inv_operator", _fake_compute)
    try:
        nf.connect_to_array(_make_array_data(array_info), array_info)
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        assert calls == []

        nf._ensure_head_model("A source modality")
        assert calls == [1]

        # Built once, then reused.
        nf._ensure_head_model("A source modality")
        assert calls == [1]
    finally:
        nf.save()


def test_head_model_without_a_baseline_says_so(tmp_path):
    """The error has to name the missing step, not the missing attribute."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(RuntimeError, match="record_baseline"):
        nf._ensure_head_model("The brain-activation display")


def test_gedai_leadfield_mode_builds_the_head_model(tmp_path, array_info, monkeypatch):
    """`fit_gedai` gates on `self.fwd`, which nothing else populates now.

    Without this it would fall through to band-filter mode -- a different
    denoising algorithm -- with only a log line to say so.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    asked = []

    def _fake_ensure(self, reason, **kw):
        asked.append(reason)
        # What the real build leaves behind, which is what the branch reads.
        # Random rather than zeros: GEDAI decomposes it, and a singular
        # leadfield yields NaNs.
        rng = np.random.default_rng(0)
        self.fwd = {"sol": {"data": rng.standard_normal((len(array_info["ch_names"]), 20))}}

    monkeypatch.setattr(type(nf), "_ensure_head_model", _fake_ensure)
    try:
        nf.connect_to_array(_make_array_data(array_info), array_info)
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf.fit_gedai(use_leadfield=True)
        assert asked == ["GEDAI leadfield mode"]
        assert nf.fwd is not None
    finally:
        nf.save()


def test_head_model_guard_checks_what_the_caller_needs(tmp_path, array_info, monkeypatch):
    """`compute_inv_operator(make_inverse=False)` leaves `src` set but `inv` None.

    Testing only `src` would report the model present and let the
    brain-activation display fail silently later.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    calls = []

    def _fake_compute(self, *a, **kw):
        calls.append(kw.get("make_inverse"))
        self.src = object()
        self.inv = object() if kw.get("make_inverse") else None

    monkeypatch.setattr(type(nf), "compute_inv_operator", _fake_compute)
    nf.raw_baseline = object()

    # A beamformer needs no inverse, so a forward-only model satisfies it.
    nf.src, nf.inv = object(), None
    nf._ensure_head_model("A beamformer")
    assert calls == []

    # The brain display dereferences `inv`, so it must not be told to proceed.
    nf._ensure_head_model("The brain-activation display", need_inverse=True)
    assert calls == [True]


def test_filters_are_applied_once_per_connection(tmp_path, array_info):
    """mne-lsl appends to the filter chain, so a second block would band-pass twice."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1", bandpass_freq=(1.0, 40.0))
    nf.connect_to_array(_make_array_data(array_info, duration=40.0), array_info, n_repeat=np.inf)
    applied = []
    original = type(nf.stream).filter

    def _spy(self, *a, **kw):
        applied.append((a, kw))
        return original(self, *a, **kw)

    nf.stream.filter = _spy.__get__(nf.stream)
    nf.run_blocks(
        blocks=[
            dict(
                duration=3.0,
                winsize=1.0,
                modality="sensor_power",
                show_raw_signal=False,
                show_nf_signal=False,
            ),
            dict(
                duration=3.0,
                winsize=1.0,
                modality="sensor_power",
                show_raw_signal=False,
                show_nf_signal=False,
            ),
        ],
        rest_duration=0.0,
    )

    assert len(applied) == 1


def test_disconnecting_clears_the_filter_flag(tmp_path, array_info):
    """mne-lsl empties its filter chain on disconnect.

    Leaving the flag set would run the next session unfiltered, silently.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1", bandpass_freq=(1.0, 40.0))
    nf.connect_to_array(_make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf)
    nf.record_main(
        duration=3.0,
        winsize=1.0,
        modality="sensor_power",
        show_raw_signal=False,
        show_nf_signal=False,
    )

    assert nf._filters_applied is False


def test_executor_survives_a_thread_that_outlives_the_join(tmp_path, array_info):
    """The pool is shut down but not unbound.

    Rebinding it to None would make a still-running acquisition thread submit
    into `None`, killing it before it stores its raw chunks.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=3.0,
            winsize=1.0,
            modality="sensor_power",
            show_raw_signal=False,
            show_nf_signal=False,
        )
        assert nf.executor is not None
        assert nf.executor._shutdown
    finally:
        if nf.stream.connected:
            nf.stream.disconnect()


def test_executor_is_shut_down_when_record_main_returns(tmp_path, array_info):
    """Nothing joined the pool, so the brain-plot task could still be running."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=3.0,
            winsize=1.0,
            modality="sensor_power",
            show_raw_signal=False,
            show_nf_signal=False,
        )
        assert nf.executor._shutdown
    finally:
        if nf.stream.connected:
            nf.stream.disconnect()


@pytest.mark.parametrize(
    "kwargs", [{"subject_fs_id": "sample"}, {"source_space": "volume"}], ids=["subject", "volume"]
)
def test_brain_kernel_refuses_what_the_display_cannot_draw(tmp_path, kwargs):
    """The display draws fsaverage's cortex: other subjects or a volume source
    space indexed out of range and left it blank, silently."""
    nf = _make_rt_stream(tmp_path)
    for key, value in kwargs.items():
        setattr(nf, key, value)
    with pytest.raises(ValueError, match="fsaverage"):
        nf._brain_kernel()


@pytest.mark.slow
def test_brain_kernel_matches_apply_inverse_raw(tmp_path, array_info):
    """The brain display applies the inverse as one precomputed matrix instead
    of preparing it again for every window (73 ms -> under 1 ms)."""
    import mne
    from mne.minimum_norm import apply_inverse_raw

    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    nf = _make_rt_stream(tmp_path, subjects_fs_dir=str(fs_dir.parent))
    try:
        nf.connect_to_array(_make_array_data(array_info, 20.0), array_info, n_repeat=np.inf)
        nf.rec_info["bads"] = [array_info["ch_names"][2]]
        nf.record_baseline(baseline_duration=4.0, winsize=1.0)
        nf._ensure_head_model("test", need_inverse=True)
        kernel, vertices = nf._brain_kernel()
        w = np.random.default_rng(0).standard_normal((kernel.shape[1], 256)) * 1e-5
        raw = nf._prepare_raw_array(w, use_picks=False)
        ref = apply_inverse_raw(raw, nf.inv, lambda2=1.0 / 9, pick_ori="normal", verbose=False)
        # Relative to the largest value: entries near zero differ by rounding only.
        scale = np.abs(ref.data).max()
        np.testing.assert_allclose(kernel @ w, ref.data, rtol=0, atol=1e-10 * scale)
        assert [len(v) for v in vertices] == [10242, 10242]
    finally:
        nf.save()


def test_brain_source_values_power_activation_and_edge_cases():
    """Above Nyquist the band raised on every window; a very short window
    raised in sosfiltfilt's default padding."""
    from mne_rt.rt_stream import _brain_source_values

    rng = np.random.default_rng(0)
    kernel = rng.standard_normal((50, 4))
    w = rng.standard_normal((4, 250))
    cache = {}
    power = _brain_source_values(kernel, w, (8.0, 13.0), 250.0, cache)
    assert power.shape == (50,) and np.nanpercentile(power, 98) == pytest.approx(1.0)
    assert (8.0, 13.0) in cache  # the filter is built once per band
    act = _brain_source_values(kernel, w, None, 250.0, cache)
    np.testing.assert_allclose(
        act * np.percentile(np.abs((kernel @ w).mean(1)), 98), np.abs((kernel @ w).mean(1))
    )
    short = _brain_source_values(kernel, w[:, :12], (8.0, 13.0), 250.0, cache)
    assert np.all(np.isfinite(short))
    with pytest.raises(ValueError, match="above what"):
        _brain_source_values(kernel, w, (150.0, 200.0), 250.0, cache)


def test_brain_source_values_ignore_a_nan_source_for_the_scale():
    """One NaN made every value NaN through the percentile."""
    from mne_rt.rt_stream import _brain_source_values

    kernel = np.eye(3)
    kernel[2] = np.nan  # one source without a usable estimate
    w = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    vals = _brain_source_values(kernel, w, None, 250.0, {})
    assert np.isfinite(vals[:2]).all() and np.isnan(vals[2])
    # Still scaled to the 98th percentile of the finite ones, not left raw.
    assert np.nanpercentile(vals, 98) == pytest.approx(1.0)
