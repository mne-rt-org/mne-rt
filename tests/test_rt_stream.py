"""Tests for RTStream instantiation and validation (no LSL required)."""

import json
import pathlib
import time

import numpy as np
import pytest


@pytest.fixture()
def subjects_dir(tmp_path):
    return str(tmp_path)


def _make_rt_stream(tmp_path, **kwargs):
    from mne_rt import RTStream

    defaults = dict(
        subject_id="sub01",
        session="01",
        subjects_dir=str(tmp_path),
        montage="easycap-M1",
    )
    defaults.update(kwargs)
    return RTStream(**defaults)


def _make_baseline_raw(n_channels=8, sfreq=256.0, duration=2.0):
    import mne

    rng = np.random.default_rng(0)
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    data = rng.standard_normal((n_channels, int(sfreq * duration))) * 1e-6
    return mne.io.RawArray(data, info, verbose=False)


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
    from pathlib import Path

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


# ─────────────────────────────────────────────────────────────────────────────
# NFRealtime: artifact_rate and snr_data attributes
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# fit_asr / fit_gedai / fit_maxwell / run_orica
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# save / load_nf_data
# ─────────────────────────────────────────────────────────────────────────────


def test_save_with_no_data_returns_empty_dict(tmp_path):
    nf = _make_rt_stream(tmp_path)
    saved = nf.save()
    assert saved == {}
    assert (nf.subject_dir / "beh").is_dir()


def test_save_writes_nf_data_json(tmp_path):
    nf = _make_rt_stream(tmp_path)
    nf.nf_data = {"sensor_power": [0.1, 0.2, 0.3]}
    nf._sfreq = 256.0
    nf.winsize = 1.0
    nf.duration = 3.0

    saved = nf.save()

    assert "nf_data" in saved
    with open(saved["nf_data"]) as fh:
        payload = json.load(fh)
    assert payload["data"]["sensor_power"] == [0.1, 0.2, 0.3]
    assert payload["meta"]["subject_id"] == "sub01"
    assert payload["meta"]["modalities"] == ["sensor_power"]


def test_save_writes_bids_tsv_when_requested(tmp_path):
    nf = _make_rt_stream(tmp_path)
    nf.nf_data = {"sensor_power": [0.1, 0.2]}
    nf._sfreq = 256.0
    nf.winsize = 1.0
    nf.duration = 2.0

    saved = nf.save(bids_tsv=True)

    assert "nf_tsv" in saved
    lines = saved["nf_tsv"].read_text().splitlines()
    assert lines[0] == "sensor_power"
    assert len(lines) == 3  # header + 2 rows


def test_load_nf_data_round_trip(tmp_path):
    from mne_rt import RTStream

    payload = {"meta": {"subject_id": "sub01"}, "data": {"sensor_power": [1.0, 2.0]}}
    fname = tmp_path / "beh.json"
    with open(fname, "w") as fh:
        json.dump(payload, fh)

    loaded = RTStream.load_nf_data(fname)
    assert loaded == payload


# ─────────────────────────────────────────────────────────────────────────────
# create_report
# ─────────────────────────────────────────────────────────────────────────────


def _report_ready(nf, **kwargs):
    """Give an RTStream the attributes record_main would have left behind."""
    nf.raw_baseline = _make_baseline_raw()
    nf.modality = "sensor_power"
    nf.nf_data = {"sensor_power": [0.1, 0.2, 0.3, 0.4]}
    nf.winsize = 1.0
    nf._sfreq = 256.0
    nf.n_total_windows = 4
    for key, value in kwargs.items():
        setattr(nf, key, value)
    return nf


def test_create_report_happy_path(tmp_path):
    nf = _report_ready(_make_rt_stream(tmp_path))

    report_path = nf.create_report(include_psd=True, include_nf_signal=True)

    assert report_path.exists()
    assert report_path.suffix == ".html"
    html = report_path.read_text(encoding="utf-8")
    assert "Session summary" in html
    assert "sub-sub01" in html
    assert "Feature time-series" in html


def test_create_report_psd_survives_extra_channel_types(tmp_path):
    # compute_psd drops ECG/EMG/misc, so counting channel types on the raw
    # over-counts and the spectrum plot refuses the axes list. An ordinary EEG
    # cap with one ECG lead is enough to hit it.
    import mne

    rng = np.random.default_rng(0)
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(8)] + ["ECG"],
        sfreq=256.0,
        ch_types=["eeg"] * 8 + ["ecg"],
    )
    raw = mne.io.RawArray(rng.standard_normal((9, 512)) * 1e-6, info, verbose=False)
    nf = _report_ready(_make_rt_stream(tmp_path))
    nf.raw_baseline = raw

    assert nf.create_report(include_psd=True).exists()


def test_create_report_without_a_baseline_raises_the_documented_error(tmp_path):
    # Previously an AttributeError from deep inside, despite the docstring and
    # the changelog both promising a RuntimeError here.
    nf = _make_rt_stream(tmp_path)

    with pytest.raises(RuntimeError, match="record_baseline"):
        nf.create_report()


def test_create_report_uses_real_window_onsets(tmp_path):
    from mne_lsl.lsl import local_clock

    t0 = local_clock()
    nf = _report_ready(
        _make_rt_stream(tmp_path),
        window_onsets=[t0, t0 + 0.5, t0 + 1.0, t0 + 1.5],
        window_durations=[1.0] * 4,
    )

    times, measured = nf._report_times(4)

    assert measured
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(1.5)


def test_create_report_falls_back_to_the_nominal_grid_and_says_so(tmp_path):
    nf = _report_ready(_make_rt_stream(tmp_path))  # no window_onsets at all

    times, measured = nf._report_times(4)
    html = nf.create_report().read_text(encoding="utf-8")

    assert not measured
    assert times[1] == pytest.approx(0.5)  # winsize / 2
    assert "nominal" in html.lower()


def _gated(nf, conditions, gate=("speech",)):
    """Set the gating attributes exactly as record_main would.

    `n_gated_windows` counts the windows gated *out* (feedback withheld), and
    `window_gated` is 1 for those same windows -- deriving both from one list of
    conditions is what keeps a test from asserting against a state the loop
    cannot produce.
    """
    from mne_lsl.lsl import local_clock

    t0 = local_clock()
    out = [0 if c in gate else 1 for c in conditions]
    nf.gate_conditions = list(gate)
    nf.window_conditions = list(conditions)
    nf.window_gated = out
    nf.window_marker_counts = [1] * len(conditions)
    nf.n_gated_windows = sum(out)
    nf.n_total_windows = len(conditions)
    nf.window_onsets = [t0 + 0.5 * i for i in range(len(conditions))]
    nf.window_durations = [1.0] * len(conditions)
    nf.marker_id = {"rest": 1, "speech": 2}
    nf.markers = [(t0 + 0.5 * i, 2 if c in gate else 1) for i, c in enumerate(conditions)]
    nf.nf_data = {"sensor_power": [0.1] * len(conditions)}
    return nf


def test_create_report_reports_gating(tmp_path):
    nf = _gated(_report_ready(_make_rt_stream(tmp_path)), ["rest", "speech", "speech", "rest"])

    html = nf.create_report().read_text(encoding="utf-8")

    assert "Gating" in html
    assert "speech" in html
    assert "Marker raster" in html
    # Two of four were in the gate, so two received feedback and two did not.
    assert nf.n_gated_windows == 2
    assert "2 of 4" in html
    assert "No feedback was delivered" not in html


def test_create_report_shouts_when_the_gate_never_opened(tmp_path):
    # The failure this whole section exists for: features are computed and saved,
    # the plot updates, and the subject simply never receives anything.
    nf = _gated(_report_ready(_make_rt_stream(tmp_path)), ["rest"] * 4)

    html = nf.create_report().read_text(encoding="utf-8")

    # n_gated_windows counts the *withheld* windows, so "all withheld" is the
    # alarm and "none withheld" is the benign case -- not the other way round.
    assert nf.n_gated_windows == 4
    assert "No feedback was delivered" in html


def test_create_report_does_not_alarm_when_every_window_was_in_gate(tmp_path):
    nf = _gated(_report_ready(_make_rt_stream(tmp_path)), ["speech"] * 4)

    html = nf.create_report().read_text(encoding="utf-8")

    assert nf.n_gated_windows == 0
    assert "No feedback was delivered" not in html
    assert "gating never suppressed feedback" in html


def test_create_report_describes_the_timing_and_marker_columns(tmp_path):
    # The data dictionary is built from the same windows table save() writes;
    # without it, the six columns a reader most needs explained were missing.
    nf = _gated(_report_ready(_make_rt_stream(tmp_path)), ["rest", "speech", "speech", "rest"])

    html = nf.create_report().read_text(encoding="utf-8")

    for column in ("onset", "duration", "condition", "n_markers", "gated"):
        assert column in html
    assert "Feedback withheld" in html


def test_session_windows_is_shared_with_save(tmp_path):
    nf = _gated(_report_ready(_make_rt_stream(tmp_path)), ["rest", "speech", "speech", "rest"])

    windows = nf._session_windows()
    saved = json.loads(nf.save()["nf_data"].read_text())["windows"]

    assert windows["condition"] == saved["condition"]
    assert windows["gated"] == saved["gated"]
    assert windows["onset"][0] == 0.0


def test_report_times_keeps_measured_onsets_despite_one_unknown(tmp_path):
    # A window the stream had not timestamped yet is routine at session start;
    # discarding every real onset because of it made the caption claim none were
    # recorded and sent the reader after a clock-sync problem that is not there.
    from mne_lsl.lsl import local_clock

    t0 = local_clock()
    nf = _report_ready(
        _make_rt_stream(tmp_path),
        window_onsets=[float("nan"), t0, t0 + 0.5, t0 + 1.0],
        window_durations=[1.0] * 4,
    )

    times, measured = nf._report_times(4)

    assert measured
    assert np.isnan(times[0])
    assert times[1] == 0.0
    assert times[-1] == pytest.approx(1.0)


def test_create_report_omits_marker_and_source_sections_when_absent(tmp_path):
    nf = _report_ready(_make_rt_stream(tmp_path))

    html = nf.create_report().read_text(encoding="utf-8")

    assert "Marker raster" not in html
    assert "Source-space configuration" not in html


def test_create_report_renders_instanced_names_and_the_combined_trace(tmp_path):
    from mne_rt._naming import parse_modality

    nf = _make_rt_stream(tmp_path)
    _report_ready(nf)
    nf._mod_specs = [parse_modality("sensor_power@alpha"), parse_modality("sensor_power@theta")]
    nf.nf_data = {
        "sensor_power@alpha": [1.0, 2.0, 3.0],
        "sensor_power@theta": [3.0, 2.0, 1.0],
        "reward": [0.5, 0.5, 0.5],
    }
    nf._combined_name = "reward"

    html = nf.create_report().read_text(encoding="utf-8")

    assert "sensor_power@alpha" in html
    assert "sensor_power@theta" in html
    # _label_for already appends the instance label; adding it again rendered
    # "Power (alpha)\n(alpha)" on the y-axis.
    from mne_rt.viz.nf_plot import _label_for

    assert _label_for("sensor_power@alpha").count("alpha") == 1
    # The combined trace is the signal the subject actually saw; it is not in
    # _mod_specs, so it has to be added back explicitly.
    assert "reward" in html


def test_create_report_names_the_source_configuration(tmp_path):
    from mne_rt._naming import parse_modality

    nf = _make_rt_stream(tmp_path, source_space="volume", source_atlas="aparc+aseg")
    _report_ready(nf)
    nf._mod_specs = [parse_modality("source_connectivity@bw")]
    nf.nf_data = {"source_connectivity@bw": [0.1, 0.2, 0.3]}
    nf.mod_params_dict = {
        "source_connectivity@bw": {
            "atlas": "aparc+aseg",
            "inverse_method": "LCMV",
            "frange": [8.0, 13.0],
            "rois": [
                {"Broca": ["ctx-lh-parsopercularis"]},
                {"Wernicke": ["ctx-lh-superiortemporal"]},
            ],
            "pairs": [["Broca", "Wernicke"]],
        }
    }

    html = nf.create_report().read_text(encoding="utf-8")

    assert "Source-space configuration" in html
    assert "Broca" in html
    assert "LCMV" in html


def test_create_report_flags_a_run_that_cannot_keep_up(tmp_path):
    nf = _report_ready(
        _make_rt_stream(tmp_path),
        method_delays={"sensor_power": [0.9] * 10},  # 900 ms against a 500 ms hop
    )

    html = nf.create_report().read_text(encoding="utf-8")

    assert "cannot keep up" in html


def test_create_report_does_not_cry_wolf_over_the_hop(tmp_path):
    # Four modalities at 140 ms each: the per-window total is 560 ms only if all
    # four peak on the same window. Summing each one's own p95 assumes exactly
    # that, and raises the alarm on a run that never approached the hop.
    rng = np.random.default_rng(0)
    delays = {f"m{i}": list(rng.normal(0.10, 0.02, 200).clip(0.02, 0.14)) for i in range(4)}
    nf = _report_ready(_make_rt_stream(tmp_path), method_delays=delays)

    html = nf.create_report().read_text(encoding="utf-8")

    assert "cannot keep up" not in html


def test_create_report_writes_one_file_per_run(tmp_path):
    nf = _report_ready(_make_rt_stream(tmp_path))

    first = nf.create_report(run=1)
    second = nf.create_report(run=2)

    assert first != second
    assert "run-01" in first.name and "run-02" in second.name
    assert first.exists() and second.exists()


def test_create_report_refuses_a_run_it_does_not_hold(tmp_path):
    # run_blocks rebinds nf_data on every block, so the stream only ever holds
    # the last one; naming an earlier run would file those traces under the
    # wrong block.
    nf = _report_ready(_make_rt_stream(tmp_path))
    nf._session_stem = "sub-sub01_ses-01_task-neurofeedback_run-03"

    with pytest.raises(ValueError, match="does not match"):
        nf.create_report(run=1)


def test_create_report_reuses_the_stem_record_main_built(tmp_path):
    nf = _report_ready(_make_rt_stream(tmp_path))
    nf._session_stem = "sub-sub01_ses-01_task-neurofeedback_run-03"

    assert "run-03" in nf.create_report().name


def test_session_meta_is_shared_with_save(tmp_path):
    # The report renders the same dict save() serialises, so they cannot drift.
    nf = _report_ready(_make_rt_stream(tmp_path), gate_conditions=["speech"], n_gated_windows=2)

    meta = nf._session_meta()
    saved = json.loads(nf.save()["nf_data"].read_text())["meta"]

    assert meta["gate_conditions"] == saved["gate_conditions"] == ["speech"]
    assert meta["n_gated_windows"] == saved["n_gated_windows"] == 2
    assert meta["modalities"] == saved["modalities"]


# ─────────────────────────────────────────────────────────────────────────────
# connect_to_lsl + record_baseline (mock LSL streaming)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def short_mock_fif(tmp_path):
    import mne

    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[:8]
    sfreq = 256.0
    duration = 8.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)

    rng = np.random.default_rng(0)
    n_samples = int(duration * sfreq)
    data = rng.standard_normal((len(ch_names), n_samples)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)

    fname = tmp_path / "mock-raw.fif"
    raw.save(fname, verbose=False)
    return fname


def test_connect_to_lsl_and_record_baseline(tmp_path, short_mock_fif):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_lsl(mock_lsl=True, fname=str(short_mock_fif), timeout=15.0)
        assert nf.sfreq == pytest.approx(256.0)
        assert len(nf.rec_info["ch_names"]) == 8

        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        assert nf.raw_baseline is not None
        assert nf.raw_baseline.info["sfreq"] == pytest.approx(256.0)
        assert nf.raw_baseline.n_times >= int(1.0 * 256.0)
    finally:
        nf.save()  # disconnects the stream / mock player


# ─────────────────────────────────────────────────────────────────────────────
# connect_to_array (plain numpy-array input, no LSL required)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def array_info():
    import mne

    montage = mne.channels.make_standard_montage("easycap-M1")
    ch_names = montage.ch_names[:8]
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    info.set_montage(montage)
    return info


def _make_array_data(info, duration=8.0):
    rng = np.random.default_rng(0)
    n_samples = int(duration * info["sfreq"])
    return rng.standard_normal((len(info["ch_names"]), n_samples)) * 1e-6


def test_connect_to_array_invalid_shape(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(ValueError, match="2D"):
        nf.connect_to_array(np.zeros((8, 10, 2)), array_info)


def test_connect_to_array_channel_mismatch(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    with pytest.raises(ValueError, match="channels"):
        nf.connect_to_array(np.zeros((4, 100)), array_info)


def test_connect_to_array_and_record_baseline(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    try:
        nf.connect_to_array(data, array_info)
        assert nf.sfreq == pytest.approx(256.0)
        assert len(nf.rec_info["ch_names"]) == 8
        assert nf.stream.connected

        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        assert nf.raw_baseline is not None
        assert nf.raw_baseline.info["sfreq"] == pytest.approx(256.0)
        assert nf.raw_baseline.n_times >= int(1.0 * 256.0)
    finally:
        nf.save()  # disconnects the stream


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


def test_connect_to_array_pick_types(tmp_path):
    import mne

    ch_names = [f"EEG{i:03d}" for i in range(4)] + ["EOG001"]
    ch_types = ["eeg"] * 4 + ["eog"]
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types=ch_types)
    data = np.random.default_rng(0).standard_normal((5, 256)) * 1e-6

    nf = _make_rt_stream(tmp_path, montage=None)
    try:
        nf.connect_to_array(data, info, pick_types="eeg")
        assert nf.rec_info["ch_names"] == ch_names[:4]
    finally:
        nf.save()


def test_connect_to_array_open_stream_viewer_raises(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info)
    try:
        nf.connect_to_array(data, array_info)
        with pytest.raises(RuntimeError, match="connect_to_array"):
            nf.open_stream_viewer()
    finally:
        nf.save()


# ─────────────────────────────────────────────────────────────────────────────
# set_decoder / "decode" modality (RTDecode, no LSL required)
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# ArrayStream (standalone duck-typed LSL-stream shim behind connect_to_array)
# ─────────────────────────────────────────────────────────────────────────────


def _make_array_stream(n_channels=4, sfreq=100.0, n_samples=500, **kwargs):
    import mne

    from mne_rt.rt_stream import ArrayStream

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    data = np.random.default_rng(0).standard_normal((n_channels, n_samples)) * 1e-6
    return ArrayStream(data, info, **kwargs)


class TestArrayStream:
    def test_not_connected_raises(self):
        stream = _make_array_stream()
        assert not stream.connected
        with pytest.raises(RuntimeError, match="not connected"):
            stream.get_data()
        with pytest.raises(RuntimeError, match="connect"):
            _ = stream.info

    def test_connect_get_data_resets_n_new_samples(self):
        import time

        stream = _make_array_stream(sfreq=200.0, bufsize=1.0, chunk_size=10, n_repeat=np.inf)
        stream.connect()
        try:
            time.sleep(0.3)
            data, ts = stream.get_data(0.2)
            assert data.shape == (4, 40)
            assert ts.shape == (40,)
            assert stream.n_new_samples == 0
            time.sleep(0.2)
            assert stream.n_new_samples > 0
        finally:
            stream.disconnect()
        assert not stream.connected

    def test_pick_reduces_channels_before_connect(self):
        stream = _make_array_stream(n_channels=4)
        stream.pick(["EEG000", "EEG002"])
        assert stream._info["ch_names"] == ["EEG000", "EEG002"]
        stream.connect()
        try:
            data, _ = stream.get_data(0.1)
            assert data.shape[0] == 2
        finally:
            stream.disconnect()

    def test_pick_preserves_requested_order(self):
        stream = _make_array_stream(n_channels=4)
        stream.pick(["EEG002", "EEG000"])
        assert stream._info["ch_names"] == ["EEG002", "EEG000"]

    def test_pick_after_connect_raises(self):
        stream = _make_array_stream()
        stream.connect()
        try:
            with pytest.raises(RuntimeError, match="before connect"):
                stream.pick("eeg")
        finally:
            stream.disconnect()

    def test_get_data_excludes_bads(self):
        stream = _make_array_stream(n_channels=4)
        stream._info["bads"] = ["EEG001"]
        stream.connect()
        try:
            data, _ = stream.get_data(0.1)
            assert data.shape[0] == 3
        finally:
            stream.disconnect()

    def test_connect_twice_does_not_leak_thread(self):
        stream = _make_array_stream(sfreq=200.0, n_repeat=np.inf)
        stream.connect()
        first_thread = stream._thread
        stream.connect()
        try:
            assert not first_thread.is_alive()
            assert stream.connected
        finally:
            stream.disconnect()

    def test_filter_and_notch_filter_return_self(self):
        stream = _make_array_stream(sfreq=256.0, n_samples=2560)
        assert stream.filter(1.0, 40.0) is stream
        assert stream.notch_filter(50.0) is stream

    def test_n_repeat_stops_advancing(self):
        import time

        stream = _make_array_stream(
            sfreq=100.0, n_samples=20, chunk_size=5, n_repeat=1, bufsize=0.5
        )
        stream.connect()
        try:
            time.sleep(0.5)  # long enough to exhaust the 20-sample array once
            assert stream.n_new_samples == 20
        finally:
            stream.disconnect()


# ------------------------------------------------------------------
# Source-space configuration
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Feature combiners in the live loop
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Bad channels
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Instanced modalities — the same measure, several times over
# ------------------------------------------------------------------


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


# ---- protocols ---------------------------------------------------


class _CountingProtocol:
    """Records how many times it was evaluated, and with what."""

    def __init__(self, threshold=0.0):
        self.current_threshold = threshold
        self.values = []

    def evaluate(self, value):
        self.values.append(float(value))
        return (float(value) > self.current_threshold, abs(float(value)))


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


# ---- combiner ----------------------------------------------------


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


# ---- outputs -----------------------------------------------------


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


# ------------------------------------------------------------------
# Z-scoring is scale-free
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Per-window timestamps
# ------------------------------------------------------------------


def test_array_stream_timestamps_use_the_lsl_clock(array_info):
    """The test double must share a time base with a real LSL stream.

    `time.time()` and `local_clock()` are ~1.8e9 s apart, so an onset derived
    under this double would otherwise land on a different clock entirely.
    """
    from mne_lsl.lsl import local_clock

    from mne_rt.rt_stream import ArrayStream

    stream = ArrayStream(_make_array_data(array_info, duration=2.0), array_info, n_repeat=np.inf)
    stream.connect()
    try:
        _wait_for = time.time() + 2.0
        while stream.n_new_samples < 10 and time.time() < _wait_for:
            time.sleep(0.01)
        ts = stream.get_data()[1]
        assert ts.size
        assert abs(float(ts[-1]) - local_clock()) < 1.0
    finally:
        stream.disconnect()


def _timed_session(tmp_path, array_info, **kwargs):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=30.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=4.0,
            winsize=1.0,
            modality="sensor_power",
            show_raw_signal=False,
            show_nf_signal=False,
            **kwargs,
        )
    finally:
        nf.save(bids_tsv=True)
    return nf


def test_window_onsets_are_recorded_per_window(tmp_path, array_info):
    nf = _timed_session(tmp_path, array_info)
    onsets = nf.window_onsets
    assert len(onsets) == len(nf.nf_data["sensor_power"]) > 2
    assert all(b > a for a, b in zip(onsets, onsets[1:]))  # strictly increasing
    assert all(d == pytest.approx(1.0, rel=1e-3) for d in nf.window_durations)
    # A band, not equality: the hop is a busy-wait that re-reads the clock after
    # the previous window's compute, which is exactly the drift being recorded.
    for a, b in zip(onsets, onsets[1:]):
        assert 0.4 <= (b - a) <= 0.9


def test_window_timing_round_trips_through_save(tmp_path, array_info):
    nf = _timed_session(tmp_path, array_info)
    beh = next(pathlib.Path(nf.subject_dir).rglob("*_beh.json"))
    payload = json.loads(beh.read_text())

    w = payload["windows"]
    n = len(nf.nf_data["sensor_power"])
    assert len(w["onset"]) == len(w["duration"]) == len(w["onset_lsl"]) == n
    assert w["onset"][0] == 0.0  # session-relative
    assert w["onset_lsl"][0] == pytest.approx(payload["meta"]["t0_lsl"])
    assert payload["meta"]["clock"] == "lsl_stream_clock"

    # The data block must stay modality-only: meta["modalities"], the TSV
    # columns and TransferProtocol all derive from its keys.
    assert set(payload["data"]) == {"sensor_power"}
    assert payload["meta"]["modalities"] == ["sensor_power"]


def test_beh_tsv_leads_with_onset_and_duration(tmp_path, array_info):
    nf = _timed_session(tmp_path, array_info)
    tsv = next(pathlib.Path(nf.subject_dir).rglob("*_beh.tsv"))
    lines = tsv.read_text().splitlines()
    header = lines[0].split("\t")
    assert header[:2] == ["onset", "duration"]  # BIDS ordering
    assert len(lines) - 1 == len(nf.nf_data["sensor_power"])
    # Sub-millisecond resolution: %.6g would round a ~1e3 s onset to 10 ms.
    assert len(lines[2].split("\t")[0].split(".")[1]) >= 3


def test_session_json_describes_its_columns(tmp_path, array_info):
    """BIDS wants a sidecar; its filename is taken by the session payload."""
    nf = _timed_session(tmp_path, array_info)
    payload = json.loads(next(pathlib.Path(nf.subject_dir).rglob("*_beh.json")).read_text())
    cols = payload["columns"]
    assert cols["onset"]["Units"] == "s"
    assert cols["duration"]["Units"] == "s"
    assert "sensor_power" in cols


def test_record_main_writes_the_tsv_by_default(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    nf.record_main(
        duration=2.0,
        winsize=1.0,
        modality="sensor_power",
        show_raw_signal=False,
        show_nf_signal=False,
    )
    assert list(pathlib.Path(nf.subject_dir).rglob("*_beh.tsv"))


def test_non_integral_winsize_still_produces_windows(tmp_path, array_info):
    """`winsize * sfreq` need not be a whole number.

    The window was sized with int() but fetched with ceil(), so the two
    disagreed by one sample and the length check discarded every window —
    a silently empty session.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=3.0,
            winsize=1.501,  # 1.501 * 256 = 384.256
            modality="sensor_power",
            show_raw_signal=False,
            show_nf_signal=False,
        )
    finally:
        nf.save()
    assert len(nf.nf_data["sensor_power"]) > 0
    assert nf.n_short_windows == 0


def test_short_window_count_is_saved_even_with_no_windows(tmp_path, array_info):
    """The diagnostic must survive the case it exists for: an empty session."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    data = _make_array_data(array_info, duration=20.0)
    nf.connect_to_array(data, array_info, n_repeat=np.inf)
    # winsize beyond the buffer: get_data can never return a whole window.
    try:
        nf.record_main(
            duration=2.0,
            winsize=float(nf.bufsize) + 5.0,
            modality="sensor_power",
            show_raw_signal=False,
            show_nf_signal=False,
        )
    finally:
        nf.save()
    assert nf.nf_data["sensor_power"] == []
    payload = json.loads(next(pathlib.Path(nf.subject_dir).rglob("*_beh.json")).read_text())
    assert payload["meta"]["n_short_windows"] > 0


# ------------------------------------------------------------------
# The head model is built on demand, not by every baseline
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Marker ingestion and feedback gating
# ------------------------------------------------------------------


class _RecordingSender:
    """Stands in for an OSC or LSL sender, counting what actually goes out."""

    def __init__(self):
        self.sends = []

    def send_all(self, names, values):
        self.sends.append((list(names), list(values)))

    def push(self, names, values):
        self.sends.append((list(names), list(values)))


def _gated_session(tmp_path, array_info, *, codes, onsets, duration=12.0, **kwargs):
    """Run a short gated session against an array stream and a marker schedule."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=40.0), array_info, n_repeat=np.inf)
    nf.connect_marker_array(codes=codes, onsets=onsets, marker_id={"speech": 1, "rest": 2})
    nf.record_main(
        duration=duration,
        winsize=1.0,
        modality="sensor_power",
        show_raw_signal=False,
        show_nf_signal=False,
        **kwargs,
    )
    return nf


# ---- MarkerArrayStream -------------------------------------------


def test_marker_array_stream_delivers_each_marker_once():
    """The high-water mark is what stops a marker being consumed twice.

    ``get_data`` returns the *entire* buffer on every call, so a poll loop
    without the mark would re-deliver everything it had already seen.
    """
    from mne_rt import MarkerArrayStream

    stream = MarkerArrayStream([1, 2, 3], [0.05, 0.15, 0.25], bufsize=16)
    stream.connect()
    try:
        seen, hwm = [], 0.0
        deadline = time.time() + 3.0
        while time.time() < deadline and len(seen) < 3:
            data, ts = stream.get_data()
            for i in range(ts.size):
                if ts[i] > hwm:
                    seen.append(int(data[0, i]))
            if ts.size:
                hwm = max(hwm, float(ts.max()))
            time.sleep(0.01)
    finally:
        stream.disconnect()

    assert seen == [1, 2, 3]


def test_marker_array_stream_uses_the_lsl_clock():
    """Markers and window onsets must be comparable without conversion."""
    from mne_lsl.lsl import local_clock

    from mne_rt import MarkerArrayStream

    stream = MarkerArrayStream([7], [0.0], bufsize=4)
    stream.connect()
    try:
        deadline = time.time() + 2.0
        ts = np.zeros(1)
        while time.time() < deadline and not ts.max():
            ts = stream.get_data()[1]
            time.sleep(0.01)
        assert ts.max() > 0
        assert abs(float(ts.max()) - local_clock()) < 1.0
    finally:
        stream.disconnect()


def test_marker_array_stream_zero_winsize_returns_nothing():
    """`ArrayStream` gets this wrong: a `[-0:]` slice returns the whole buffer."""
    from mne_rt import MarkerArrayStream

    stream = MarkerArrayStream([1], [0.0], bufsize=8).connect()
    try:
        data, ts = stream.get_data(0)
        assert data.shape == (1, 0) and ts.size == 0
    finally:
        stream.disconnect()


def test_marker_array_stream_rejects_mismatched_schedule():
    from mne_rt import MarkerArrayStream

    with pytest.raises(ValueError, match="same length"):
        MarkerArrayStream([1, 2], [0.0])
    with pytest.raises(ValueError, match="non-decreasing"):
        MarkerArrayStream([1, 2], [1.0, 0.0])


# ---- connection ---------------------------------------------------


def test_marker_stream_is_not_spliced_onto_self(tmp_path, array_info):
    """`_finalize_stream` copies every non-dunder callable onto the RTStream.

    Running the marker stream through it would rebind `self.get_data`,
    `self.disconnect` and `self._push` to the marker stream, quietly cutting
    the signal path.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info), array_info, n_repeat=np.inf)
    signal_get_data = nf.get_data
    try:
        nf.connect_marker_array(codes=[1], onsets=[0.0], marker_id={"speech": 1})
        assert nf.get_data is signal_get_data
        assert nf.get_data.__self__ is nf.stream
    finally:
        nf.stream.disconnect()
        nf.marker_stream.disconnect()


def test_unknown_marker_channel_names_what_is_available(tmp_path):
    """mne-lsl's own message does not list the names the stream does publish."""

    class _Stub:
        info = {"nchan": 1, "sfreq": 0.0, "ch_names": ["0"]}
        disconnected = False

        def disconnect(self):
            self.disconnected = True

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    stub = _Stub()
    with pytest.raises(ValueError, match=r"\['0'\]") as excinfo:
        nf._attach_marker_stream(stub, marker_id={"go": 1}, ch_name="markers")

    # An outlet that does not label its channel surfaces as "0".
    assert "'0', '1'" in str(excinfo.value)
    assert stub.disconnected


def test_gate_conditions_without_a_marker_stream_is_refused(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info), array_info, n_repeat=np.inf)
    try:
        with pytest.raises(RuntimeError, match="needs a marker stream"):
            nf.record_main(
                duration=1.0,
                modality="sensor_power",
                gate_conditions=["speech"],
                show_raw_signal=False,
                show_nf_signal=False,
            )
    finally:
        nf.stream.disconnect()


def test_gate_condition_that_can_never_match_is_refused(tmp_path, array_info):
    """A typo here would otherwise mute the whole session in silence."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info), array_info, n_repeat=np.inf)
    nf.connect_marker_array(codes=[1], onsets=[0.0], marker_id={"speech": 1})
    try:
        with pytest.raises(ValueError, match="not in the marker stream"):
            nf.record_main(
                duration=1.0,
                modality="sensor_power",
                gate_conditions=["speach"],
                show_raw_signal=False,
                show_nf_signal=False,
            )
    finally:
        nf.stream.disconnect()
        nf.marker_stream.disconnect()


# ---- matching and gating -------------------------------------------


def test_condition_latches_and_switches_with_the_markers(tmp_path, array_info):
    """Latch, not containment: every window after the first marker is labelled.

    With 50 % overlap most windows contain no marker at all, so a containment
    rule would leave them unlabelled.
    """
    nf = _gated_session(tmp_path, array_info, codes=[1, 2], onsets=[1.0, 5.0], duration=10.0)

    conditions = nf.window_conditions
    assert set(conditions) <= {None, "speech", "rest"}
    # No condition until the first marker: "nothing has been announced yet" is
    # not the same as a condition, and a gated session must not treat it as one.
    assert conditions[0] is None
    assert conditions[-1] == "rest"
    # Each state entered once, in order -- the latch never flaps back.
    switches = [(a, b) for a, b in zip(conditions, conditions[1:]) if a != b]
    assert switches == [(None, "speech"), ("speech", "rest")]


def test_every_marker_is_counted_exactly_once(tmp_path, array_info):
    """Overlapping windows must not double-count a contained marker."""
    nf = _gated_session(
        tmp_path, array_info, codes=[1, 2, 1], onsets=[2.0, 5.0, 8.0], duration=11.0
    )

    # The invariant, not a count that depends on how long setup took: every
    # marker the run saw is attributed to exactly one window.
    assert len(nf.markers) >= 2
    assert max(nf.window_marker_counts) <= 1
    assert sum(nf.window_marker_counts) == len(nf.markers)


def test_gating_mutes_the_protocol_and_the_senders(tmp_path, array_info):
    """A muted window still records a value, but drives nothing outward."""
    proto = _CountingProtocol(threshold=-1e30)
    osc, lsl = _RecordingSender(), _RecordingSender()
    nf = _gated_session(
        tmp_path,
        array_info,
        codes=[1, 2],
        onsets=[0.2, 5.0],
        duration=10.0,
        gate_conditions=["speech"],
        protocol={"sensor_power": proto},
        osc_sender=osc,
        lsl_sender=lsl,
    )

    n_windows = len(nf.nf_data["sensor_power"])
    n_live = n_windows - nf.n_gated_windows
    assert 0 < n_live < n_windows  # the gate really did open and close
    assert len(proto.values) == n_live
    assert len(osc.sends) == n_live
    assert len(lsl.sends) == n_live
    # The trace itself stays dense -- a muted window is recorded, not skipped.
    assert len(nf.window_gated) == n_windows
    assert sum(nf.window_gated) == nf.n_gated_windows


def test_muted_windows_keep_reward_data_aligned(tmp_path, array_info):
    """A missing placeholder would shift every later reward row silently."""
    proto = _CountingProtocol(threshold=-1e30)
    nf = _gated_session(
        tmp_path,
        array_info,
        codes=[1, 2],
        onsets=[0.2, 5.0],
        duration=10.0,
        gate_conditions=["speech"],
        protocol={"sensor_power": proto},
    )

    n_windows = len(nf.nf_data["sensor_power"])
    assert nf.n_gated_windows > 0
    assert len(nf.reward_data["sensor_power"]) == n_windows
    assert len(nf.window_onsets) == n_windows
    assert len(nf.window_conditions) == n_windows
    assert len(nf.window_marker_counts) == n_windows


def test_ungated_session_is_unchanged(tmp_path, array_info):
    """`gate_conditions=None` must behave exactly as it did before."""
    proto = _CountingProtocol(threshold=-1e30)
    osc = _RecordingSender()
    nf = _gated_session(
        tmp_path,
        array_info,
        codes=[1, 2],
        onsets=[0.2, 5.0],
        duration=8.0,
        protocol={"sensor_power": proto},
        osc_sender=osc,
    )

    n_windows = len(nf.nf_data["sensor_power"])
    assert nf.n_gated_windows == 0
    assert len(proto.values) == n_windows
    assert len(osc.sends) == n_windows
    # Windows are still labelled -- annotation happens whether or not it gates.
    assert nf.window_conditions[0] == "speech"


def test_warmup_no_longer_rewards_native_units(tmp_path, array_info):
    """`_apply_zscore` passes the raw value through until warmup completes.

    Evaluating those windows rewarded native-unit power (~1e-11) against a
    z-score-calibrated threshold -- random reinforcement at the start of every
    run.
    """
    proto = _CountingProtocol(threshold=-1e30)
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=40.0), array_info, n_repeat=np.inf)
    nf.record_main(
        duration=10.0,
        winsize=1.0,
        modality="sensor_power",
        protocol={"sensor_power": proto},
        zscore_normalize=True,
        zscore_warmup=6,
        show_raw_signal=False,
        show_nf_signal=False,
    )

    n_windows = len(nf.nf_data["sensor_power"])
    assert n_windows > 6
    assert len(proto.values) == n_windows - 5  # windows 1..5 are pass-through
    # Nothing the protocol saw is still in native units.
    assert max(abs(v) for v in proto.values) > 1e-6


def test_markers_and_gate_round_trip_through_save(tmp_path, array_info):
    nf = _gated_session(
        tmp_path,
        array_info,
        codes=[1, 2],
        onsets=[0.2, 5.0],
        duration=10.0,
        gate_conditions=["speech"],
    )

    path = next(pathlib.Path(nf.subject_dir).rglob("*_beh.json"))
    payload = json.loads(path.read_text())

    # The data block stays modality-only: meta["modalities"], the TSV columns
    # and TransferProtocol all derive from its keys.
    assert set(payload["data"]) == {"sensor_power"}
    assert payload["windows"]["condition"][0] == "speech"
    assert set(payload["windows"]["gated"]) == {0, 1}
    assert payload["meta"]["gate_conditions"] == ["speech"]
    assert payload["meta"]["n_gated_windows"] > 0
    assert payload["meta"]["marker_id"] == {"speech": 1, "rest": 2}
    # The raw markers, as ground truth for anything the columns cannot express.
    assert payload["markers"]["code"] == [1, 2]


def test_beh_tsv_carries_the_marker_columns(tmp_path, array_info):
    nf = _gated_session(
        tmp_path,
        array_info,
        codes=[1, 2],
        onsets=[0.2, 5.0],
        duration=10.0,
        gate_conditions=["speech"],
    )

    tsv = next(pathlib.Path(nf.subject_dir).rglob("*_beh.tsv"))
    lines = tsv.read_text().splitlines()
    header = lines[0].split("\t")
    assert header[:5] == ["onset", "duration", "condition", "n_markers", "gated"]
    assert lines[1].split("\t")[2] == "speech"


def test_reconnecting_the_signal_stream_drops_the_marker_stream_loudly(
    tmp_path, array_info, monkeypatch
):
    """Marker timestamps only mean anything against the recording they came with.

    Dropping one silently would leave a session that looks gated but never gates.
    `caplog` cannot see this: the package logger sets ``propagate = False``.
    """
    from mne_rt import rt_stream as rt_stream_mod

    warnings_seen = []
    monkeypatch.setattr(
        rt_stream_mod.logger,
        "warning",
        lambda msg, *a, **kw: warnings_seen.append(str(msg)),
    )

    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info), array_info, n_repeat=np.inf)
    nf.connect_marker_array(codes=[1], onsets=[0.0], marker_id={"speech": 1})
    try:
        nf.connect_to_array(_make_array_data(array_info), array_info, n_repeat=np.inf)
        assert nf.marker_stream is None
        assert any("drops the marker stream" in w for w in warnings_seen)
    finally:
        nf.stream.disconnect()


def test_no_marker_stream_leaves_the_table_unchanged(tmp_path, array_info):
    """The marker columns must not appear for a session that had no markers.

    They are appended per window, so placeholders would give every existing
    user's table three new all-empty columns.
    """
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf)
    nf.record_main(
        duration=4.0,
        winsize=1.0,
        modality="sensor_power",
        show_raw_signal=False,
        show_nf_signal=False,
    )

    assert nf.window_conditions == []
    payload = json.loads(next(pathlib.Path(nf.subject_dir).rglob("*_beh.json")).read_text())
    assert set(payload["windows"]) == {"onset", "duration", "onset_lsl"}
    assert "markers" not in payload

    tsv = next(pathlib.Path(nf.subject_dir).rglob("*_beh.tsv"))
    header = tsv.read_text().splitlines()[0].split("\t")
    assert header == ["onset", "duration", "sensor_power"]


def test_marker_overflow_is_detectable_before_the_read():
    """`n_new_samples` is reset by `get_data`, so it has to be read first.

    Read afterwards it is always 0, and the one diagnostic that would tell a
    user their markers were overwritten never fires.
    """
    from mne_rt import MarkerArrayStream

    stream = MarkerArrayStream([1, 2, 3, 4], [0.0, 0.01, 0.02, 0.03], bufsize=2)
    stream.connect()
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and stream.n_new_samples < 2:
            time.sleep(0.01)
        assert stream.n_new_samples >= 2
        stream.get_data()
        assert stream.n_new_samples == 0
    finally:
        stream.disconnect()


def test_condition_ignores_markers_arriving_after_the_window_ends(tmp_path, array_info):
    """`condition` and `n_markers` must agree about which window a marker is in.

    The post-fetch poll also consumes markers newer than the window's last
    sample; letting one of those label the window would open the gate early.
    """
    nf = _gated_session(tmp_path, array_info, codes=[1, 2], onsets=[1.0, 5.0], duration=10.0)

    counts = nf.window_marker_counts
    conditions = nf.window_conditions
    first_rest = next(i for i, c in enumerate(conditions) if c == "rest")
    marker_windows = [i for i, n in enumerate(counts) if n]
    assert marker_windows
    # The switch cannot precede the window the marker was attributed to.
    assert first_rest >= marker_windows[-1]


def test_sidecar_omits_empty_levels():
    """`"Levels": null` would break a reader doing `Levels.items()`."""
    from mne_rt.tools.bids_io import describe_nf_columns

    cols = describe_nf_columns(
        nf_data={"sensor_power": [1.0]},
        windows={"condition": ["speech"], "n_markers": [1], "gated": [0]},
    )
    assert "Levels" not in cols["condition"]

    cols = describe_nf_columns(
        nf_data={"sensor_power": [1.0]},
        windows={"condition": ["speech"], "n_markers": [1], "gated": [0]},
        meta={"marker_id": {"speech": 1, "rest": 2}},
    )
    assert cols["condition"]["Levels"] == {"rest": "rest", "speech": "speech"}


# ------------------------------------------------------------------
# Multi-block sessions
# ------------------------------------------------------------------


def _two_block_session(tmp_path, array_info, **block_kwargs):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=40.0), array_info, n_repeat=np.inf)
    blocks = [
        dict(duration=3.0, winsize=1.0, modality="sensor_power", **block_kwargs),
        dict(duration=3.0, winsize=1.0, modality="sensor_power", **block_kwargs),
    ]
    nf.run_blocks(blocks=blocks, rest_duration=0.0)
    return nf


def test_run_blocks_runs_every_block(tmp_path, array_info):
    """`record_main` called `save()`, which disconnected the stream.

    Block 2 then fetched from a disconnected stream: headless that wrote an
    *empty* session over block 1's files, and with the Qt windows open it hung
    forever, because the event loop only quits once the acquisition thread
    signals it is done.
    """
    nf = _two_block_session(tmp_path, array_info, show_raw_signal=False, show_nf_signal=False)

    assert len(nf.block_nf_data) == 2
    for block in nf.block_nf_data:
        assert len(block["sensor_power"]) > 0


def test_run_blocks_keeps_the_stream_up_until_the_last_block(tmp_path, array_info):
    """Reconnecting would not be enough: `save()` also stops the mock player."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=40.0), array_info, n_repeat=np.inf)
    connected_during = []

    original = type(nf).record_main

    def _spy(self, *a, **kw):
        connected_during.append(self.stream.connected)
        return original(self, *a, **kw)

    nf.record_main = _spy.__get__(nf)
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

    assert connected_during == [True, True]
    # ...and the last block does disconnect.
    assert not nf.stream.connected


def test_run_blocks_writes_one_file_per_block(tmp_path, array_info):
    """No `run-` entity meant block N clobbered block N-1's JSON, TSV and raw."""
    nf = _two_block_session(tmp_path, array_info, show_raw_signal=False, show_nf_signal=False)

    names = sorted(p.name for p in pathlib.Path(nf.subject_dir).rglob("*_beh.json"))
    assert len(names) == 2
    assert any("run-01" in n for n in names)
    assert any("run-02" in n for n in names)


def test_single_session_filenames_are_unchanged(tmp_path, array_info):
    """Back-compat guard: without `run`, the BIDS stem must not gain an entity."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf)
    nf.record_main(
        duration=3.0,
        winsize=1.0,
        modality="sensor_power",
        show_raw_signal=False,
        show_nf_signal=False,
    )

    names = [p.name for p in pathlib.Path(nf.subject_dir).rglob("*_beh.json")]
    assert names == ["sub-sub01_ses-01_task-neurofeedback_beh.json"]


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


def test_run_blocks_keeps_every_per_block_result(tmp_path, array_info):
    """Only three of nine were captured; the rest were overwritten and lost."""
    nf = _two_block_session(tmp_path, array_info, show_raw_signal=False, show_nf_signal=False)

    assert len(nf.block_results) == 2
    for i, res in enumerate(nf.block_results):
        assert res["run"] == i + 1
        assert len(res["window_onsets"]) == len(res["nf_data"]["sensor_power"])
        assert "reward_data" in res and "n_short_windows" in res


def test_run_blocks_records_the_run_index_it_actually_wrote(tmp_path, array_info):
    """A block may override its own index; the in-memory record must match."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf)
    nf.run_blocks(
        blocks=[
            dict(
                duration=3.0,
                winsize=1.0,
                modality="sensor_power",
                run=7,
                show_raw_signal=False,
                show_nf_signal=False,
            )
        ],
        rest_duration=0.0,
    )

    assert nf.block_results[0]["run"] == 7
    assert any("run-07" in p.name for p in pathlib.Path(nf.subject_dir).rglob("*_beh.json"))


def test_block_results_carry_the_delay_traces(tmp_path, array_info):
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf)
    nf.run_blocks(
        blocks=[
            dict(
                duration=3.0,
                winsize=1.0,
                modality="sensor_power",
                estimate_delays=True,
                show_raw_signal=False,
                show_nf_signal=False,
            )
        ],
        rest_duration=0.0,
    )

    res = nf.block_results[0]
    assert res["acq_delays"] and res["method_delays"]["sensor_power"]
    assert "n_artifact_windows" in res


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
