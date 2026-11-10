"""Saving a session (JSON, BIDS TSV, raw) and the HTML report."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from mne_rt.tests._helpers import _make_array_data, _make_baseline_raw, _make_rt_stream

pytestmark = pytest.mark.usefixtures("fast_clock")


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


def test_save_rejects_an_unsupported_format_before_writing(tmp_path):
    """``format`` was accepted and ignored, so ``"csv"`` silently wrote JSON."""
    nf = _make_rt_stream(tmp_path)
    nf.nf_data = {"sensor_power": [0.1, 0.2, 0.3]}
    with pytest.raises(ValueError, match="format"):
        nf.save(format="csv")
    assert not list(tmp_path.rglob("*_beh.json"))


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


@pytest.mark.real_time
def test_window_onsets_are_recorded_per_window(tmp_path, array_info):
    nf = _timed_session(tmp_path, array_info)
    onsets = nf.window_onsets
    assert len(onsets) == len(nf.nf_data["sensor_power"]) > 2
    assert all(b > a for a, b in zip(onsets, onsets[1:]))  # strictly increasing
    assert all(d == pytest.approx(1.0, rel=1e-3) for d in nf.window_durations)
    # The mean gap, not each one: an onset is the newest sample's timestamp, so a
    # delivery the runner delays (macOS CI) shortens one gap and lengthens the
    # next. A band, not equality: the hop re-reads the clock after each compute.
    gaps = np.diff(onsets)
    assert 0.4 <= gaps.mean() <= 0.9


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
