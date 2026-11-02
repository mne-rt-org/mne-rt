"""Wearable Sensing DSI support and the viewers' channel sets.

``dsi2lsl`` publishes channels named after their reference ("O1-A1/2-A2/2"),
declares trigger/aux inputs as EEG and streams microvolts.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from mne_rt._montage import make_standard_montage

DSI24_SCALP = "P3 C3 F3 Fz F4 C4 P4 Cz Pz Fp1 Fp2 T3 T5 O1 O2 F7 F8 T6 T4".split()
DSI24_NAMES = [f"{c}-A1/2-A2/2" for c in DSI24_SCALP[:17]] + [
    "X3",
    "X2",
    "A2-A1/2-A2/2",
    "T6-A1/2-A2/2",
    "T4-A1/2-A2/2",
    "X1",
    "TRG",
]


class _FakeStream:
    """Records the calls _setup_dsi_channels makes on an mne-lsl stream."""

    def __init__(self, names):
        self.info = {"ch_names": list(names)}
        self.renamed = None
        self.types = None

    def rename_channels(self, mapping):
        self.renamed = mapping
        self.info["ch_names"] = [mapping.get(ch, ch) for ch in self.info["ch_names"]]

    def set_channel_types(self, mapping, *, on_unit_change="warn"):
        self.types = mapping
        self.on_unit_change = on_unit_change


def test_setup_dsi_channels_strips_reference_and_retypes_aux():
    from mne_rt.rt_stream import _setup_dsi_channels

    stream = _FakeStream(DSI24_NAMES)
    _setup_dsi_channels(stream)
    assert stream.info["ch_names"][:3] == ["P3", "C3", "F3"]
    assert "O1" in stream.info["ch_names"] and "A2" in stream.info["ch_names"]
    assert stream.types == {
        "X3": "misc",
        "X2": "misc",
        "A2": "misc",
        "X1": "misc",
        "TRG": "stim",
    }
    # TRG's unit changes from V to none: intended, so no warning on every connect.
    assert stream.on_unit_change == "ignore"


def test_setup_dsi_channels_refuses_ambiguous_names():
    from mne_rt.rt_stream import _setup_dsi_channels

    with pytest.raises(RuntimeError, match="not be unique"):
        _setup_dsi_channels(_FakeStream(["O1-A1", "O1-A2"]))


def test_connect_to_lsl_rejects_unknown_device(tmp_path):
    from mne_rt import RTStream

    nf = RTStream("sub01", "01", str(tmp_path), montage=None)
    with pytest.raises(ValueError, match="device"):
        nf.connect_to_lsl(mock_lsl=True, device="openbci")


@pytest.fixture()
def dsi_fif(tmp_path):
    """A short recording laid out the way dsi2lsl publishes a DSI-24."""
    import mne

    sfreq = 300.0
    data = np.random.default_rng(0).standard_normal((len(DSI24_NAMES), int(8 * sfreq)))
    raw = mne.io.RawArray(data * 1e-5, mne.create_info(DSI24_NAMES, sfreq, "eeg"), verbose=False)
    fname = tmp_path / "dsi-raw.fif"
    raw.save(fname, verbose=False)
    return fname


def test_connect_to_lsl_dsi_device(tmp_path, dsi_fif):
    from mne_rt import RTStream

    nf = RTStream("sub01", "01", str(tmp_path), montage="colin27_1020")
    try:
        nf.connect_to_lsl(
            mock_lsl=True, fname=str(dsi_fif), device="dsi", pick_types="eeg", timeout=15.0
        )
        names = nf.rec_info["ch_names"]
        assert sorted(names) == sorted(DSI24_SCALP)
        # The montage now applies: every scalp channel has a position.
        locs = np.array([ch["loc"][:3] for ch in nf.rec_info["chs"]])
        assert np.all(np.isfinite(locs)) and not np.any(np.all(locs == 0, axis=1))
    finally:
        nf.save()


def test_viewers_cover_all_channels_when_picks_is_set(tmp_path, monkeypatch):
    """The raw and topo viewers span every good channel; NF uses only ``picks``.

    Both used to be built for one channel set and fed another, so RawPlot
    silently dropped every chunk ("Waiting for data") and TopomapPlot raised
    IndexError on every update.
    """
    import mne

    import mne_rt.rt_stream as rs
    from mne_rt import RTStream

    montage = make_standard_montage("colin27_1020")
    names = DSI24_SCALP
    info = mne.create_info(names, 250.0, "eeg")
    info.set_montage(montage)
    data = np.random.default_rng(0).standard_normal((len(names), 250 * 24)) * 1e-5

    pushed = {"raw": [], "topo": []}

    class SpyRaw(rs.RawPlot):
        def push(self, chunk):
            pushed["raw"].append((self._n_ch, chunk.shape[0]))
            super().push(chunk)

    class SpyTopo(rs.TopomapPlot):
        def push(self, window):
            pushed["topo"].append((len(self._topo_picks), window.shape[0]))
            super().push(window)

    monkeypatch.setattr(rs, "RawPlot", SpyRaw)
    monkeypatch.setattr(rs, "TopomapPlot", SpyTopo)

    nf = RTStream("sub01", "01", str(tmp_path), montage=None)
    nf.connect_to_array(data, info, n_repeat=np.inf)
    try:
        nf.record_main(
            duration=4.0,
            winsize=1.0,
            modality="sensor_power",
            picks=["O1", "O2"],
            show_raw_signal=True,
            show_nf_signal=False,
            show_topo=True,
            save_tsv=False,
        )
        assert len(nf.nf_data["sensor_power"]) > 0
        assert pushed["raw"] and pushed["topo"]
        assert all(n_ch == len(names) == got for n_ch, got in pushed["raw"])
        assert all(got == len(names) for _, got in pushed["topo"])
    finally:
        nf.save()


def test_cli_device_dsi_defaults():
    from argparse import Namespace

    from mne_rt.cli import _resolve_montage, _stream_kwargs

    dsi = Namespace(montage=None, device="dsi", mock=False, stream_name="DSI")
    assert _resolve_montage(dsi) == "colin27_1020"
    assert _stream_kwargs(dsi) == {"stream_name": "DSI", "device": "dsi", "pick_types": "eeg"}

    plain = Namespace(montage=None, device=None, mock=True, stream_name="DSI")
    assert _resolve_montage(plain) == "easycap-M1"
    assert _stream_kwargs(plain) == {"stream_name": None}  # --mock replays a file


def test_cli_parses_stream_name_and_device():
    from mne_rt.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "run",
            "--subject",
            "s",
            "--subjects-dir",
            "d",
            "--duration",
            "60",
            "--stream-name",
            "DSI",
            "--device",
            "dsi",
        ]
    )
    assert args.stream_name == "DSI" and args.device == "dsi" and args.montage is None


def test_record_main_topomap_interpolates_bad_channels(tmp_path, monkeypatch):
    """A channel marked bad is not acquired, but the topomap still draws it."""
    import mne

    import mne_rt.rt_stream as rs
    from mne_rt import RTStream

    info = mne.create_info(DSI24_SCALP, 250.0, "eeg")
    info.set_montage(make_standard_montage("colin27_1020"))
    data = np.random.default_rng(0).standard_normal((len(DSI24_SCALP), 250 * 24)) * 1e-5

    seen = {}

    class SpyTopo(rs.TopomapPlot):
        # After push, which puts the good-channel window back into full rows.
        def _update_topomaps(self, window):
            seen.setdefault("shapes", []).append(window.shape[0])
            seen["interp"] = [self._info["ch_names"][i] for i in self._interp_picks]
            super()._update_topomaps(window)

    monkeypatch.setattr(rs, "TopomapPlot", SpyTopo)

    nf = RTStream("sub01", "01", str(tmp_path), montage=None)
    nf.connect_to_array(data, info, n_repeat=np.inf)
    nf.rec_info["bads"] = ["P3"]
    try:
        nf.record_main(
            duration=4.0,
            winsize=1.0,
            modality="sensor_power",
            picks=["O1", "O2"],
            show_raw_signal=False,
            show_nf_signal=False,
            show_topo=True,
            save_tsv=False,
        )
        assert seen["interp"] == ["P3"]
        assert set(seen["shapes"]) == {len(DSI24_SCALP)}  # P3's row is back
    finally:
        nf.save()


# ---------------------------------------------------------------------------
# From the review of #56
# ---------------------------------------------------------------------------


def test_connect_to_lsl_rejects_a_device_that_is_not_a_string(tmp_path):
    from mne_rt import RTStream

    nf = RTStream("sub01", "01", str(tmp_path), montage=None)
    with pytest.raises(ValueError, match="device"):
        nf.connect_to_lsl(mock_lsl=True, device=1)


def _nf_with_info(tmp_path, names, types, bads=()):
    import mne

    from mne_rt import RTStream

    nf = RTStream("sub01", "01", str(tmp_path), montage=None)
    nf.rec_info = mne.create_info(names, 250.0, types)
    nf.rec_info["bads"] = list(bads)
    return nf


def test_acquired_channels_resolve_a_type_string(tmp_path):
    """picks="eeg" returned every good channel, while the stream returned only
    the EEG ones, so every channel-dependent step described the wrong rows."""
    nf = _nf_with_info(tmp_path, ["C3", "EOG1", "C4", "TRG"], ["eeg", "eog", "eeg", "stim"])
    nf.picks = "eeg"
    assert nf._acquired_ch_names() == ["C3", "C4"]
    assert nf._acquired_ch_names(use_picks=False) == ["C3", "EOG1", "C4", "TRG"]


def test_acquired_channels_keep_the_order_of_a_list(tmp_path):
    """get_data returns a list of picks in the order given; this used to return
    them in recording order, swapping the channels' names."""
    nf = _nf_with_info(tmp_path, ["C3", "Cz", "C4"], "eeg", bads=["Cz"])
    nf.picks = ["C4", "Cz", "C3", "C4"]
    assert nf._acquired_ch_names() == ["C4", "C3"]  # bads out, once each


def test_si_scale_reads_the_declared_units():
    import mne

    from mne_rt.rt_stream import _si_scale

    info = mne.create_info(["C3", "C4"], 250.0, "eeg")
    assert _si_scale(info) is None
    with info._unlock():
        info["chs"][0]["unit_mul"] = -6
    np.testing.assert_allclose(_si_scale(info), [1e-6, 1.0])


def test_read_window_converts_to_si_units(tmp_path):
    """µV streams were treated as V everywhere but in the raw viewer."""
    import mne

    from mne_rt import RTStream

    info = mne.create_info(["C3", "Cz", "C4"], 100.0, "eeg")
    data = np.ones((3, 500)) * 50.0  # 50 µV, sent as µV
    nf = RTStream("sub01", "01", str(tmp_path), montage=None)
    try:
        nf.connect_to_array(data, info, n_repeat=np.inf)
        nf._si_scale = np.array([1e-6, 1e-6, 1e-6])
        nf.rec_info["bads"] = ["Cz"]
        import time

        time.sleep(0.3)
        good, _ = nf._read_window(0.1)
        every, _ = nf._read_window(0.1, all_channels=True)
        assert good.shape[0] == 2 and every.shape[0] == 3
        np.testing.assert_allclose(good, 50e-6)
        np.testing.assert_allclose(every, 50e-6)
    finally:
        nf.save()


def test_find_signal_stream_leaves_marker_streams_out(monkeypatch):
    """A marker outlet next to the EEG made a connection without a name fail
    ("2 were found") or connect to the markers."""
    import mne_lsl.lsl

    from mne_rt.rt_stream import _find_signal_stream

    class Info:
        def __init__(self, name, stype, sfreq):
            self.name, self.stype, self.sfreq, self.source_id = name, stype, sfreq, name + "-id"

    found = [Info("DSI", "EEG", 300.0), Info("psychopy_markers", "Markers", 0.0)]
    monkeypatch.setattr(mne_lsl.lsl, "resolve_streams", lambda timeout: found)
    assert _find_signal_stream(15.0) == {"name": "DSI", "source_id": "DSI-id"}
    found.append(Info("Other", "EEG", 500.0))
    with pytest.raises(RuntimeError, match="'DSI' \\(EEG\\), 'Other' \\(EEG\\)"):
        _find_signal_stream(15.0)


def test_baseline_describes_every_good_channel_whatever_the_picks(tmp_path):
    """The baseline reads every good channel, but was wrapped in an Info cut to
    the `picks` of an earlier record_main."""
    import mne

    from mne_rt import RTStream

    montage = make_standard_montage("colin27_1020")
    info = mne.create_info(DSI24_SCALP, 250.0, "eeg")
    info.set_montage(montage)
    data = np.random.default_rng(1).standard_normal((len(DSI24_SCALP), 2500)) * 1e-6
    nf = RTStream("sub01", "01", str(tmp_path), montage="colin27_1020")
    try:
        nf.connect_to_array(data, info, n_repeat=np.inf)
        nf.picks = ["O1", "O2"]  # left by an earlier record_main
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        assert nf.raw_baseline.ch_names == DSI24_SCALP
    finally:
        nf.save()


def test_picks_naming_a_missing_or_only_bad_channel_raise(tmp_path):
    """Dropped silently, a typo ran the session on the rest, or on nothing."""
    nf = _nf_with_info(tmp_path, ["C3", "Cz", "C4"], "eeg", bads=["Cz"])
    nf.picks = ["C3", "c4"]
    with pytest.raises(ValueError, match="does not have"):
        nf._acquired_ch_names()
    nf.picks = ["Cz"]
    with pytest.raises(ValueError, match="marked bad"):
        nf._acquired_ch_names()


def test_get_data_si_scales_only_the_rows_it_returns(tmp_path):
    import mne

    from mne_rt import RTStream

    info = mne.create_info(["C3", "C4"], 100.0, "eeg")
    nf = RTStream("sub01", "01", str(tmp_path), montage=None)
    try:
        nf.connect_to_array(np.ones((2, 500)), info, n_repeat=np.inf)
        nf._si_scale = np.array([1e-6, 1.0])
        import time

        time.sleep(0.3)
        c4, _ = nf._get_data_si(0.1, picks=["C4"])
        c3, _ = nf._get_data_si(0.1, picks=["C3"])
        np.testing.assert_allclose(c4, 1.0)
        np.testing.assert_allclose(c3, 1e-6)
    finally:
        nf.save()


def test_find_signal_stream_waits_for_a_late_outlet(monkeypatch):
    """Capping discovery at one short scan failed when dsi2lsl started late."""
    import mne_lsl.lsl

    from mne_rt.rt_stream import _find_signal_stream

    class Info:
        name, stype, sfreq, source_id = "DSI", "EEG", 300.0, "DSI-id"

    scans = []

    def resolve(timeout):
        scans.append(timeout)
        return [Info()] if len(scans) >= 3 else []

    monkeypatch.setattr(mne_lsl.lsl, "resolve_streams", resolve)
    assert _find_signal_stream(15.0)["name"] == "DSI"
    assert len(scans) == 3


def test_baseline_is_cut_to_the_picks_for_the_modalities(tmp_path):
    """The baseline holds every good channel; ERD/ERS, laterality and the
    correctors compared or fitted it against windows of `picks` only."""
    import mne

    from mne_rt import RTStream

    montage = make_standard_montage("colin27_1020")
    info = mne.create_info(DSI24_SCALP, 250.0, "eeg")
    info.set_montage(montage)
    data = np.random.default_rng(2).standard_normal((len(DSI24_SCALP), 2500)) * 1e-6
    nf = RTStream("sub01", "01", str(tmp_path), montage="colin27_1020")
    try:
        nf.connect_to_array(data, info, n_repeat=np.inf)
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf.picks = ["O2", "O1"]
        cut = nf._baseline_data()
        full = nf.raw_baseline.get_data()
        names = nf.raw_baseline.ch_names
        np.testing.assert_array_equal(cut, full[[names.index("O2"), names.index("O1")]])
    finally:
        nf.save()


def test_save_raw_with_picks_writes_the_analysed_channels(tmp_path):
    """The saved windows hold `picks`, but were wrapped in the Info of every
    channel, so RawArray raised and no raw FIF was written."""
    import mne

    from mne_rt import RTStream

    montage = make_standard_montage("colin27_1020")
    info = mne.create_info(DSI24_SCALP, 250.0, "eeg")
    info.set_montage(montage)
    data = np.random.default_rng(3).standard_normal((len(DSI24_SCALP), 5000)) * 1e-6
    nf = RTStream("sub01", "01", str(tmp_path), montage="colin27_1020")
    try:
        nf.connect_to_array(data, info, n_repeat=np.inf)
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf.record_main(
            duration=2.0,
            modality="sensor_power",
            picks=["O1", "O2"],
            save_raw=True,
            show_raw_signal=False,
            show_nf_signal=False,
        )
        (fif,) = tmp_path.rglob("*task-neurofeedback*_eeg.fif")
        assert mne.io.read_raw_fif(fif, verbose=False).ch_names == ["O1", "O2"]
    finally:
        nf.save()


def test_record_main_topomap_with_a_type_string_picks_and_other_channel_types(
    tmp_path, monkeypatch
):
    """picks="eeg" on a stream with a stim channel and a bad one gave the
    topomap windows of the wrong size, raising in the Qt timer every 200 ms."""
    import mne

    import mne_rt.rt_stream as rs
    from mne_rt import RTStream

    names = DSI24_SCALP + ["TRG"]
    info = mne.create_info(names, 250.0, ["eeg"] * len(DSI24_SCALP) + ["stim"])
    info.set_montage(make_standard_montage("colin27_1020"))
    data = np.random.default_rng(4).standard_normal((len(names), 250 * 24)) * 1e-5
    shapes = []

    class SpyTopo(rs.TopomapPlot):
        def _update_topomaps(self, window):
            shapes.append(window.shape[0])
            super()._update_topomaps(window)

    monkeypatch.setattr(rs, "TopomapPlot", SpyTopo)
    nf = RTStream("sub01", "01", str(tmp_path), montage=None)
    nf.connect_to_array(data, info, n_repeat=np.inf)
    nf.rec_info["bads"] = ["P3"]
    try:
        nf.record_main(
            duration=4.0,
            winsize=1.0,
            modality="sensor_power",
            picks="eeg",
            show_raw_signal=False,
            show_nf_signal=False,
            show_topo=True,
            save_tsv=False,
        )
        assert shapes and set(shapes) == {len(names)}
    finally:
        nf.save()


@pytest.mark.parametrize("name", ["standard_1020", "colin27_1020"])
def test_either_name_of_a_renamed_montage_works(tmp_path, name):
    """MNE 1.13 renamed standard_1020 to colin27_1020: the old name is no
    longer a listed built-in there, and the new one does not exist before."""
    import mne

    from mne_rt import RTStream
    from mne_rt._montage import builtin_montage

    resolved = builtin_montage(name)
    assert resolved in mne.channels.get_builtin_montages()
    nf = RTStream("sub01", "01", str(tmp_path), montage=name)
    assert nf.montage == resolved
