"""Wearable Sensing DSI support and the viewers' channel sets.

``dsi2lsl`` publishes channels named after their reference ("O1-A1/2-A2/2"),
declares trigger/aux inputs as EEG and streams microvolts.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

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

    def set_channel_types(self, mapping):
        self.types = mapping


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

    montage = mne.channels.make_standard_montage("colin27_1020")
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
    info.set_montage(mne.channels.make_standard_montage("colin27_1020"))
    data = np.random.default_rng(0).standard_normal((len(DSI24_SCALP), 250 * 24)) * 1e-5

    seen = {}

    class SpyTopo(rs.TopomapPlot):
        def push(self, window):
            seen.setdefault("shapes", []).append(window.shape[0])
            seen["interp"] = [self._info["ch_names"][i] for i in self._interp_picks]
            super().push(window)

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
