"""Multi-block sessions (run_blocks)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from mne_rt.tests._helpers import _HEADLESS, _failing_after, _make_array_data, _make_rt_stream

pytestmark = pytest.mark.usefixtures("fast_clock")


def test_run_blocks_accepts_one_shot_iterables_as_blocks(tmp_path, array_info):
    """Reading the last block's ``disconnect`` up front used to consume it, so
    that block ran with none of its settings."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        block = {"duration": 1.0, "modality": "band_ratio", **_HEADLESS}
        nf.run_blocks([iter(block.items())], rest_duration=0)
        assert list(nf.nf_data) == ["band_ratio"]
        # A block dict() cannot read fails before anything runs, and leaves the
        # previous call's results in place.
        with pytest.raises(TypeError):
            nf.run_blocks([block, 42], rest_duration=0)
        assert len(nf.block_results) == 1
    finally:
        nf.save()


def test_run_blocks_disconnects_when_an_early_block_fails(tmp_path, array_info):
    """Only the last block disconnects normally; a failure before it must not
    leave the stream up once run_blocks has raised."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf._band_ratio = _failing_after(0, RuntimeError("block 1 failed"))
        blocks = [
            {"duration": 2.0, "modality": "band_ratio", **_HEADLESS},
            {"duration": 2.0, "modality": "sensor_power", **_HEADLESS},
        ]
        with pytest.raises(RuntimeError, match="block 1 failed"):
            nf.run_blocks(blocks, rest_duration=0)
        assert nf.block_results == []
        assert not nf.stream.connected
    finally:
        nf.save()


def test_run_blocks_failing_cleanup_does_not_hide_the_block_error(
    tmp_path, array_info, monkeypatch
):
    """The disconnect after a failed block must not replace the block's error,
    and the last block need not be a Mapping to read its ``disconnect``."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf._band_ratio = _failing_after(0, RuntimeError("block 1 failed"))

        def broken_disconnect(**_):
            raise OSError("disconnect failed")

        monkeypatch.setattr(nf, "_disconnect", broken_disconnect)
        blocks = [
            list({"duration": 2.0, "modality": "band_ratio", **_HEADLESS}.items()),
            list({"duration": 2.0, "modality": "sensor_power", **_HEADLESS}.items()),
        ]
        with pytest.raises(RuntimeError, match="block 1 failed"):
            nf.run_blocks(blocks, rest_duration=0)
    finally:
        monkeypatch.undo()
        nf.save()


def test_run_blocks_disconnects_when_the_last_block_fails_before_saving(tmp_path, array_info):
    """A last block that raises before its own save never reaches its
    disconnect, so run_blocks has to do it."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        with pytest.raises(ValueError):
            nf.run_blocks(
                [{"duration": 2.0, "modality": "not_a_modality", **_HEADLESS}],
                rest_duration=0,
            )
        assert not nf.stream.connected
    finally:
        nf.save()


def test_run_blocks_keeps_finished_blocks_when_one_fails(tmp_path, array_info):
    """A failed block now raises, so run_blocks must not lose the blocks that
    finished: it used to set block_results only after the last block."""
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    try:
        nf.connect_to_array(
            _make_array_data(array_info, duration=20.0), array_info, n_repeat=np.inf
        )
        nf.record_baseline(baseline_duration=1.0, winsize=0.5)
        nf._band_ratio = _failing_after(0, RuntimeError("block 2 failed"))
        blocks = [
            {"duration": 2.0, "modality": "sensor_power", **_HEADLESS},
            {"duration": 2.0, "modality": "band_ratio", **_HEADLESS},
        ]
        with pytest.raises(RuntimeError, match="block 2 failed"):
            nf.run_blocks(blocks, rest_duration=0)
        assert [b["run"] for b in nf.block_results] == [1]
        assert len(nf.block_results[0]["nf_data"]["sensor_power"]) > 0
        assert len(nf.block_nf_data) == 1
    finally:
        nf.save()


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


def test_run_blocks_rests_between_blocks_not_after_the_last(tmp_path, array_info, monkeypatch):
    from mne_rt import rt_stream

    rests = []
    monkeypatch.setattr(rt_stream, "_sleep", rests.append)
    nf = _make_rt_stream(tmp_path, montage="easycap-M1")
    nf.connect_to_array(_make_array_data(array_info, duration=40.0), array_info, n_repeat=np.inf)
    block = dict(duration=2.0, winsize=1.0, modality="sensor_power", **_HEADLESS)
    try:
        nf.run_blocks(blocks=[block, dict(block), dict(block)], rest_duration=5.0)
    finally:
        nf.save()

    assert rests == [5.0, 5.0]
