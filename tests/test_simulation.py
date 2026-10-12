"""Clock and bounded-queue behavior without wall-clock sleeps or sensors."""
import multiprocessing as mp
import queue

import pytest

from common import simscript
from common.messages import make_event, make_status
from common.qutils import drain, get_latest, put_latest
from modules.su_depth.integration import put_latest as put_latest_packet


@pytest.mark.parametrize("elapsed, cycle_time, cycle", [
    (-1.0, 0.0, 0), (0.0, 0.0, 0), (10.0, 10.0, 0),
    (13.0, 13.0, 0), (45.0, 0.0, 1), (103.0, 13.0, 2),
])
def test_scenario_clock_boundaries(elapsed, cycle_time, cycle):
    state = simscript.make_state(now=1000.0)
    assert simscript.sim_clock(state, now=1000.0 + elapsed) == (
        cycle_time, cycle)


def test_walking_fall_help_and_cycle_reset_share_one_clock():
    state = simscript.make_state(now=1000.0)
    expectations = [
        (0.0, False, False, 0.15),
        (5.0, False, False, 0.50),
        (9.999, False, False, 0.84993),
        (10.0, True, False, 0.90),
        (13.0, True, True, 0.90),
        (45.0, False, False, 0.15),
    ]
    for elapsed, fallen, helping, depth in expectations:
        t, _ = simscript.sim_clock(state, now=1000.0 + elapsed)
        assert simscript.is_fallen(state, t) is fallen
        assert simscript.is_helping(state, t) is helping
        assert simscript.phase(state, t) == (
            "fallen" if fallen else "walking")
        assert simscript.depth(state, t) == pytest.approx(depth)
        assert simscript.near_score(state, t) == pytest.approx(depth)
        assert simscript.screen_y(depth) == depth


def test_manual_fall_help_and_reset_override_scene_without_advancing_time():
    state = simscript.make_state(now=1000.0)
    t, _ = simscript.sim_clock(state, now=1001.0)
    state[simscript.I_FORCED_FALL] = 1.0
    assert simscript.is_fallen(state, t)
    assert simscript.depth(state, t) == 0.90
    assert not simscript.is_helping(state, t)
    state[simscript.I_FORCED_HELP] = 1.0
    assert simscript.is_helping(state, t)
    state[:] = simscript.make_state(now=1001.0)
    t, cycle = simscript.sim_clock(state, now=1001.0)
    assert (t, cycle) == (0.0, 0)
    assert not simscript.is_fallen(state, t)
    assert not simscript.is_helping(state, t)


def test_latest_frame_replaces_unconsumed_frame_without_backlog():
    frames = queue.Queue(maxsize=1)
    put_latest(frames, "old")
    put_latest(frames, "new")
    assert get_latest(frames, timeout=0.0) == "new"
    assert get_latest(frames, timeout=0.0) is None


def test_pose_queue_reports_dropped_packet_and_preserves_latest():
    packets = queue.Queue(maxsize=1)
    assert put_latest_packet(packets, {"frame_id": 1}) is False
    assert put_latest_packet(packets, {"frame_id": 2}) is True
    assert packets.get_nowait() == {"frame_id": 2}
    assert packets.empty()


def test_drain_preserves_event_order_and_respects_batch_limit():
    events = queue.Queue()
    messages = [make_status("gesture"),
                make_event("gesture", {"gesture": "lay_down"}),
                make_event("speech", {"keyword": "help"})]
    for message in messages:
        events.put_nowait(message)
    assert drain(events, max_items=2) == messages[:2]
    assert drain(events) == messages[2:]
    assert drain(events) == []


def _crash_video_worker(*args):
    raise RuntimeError("controlled smoke-test worker failure")


def _noncooperative_worker():
    # An OS wait deliberately ignores the pipeline's stop_event. The
    # shutdown helper must terminate and reap this child within its bound.
    mp.Event().wait()


def test_smoke_detects_failed_worker_and_reaps_all_children(monkeypatch):
    from tests import smoke_sim

    before = {child.pid for child in mp.active_children()}
    monkeypatch.setattr(smoke_sim, "video_capture_process", _crash_video_worker)
    with pytest.raises(AssertionError, match="Worker video exited early"):
        smoke_sim.main(duration_s=3.0, ctx=mp.get_context("spawn"))
    assert {child.pid for child in mp.active_children()} == before


def test_shutdown_timeout_terminates_and_reaps_noncooperative_worker():
    from tests.smoke_sim import _stop_workers

    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    events = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_noncooperative_worker, name="stuck")
    proc.start()
    try:
        assert _stop_workers([proc], stop, [events], timeout=0.05) == ["stuck"]
        assert stop.is_set()
        assert not proc.is_alive()
        assert proc.exitcode != 0
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
