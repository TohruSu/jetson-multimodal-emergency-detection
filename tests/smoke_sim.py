# -*- coding: utf-8 -*-
"""
Headless integration smoke test (no display, no hardware).
Starts the original five worker functions in sim mode, listens for
16 s, and checks L1 (~10 s), then L2 (~13 s), and clean worker shutdown.

Usage: python3 -m tests.smoke_sim
The pure scenario tests use a virtual clock; this test intentionally
exercises real multiprocessing queues and the workers' normal pacing.
"""
import math
import multiprocessing as mp
import time

from common.messages import TYPE_EVENT, TYPE_STATUS
from common.qutils import drain
from common import simscript
from fusion.state_machine import FusionStateMachine
from capture.video_capture import video_capture_process
from capture.audio_capture import audio_capture_process
from modules.gesture_song import gesture_process
from modules.depth_su import depth_process
from modules.speech_huo import speech_process

DURATION_S = 16.0
WORKER_NAMES = {"video", "audio", "gesture", "depth", "speech"}


def _assert_workers_running(procs):
    for proc in procs:
        assert proc.is_alive(), "Worker {} exited early (exitcode={})".format(
            proc.name, proc.exitcode)


def _stop_workers(procs, stop_event, queues, timeout=3.0):
    """Drain pending feeder writes while joining, then reap any stuck child."""
    stop_event.set()
    deadline = time.monotonic() + timeout
    while any(proc.is_alive() for proc in procs) and time.monotonic() < deadline:
        for queue in queues:
            drain(queue)
        for proc in procs:
            proc.join(timeout=0.01)
    forced = []
    for proc in procs:
        if proc.is_alive():
            forced.append(proc.name)
            proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            # Modern synthetic/PC Python supports kill(); retain a safe
            # fallback for interpreters without it.
            getattr(proc, "kill", proc.terminate)()
            proc.join(timeout=1.0)
    for queue in queues:
        queue.close()
        queue.join_thread()
    return forced


def main(duration_s=DURATION_S, ctx=None):
    # Spawn verifies that worker imports do not inherit hardware libraries
    # from the parent, exercising fresh imports rather than inherited modules.
    ctx = ctx or mp.get_context("spawn")
    frame_q_gesture = ctx.Queue(maxsize=1)
    frame_q_ui = ctx.Queue(maxsize=1)
    pose_packet_q = ctx.Queue(maxsize=1)
    audio_q = ctx.Queue(maxsize=32)
    event_q = ctx.Queue(maxsize=256)
    queues = [event_q, frame_q_gesture, frame_q_ui, pose_packet_q, audio_q]
    stop_event = ctx.Event()
    sim_state = ctx.Array("d", simscript.make_state(time.time()))

    procs = [
        ctx.Process(target=video_capture_process, name="video",
                    args=([frame_q_gesture], event_q, stop_event, 0, True,
                          sim_state, frame_q_ui, None, None)),
        ctx.Process(target=audio_capture_process, name="audio",
                    args=(audio_q, event_q, stop_event, True, sim_state)),
        ctx.Process(target=gesture_process, name="gesture",
                    args=(frame_q_gesture, event_q, stop_event, True,
                          sim_state, pose_packet_q, None, None)),
        ctx.Process(target=depth_process, name="depth",
                    args=(pose_packet_q, event_q, stop_event, True, sim_state)),
        ctx.Process(target=speech_process, name="speech",
                    args=(audio_q, event_q, stop_event, True, sim_state)),
    ]
    started = []
    fusion = FusionStateMachine()
    alerts = []
    events = []
    sources_seen = set()
    heartbeats_seen = set()
    ui_frames_seen = 0
    t0 = time.monotonic()

    print("Listening for {:.0f} s (L1 alert expected ~10 s, "
          "L2 ~13 s)...".format(duration_s))
    try:
        for proc in procs:
            proc.daemon = True
            proc.start()
            started.append(proc)
        while time.monotonic() - t0 < duration_s:
            _assert_workers_running(started)
            for msg in drain(event_q):
                assert isinstance(msg, dict), "Invalid event queue message"
                assert isinstance(msg.get("payload"), dict), "Missing payload"
                assert isinstance(msg.get("ts"), (int, float))
                assert math.isfinite(msg["ts"]), "Non-finite message timestamp"
                source = msg.get("source")
                sources_seen.add(source)
                if msg.get("type") == TYPE_STATUS:
                    assert msg["payload"].get("state") == "alive"
                    worker = (msg["payload"].get("module")
                              if source == "system" else source)
                    heartbeats_seen.add(worker)
                    assert msg["payload"].get("health") == "simulated", (
                        "Synthetic worker did not declare simulation: {}".format(msg))
                    assert msg["payload"].get("backend"), "Missing backend health context"
                elif msg.get("type") == TYPE_EVENT:
                    events.append(msg)
                else:
                    raise AssertionError("Unexpected synthetic message type")
                alert = fusion.update(msg)
                if alert:
                    alerts.append(alert)
                    print("  t={:5.1f}s  L{}  {}".format(
                        time.monotonic() - t0, alert["level"], alert["sentence"]))
            # Validate the actual video queue in addition to worker heartbeats.
            for frame in drain(frame_q_ui, max_items=5):
                assert frame.shape == (480, 640, 3)
                assert frame.dtype.name == "uint8"
                ui_frames_seen += 1
            fusion.tick()
            time.sleep(0.05)
        _assert_workers_running(started)
    finally:
        forced = _stop_workers(started, stop_event, queues)

    assert not forced, "Workers required termination: {}".format(forced)
    for proc in started:
        assert not proc.is_alive(), "Worker {} leaked".format(proc.name)
        assert proc.exitcode == 0, "Worker {} failed (exitcode={})".format(
            proc.name, proc.exitcode)
    assert heartbeats_seen == WORKER_NAMES, "Missing heartbeats: {}".format(
        WORKER_NAMES - heartbeats_seen)
    assert sources_seen == {"system", "gesture", "depth", "speech"}
    assert ui_frames_seen > 0, "No synthetic video reached the UI queue"
    assert [alert["level"] for alert in alerts] == [1, 2], (
        "Expected one L1 followed by one L2, got {}".format(alerts))
    assert all(alert["depth_level"] in {"FAR", "MIDDLE", "NEAR"}
               for alert in alerts), "Alert has no usable relative depth"
    gesture = [msg for msg in events if msg["source"] == "gesture"]
    speech = [msg for msg in events if msg["source"] == "speech"]
    assert len(gesture) == len(speech) == 1, "Repeated/missing scripted edges"
    assert gesture[0]["payload"]["gesture"] == "lay_down"
    assert speech[0]["payload"]["keyword"] == "help"
    assert events.index(gesture[0]) < events.index(speech[0])
    assert 0.0 <= speech[0]["ts"] - gesture[0]["ts"] <= 10.0
    depth_levels = {msg["payload"]["depth_level"]
                    for msg in events if msg["source"] == "depth"}
    assert depth_levels == {"FAR", "MIDDLE", "NEAR"}, depth_levels
    print("Sources seen: {}".format(sorted(sources_seen)))
    print("Worker exit codes: {}".format(
        {proc.name: proc.exitcode for proc in started}))
    print("SMOKE TEST: OK — synthetic five-process pipeline validated.")
    return {"alerts": alerts, "sources": sources_seen,
            "heartbeats": heartbeats_seen, "ui_frames": ui_frames_seen,
            "exitcodes": {proc.name: proc.exitcode for proc in started}}


if __name__ == "__main__":
    main()
