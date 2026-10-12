# -*- coding: utf-8 -*-
"""
MODULE 2 — Depth / relative localization (owner: SU)

INTEGRATED (7/20): wraps SU's depth module (Depth Anything V2 Small),
delivered under modules/su_depth/. SU's model outputs a RELATIVE depth
level FAR / MIDDLE / NEAR for the person — not metric distance, and not
an (x, y) map position. The UI shows a depth indicator accordingly.

Contract (published to event_queue):
    {"depth_level": "FAR"|"MIDDLE"|"NEAR"|"NO_TARGET"|"UNAVAILABLE",
     "near_score": float,     # relative score in [0, 1], NOT meters
     "status": "OK"|"NO_TARGET"|"STALE_PACKET"|"ERROR"}

WIRED (7/28): in real mode this module consumes (frame + person bbox)
packets produced by the gesture module (SONG) through
pose_packet_queue — SU's intended pose->depth protocol. No person in
frame -> bbox=None -> adapter reports NO_TARGET, as designed.

Real depth is on demand, with a 20 s idle refresh by default. Conflicting
historical Nano timings were not remeasured in Phase C1.

Real-mode requirements (unverified Jetson path): torch + OpenCV + the
external Depth-Anything-V2 repo on PYTHONPATH and a trusted local
models/depth_anything_v2_vits.pth (excluded from the archive). If missing,
this standalone adapter reports UNAVAILABLE; it does not simulate depth.
Both workers report backend health separately from liveness; see docs/SETUP.md.
"""
import os
import time

from common.qutils import get_latest, ignore_sigint
from common.messages import make_event, SOURCE_DEPTH
from common.health import HealthReporter
from common import simscript

# Historical notes report ~6 s versus ~0.34 s per Nano inference; neither
# timing was reproduced in Phase C1. Preserve the original on-demand
# schedule to limit work between pose updates on constrained hardware.
# Otherwise refresh only every IDLE_PERIOD_S.
IDLE_PERIOD_S = 20.0
DEFAULT_INPUT_SIZE = 154  # historical main default; performance unverified

SIM_FAR_MIDDLE = 0.5142    # historical SU calibration; not revalidated here
SIM_MIDDLE_NEAR = 0.8364


def _sim_level(score):
    if score >= SIM_MIDDLE_NEAR:
        return "NEAR"
    if score >= SIM_FAR_MIDDLE:
        return "MIDDLE"
    return "FAR"


HERE = os.path.dirname(os.path.abspath(__file__))
DEPTH_MODEL_PATH = os.path.join(HERE, "..", "models",
                                "depth_anything_v2_vits.pth")


def _make_real_processor(input_size=DEFAULT_INPUT_SIZE):
    """Build SU's DepthProcessor + adapter, or (None, None, reason)."""
    try:
        from modules.su_depth import (DepthProcessor,
                                      DepthIntegrationAdapter)
        # The local integration expects models/ (team convention), so we
        # override SU's default checkpoints/ path via his model_path arg.
        processor = DepthProcessor(model_path=DEPTH_MODEL_PATH,
                                   input_size=input_size)
        adapter = DepthIntegrationAdapter(processor=processor,
                                          max_packet_age_ms=1000.0)
        return processor, adapter, None
    except Exception:
        return None, None, "depth_initialization_failed"


def _set_depth_result_health(health, result):
    """Keep depth availability distinct from an absent or stale target."""
    status = result.get("integration_status", "ERROR")
    if status == "STALE_PACKET":
        health.set("degraded", backend="depth_anything_v2", reason="stale_packet")
    elif status == "ERROR" or result.get("depth_level") == "UNAVAILABLE":
        health.set("degraded", backend="depth_anything_v2",
                   reason="inference_failed")
    else:
        # NO_TARGET is a valid observation, not a failed depth backend.
        health.set("ready", backend="depth_anything_v2")


def depth_process(pose_packet_queue, event_queue, stop_event, sim=False,
                  sim_state=None, depth_request=None,
                  input_size=DEFAULT_INPUT_SIZE, idle_period=IDLE_PERIOD_S,
                  gesture_ready=None):
    ignore_sigint()
    health = HealthReporter(SOURCE_DEPTH, event_queue)
    backend = "synthetic" if sim else "depth_anything_v2"
    health.set("simulated" if sim else "initializing", backend=backend)
    processor = adapter = None
    if not sim:
        # Never load at the same time as the pose model: two CUDA
        # contexts + two model loads at once will thrash a 4 GB Nano.
        if gesture_ready is not None:
            waited = 0.0
            while (gesture_ready.value < 0.5 and waited < 300.0
                   and not stop_event.is_set()):
                health.heartbeat()
                time.sleep(1.0)
                waited += 1.0
            if stop_event.is_set():
                return
            print("[depth] pose initialization wait complete; loading depth")
        print("[depth] loading Depth Anything V2 (input {}px)..."
              .format(input_size))
        processor, adapter, reason = _make_real_processor(input_size)
        if processor is not None:
            health.set("ready", backend=backend)
            print("[depth] depth model ready (first inference is slow)")
        if processor is None:
            health.set("unavailable", backend=backend,
                       reason="depth_initialization_failed")
            print("[depth] depth_initialization_failed -> unavailable")

    last_infer = 0.0
    warmed_up = False
    last_request_token = (depth_request.value
                          if depth_request is not None else 0.0)

    def emit(depth_level, near_score, status):
        event_queue.put(make_event(SOURCE_DEPTH, {
            "depth_level": depth_level,
            "near_score": float(near_score),
            "status": status,
        }))

    while not stop_event.is_set():
        now = time.time()
        health.heartbeat()

        if not sim and processor is None:
            # Real mode, model missing: report UNAVAILABLE instead of
            # publishing a fake depth level.
            if (now - last_infer) >= 2.0:
                last_infer = now
                emit("UNAVAILABLE", 0.0, "ERROR")
            time.sleep(0.2)
            continue

        if sim:
            # No packets needed in sim: the score comes straight from
            # the shared scenario, so the bar matches the video figure.
            time.sleep(0.05)
            if (now - last_infer) < 0.2:
                continue
            last_infer = now
            t_cycle, _ = simscript.sim_clock(sim_state)
            score = simscript.near_score(sim_state, t_cycle)
            emit(_sim_level(score), score, "OK")
            continue

        # ---- real mode: consume SONG's (frame + bbox) pose packets ---
        packet = get_latest(pose_packet_queue, timeout=0.5)
        if packet is None:
            continue

        # Run only when asked (a fall was just detected) or after a long
        # idle period. Historical inference timing is unverified here.
        requested = False
        if depth_request is not None:
            token = depth_request.value
            if token != last_request_token:
                last_request_token = token
                requested = True
        if not requested and (now - last_infer) < idle_period:
            continue
        last_infer = now
        try:
            if not warmed_up:
                # skip processor.warmup(): the first real inference pays
                # that cost anyway, no need to pay it twice.
                warmed_up = True
            result = adapter.process_packet(packet, now=time.monotonic())
            level = result.get("depth_level", "UNAVAILABLE")
            score = result.get("smoothed_near_score") or 0.0
            status = result.get("integration_status", "ERROR")
            emit(level, score, status)
            _set_depth_result_health(health, result)
        except Exception:
            if health.set("degraded", backend=backend,
                          reason="inference_failed"):
                print("[depth] inference_failed; retrying next packet")
            emit("UNAVAILABLE", 0.0, "ERROR")

    print("[depth] stopped cleanly")
