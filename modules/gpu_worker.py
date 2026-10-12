# -*- coding: utf-8 -*-
"""
MERGED GPU WORKER (real mode only) — pose + depth in ONE process.

Why this exists
---------------
Historical team notes report high CUDA context/allocator memory use on
a 4 GB Jetson Nano and instability with separate pose/depth processes.
Those estimates were not remeasured in Phase C1. This worker retains
their memory-saving design: both models share one process and run
sequentially here:

    frame -> pose (every frame) -> boxes/fall events
                               -> depth (ON DEMAND only) -> depth level

Contract unchanged: this worker publishes exactly the same messages as
the separate gesture/depth modules (SOURCE_GESTURE events + detections,
SOURCE_DEPTH events, heartbeats for both), so fusion and UI don't care.

Sim mode still uses the two separate light modules — nothing to gain
here when no model is loaded.
"""
import time

from common.qutils import get_latest, ignore_sigint
from common.messages import (make_event, make_detections,
                             SOURCE_GESTURE, SOURCE_DEPTH)
from common.health import HealthReporter

from modules.gesture_song import (_make_real_detector, _best_track_bbox,
                                  MODEL_SIZE)
from modules.depth_su import _make_real_processor, _set_depth_result_health

FALLEN_AUTO_RESET_S = 25.0   # never leave a box stuck red on screen


def gpu_worker_process(frame_queue, event_queue, stop_event,
                       reset_request=None, depth_request=None,
                       use_depth=False, depth_input_size=154,
                       depth_idle_period=20.0):
    ignore_sigint()
    pose_health = HealthReporter(SOURCE_GESTURE, event_queue)
    depth_health = HealthReporter(SOURCE_DEPTH, event_queue)
    pose_backend = "trt_pose"
    pose_health.set("initializing", backend=pose_backend)
    depth_health.set("initializing" if use_depth else "disabled",
                     backend="depth_anything_v2" if use_depth else "none")

    # ---------------- load pose first ---------------------------------
    ctx, reason = _make_real_detector()
    if ctx is None:
        pose_health.set("unavailable", backend=pose_backend,
                        reason="pose_initialization_failed")
        print("[gpu] pose_initialization_failed -> worker idle")
    else:
        pose_backend = ctx.get("backend", "trt_pose")
        pose_health.set("ready", backend=pose_backend)

    # ---------------- then depth, same CUDA context -------------------
    processor = adapter = None
    if use_depth and ctx is not None:
        print("[gpu] loading depth model (input {}px)..."
              .format(depth_input_size))
        processor, adapter, dreason = _make_real_processor(depth_input_size)
        if processor is None:
            depth_health.set("unavailable", backend="depth_anything_v2",
                             reason="depth_initialization_failed")
            print("[gpu] depth_initialization_failed")
        else:
            depth_health.set("ready", backend="depth_anything_v2")
            print("[gpu] depth ready (first inference is slow)")
    elif not use_depth:
        print("[gpu] depth disabled (--depth to enable)")
    else:
        depth_health.set("unavailable", backend="depth_anything_v2",
                         reason="pose_unavailable")

    torch = ctx.get("torch") if ctx is not None else None
    last_depth = 0.0
    last_reset_seen = reset_request.value if reset_request is not None else 0.0
    last_depth_token = (depth_request.value
                        if depth_request is not None else 0.0)
    fallen_since = None
    fps_count = 0
    fps_t0 = time.time()
    fps_value = 0.0

    while not stop_event.is_set():
        frame = get_latest(frame_queue, timeout=0.5)

        now = time.time()
        pose_health.heartbeat()
        depth_health.heartbeat()

        if ctx is None or frame is None:
            continue

        try:
            import cv2
            fd = ctx["fd"]

            # ---- manual / automatic tracker reset --------------------
            force_reset = False
            if (reset_request is not None
                    and reset_request.value != last_reset_seen):
                last_reset_seen = reset_request.value
                force_reset = True
            if (fallen_since is not None
                    and (now - fallen_since) > FALLEN_AUTO_RESET_S):
                print("[gpu] track stuck in FALLEN -> auto reset")
                force_reset = True
            if force_reset:
                ctx["tracker"] = fd.SimpleTracker(ctx["config"])
                fallen_since = None

            # ---- pose inference -------------------------------------
            h, w = frame.shape[:2]
            model_img = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
            with torch.no_grad():
                tensor = ctx["pre"](model_img)
                if ctx.get("half"):
                    tensor = tensor.half()
                cmap, paf = ctx["model"](tensor)
            counts, objects, peaks = ctx["parse"](
                cmap.detach().float().cpu(), paf.detach().float().cpu())
            detections = fd.extract_detections(
                counts, objects, peaks, ctx["names"], w, h,
                ctx["config"]["min_keypoints"])
            tracks, events = ctx["tracker"].update(detections, now, w, h)

            # ---- fall events ----------------------------------------
            fall_now = False
            for ev in events:
                pos = ev.get("position", {})
                cn = pos.get("center_norm", {})
                event_queue.put(make_event(SOURCE_GESTURE, {
                    "gesture": "lay_down", "confidence": 0.9,
                    "region": pos.get("region"),
                    "center_norm": [cn.get("x"), cn.get("y")]}))
                fall_now = True

            if any(t.state == "FALLEN" for t in tracks):
                if fallen_since is None:
                    fallen_since = now
            else:
                fallen_since = None

            # ---- boxes for the UI (tiny payload, never frames) ------
            fps_count += 1
            if now - fps_t0 >= 1.0:
                fps_value = fps_count / (now - fps_t0)
                fps_count = 0
                fps_t0 = now
            box_list = []
            for tr in tracks:
                feats = getattr(tr, "features", None)
                if not feats or feats.get("bbox") is None:
                    continue
                x1, y1, x2, y2 = feats["bbox"]
                box_list.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "state": tr.state,
                    "track_id": tr.track_id,
                })
            event_queue.put(make_detections(SOURCE_GESTURE, box_list,
                                            fps_value))
            pose_health.set("ready", backend=pose_backend)

            # ---- depth, inline and ON DEMAND ------------------------
            if processor is not None:
                requested = fall_now
                if depth_request is not None:
                    token = depth_request.value
                    if token != last_depth_token:
                        last_depth_token = token
                        requested = True
                if requested or (now - last_depth) >= depth_idle_period:
                    last_depth = now
                    try:
                        best = _best_track_bbox(tracks)
                        bbox, track_id = (best if best is not None
                                          else (None, None))
                        from modules.su_depth import make_pose_depth_packet
                        packet = make_pose_depth_packet(
                            frame=frame, frame_id=int(now * 10) % 100000,
                            captured_at=time.monotonic(), bbox=bbox,
                            pose_inferred_at=time.monotonic(),
                            pose_confidence=None, track_id=track_id)
                        result = adapter.process_packet(packet,
                                                        now=time.monotonic())
                        event_queue.put(make_event(SOURCE_DEPTH, {
                            "depth_level": result.get("depth_level",
                                                      "UNAVAILABLE"),
                            "near_score": float(
                                result.get("smoothed_near_score") or 0.0),
                            "status": result.get("integration_status", "ERROR"),
                        }))
                        _set_depth_result_health(depth_health, result)
                        # give the memory back between measurements
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        if depth_health.set("degraded", backend="depth_anything_v2",
                                            reason="inference_failed"):
                            print("[gpu] depth inference_failed; retrying next request")
                        event_queue.put(make_event(SOURCE_DEPTH, {
                            "depth_level": "UNAVAILABLE", "near_score": 0.0,
                            "status": "ERROR"}))
        except Exception:
            if pose_health.set("degraded", backend=pose_backend,
                               reason="inference_failed"):
                print("[gpu] pose inference_failed; retrying next frame")
            time.sleep(0.2)

    print("[gpu] stopped cleanly")
