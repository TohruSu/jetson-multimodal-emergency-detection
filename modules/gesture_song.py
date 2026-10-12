# -*- coding: utf-8 -*-
"""
MODULE 1 — Gesture / fall detection (owner: SONG)

INTEGRATED (7/28): wraps SONG's trt_pose fall detector, delivered as
modules/song/fall_detector.py (his file, unchanged). His code runs
pose estimation (trt_pose ResNet-18, TensorRT) plus a per-person
NORMAL -> FALLING -> FALLEN state machine (head-drop speed + bbox
aspect ratio + torso tilt). We reuse his classes as a library:
frames come from OUR capture queue, display goes to OUR UI, and his
confirmed-fall events become our contract events.

This module is ALSO the producer for SU's depth module: after each
pose inference it publishes a (frame + person bbox) packet into
pose_packet_queue — exactly the pose->depth protocol SU designed.

Contract (event_queue):
    {"gesture": "lay_down", "confidence": float,
     "region": "center-bottom",          # SONG's 3x3 frame region
     "center_norm": [x, y]}              # normalized body center

Real-mode requirements (unverified Jetson path): torch, torchvision,
OpenCV, Pillow, trt_pose, torch2trt and trusted local model files.
The wrapper prefers a separate *_trt_engine.pth; otherwise it expects
plain PyTorch weights at the historically named *_trt.pth path.
The absent historical checkpoint's contents were not verified.
See docs/MODELS.md and docs/SETUP.md for dependencies and failure behavior.
"""
import os
import time

from common.qutils import get_latest, put_latest, ignore_sigint
from common.messages import make_event, make_detections, SOURCE_GESTURE
from common.health import HealthReporter
from common import simscript

HERE = os.path.dirname(os.path.abspath(__file__))
POSE_JSON = os.path.join(HERE, "song", "human_pose.json")
MODELS_DIR = os.path.join(HERE, "..", "models")
# Historical integration notes describe "_trt.pth" as plain weights;
# the file is absent from the source archive, so this is unverified.
# Support either a separate locally built engine or those plain weights.
ENGINE_PATH = os.path.join(
    MODELS_DIR, "resnet18_baseline_att_224x224_A_epoch_249_trt_engine.pth")
WEIGHTS_PATH = os.path.join(
    MODELS_DIR, "resnet18_baseline_att_224x224_A_epoch_249_trt.pth")
MODEL_SIZE = 224  # matches the 224x224 model
PACKET_PERIOD_S = 0.5  # how often we hand a frame+bbox to the depth module

# SONG's recovery (FALLEN -> NORMAL) needs torso angle AND aspect ratio
# under threshold for 1.2 s. The angle needs shoulders AND hips visible:
# standing close to the webcam often hides the hips, so the box can stay
# red forever. We loosen the thresholds and add a safety auto-reset.
RECOVERY_ANGLE = 45.0      # SONG default 35.0
RECOVERY_RATIO = 1.05      # SONG default 0.95
RECOVERY_SECONDS = 1.00    # SONG default 1.20
FALLEN_AUTO_RESET_S = 25.0  # force a track back to NORMAL after this


def _tuned_config(base):
    cfg = dict(base)
    cfg["recovery_angle"] = RECOVERY_ANGLE
    cfg["recovery_ratio"] = RECOVERY_RATIO
    cfg["recovery_seconds"] = RECOVERY_SECONDS
    return cfg


def _make_real_detector():
    """Load SONG's components, or return (None, reason)."""
    try:
        import json
        import torch
        import trt_pose.coco
        from trt_pose.parse_objects import ParseObjects
        from trt_pose.draw_objects import DrawObjects
        from modules.song import fall_detector as fd

        with open(POSE_JSON, "r") as f:
            human_pose = json.load(f)
        names = [fd.canonical_name(n) for n in human_pose["keypoints"]]
        topology = trt_pose.coco.coco_category_to_topology(human_pose)

        if os.path.exists(ENGINE_PATH):
            print("[gesture] loading TensorRT engine...")
            from torch2trt import TRTModule
            model = TRTModule()
            model.load_state_dict(torch.load(ENGINE_PATH))
            model = model.cuda().eval()
            use_half = False
            backend = "trt_pose_tensorrt"
            print("[gesture] trt_pose ready (TensorRT)")
        else:
            backend = "trt_pose_pytorch"
            print("[gesture] no TensorRT engine found -> plain PyTorch "
                  "(throughput unverified). Optional local conversion: "
                  "python3 tools/build_trt_engine.py")
            import trt_pose.models
            num_parts = len(human_pose["keypoints"])
            num_links = len(human_pose["skeleton"])
            model = trt_pose.models.resnet18_baseline_att(
                num_parts, 2 * num_links).cuda().eval()
            model.load_state_dict(torch.load(WEIGHTS_PATH,
                                             map_location="cuda"))
            # Try the historical fp16 path; throughput is unverified.
            try:
                model = model.half()
                use_half = True
            except Exception:
                use_half = False
            print("[gesture] trt_pose ready (PyTorch{})".format(
                ", fp16" if use_half else ""))

        parse = ParseObjects(topology)
        draw = DrawObjects(topology)
        ctx = {
            "fd": fd,
            "torch": torch,
            "names": names,
            "model": model,
            "parse": parse,
            "draw": draw,
            "pre": fd.PosePreprocessor(torch.device("cuda")),
            "half": use_half,
            "backend": backend,
            "tracker": fd.SimpleTracker(dict(fd.DEFAULT_CONFIG)),
            "config": _tuned_config(fd.DEFAULT_CONFIG),
        }
        return ctx, None
    except Exception:
        return None, "pose_initialization_failed"


def _best_track_bbox(tracks):
    """Pick the person bbox for the depth module: prefer a FALLEN
    track, else the largest freshly-updated one. Returns bbox or None."""
    best = None
    best_key = None
    for t in tracks:
        if not getattr(t, "updated_this_frame", False):
            continue
        feats = getattr(t, "features", None)
        if not feats or feats.get("bbox") is None:
            continue
        x1, y1, x2, y2 = feats["bbox"]
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        fallen = 1 if t.state == "FALLEN" else 0
        key = (fallen, area)
        if best_key is None or key > best_key:
            best_key = key
            best = (feats["bbox"], t.track_id)
    return best


def gesture_process(frame_queue, event_queue, stop_event, sim=False,
                    sim_state=None, pose_packet_queue=None,
                    ui_queue=None, annotated_active=None,
                    reset_request=None, depth_request=None,
                    gesture_ready=None):
    ignore_sigint()
    health = HealthReporter(SOURCE_GESTURE, event_queue)
    backend = "synthetic" if sim else "trt_pose"
    health.set("simulated" if sim else "initializing", backend=backend)
    ctx = None
    if not sim:
        ctx, reason = _make_real_detector()
        if gesture_ready is not None:
            gesture_ready.value = 1.0   # depth may load now
        if ctx is None:
            health.set("unavailable", backend=backend,
                       reason="pose_initialization_failed")
            print("[gesture] pose_initialization_failed -> inactive")
        else:
            backend = ctx.get("backend", "trt_pose")
            health.set("ready", backend=backend)

    prev_fallen = False
    frame_id = 0
    last_packet_ts = 0.0
    last_reset_seen = 0.0
    fallen_since = None
    fps_count = 0
    fps_t0 = time.time()
    fps_value = 0.0

    def emit(payload):
        event_queue.put(make_event(SOURCE_GESTURE, payload))

    while not stop_event.is_set():
        frame = get_latest(frame_queue, timeout=0.5)

        now = time.time()
        health.heartbeat()

        if not sim and ctx is None:
            # Real mode with no model: stay INACTIVE. Never invent
            # detections — a demo must not show fake alerts.
            time.sleep(0.2)
            continue

        if sim:
            # Fire on the rising edge of "fallen" (clock or forced by F).
            t_cycle, _ = simscript.sim_clock(sim_state)
            fallen = simscript.is_fallen(sim_state, t_cycle)
            if fallen and not prev_fallen:
                emit({"gesture": "lay_down", "confidence": 0.91,
                      "region": "center-bottom"})
            prev_fallen = fallen
            continue

        if frame is None:
            continue

        # ---------------- real mode: SONG's pipeline ------------------
        try:
            import cv2

            # R key (or auto-reset) -> forget every track, back to NORMAL
            force_reset = False
            if reset_request is not None and reset_request.value != last_reset_seen:
                last_reset_seen = reset_request.value
                force_reset = True
            if fallen_since is not None and (now - fallen_since) > FALLEN_AUTO_RESET_S:
                print("[gesture] track stuck in FALLEN -> auto reset")
                force_reset = True
            if force_reset:
                ctx["tracker"] = ctx["fd"].SimpleTracker(ctx["config"])
                fallen_since = None
                prev_fallen = False
            fd = ctx["fd"]
            torch = ctx["torch"]
            h, w = frame.shape[:2]
            model_img = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
            with torch.no_grad():
                tensor = ctx["pre"](model_img)
                if ctx.get("half"):
                    tensor = tensor.half()
                cmap, paf = ctx["model"](tensor)
            # parse_objects runs on CPU float32
            counts, objects, peaks = ctx["parse"](
                cmap.detach().float().cpu(), paf.detach().float().cpu())
            detections = fd.extract_detections(
                counts, objects, peaks, ctx["names"], w, h,
                ctx["config"]["min_keypoints"])
            tracks, events = ctx["tracker"].update(detections, now, w, h)

            # confirmed falls -> our contract events
            for ev in events:
                pos = ev.get("position", {})
                cn = pos.get("center_norm", {})
                emit({"gesture": "lay_down", "confidence": 0.9,
                      "region": pos.get("region"),
                      "center_norm": [cn.get("x"), cn.get("y")]})
                # a fall is exactly when the depth level matters
                if depth_request is not None:
                    depth_request.value += 1.0

            # ---- detections for the UI (boxes only, not frames) -------
            # Shipping annotated frames would tie the video framerate to
            # pose inference (historical ~1 fps, unverified). Send boxes
            # instead and let the UI draw them over the live feed.
            fps_count += 1
            if now - fps_t0 >= 1.0:
                fps_value = fps_count / (now - fps_t0)
                fps_count = 0
                fps_t0 = now
            any_fallen = any(getattr(t, "state", "") == "FALLEN"
                             for t in tracks)
            if any_fallen and fallen_since is None:
                fallen_since = now
            elif not any_fallen:
                fallen_since = None

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

            # ---- pose packet for SU's depth module (throttled) --------
            # Depth runs on demand or after its idle period, so sending a
            # full frame after every pose inference wastes IPC bandwidth.
            if (pose_packet_queue is not None
                    and (now - last_packet_ts) >= PACKET_PERIOD_S):
                last_packet_ts = now
                from modules.su_depth import make_pose_depth_packet
                best = _best_track_bbox(tracks)
                bbox, track_id = (best if best is not None
                                  else (None, None))
                frame_id += 1
                packet = make_pose_depth_packet(
                    frame=frame,
                    frame_id=frame_id,
                    captured_at=time.monotonic(),
                    bbox=bbox,
                    pose_inferred_at=time.monotonic(),
                    pose_confidence=None,
                    track_id=track_id,
                )
                put_latest(pose_packet_queue, packet)
            health.set("ready", backend=backend)
        except Exception:
            if health.set("degraded", backend=backend,
                          reason="inference_failed"):
                print("[gesture] inference_failed; retrying next frame")

    print("[gesture] stopped cleanly")
