# -*- coding: utf-8 -*-
"""
Video capture process (OpenCV).
One source (USB webcam) -> N "latest frame" queues (one per consumer).
Sim mode: draws a synthetic scene — a stick figure walking around the
room, falling at t=10 s (matching the scripted gesture/speech events).
"""
import time

import numpy as np

from common.qutils import put_latest, ignore_sigint
from common.messages import SOURCE_SYSTEM
from common.health import HealthReporter
from common import simscript

TARGET_FPS = 12.0     # real-mode default; lower it if the Nano struggles
FRAME_W = 640
FRAME_H = 480
RETRY_PERIOD_S = 2.0


def _draw_sim_frame(sim_state):
    """
    Synthetic camera view. Depth is the single driver:
      far  -> person near the TOP,    small
      near -> person near the BOTTOM, large
    so vertical position, size and the depth score always agree.
    """
    import math
    t_cycle, _ = simscript.sim_clock(sim_state)
    d = simscript.depth(sim_state, t_cycle)          # 0 far .. 1 near
    fallen = simscript.is_fallen(sim_state, t_cycle)

    # vertical position from depth; slight horizontal sway while walking
    cy_frac = 0.18 + 0.64 * simscript.screen_y(d)    # top->bottom
    sway = 0.0 if fallen else 0.05 * math.sin(t_cycle * 2.2)
    px = int((0.5 + sway) * FRAME_W)
    py = int(cy_frac * FRAME_H)
    scale = 0.5 + 1.0 * d                            # far small, near large

    frame = np.full((FRAME_H, FRAME_W, 3), 45, dtype=np.uint8)
    try:
        import cv2
    except ImportError:
        color = (60, 60, 230) if fallen else (90, 200, 120)
        bl = int(24 * scale)
        if fallen:
            frame[py:py + bl, max(0, px - 2 * bl):px + 2 * bl] = color
        else:
            frame[max(0, py - 3 * bl):py + bl, max(0, px - bl):px + bl] = color
        return frame

    # floor grid: horizontal lines closer together near the top,
    # giving a simple sense of depth/perspective.
    for i in range(1, 9):
        gy = int(FRAME_H * (0.15 + 0.85 * (i / 9.0) ** 1.6))
        cv2.line(frame, (0, gy), (FRAME_W, gy), (65, 65, 65), 1)

    body = int(70 * scale)
    head_r = max(4, int(14 * scale))
    color = (60, 60, 230) if fallen else (90, 200, 120)  # BGR

    if not fallen:
        # standing figure, feet at py
        cv2.circle(frame, (px, py - body), head_r, color, -1)
        cv2.line(frame, (px, py - body + head_r), (px, py), color, 3)
        cv2.line(frame, (px, py), (px - body // 3, py + body // 2), color, 3)
        cv2.line(frame, (px, py), (px + body // 3, py + body // 2), color, 3)
        cv2.line(frame, (px, py - body // 2),
                 (px - body // 3, py - body // 4), color, 3)
        cv2.line(frame, (px, py - body // 2),
                 (px + body // 3, py - body // 4), color, 3)
        x1, y1 = px - body // 2, py - body - head_r
        x2, y2 = px + body // 2, py + body // 2
        label = "pose: walking"
        box_color = (90, 200, 120)
    else:
        # lying horizontally on the floor
        cv2.circle(frame, (px - body, py), head_r, color, -1)
        cv2.line(frame, (px - body + head_r, py), (px + body // 2, py),
                 color, 4)
        x1, y1 = px - body - head_r, py - head_r - 4
        x2, y2 = px + body // 2 + 6, py + head_r + 4
        label = "pose: LAY_DOWN (0.91)"
        box_color = (60, 60, 230)

    cv2.putText(frame, "SIMULATION  t={:4.1f}s".format(t_cycle), (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # ---- overlays mimicking the AI modules' output -------------------
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    if d >= simscript.DEPTH_FALL - 0.05:
        level = "NEAR"
    elif d >= 0.5:
        level = "MIDDLE"
    else:
        level = "FAR"
    cv2.putText(frame, "depth: {} ({:.2f})".format(level, d),
                (x1, min(FRAME_H - 12, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 210, 120), 2)

    if fallen and simscript.is_helping(sim_state, t_cycle):
        cv2.putText(frame, 'mic: keyword "HELP" (0.88)', (15, FRAME_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 230), 2)
    else:
        cv2.putText(frame, "mic: listening...", (15, FRAME_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)
    return frame

def _release_camera(cap):
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def video_capture_process(frame_queues, event_queue, stop_event,
                          cam_index=0, sim=False, sim_state=None,
                          ui_queue=None, annotated_active=None, fps=None):
    ignore_sigint()
    """
    frame_queues: list of multiprocessing.Queue(maxsize=1)
                  (gesture, depth, ui — each gets the latest frame)
    """
    cap = None
    backend = "synthetic" if sim else "opencv"
    health = HealthReporter(SOURCE_SYSTEM, event_queue, module="video",
                            clock=time.monotonic)
    health.set("simulated" if sim else "initializing", backend=backend)
    period = 1.0 / float(fps or TARGET_FPS)
    retry_at = 0.0
    had_frame = False

    def failed(state, reason):
        if health.set(state, backend=backend, reason=reason):
            print("[video] {} ({})".format(state, reason))

    try:
        while not stop_event.is_set():
            loop_start = time.monotonic()
            frame = None

            if sim:
                frame = _draw_sim_frame(sim_state)
            else:
                if cap is None and loop_start >= retry_at:
                    reason = "opencv_unavailable"
                    try:
                        import cv2
                        reason = "camera_open_failed"
                        cap = cv2.VideoCapture(cam_index)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                        if not cap.isOpened():
                            raise RuntimeError("camera did not open")
                    except Exception:
                        _release_camera(cap)
                        cap = None
                        retry_at = time.monotonic() + RETRY_PERIOD_S
                        failed("unavailable", reason)

                if cap is not None:
                    try:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            raise RuntimeError("camera did not return a frame")
                    except Exception:
                        # A read failure contributes no observation to pose/depth.
                        # Reopening also handles unplugged/reconnected devices.
                        frame = None
                        _release_camera(cap)
                        cap = None
                        retry_at = time.monotonic() + RETRY_PERIOD_S
                        failed("degraded" if had_frame else "unavailable",
                               "camera_read_failed")
                    else:
                        had_frame = True
                        if health.set("ready", backend=backend):
                            print("[video] camera {} ready".format(cam_index))

            if frame is not None:
                for q in frame_queues:
                    put_latest(q, frame)

                # Missing hardware leaves the UI's last genuine frame intact;
                # its video health label describes the failed live capture.
                if ui_queue is not None:
                    put_latest(ui_queue, frame)

            health.heartbeat()
            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        _release_camera(cap)
    print("[video] stopped cleanly")
