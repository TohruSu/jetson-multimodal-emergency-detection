# -*- coding: utf-8 -*-
"""
System entry point (owner: Martin).

Launches five simulated workers, or four real workers with merged GPU
pose/depth inference, then
the PyQt UI in the main process. A 20 Hz QTimer drains the event
queue -> fusion -> UI.

Usage:
    python3 main.py            # real webcam + mic, real models
    python3 main.py --sim      # everything simulated (no hardware)
    python3 main.py --cam 1    # different camera index

Expected in --sim (45 s scripted loop, all modules synchronized):
    t = 0-10 s : stick figure walks, relative depth indicator changes
    t = 10 s   : figure falls -> ALERT level 1 (orange)
    t = 13 s   : "help" -> ALERT level 2 (red, emergency sentence)
    t = 43 s   : 30 s timeout -> back to "System ready"

Demo keys: F/H force simulated fall/help in --sim; R resets any mode.
"""
import argparse
import multiprocessing as mp
import os
import sys
import signal
import time

from common.qutils import drain, get_latest
from fusion.state_machine import FusionStateMachine, STATE_IDLE

from capture.video_capture import video_capture_process
from capture.audio_capture import audio_capture_process
from modules.gesture_song import gesture_process
from modules.depth_su import depth_process
from modules.gpu_worker import gpu_worker_process
from modules.speech_huo import speech_process

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERT_LOG = os.path.join(BASE_DIR, "logs", "alerts.log")

UI_REFRESH_MS = 50  # 20 Hz


def _stop_processes(procs, stop_event, queues, timeout=2.0):
    """Stop only our workers, draining queues so feeder threads can exit."""
    stop_event.set()
    deadline = time.monotonic() + timeout
    while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
        for q in queues:
            drain(q, max_items=256)
        for p in procs:
            p.join(timeout=0.02)
    for p in procs:
        if p.is_alive():
            p.terminate()
        p.join(timeout=1.0)
        if p.is_alive():
            getattr(p, "kill", p.terminate)()
            p.join(timeout=1.0)
        if p.is_alive():
            raise RuntimeError("Worker {} could not be stopped".format(p.name))
    for q in queues:
        q.close()
        q.join_thread()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true",
                        help="full simulation mode (no hardware)")
    parser.add_argument("--cam", type=int, default=0,
                        help="webcam index (default 0)")
    parser.add_argument("--fps", type=float, default=None,
                        help="capture fps (lower it if the Nano lags)")
    parser.add_argument("--depth-size", type=int, default=154,
                        help="depth model input size (default 154; throughput unverified)")
    parser.add_argument("--depth-period", type=float, default=20.0,
                        help="seconds between idle depth refreshes")
    parser.add_argument("--mic", type=int, default=None,
                        help="input device index (see tools/check_mic.py)")
    parser.add_argument("--depth", action="store_true",
                        help="enable the hardware depth module "
                             "(compute intensive, off by default)")
    args = parser.parse_args(argv)

    # ---------------- Queues and stop flag ----------------------------
    frame_q_gesture = mp.Queue(maxsize=1)
    frame_q_ui = mp.Queue(maxsize=1)
    pose_packet_q = mp.Queue(maxsize=1)  # gesture -> depth (frame+bbox)
    audio_q = mp.Queue(maxsize=32)
    event_q = mp.Queue(maxsize=256)
    stop_event = mp.Event()
    # set to 1.0 by the gesture module when it publishes annotated frames
    annotated_active = mp.Value("d", 0.0)
    # bumped by the gesture module on a confirmed fall -> depth measures
    depth_request = mp.Value("d", 0.0)
    # bumped by the R key -> the gesture tracker forgets every track
    reset_request = mp.Value("d", 0.0)
    # depth waits for this before loading its own CUDA context
    gesture_ready = mp.Value("d", 0.0)

    # Shared sim state [t0, forced_fall, forced_help]: all simulated
    # modules read it, and the demo keys (F/H/R) drive it live.
    from common import simscript
    init = simscript.make_state(time.time() + 1.0)
    sim_state = mp.Array("d", init)

    # ---------------- Processes (start BEFORE Qt) ----------------------
    procs = [
        mp.Process(target=video_capture_process, name="video",
                   args=([frame_q_gesture],
                         event_q, stop_event, args.cam, args.sim, sim_state,
                         frame_q_ui, annotated_active, args.fps)),
        mp.Process(target=audio_capture_process, name="audio",
                   args=(audio_q, event_q, stop_event, args.sim, sim_state,
                         args.mic)),
    ]

    if args.sim:
        # sim mode: the two light modules, nothing loads on the GPU
        procs += [
            mp.Process(target=gesture_process, name="gesture",
                       args=(frame_q_gesture, event_q, stop_event, True,
                             sim_state, pose_packet_q, frame_q_ui,
                             annotated_active, reset_request,
                             depth_request, gesture_ready)),
            mp.Process(target=depth_process, name="depth",
                       args=(pose_packet_q, event_q, stop_event, True,
                             sim_state, depth_request, args.depth_size,
                             args.depth_period, gesture_ready)),
        ]
    else:
        # real mode: pose + depth share ONE process = one CUDA context
        procs += [
            mp.Process(target=gpu_worker_process, name="gpu",
                       args=(frame_q_gesture, event_q, stop_event,
                             reset_request, depth_request, args.depth,
                             args.depth_size, args.depth_period)),
        ]

    procs += [
        mp.Process(target=speech_process, name="speech",
                   args=(audio_q, event_q, stop_event, args.sim, sim_state)),
    ]
    started = []
    queues = [frame_q_gesture, frame_q_ui, pose_packet_q, audio_q, event_q]
    try:
        for p in procs:
            p.daemon = True
            p.start()
            started.append(p)
        print("[main] {} processes started (sim={})".format(len(procs), args.sim))

        # ---------------- PyQt UI (main process) ---------------------------
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
        from ui.main_window import MainWindow

        app = QApplication(sys.argv)
        window = MainWindow()
        window.resize(900, 560)
        window.show()

        os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)
        fusion = FusionStateMachine(log_path=ALERT_LOG)

        def tick():
            if fusion.tick():
                window.clear_alert()
            # 1) latest frame -> display
            frame = get_latest(frame_q_ui, timeout=0.0)
            if frame is not None:
                window.update_frame(frame)

            # 2) events (processes + demo keys) -> fusion + UI
            manual = window.manual_events
            window.manual_events = []
            for msg in manual:
                key = msg.get("demo_key")
                if key == "F" and args.sim:
                    sim_state[simscript.I_FORCED_FALL] = 1.0
                elif key == "H" and args.sim:
                    sim_state[simscript.I_FORCED_HELP] = 1.0
                elif key == "R":
                    fusion.reset()
                    window.clear_alert()
                    reset_request.value += 1.0   # clear stuck FALLEN boxes
                    if args.sim:
                        sim_state[simscript.I_T0] = time.time()
                        sim_state[simscript.I_FORCED_FALL] = 0.0
                        sim_state[simscript.I_FORCED_HELP] = 0.0
            for msg in drain(event_q):
                window.handle_message(msg)
                previous_state = fusion.state
                alert = fusion.update(msg)
                if previous_state != STATE_IDLE and fusion.state == STATE_IDLE:
                    window.clear_alert()
                if alert is not None:
                    print("[fusion] {}".format(alert["sentence"]))
                    window.show_alert(alert)

            # 3) alert timeout
            if fusion.tick():
                window.clear_alert()

            # 4) module status
            window.refresh_status()

        timer = QTimer()
        timer.timeout.connect(tick)
        timer.start(UI_REFRESH_MS)

        # Clean Ctrl+C from the terminal
        signal.signal(signal.SIGINT, lambda *a: app.quit())

        code = app.exec_()
        return code
    finally:
        print("[main] shutting down...")
        _stop_processes(started, stop_event, queues)
        print("[main] done")

if __name__ == "__main__":
    sys.exit(main())
