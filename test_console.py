# -*- coding: utf-8 -*-
"""
Test console (SANS interface graphique) — a lancer en SSH simple.

    python3 test_console.py            # 60 s
    python3 test_console.py 120        # 120 s
    python3 test_console.py 120 --depth  # avec la profondeur (lourd)

Affiche en direct :
  - le nombre de personnes detectees et leur etat (NORMAL/FALLING/FALLEN)
  - la cadence reelle de l'inference pose
  - les alertes L1 / L2 quand la fusion les declenche

A placer a la racine du projet (a cote de main.py).
"""
from __future__ import print_function

import multiprocessing as mp
import sys
import time

from common.qutils import drain
from common.messages import TYPE_DETECTIONS, TYPE_EVENT, TYPE_STATUS
from fusion.state_machine import FusionStateMachine
from capture.video_capture import video_capture_process
from capture.audio_capture import audio_capture_process
from modules.gpu_worker import gpu_worker_process
from modules.speech_huo import speech_process

FPS = 8


def main():
    # Keep command-line parsing inside the entry point: pytest and spawn
    # import this module with their own argv.
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    use_depth = "--depth" in sys.argv
    frame_q = mp.Queue(maxsize=1)
    ui_q = mp.Queue(maxsize=1)
    audio_q = mp.Queue(maxsize=32)
    event_q = mp.Queue(maxsize=256)
    stop = mp.Event()
    depth_req = mp.Value("d", 0.0)

    procs = [
        mp.Process(target=video_capture_process, name="video",
                   args=([frame_q], event_q, stop, 0, False, None,
                         ui_q, None, FPS)),
        mp.Process(target=audio_capture_process, name="audio",
                   args=(audio_q, event_q, stop, False, None)),
        # pose + depth in ONE process = one CUDA context
        mp.Process(target=gpu_worker_process, name="gpu",
                   args=(frame_q, event_q, stop, None, depth_req,
                         use_depth, 154, 20.0)),
        mp.Process(target=speech_process, name="speech",
                   args=(audio_q, event_q, stop, False, None)),
    ]
    for p in procs:
        p.daemon = True
        p.start()

    fusion = FusionStateMachine()
    print("")
    print("=== chargement des modeles, duree non verifiee ===")
    print("Ensuite : bouge devant la camera, puis simule une chute,")
    print("puis crie 'help' dans les 10 s qui suivent.")
    print("")

    t0 = time.time()
    last_print = 0.0
    depth_level = "?"
    try:
        while time.time() - t0 < duration:
            for msg in drain(event_q):
                t = time.time() - t0
                mtype = msg.get("type")
                payload = msg.get("payload", {})

                if mtype == TYPE_DETECTIONS:
                    tracks = payload.get("tracks", [])
                    if t - last_print > 1.0:
                        last_print = t
                        states = [x.get("state") for x in tracks]
                        print("  %5.1fs  pose %.1f fps | %d personne(s) %s "
                              "| depth %s" % (t, payload.get("fps", 0.0),
                                              len(tracks), states,
                                              depth_level))
                elif mtype == TYPE_EVENT:
                    src = msg.get("source")
                    if src == "depth":
                        depth_level = payload.get("depth_level", "?")
                    elif src == "gesture":
                        print("  %5.1fs  >>> CHUTE detectee (%s)"
                              % (t, payload.get("region")))
                    elif src == "speech":
                        print("  %5.1fs  >>> MOT-CLE '%s' (%.2f)"
                              % (t, payload.get("keyword"),
                                 payload.get("confidence", 0)))
                    alert = fusion.update(msg)
                    if alert:
                        print("  %5.1fs  *** ALERTE L%d : %s"
                              % (t, alert["level"], alert["sentence"]))
                elif mtype == TYPE_STATUS:
                    pass
            drain(ui_q, max_items=3)   # ne pas bloquer la capture
            fusion.tick()
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\ninterrompu")

    print("\narret...")
    stop.set()
    for p in procs:
        p.join(timeout=3.0)
    print("termine")


if __name__ == "__main__":
    main()
