# -*- coding: utf-8 -*-
"""
Diagnostic micro — a lancer sur le Jetson :

    python3 tools/check_mic.py            # liste + teste le micro par defaut
    python3 tools/check_mic.py 2          # teste le peripherique numero 2

Affiche le niveau sonore (RMS) capte pendant 4 s. Parle/crie "help"
pendant le test.

  RMS < 0.005  -> quasi silence : mauvais peripherique, ou micro coupe
  RMS > 0.02   -> bon niveau, la garde d'energie laissera passer
"""
from __future__ import print_function

import sys
import time

import numpy as np

RATE = 16000
CHUNK = 4000
SECONDS = 4.0


def main():
    try:
        import pyaudio
    except ImportError:
        print("pyaudio non installe: pip3 install --user pyaudio")
        return 1

    pa = pyaudio.PyAudio()

    print("\n--- peripheriques d'ENTREE disponibles ---")
    default_in = None
    try:
        default_in = pa.get_default_input_device_info()["index"]
    except Exception:
        pass
    inputs = []
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
        except Exception as exc:
            print("  [{}] (illisible: {})".format(i, exc))
            continue
        if int(info.get("maxInputChannels", 0)) > 0:
            inputs.append(i)
            mark = "  <-- defaut" if i == default_in else ""
            print("  [{}] {} ({} ch, {:.0f} Hz){}".format(
                i, info.get("name"), info.get("maxInputChannels"),
                info.get("defaultSampleRate"), mark))
    if not inputs:
        print("  (aucun peripherique d'entree vu par PyAudio)")
        print("  -> utilise plutot : arecord -l")

    index = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if index is not None and index not in inputs:
        print("\nATTENTION: [{}] n'est pas une entree valide.".format(index))
    print("\n--- test de capture {} ---".format(
        "peripherique {}".format(index) if index is not None
        else "peripherique par defaut"))
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                         input=True, frames_per_buffer=CHUNK,
                         input_device_index=index)
    except Exception as exc:
        print("ERREUR ouverture: {}".format(exc))
        print("-> essaie un autre numero dans la liste ci-dessus")
        pa.terminate()
        return 1

    print("PARLE MAINTENANT (crie 'help')...  (Ctrl+C pour arreter)")
    peak = 0.0
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except Exception as exc:
            print("erreur lecture: {}".format(exc))
            break
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        peak = max(peak, rms)
        bar = "#" * min(40, int(rms * 400))
        print("  RMS %.4f  %s" % (rms, bar))

    stream.stop_stream()
    stream.close()
    pa.terminate()

    print("\nRMS max: %.4f" % peak)
    if peak < 0.005:
        print("=> QUASI SILENCE : mauvais peripherique ou micro muet.")
        print("   Relance avec un autre numero, ou verifie alsamixer.")
    elif peak < 0.02:
        print("=> FAIBLE : baisse le seuil ->")
        print("   export SPEECH_ENERGY_GATE=%.4f" % (peak / 3.0))
    else:
        print("=> BON NIVEAU. Le micro n'est pas le probleme.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
