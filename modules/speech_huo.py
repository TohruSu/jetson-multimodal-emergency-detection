# -*- coding: utf-8 -*-
"""
MODULE 3 — Speech / keyword spotting (owner: HUO)

Contract:
- Consumes: audio_queue (PCM int16 chunks, 16 kHz mono, 0.25 s)
- Publishes: event_queue, payload {"keyword": "help", "confidence": float}

INTEGRATED (7/15): HUO's transfer-learning model (ResNet-18,
"hello"/"help", models/speech.pth) is plugged in as the primary
detector, wrapped for streaming in modules/huo/classifier.py
(sliding window + energy gate + cooldown).
Real-mode fallback order: HUO model -> Vosk (models/vosk, optional)
-> inactive. Synthetic keywords are emitted only with sim=True.

HUO: thresholds to tune with the real webcam mic: ENERGY_GATE_RMS,
CONF_THRESHOLD, WINDOW_S in modules/huo/classifier.py.
"""
import json
import queue
import time

from common.qutils import ignore_sigint
from common.messages import make_event, SOURCE_SPEECH
from common.health import HealthReporter
from common import simscript

KEYWORDS = ("help", "aide", "tasukete")


def speech_process(audio_queue, event_queue, stop_event, sim=False,
                   sim_state=None, model_path="models/vosk"):
    ignore_sigint()
    health = HealthReporter(SOURCE_SPEECH, event_queue)
    backend = "synthetic" if sim else "huo_keyword"
    health.set("simulated" if sim else "initializing", backend=backend)
    classifier = None
    recognizer = None
    if not sim:
        # 1st choice: HUO's trained keyword model
        try:
            from modules.huo.classifier import StreamingKeywordClassifier
            print("[speech] loading keyword model...")
            classifier = StreamingKeywordClassifier("models/speech.pth")
            health.set("ready", backend=backend)
            print("[speech] using HUO's keyword model")
        except Exception:
            print("[speech] primary_unavailable; trying Vosk")
        # 2nd choice: Vosk generic recognizer
        if classifier is None:
            try:
                from vosk import Model, KaldiRecognizer
                model = Model(model_path)
                recognizer = KaldiRecognizer(model, 16000)
                backend = "vosk"
                health.set("degraded", backend=backend,
                           reason="primary_unavailable")
                print("[speech] using Vosk fallback")
            except Exception:
                backend = "none"
                health.set("unavailable", backend=backend,
                           reason="speech_initialization_failed")
                print("[speech] speech_initialization_failed -> inactive")

    prev_helping = False

    def emit(keyword, confidence):
        event_queue.put(make_event(SOURCE_SPEECH, {
            "keyword": keyword,
            "confidence": float(confidence),
        }))

    while not stop_event.is_set():
        try:
            data = audio_queue.get(timeout=0.5)
        except queue.Empty:
            data = None

        health.heartbeat()

        if not sim and classifier is None and recognizer is None:
            # Real mode with no model: stay inactive, never fake a keyword.
            time.sleep(0.2)
            continue

        if sim:
            # Fire on the rising edge of "helping" (clock or forced by H).
            t_cycle, _ = simscript.sim_clock(sim_state)
            helping = simscript.is_helping(sim_state, t_cycle)
            if helping and not prev_helping:
                emit("help", 0.88)
            prev_helping = helping
            continue

        if data is None:
            continue

        try:
            # HUO's streaming classifier
            if classifier is not None:
                det = classifier.accept_chunk(data)
                if det is not None:
                    emit(det["keyword"], det["confidence"])
                health.set("ready", backend=backend)
                continue

            # Vosk remains a functional fallback with degraded health.
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                for kw in KEYWORDS:
                    if kw in text:
                        emit(kw, 0.9)
            else:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "").lower()
                for kw in KEYWORDS:
                    if kw in text:
                        emit(kw, 0.7)
            health.set("degraded", backend=backend, reason="primary_unavailable")
        except Exception:
            if health.set("degraded", backend=backend, reason="inference_failed"):
                print("[speech] inference_failed; retrying next audio chunk")

    print("[speech] stopped cleanly")
