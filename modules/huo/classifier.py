# -*- coding: utf-8 -*-
"""
Streaming keyword classifier (integration wrapper around HUO's model).

HUO's original API is predict(wav_file): one file -> one label.
Our pipeline streams 0.25 s PCM chunks, so this wrapper adds:
- a sliding window (2.5 s, matching his training clip durations),
- an ENERGY GATE: the model only knows "hello" vs "help", so on pure
  silence it would still pick one of them with high confidence —
  we only run inference when there is actual sound,
- a cooldown so one shout does not emit 5 events.

Model: ResNet-18 (2 classes) on log-mel spectrograms — see
modules/huo/feature.py (HUO's file, unchanged) and models/speech.pth.
"""
import collections

import numpy as np

LABELS = {0: "hello", 1: "help"}

WINDOW_S = 2.5          # matches HUO's training clips (1.6-3.9 s)
HOP_S = 1.0             # run inference every 1 s (Nano CPU budget)
SAMPLE_RATE = 16000
# Tunable at runtime, since it depends on the mic and the room:
#   export SPEECH_ENERGY_GATE=0.004   (lower = more sensitive)
#   export SPEECH_CONF=0.60           (lower = more permissive)
#   export SPEECH_DEBUG=1             (print RMS + class each inference)
import os as _os
# Historical team calibration notes (not remeasured) describe microphone
# RMS and speaker-dependent confidence differences. These defaults preserve
# that integration: gate 0.006, confidence 0.50. No accuracy is established.
# Fusion also requires a fall, but a false keyword can still escalate a
# coincident fall; this condition is not a safety guarantee.
ENERGY_GATE_RMS = float(_os.environ.get("SPEECH_ENERGY_GATE", "0.006"))
COOLDOWN_S = 2.0        # min delay between two emitted detections
CONF_THRESHOLD = float(_os.environ.get("SPEECH_CONF", "0.50"))
DEBUG = _os.environ.get("SPEECH_DEBUG") == "1"


def build_model(num_classes):
    """HUO's build_model, without the network download.

    His version pulls resnet18 weights from torch.hub at every start —
    on an offline Nano that fails. We build the architecture locally
    (random init) since load_state_dict overwrites all weights anyway.
    """
    import torch.nn as nn
    from torchvision import models
    model = models.resnet18(weights=None) if hasattr(models, "ResNet18_Weights") \
        else models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class StreamingKeywordClassifier(object):

    def __init__(self, model_path="models/speech.pth"):
        import torch
        from modules.huo.feature import AudioFeatureExtractor

        self.torch = torch
        # CPU on purpose: on a 4 GB Nano every extra CUDA context costs
        # hundreds of MB, and this model is small enough for the CPU.
        # Set SPEECH_DEVICE=cuda to override.
        import os as _os
        self.device = torch.device(_os.environ.get("SPEECH_DEVICE", "cpu"))
        self.model = build_model(len(LABELS))
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.feat = AudioFeatureExtractor()

        self._window = collections.deque(
            maxlen=int(WINDOW_S * SAMPLE_RATE))
        self._samples_since_infer = 0
        self._hop_samples = int(HOP_S * SAMPLE_RATE)
        self._cooldown_chunks = 0

    # ------------------------------------------------------------------
    def accept_chunk(self, pcm_bytes):
        """
        Feed one PCM int16 mono chunk. Returns None, or
        {"keyword": str, "confidence": float} when a keyword is
        detected with enough confidence.
        """
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(
            np.float32) / 32768.0
        self._window.extend(samples)
        self._samples_since_infer += len(samples)

        if self._cooldown_chunks > 0:
            self._cooldown_chunks -= 1
            return None
        if self._samples_since_infer < self._hop_samples:
            return None
        if len(self._window) < self._window.maxlen:
            return None  # window not full yet
        self._samples_since_infer = 0

        wav = np.asarray(self._window, dtype=np.float32)
        rms = float(np.sqrt(np.mean(wav ** 2)))
        if rms < ENERGY_GATE_RMS:
            if DEBUG:
                print("[speech] rms %.4f < gate %.4f -> skipped"
                      % (rms, ENERGY_GATE_RMS))
            return None  # silence: 2-class model would hallucinate

        torch = self.torch
        with torch.no_grad():
            mel = self.feat.extract(torch.tensor(wav))
            mel = mel.repeat(3, 1, 1)          # same as HUO's inference.py
            out = self.model(mel.unsqueeze(0).to(self.device))
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, dim=1)

        label = LABELS[int(pred.item())]
        confidence = float(conf.item())
        if DEBUG:
            print("[speech] rms %.4f -> %s (%.2f)"
                  % (rms, label, confidence))
        if label == "help" and confidence >= CONF_THRESHOLD:
            chunk_s = len(samples) / float(SAMPLE_RATE)
            self._cooldown_chunks = int(COOLDOWN_S / max(chunk_s, 1e-3))
            self._window.clear()
            return {"keyword": "help", "confidence": confidence}
        return None
