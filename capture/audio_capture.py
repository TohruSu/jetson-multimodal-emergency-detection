# -*- coding: utf-8 -*-
"""
Audio capture process (PyAudio, webcam built-in mic).
Publishes 16-bit PCM / 16 kHz mono chunks into audio_queue —
exactly the format Vosk expects. Sim mode: silence chunks.
"""
import time
import queue

from common.qutils import ignore_sigint
from common.messages import SOURCE_SYSTEM
from common.health import HealthReporter

SAMPLE_RATE = 16000
CHUNK_FRAMES = 4000  # 0.25 s of audio per chunk
RETRY_PERIOD_S = 2.0


def _find_usb_mic(pa):
    """Prefer a USB/webcam microphone over the (mic-less) onboard card."""
    keywords = ("usb", "cam", "mic", "hd pro", "logitech")
    best = None
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
        except Exception:
            continue
        if int(info.get("maxInputChannels", 0)) <= 0:
            continue
        name = str(info.get("name", "")).lower()
        if any(k in name for k in keywords):
            best = i
            break
        if best is None:
            best = i
    return best


def _close_audio(stream, pa):
    # Each resource is independent: failure to stop a disconnected stream
    # must not prevent closing it or terminating its PyAudio owner.
    if stream is not None:
        for operation in (stream.stop_stream, stream.close):
            try:
                operation()
            except Exception:
                pass
    if pa is not None:
        try:
            pa.terminate()
        except Exception:
            pass


def audio_capture_process(audio_queue, event_queue, stop_event, sim=False,
                          sim_state=None, device_index=None):
    ignore_sigint()
    pa = None
    stream = None
    backend = "synthetic" if sim else "pyaudio"
    health = HealthReporter(SOURCE_SYSTEM, event_queue, module="audio",
                            clock=time.monotonic)
    health.set("simulated" if sim else "initializing", backend=backend)
    silence = b"\x00" * (CHUNK_FRAMES * 2)  # int16 -> 2 bytes/frame
    retry_at = 0.0
    period = CHUNK_FRAMES / float(SAMPLE_RATE)
    had_audio = False

    def failed(state, reason):
        if health.set(state, backend=backend, reason=reason):
            print("[audio] {} ({})".format(state, reason))

    try:
        while not stop_event.is_set():
            data = None
            if sim:
                time.sleep(period)
                data = silence
            else:
                if stream is None and time.monotonic() >= retry_at:
                    reason = "pyaudio_unavailable"
                    try:
                        import pyaudio
                        reason = "microphone_open_failed"
                        pa = pyaudio.PyAudio()
                        # Re-evaluate automatic selection after a reconnect.
                        selected = (device_index if device_index is not None
                                    else _find_usb_mic(pa))
                        stream = pa.open(format=pyaudio.paInt16, channels=1,
                                         rate=SAMPLE_RATE, input=True,
                                         frames_per_buffer=CHUNK_FRAMES,
                                         input_device_index=selected)
                    except Exception:
                        _close_audio(stream, pa)
                        stream, pa = None, None
                        retry_at = time.monotonic() + RETRY_PERIOD_S
                        failed("unavailable", reason)

                if stream is not None:
                    try:
                        data = stream.read(CHUNK_FRAMES,
                                           exception_on_overflow=False)
                        if not data:
                            raise RuntimeError("microphone returned no audio")
                    except Exception:
                        # Fabricated silence would conceal a dead microphone.
                        data = None
                        _close_audio(stream, pa)
                        stream, pa = None, None
                        retry_at = time.monotonic() + RETRY_PERIOD_S
                        failed("degraded" if had_audio else "unavailable",
                               "microphone_read_failed")
                    else:
                        had_audio = True
                        if health.set("ready", backend=backend):
                            print("[audio] microphone ready")
                else:
                    time.sleep(period)

            if data is not None:
                # Bounded queue: drop an old PCM chunk when Vosk falls behind.
                try:
                    audio_queue.put_nowait(data)
                except queue.Full:
                    try:
                        audio_queue.get_nowait()
                        audio_queue.put_nowait(data)
                    except (queue.Empty, queue.Full):
                        pass

            health.heartbeat()
    finally:
        _close_audio(stream, pa)
    print("[audio] stopped cleanly")
