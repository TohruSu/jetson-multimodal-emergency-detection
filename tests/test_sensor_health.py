"""Capture failure/recovery behavior using virtual time and fake devices."""
import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from capture import audio_capture, video_capture


class Clock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    monotonic = time

    def sleep(self, seconds):
        self.now += seconds


class TimedStop:
    def __init__(self, clock, duration):
        self.clock = clock
        self.deadline = clock.now + duration

    def is_set(self):
        return self.clock.now >= self.deadline


class Camera:
    def __init__(self, opened=True, reads=None):
        self.opened = opened
        self.reads = list(reads or [])
        self.released = False

    def set(self, *args):
        pass

    def isOpened(self):
        return self.opened

    def read(self):
        result = self.reads.pop(0) if self.reads else (False, None)
        if isinstance(result, Exception):
            raise result
        return result

    def release(self):
        self.released = True


def fake_cameras(monkeypatch, clock, cameras):
    attempts = []

    def open_camera(index):
        attempts.append(clock.now)
        return cameras[min(len(attempts) - 1, len(cameras) - 1)]

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=open_camera, CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4))
    return attempts


def capture_setup(monkeypatch, module, duration):
    clock = Clock()
    monkeypatch.setattr(module, "time", clock)
    monkeypatch.setattr(module, "ignore_sigint", lambda: None)
    return clock, TimedStop(clock, duration), queue.Queue()


def statuses(events, module):
    values = []
    while not events.empty():
        message = events.get_nowait()
        if message["type"] == "status" and message["payload"].get("module") == module:
            values.append(message["payload"])
    return values


class RecordingQueue(queue.Queue):
    def __init__(self):
        super().__init__(maxsize=1)
        self.observed = []

    def put_nowait(self, item):
        super().put_nowait(item)
        self.observed.append(item)


def health_transitions(events, module):
    transitions = []
    for payload in statuses(events, module):
        if not transitions or transitions[-1] != payload["health"]:
            transitions.append(payload["health"])
    return transitions


def test_missing_real_camera_never_feeds_synthetic_frames(monkeypatch, capsys):
    clock, stop, events = capture_setup(monkeypatch, video_capture, 4.5)
    cameras = [Camera(opened=False) for _ in range(3)]
    attempts = fake_cameras(monkeypatch, clock, cameras)
    generated = np.full((2, 2, 3), 99, dtype=np.uint8)
    monkeypatch.setattr(video_capture, "_draw_sim_frame", lambda state: generated)
    frames, ui = queue.Queue(maxsize=1), queue.Queue(maxsize=1)

    video_capture.video_capture_process([frames], events, stop, ui_queue=ui)

    assert frames.empty(), "missing real camera must not feed synthetic frames to inference"
    assert ui.empty(), "missing real camera must not appear as a live synthetic camera"
    assert len(attempts) == 3
    assert all(second - first >= 2.0 for first, second in zip(attempts, attempts[1:]))
    assert all(camera.released for camera in cameras)
    health = statuses(events, "video")
    assert health[-1]["health"] == "unavailable"
    assert health[-1]["reason"] == "camera_open_failed"
    assert capsys.readouterr().out.count("unavailable (camera_open_failed)") == 1


class AudioStream:
    def __init__(self, clock, reads):
        self.clock = clock
        self.reads = list(reads)
        self.stopped = False
        self.closed = False

    def read(self, *args, **kwargs):
        self.clock.sleep(0.25)
        result = self.reads.pop(0) if self.reads else b"real microphone data"
        if isinstance(result, Exception):
            raise result
        return result

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class AudioDevice:
    def __init__(self, result):
        self.result = result
        self.terminated = False

    def open(self, **kwargs):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def get_device_info_by_index(self, index):
        return {"index": index, "name": "fake test microphone"}

    def terminate(self):
        self.terminated = True


def fake_audio(monkeypatch, clock, devices):
    attempts = []

    def open_device():
        attempts.append(clock.now)
        return devices[min(len(attempts) - 1, len(devices) - 1)]

    monkeypatch.setitem(sys.modules, "pyaudio", SimpleNamespace(
        PyAudio=open_device, paInt16=8))
    return attempts


def test_missing_real_microphone_never_feeds_fake_silence(monkeypatch, capsys):
    clock, stop, events = capture_setup(monkeypatch, audio_capture, 4.5)
    devices = [AudioDevice(OSError("missing microphone")) for _ in range(3)]
    attempts = fake_audio(monkeypatch, clock, devices)
    audio = queue.Queue(maxsize=1)

    audio_capture.audio_capture_process(audio, events, stop, device_index=0)

    assert audio.empty(), "failed real microphone must not feed fabricated silence"
    assert len(attempts) == 3
    assert all(second - first >= 2.0 for first, second in zip(attempts, attempts[1:]))
    assert all(device.terminated for device in devices)
    health = statuses(events, "audio")
    assert health[-1]["health"] == "unavailable"
    assert health[-1]["reason"] == "microphone_open_failed"
    assert capsys.readouterr().out.count("unavailable (microphone_open_failed)") == 1


@pytest.mark.parametrize("read_failure", [(False, None), OSError("disconnected")])
def test_real_camera_read_failure_reopens_and_recovers(monkeypatch, read_failure):
    clock, stop, events = capture_setup(monkeypatch, video_capture, 3.0)
    first = np.full((2, 2, 3), 10, dtype=np.uint8)
    recovered = np.full((2, 2, 3), 20, dtype=np.uint8)
    cameras = [Camera(reads=[(True, first), read_failure]),
               Camera(reads=[(True, recovered)] * 30)]
    attempts = fake_cameras(monkeypatch, clock, cameras)
    monkeypatch.setattr(video_capture, "_draw_sim_frame", lambda state: pytest.fail(
        "real camera failure must not draw a simulated frame"))
    frames, ui = RecordingQueue(), RecordingQueue()

    video_capture.video_capture_process([frames], events, stop, ui_queue=ui)

    assert len(attempts) == 2
    assert attempts[1] - attempts[0] >= 2.0
    assert all(camera.released for camera in cameras)
    assert frames.observed[0] is first
    assert frames.observed[-1] is recovered
    assert len(frames.observed) == len(ui.observed)
    assert all(frame is first or frame is recovered for frame in frames.observed)
    assert health_transitions(events, "video") == [
        "initializing", "ready", "degraded", "ready"]


def test_real_microphone_read_failure_reopens_and_recovers(monkeypatch):
    clock, stop, events = capture_setup(monkeypatch, audio_capture, 3.5)
    first = b"first real microphone chunk"
    recovered = b"reconnected real microphone chunk"
    streams = [AudioStream(clock, [first, OSError("disconnected")]),
               AudioStream(clock, [recovered] * 20)]
    devices = [AudioDevice(stream) for stream in streams]
    attempts = fake_audio(monkeypatch, clock, devices)
    audio = RecordingQueue()

    audio_capture.audio_capture_process(audio, events, stop, device_index=0)

    assert len(attempts) == 2
    assert attempts[1] - attempts[0] >= 2.0
    assert all(stream.stopped and stream.closed for stream in streams)
    assert all(device.terminated for device in devices)
    assert audio.observed[0] == first
    assert audio.observed[-1] == recovered
    assert set(audio.observed) == {first, recovered}
    assert health_transitions(events, "audio") == [
        "initializing", "ready", "degraded", "ready"]


@pytest.mark.parametrize("module", [video_capture, audio_capture])
def test_initial_open_failure_recovers_instead_of_permanently_disabling_sensor(
        monkeypatch, module):
    clock, stop, events = capture_setup(monkeypatch, module, 3.0)
    output = RecordingQueue()
    if module is video_capture:
        genuine = np.full((2, 2, 3), 7, dtype=np.uint8)
        devices = [Camera(opened=False), Camera(reads=[(True, genuine)] * 30)]
        attempts = fake_cameras(monkeypatch, clock, devices)
        module.video_capture_process([output], events, stop)
        assert all(device.released for device in devices)
        assert all(frame is genuine for frame in output.observed)
        name = "video"
    else:
        genuine = b"reconnected microphone data"
        stream = AudioStream(clock, [genuine] * 20)
        devices = [AudioDevice(OSError("temporarily missing")), AudioDevice(stream)]
        attempts = fake_audio(monkeypatch, clock, devices)
        module.audio_capture_process(output, events, stop, device_index=0)
        assert all(device.terminated for device in devices)
        assert stream.stopped and stream.closed
        assert all(data == genuine for data in output.observed)
        name = "audio"

    assert output.observed
    assert len(attempts) == 2
    assert attempts[1] - attempts[0] >= 2.0
    assert health_transitions(events, name) == ["initializing", "unavailable", "ready"]


@pytest.mark.parametrize("module, dependency, reason", [
    (video_capture, "cv2", "opencv_unavailable"),
    (audio_capture, "pyaudio", "pyaudio_unavailable"),
])
def test_missing_capture_dependency_is_explicit_and_emits_no_data(
        monkeypatch, module, dependency, reason):
    clock, stop, events = capture_setup(monkeypatch, module, 2.5)
    monkeypatch.setitem(sys.modules, dependency, None)
    output = RecordingQueue()
    if module is video_capture:
        module.video_capture_process([output], events, stop)
        name = "video"
    else:
        module.audio_capture_process(output, events, stop)
        name = "audio"

    assert output.observed == []
    health = statuses(events, name)
    assert health[-1]["health"] == "unavailable"
    assert health[-1]["reason"] == reason


def test_simulated_camera_retains_frames_and_explicit_simulated_health(monkeypatch):
    clock, stop, events = capture_setup(monkeypatch, video_capture, 0.3)
    monkeypatch.setitem(sys.modules, "cv2", None)
    generated = np.full((2, 2, 3), 99, dtype=np.uint8)
    monkeypatch.setattr(video_capture, "_draw_sim_frame", lambda state: generated)
    frames = RecordingQueue()

    video_capture.video_capture_process([frames], events, stop, sim=True)

    assert frames.observed
    assert all(frame is generated for frame in frames.observed)
    assert health_transitions(events, "video") == ["simulated"]


def test_simulated_microphone_retains_silence_and_explicit_simulated_health(monkeypatch):
    clock, stop, events = capture_setup(monkeypatch, audio_capture, 0.6)
    monkeypatch.setitem(sys.modules, "pyaudio", None)
    audio = RecordingQueue()

    audio_capture.audio_capture_process(audio, events, stop, sim=True)

    assert audio.observed
    assert all(data == b"\x00" * 8000 for data in audio.observed)
    assert health_transitions(events, "audio") == ["simulated"]


def test_audio_cleanup_closes_and_terminates_when_stop_stream_fails(monkeypatch):
    clock, stop, events = capture_setup(monkeypatch, audio_capture, 0.3)
    stream = AudioStream(clock, [OSError("disconnected")])

    def failed_stop():
        raise OSError("device already removed")

    stream.stop_stream = failed_stop
    device = AudioDevice(stream)
    fake_audio(monkeypatch, clock, [device])

    audio_capture.audio_capture_process(queue.Queue(), events, stop, device_index=0)

    assert stream.closed
    assert device.terminated
    health = statuses(events, "audio")
    assert health[-1]["health"] == "unavailable"
    assert health[-1]["reason"] == "microphone_read_failed"
