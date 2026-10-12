"""Model health and recovery with no CUDA, weights, microphones or display."""
from contextlib import nullcontext
import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from modules import depth_su, gesture_song, gpu_worker, speech_huo


class Stop:
    stopped = False

    def is_set(self):
        return self.stopped


class Inputs:
    """Stop after the final input has been processed, with bounded fake IO."""
    def __init__(self, values, stop):
        self.values = iter(values)
        self.remaining = len(values)
        self.stop = stop

    def get(self, timeout=None):
        self.remaining -= 1
        if self.remaining <= 0:
            self.stop.stopped = True
        return next(self.values)


def payloads(events, source, kind="status"):
    return [m["payload"] for m in list(events.queue)
            if m["source"] == source and m["type"] == kind]


def healths(events, source):
    statuses = payloads(events, source)
    assert statuses, "the UI must receive backend health even when idle"
    assert all(p["state"] == "alive" for p in statuses)
    return [p.get("health") for p in statuses]


@pytest.fixture
def runtime(monkeypatch):
    stop = Stop()
    events = queue.Queue()
    # A bounded fake replaces any idle/retry delays and real camera IO.
    monkeypatch.setattr(gesture_song.time, "sleep", lambda _: None)
    monkeypatch.setattr(gpu_worker, "ignore_sigint", lambda: None)
    monkeypatch.setattr(gesture_song, "ignore_sigint", lambda: None)
    monkeypatch.setattr(depth_su, "ignore_sigint", lambda: None)
    monkeypatch.setattr(speech_huo, "ignore_sigint", lambda: None)
    return stop, events


def fake_frame_input(monkeypatch, worker, values, stop):
    inputs = Inputs(values, stop)
    monkeypatch.setattr(worker, "get_latest",
                        lambda _queue, timeout=None: inputs.get(timeout))


def fake_pose(monkeypatch, failures=0):
    """Return the original pose interface while controlling inference errors."""
    class Tensor:
        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

    calls = []

    def model(_tensor):
        calls.append(True)
        if len(calls) <= failures:
            raise RuntimeError("private checkpoint detail")
        return Tensor(), Tensor()

    torch = SimpleNamespace(no_grad=nullcontext,
                            cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        resize=lambda frame, *args, **kwargs: frame, INTER_LINEAR=1))
    tracker = SimpleNamespace(update=lambda *args: ([], []))
    return {"torch": torch, "fd": SimpleNamespace(
        extract_detections=lambda *args: []), "pre": lambda image: image,
        "model": model, "parse": lambda *args: ([], [], []),
        "names": [], "config": {"min_keypoints": 3},
        "tracker": tracker, "backend": "trt_pose_pytorch"}, calls


def speech_dependencies(monkeypatch, classifier_factory, recognizer_factory):
    monkeypatch.setitem(sys.modules, "modules.huo.classifier",
                        SimpleNamespace(StreamingKeywordClassifier=classifier_factory))
    monkeypatch.setitem(sys.modules, "vosk", SimpleNamespace(
        Model=lambda path: object(),
        KaldiRecognizer=lambda model, rate: recognizer_factory()))


def unavailable(*args):
    raise RuntimeError("private checkpoint detail")


@pytest.mark.parametrize("use_depth, expected, reason", [
    (False, "disabled", None),
    (True, "unavailable", "pose_unavailable"),
])
def test_gpu_missing_pose_distinguishes_disabled_and_blocked_depth(
        monkeypatch, runtime, use_depth, expected, reason):
    stop, events = runtime
    monkeypatch.setattr(gpu_worker, "_make_real_detector",
                        lambda: (None, "sensitive initializer detail"))
    fake_frame_input(monkeypatch, gpu_worker, [None], stop)
    gpu_worker.gpu_worker_process(None, events, stop, use_depth=use_depth)
    assert healths(events, "gesture")[-1] == "unavailable"
    assert healths(events, "depth")[-1] == expected
    assert payloads(events, "depth")[-1].get("reason") == reason
    assert not payloads(events, "gesture", "event")


def test_missing_standalone_pose_is_unavailable_and_publishes_no_detections(
        monkeypatch, runtime):
    stop, events = runtime
    monkeypatch.setattr(gesture_song, "_make_real_detector",
                        lambda: (None, "sensitive initializer detail"))
    fake_frame_input(monkeypatch, gesture_song, [None], stop)
    gesture_song.gesture_process(None, events, stop)
    assert healths(events, "gesture")[-1] == "unavailable"
    assert not payloads(events, "gesture", "detections")


def test_missing_depth_reports_unavailable_with_existing_event_contract(
        monkeypatch, runtime):
    stop, events = runtime
    monkeypatch.setattr(depth_su, "_make_real_processor",
                        lambda size: (None, None, "sensitive initializer detail"))
    monkeypatch.setattr(depth_su.time, "sleep",
                        lambda _: setattr(stop, "stopped", True))
    depth_su.depth_process(None, events, stop)
    assert healths(events, "depth")[-1] == "unavailable"
    assert payloads(events, "depth", "event")[-1] == {
        "depth_level": "UNAVAILABLE", "near_score": 0.0, "status": "ERROR"}


def test_missing_both_speech_backends_is_unavailable_without_keywords(
        monkeypatch, runtime):
    stop, events = runtime
    speech_dependencies(monkeypatch, unavailable, unavailable)
    speech_huo.speech_process(Inputs([None], stop), events, stop)
    assert healths(events, "speech")[-1] == "unavailable"
    assert not payloads(events, "speech", "event")


def test_primary_speech_ready_and_keywords_keep_original_payload(
        monkeypatch, runtime):
    stop, events = runtime
    classifier = SimpleNamespace(accept_chunk=lambda data: {
        "keyword": "help", "confidence": 0.85})
    speech_dependencies(monkeypatch, lambda path: classifier, unavailable)
    speech_huo.speech_process(Inputs([b"pcm"], stop), events, stop)
    assert healths(events, "speech")[-1] == "ready"
    assert payloads(events, "speech", "event") == [
        {"keyword": "help", "confidence": 0.85}]


def test_vosk_fallback_is_degraded_but_still_recognizes(monkeypatch, runtime):
    stop, events = runtime
    recognizer = SimpleNamespace(AcceptWaveform=lambda data: True,
                                 Result=lambda: '{"text":"help"}')
    speech_dependencies(monkeypatch, unavailable, lambda: recognizer)
    speech_huo.speech_process(Inputs([b"pcm"], stop), events, stop)
    assert healths(events, "speech")[-1] == "degraded"
    assert payloads(events, "speech")[-1]["backend"] == "vosk"
    assert payloads(events, "speech")[-1]["reason"] == "primary_unavailable"
    assert payloads(events, "speech", "event") == [
        {"keyword": "help", "confidence": 0.9}]


@pytest.mark.parametrize("fallback", [False, True])
def test_speech_inference_failure_retries_and_recovers_without_raw_logs(
        monkeypatch, runtime, capsys, fallback):
    stop, events = runtime
    calls = []

    def process(data):
        calls.append(data)
        if len(calls) <= 2:
            raise RuntimeError("private checkpoint detail")
        return {"keyword": "help", "confidence": 0.8}

    if fallback:
        recognizer = SimpleNamespace(AcceptWaveform=lambda data: bool(process(data)),
                                     Result=lambda: '{"text":"help"}')
        speech_dependencies(monkeypatch, unavailable, lambda: recognizer)
    else:
        classifier = SimpleNamespace(accept_chunk=process)
        speech_dependencies(monkeypatch, lambda path: classifier, unavailable)
    speech_huo.speech_process(Inputs([b"a", b"b", b"c"], stop), events, stop)
    assert len(calls) == 3
    assert any(p.get("reason") == "inference_failed"
               for p in payloads(events, "speech"))
    statuses = payloads(events, "speech")
    assert sum(p.get("reason") == "inference_failed" for p in statuses) == 1
    assert statuses[-1]["health"] == ("degraded" if fallback else "ready")
    assert statuses[-1].get("reason") == ("primary_unavailable" if fallback else None)
    assert len(payloads(events, "speech", "event")) == 1
    assert "private checkpoint detail" not in capsys.readouterr().out


@pytest.mark.parametrize("worker", [gesture_song, gpu_worker])
def test_pose_inference_failure_recovers_on_next_frame(
        monkeypatch, runtime, capsys, worker):
    stop, events = runtime
    ctx, calls = fake_pose(monkeypatch, failures=2)
    monkeypatch.setattr(worker, "_make_real_detector", lambda: (ctx, None))
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    fake_frame_input(monkeypatch, worker, [frame, frame, frame], stop)
    entry = (gesture_song.gesture_process if worker is gesture_song
             else gpu_worker.gpu_worker_process)
    entry(None, events, stop)
    assert len(calls) == 3
    assert "degraded" in healths(events, "gesture")
    assert healths(events, "gesture")[-1] == "ready"
    assert sum(p.get("reason") == "inference_failed"
               for p in payloads(events, "gesture")) == 1
    assert len(payloads(events, "gesture", "detections")) == 1
    assert "private checkpoint detail" not in capsys.readouterr().out


@pytest.mark.parametrize("merged", [False, True])
def test_depth_failure_recovers_and_does_not_degrade_working_pose(
        monkeypatch, runtime, capsys, merged):
    stop, events = runtime
    calls = []

    def process(packet, now):
        calls.append(packet)
        if len(calls) == 1:
            raise RuntimeError("private checkpoint detail")
        return {"depth_level": "NEAR", "smoothed_near_score": 0.9,
                "integration_status": "OK"}

    adapter = SimpleNamespace(process_packet=process)
    worker = gpu_worker if merged else depth_su
    monkeypatch.setattr(worker, "_make_real_processor",
                        lambda size: (object(), adapter, None))
    if merged:
        ctx, _ = fake_pose(monkeypatch)
        monkeypatch.setattr(worker, "_make_real_detector", lambda: (ctx, None))
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        fake_frame_input(monkeypatch, worker, [frame, frame], stop)
        worker.gpu_worker_process(None, events, stop, use_depth=True,
                                  depth_idle_period=0)
        assert "degraded" not in healths(events, "gesture")
    else:
        fake_frame_input(monkeypatch, worker, [{}, {}], stop)
        worker.depth_process(None, events, stop, idle_period=0)
    assert len(calls) == 2
    assert "degraded" in healths(events, "depth")
    assert healths(events, "depth")[-1] == "ready"
    assert payloads(events, "depth", "event")[-1] == {
        "depth_level": "NEAR", "near_score": 0.9, "status": "OK"}
    assert "private checkpoint detail" not in capsys.readouterr().out


def test_gpu_depth_initialization_failure_keeps_pose_ready(monkeypatch, runtime):
    stop, events = runtime
    ctx, _ = fake_pose(monkeypatch)
    monkeypatch.setattr(gpu_worker, "_make_real_detector", lambda: (ctx, None))
    monkeypatch.setattr(gpu_worker, "_make_real_processor",
                        lambda size: (None, None, "private initializer detail"))
    fake_frame_input(monkeypatch, gpu_worker, [None], stop)
    gpu_worker.gpu_worker_process(None, events, stop, use_depth=True)
    assert healths(events, "gesture")[-1] == "ready"
    assert healths(events, "depth")[-1] == "unavailable"


@pytest.mark.parametrize("source,worker", [
    ("gesture", gesture_song), ("depth", depth_su), ("speech", speech_huo),
])
def test_model_simulation_health_is_explicit_and_skips_initializers(
        monkeypatch, runtime, source, worker):
    stop, events = runtime
    if source == "speech":
        speech_dependencies(monkeypatch, unavailable, unavailable)
        worker.speech_process(Inputs([None], stop), events, stop, sim=True)
    elif source == "depth":
        monkeypatch.setattr(worker, "_make_real_processor", unavailable)
        monkeypatch.setattr(worker.time, "sleep",
                            lambda _: setattr(stop, "stopped", True))
        worker.depth_process(None, events, stop, sim=True)
    else:
        monkeypatch.setattr(worker, "_make_real_detector", unavailable)
        fake_frame_input(monkeypatch, worker, [None], stop)
        worker.gesture_process(None, events, stop, sim=True)
    assert healths(events, source)[-1] == "simulated"
    assert payloads(events, source)[-1]["backend"] == "synthetic"


def test_disabled_depth_heartbeat_survives_missing_pose_and_fixed_wall_clock(
        monkeypatch, runtime):
    stop, events = runtime
    mono = [0.0]
    monkeypatch.setattr(gpu_worker.time, "time", lambda: 100.0)
    monkeypatch.setattr(gpu_worker.time, "monotonic", lambda: mono[0])
    monkeypatch.setattr(gpu_worker, "_make_real_detector",
                        lambda: (None, "pose_initialization_failed"))

    def input_with_elapsed_time(_queue, timeout=None):
        mono[0] += 2.0
        if mono[0] >= 4.0:
            stop.stopped = True
        return None

    monkeypatch.setattr(gpu_worker, "get_latest", input_with_elapsed_time)
    gpu_worker.gpu_worker_process(None, events, stop)
    assert healths(events, "depth") == ["disabled", "disabled", "disabled"]
    assert healths(events, "gesture") == [
        "initializing", "unavailable", "unavailable", "unavailable"]
    assert all(message["ts"] == 100.0 for message in list(events.queue))


@pytest.mark.parametrize("merged", [False, True])
def test_stale_depth_packet_recovers_to_ready_no_target(monkeypatch, runtime, merged):
    stop, events = runtime
    results = iter([
        {"depth_level": "NO_TARGET", "integration_status": "STALE_PACKET"},
        {"depth_level": "NO_TARGET", "integration_status": "NO_TARGET"},
    ])
    adapter = SimpleNamespace(process_packet=lambda packet, now: next(results))
    worker = gpu_worker if merged else depth_su
    monkeypatch.setattr(worker, "_make_real_processor",
                        lambda size: (object(), adapter, None))
    if merged:
        ctx, _ = fake_pose(monkeypatch)
        monkeypatch.setattr(worker, "_make_real_detector", lambda: (ctx, None))
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        fake_frame_input(monkeypatch, worker, [frame, frame], stop)
        worker.gpu_worker_process(None, events, stop, use_depth=True,
                                  depth_idle_period=0)
    else:
        fake_frame_input(monkeypatch, worker, [{}, {}], stop)
        worker.depth_process(None, events, stop, idle_period=0)
    assert any(p.get("reason") == "stale_packet" for p in payloads(events, "depth"))
    assert healths(events, "depth")[-1] == "ready"
    assert [p["status"] for p in payloads(events, "depth", "event")] == [
        "STALE_PACKET", "NO_TARGET"]


def test_model_initializer_failures_return_fixed_codes_without_loading_artifacts(
        monkeypatch):
    # Fail at dependency import/construction; no checkpoint can be touched.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "trt_pose", None)
    assert gesture_song._make_real_detector() == (None, "pose_initialization_failed")
    import modules.su_depth
    monkeypatch.setattr(modules.su_depth, "DepthProcessor", unavailable)
    assert depth_su._make_real_processor() == (None, None, "depth_initialization_failed")
