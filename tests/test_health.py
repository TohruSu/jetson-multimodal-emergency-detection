"""Worker health and the UI's status presentation without Qt or hardware."""
import importlib.util
from pathlib import Path
import queue
import sys
import types

import pytest


class Clock:
    def __init__(self, value=10.0):
        self.value = value

    def __call__(self):
        return self.value


@pytest.fixture
def ui_module(monkeypatch):
    # Only class definitions are imported. No widgets, display, or QApplication
    # are created; the actual status handlers below run against plain objects.
    modules = {
        "PyQt5": {},
        "PyQt5.QtCore": {"Qt": object(), "QSize": object},
        "PyQt5.QtGui": dict.fromkeys(
            ["QImage", "QPixmap", "QPainter", "QColor", "QPen"], object),
        "PyQt5.QtWidgets": dict.fromkeys(
            ["QMainWindow", "QWidget", "QLabel", "QHBoxLayout", "QVBoxLayout"],
            object),
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "ui" / "main_window.py"
    spec = importlib.util.spec_from_file_location("_health_ui_test", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Label:
    def setText(self, text):
        self.text = text


def ui_state():
    return types.SimpleNamespace(
        _heartbeats=dict.fromkeys(
            ["video", "audio", "gesture", "depth", "speech"], 0.0),
        _health_payloads={}, _health_received={}, status_label=Label())


def status(health="ready", backend="opencv", source="system",
           module="video", timestamp=100.0):
    payload = {"state": "alive", "health": health, "backend": backend}
    if module is not None:
        payload["module"] = module
    return {"type": "status", "source": source, "ts": timestamp,
            "payload": payload}


def test_ui_does_not_present_alive_unavailable_worker_as_ok(ui_module,
                                                          monkeypatch):
    window = ui_state()
    monkeypatch.setattr(ui_module.time, "time", lambda: 100.0)
    monkeypatch.setattr(ui_module.time, "monotonic", lambda: 10.0)
    ui_module.MainWindow.handle_message(window, status("unavailable"))
    ui_module.MainWindow.refresh_status(window)
    assert "video UNAVAILABLE (opencv)" in window.status_label.text
    assert "OK video" not in window.status_label.text
    assert window._heartbeats["video"] == 100.0


@pytest.mark.parametrize("wall_now", [50.0, 1000.0])
def test_ui_status_freshness_uses_monotonic_receipt_age(ui_module, monkeypatch,
                                                     wall_now):
    window = ui_state()
    mono = Clock()
    monkeypatch.setattr(ui_module.time, "time", lambda: 100.0)
    monkeypatch.setattr(ui_module.time, "monotonic", mono)
    ui_module.MainWindow.handle_message(window, status())
    monkeypatch.setattr(ui_module.time, "time", lambda: wall_now)
    mono.value += 4.0
    ui_module.MainWindow.refresh_status(window)
    assert "video READY (opencv)" in window.status_label.text
    assert "video STALE" not in window.status_label.text
    mono.value += 2.0
    ui_module.MainWindow.refresh_status(window)
    assert "video STALE - last READY (opencv)" in window.status_label.text


def test_ui_detection_does_not_refresh_backend_health(ui_module, monkeypatch):
    window = ui_state()
    mono = Clock()
    monkeypatch.setattr(ui_module.time, "monotonic", mono)
    ui_module.MainWindow.handle_message(
        window, status("unavailable", "trt_pose", source="gesture",
                       module=None))
    mono.value += 6.0
    ui_module.MainWindow.handle_message(window, {
        "source": "gesture", "type": "detections", "ts": 106.0,
        "payload": {"tracks": [], "fps": 0.0}})
    ui_module.MainWindow.refresh_status(window)
    assert "gesture STALE - last UNAVAILABLE (trt_pose)" in (
        window.status_label.text)


def test_ui_distinguishes_unreceived_status_from_stale(ui_module):
    window = ui_state()
    ui_module.MainWindow.refresh_status(window)
    assert "video WAITING" in window.status_label.text
    assert "audio WAITING" in window.status_label.text


def test_reporter_preserves_status_envelope_and_wall_timestamp(monkeypatch):
    from common.health import HealthReporter
    from common import messages

    events = queue.Queue()
    mono = Clock()
    monkeypatch.setattr(messages.time, "time", lambda: 1234.5)
    reporter = HealthReporter("system", events, module="video", clock=mono)
    assert reporter.set("unavailable", "opencv", "camera_open_failed")
    assert events.get_nowait() == {
        "ts": 1234.5, "source": "system", "type": "status",
        "payload": {"state": "alive", "module": "video",
                    "health": "unavailable", "backend": "opencv",
                    "reason": "camera_open_failed"}}


def test_reporter_publishes_only_transitions_between_heartbeats():
    from common.health import HealthReporter

    events, mono = queue.Queue(), Clock()
    reporter = HealthReporter("gesture", events, clock=mono)
    assert reporter.set("ready", "trt_pose")
    assert not reporter.set("ready", "trt_pose")
    assert events.qsize() == 1
    assert reporter.set("degraded", "trt_pose", "inference_failed")
    assert not reporter.set("degraded", "trt_pose", "inference_failed")
    assert reporter.set("ready", "trt_pose")
    payloads = [events.get_nowait()["payload"] for _ in range(3)]
    assert [payload["health"] for payload in payloads] == [
        "ready", "degraded", "ready"]
    assert "reason" not in payloads[-1]
    assert "module" not in payloads[-1]


def test_reporter_heartbeat_uses_monotonic_clock(monkeypatch):
    from common.health import HealthReporter
    from common import messages

    events, mono = queue.Queue(), Clock()
    reporter = HealthReporter("speech", events, clock=mono)
    reporter.set("ready", "vosk")
    monkeypatch.setattr(messages.time, "time", lambda: -1000.0)
    mono.value += 1.999
    assert not reporter.heartbeat()
    mono.value = 12.0
    assert reporter.heartbeat()
    assert events.qsize() == 2
    assert events.get_nowait()["ts"] != -1000.0
    assert events.get_nowait()["ts"] == -1000.0


def test_full_event_queue_cannot_block_worker_and_heartbeat_retries():
    from common.health import HealthReporter

    events, mono = queue.Queue(maxsize=1), Clock()
    events.put_nowait("occupied")
    reporter = HealthReporter("depth", events, clock=mono)
    assert reporter.set("disabled", "depth_anything")
    assert events.get_nowait() == "occupied"
    assert reporter.heartbeat()
    assert events.get_nowait()["payload"]["health"] == "disabled"


@pytest.mark.parametrize("health, label", [
    ("ready", "READY"), ("simulated", "SIMULATION"),
    ("degraded", "DEGRADED"), ("unavailable", "UNAVAILABLE"),
    ("disabled", "DISABLED"), ("initializing", "INITIALIZING"),
])
def test_ui_visible_health_labels(health, label):
    from common.health import format_health

    payload = {"state": "alive", "health": health, "backend": "example"}
    assert format_health(payload) == label + " (example)"
    assert format_health(payload, fresh=False) == (
        "STALE - last " + label + " (example)")


def test_legacy_status_does_not_claim_backend_ready():
    from common.health import format_health
    from common.messages import make_status

    payload = make_status("gesture")["payload"]
    assert format_health(payload) == "ALIVE (health unknown)"
    assert format_health(payload, fresh=False) == (
        "STALE - last ALIVE (health unknown)")


def test_degraded_reason_is_visible():
    from common.health import format_health

    assert format_health({"state": "alive", "health": "degraded",
                          "backend": "pyaudio",
                          "reason": "microphone_read_failed"}) == (
        "DEGRADED (pyaudio; microphone_read_failed)")
