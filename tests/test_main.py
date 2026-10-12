"""Regression checks for process wiring and cleanup on missing GUI dependencies."""
import builtins
import inspect
import subprocess
import sys

import pytest

import main


def test_mic_index_is_only_passed_to_audio(monkeypatch):
    processes = []

    class RecordedProcess:
        def __init__(self, target, name, args):
            self.name = name
            self.arguments = inspect.signature(target).bind(*args)
            self.arguments.apply_defaults()
            processes.append(self)

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    original_import = builtins.__import__

    def without_gui(name, *args, **kwargs):
        if name.startswith("PyQt5"):
            raise ImportError("GUI intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(main.mp, "Process", RecordedProcess)
    monkeypatch.setattr(builtins, "__import__", without_gui)
    with pytest.raises(ImportError, match="intentionally unavailable"):
        main.main(["--sim", "--mic", "7"])
    audio = next(p for p in processes if p.name == "audio")
    speech = next(p for p in processes if p.name == "speech")
    assert audio.arguments.arguments["device_index"] == 7
    assert speech.arguments.arguments["model_path"] == "models/vosk"


def test_missing_gui_stops_actual_synthetic_workers():
    script = '''
import builtins
import multiprocessing as mp
import main
original_import = builtins.__import__
def without_gui(name, *args, **kwargs):
    if name.startswith("PyQt5"):
        raise ImportError("GUI intentionally unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_gui
try:
    main.main(["--sim"])
except ImportError as exc:
    assert "intentionally unavailable" in str(exc)
else:
    raise AssertionError("Expected missing GUI error")
assert not mp.active_children(), "Synthetic workers survived startup failure"
print("STARTUP FAILURE CLEANUP: OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        cwd=main.BASE_DIR, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STARTUP FAILURE CLEANUP: OK" in result.stdout
