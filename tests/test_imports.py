"""The synthetic entry points stay usable without hardware-only packages."""
import os
from pathlib import Path
import subprocess
import sys


def test_synthetic_imports_and_video_fallback_without_hardware_packages():
    root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib
import importlib.abc
import sys

blocked = {"torch", "torchvision", "trt_pose", "torch2trt",
           "depth_anything_v2", "cv2", "pyaudio", "PyQt5", "vosk",
           "librosa"}

class NoHardwareImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ImportError("hardware dependency blocked for test: " + fullname)

sys.meta_path.insert(0, NoHardwareImports())
sys.argv = ["pytest", "-q", "tests"]
for name in ("main", "test_console", "tests.smoke_sim",
             "common.messages", "common.qutils", "common.simscript",
             "capture.video_capture", "capture.audio_capture",
             "modules.gesture_song", "modules.depth_su",
             "modules.speech_huo", "modules.gpu_worker",
             "modules.su_depth", "modules.su_depth.integration",
             "modules.su_depth.depth_processor", "fusion.state_machine"):
    importlib.import_module(name)

from capture.video_capture import _draw_sim_frame, FRAME_H, FRAME_W
from common import simscript
state = simscript.make_state()
frame = _draw_sim_frame(state)
assert frame.shape == (FRAME_H, FRAME_W, 3)
assert frame.dtype.name == "uint8"
assert (frame != 45).any(), "numpy fallback must draw a person"
assert not blocked.intersection(sys.modules)
'''
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=str(root), env=env,
        text=True, capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
