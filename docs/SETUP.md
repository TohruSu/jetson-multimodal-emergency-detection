# Development setup

Run the commands below from a normal checkout of this repository. Keep the PC simulation environment separate from any Jetson platform environment.

## Synthetic / PC workflow

The PC development target is CPython 3.10–3.12. Phase C1 was validated locally on Python 3.12 and in hosted headless CI on Python 3.10 and 3.12; see [TESTING.md](TESTING.md) for the validation scope. This target is separate from the historical Jetson runtime. Synthetic mode needs no CUDA, TensorRT, sensors, PyTorch, checkpoints, or display for the headless checks.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m tests.test_fusion
python -m tests.smoke_sim
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead.

`requirements-sim.txt` supplies NumPy for capture arrays and depth processing logic. `requirements-dev.txt` includes it and the tested pytest dependency pins. These pins are for modern PC development, not JetPack 4.6. No sensor or model package belongs in the base synthetic dependency set.

The optional graphical simulation additionally needs PyQt5 and a desktop display:

```bash
python -m pip install -r requirements-ui.txt
python main.py --sim
```

OpenCV is optional for the synthetic drawing overlay: if unavailable, the capture code draws a simpler NumPy scene. Headless CI need not install it. `QT_QPA_PLATFORM=offscreen` can exercise Qt construction on a machine without a display, but is not an interactive demonstration. Do not count an offscreen check as observed fall-detection accuracy or sensor validation.

The demo uses a scripted 45-second cycle: fall at 10 seconds, help at 13 seconds, and alert expiration after the configured inactivity period. `F` and `H` drive the scenario only with `--sim`; `R` clears alert/tracker state. These events and confidence values are synthetic. Alerts are shown and logged locally, rather than sent by SMS. The current UI shows video and relative-depth bars; it does not implement a geographic map.

## Historical Jetson hardware workflow — unverified

The original team documentation targets Jetson Nano with JetPack 4.6.x / Ubuntu 18.04, system Python 3.6.9, CUDA 10.2, and a Jetson PyTorch build described as “1.10 max.” No exact reproducible hardware dependency set or external-framework revision was supplied. Those historical constraints are retained as evidence, not a compatibility guarantee. Do not install modern PC requirements on that system and infer that the hardware path has been upgraded or tested.

| Component | Required runtime or resource | Status / caveat |
| --- | --- | --- |
| Video capture | USB/webcam supported by OpenCV, camera index `--cam` | 640×480 requested; default real capture target is 12 fps. Requested capture rate is not a measured inference rate. |
| Audio capture | PyAudio, PortAudio, mono 16 kHz int16 input, `--mic` override | Input device is selected by USB/camera/microphone name where possible. `tools/check_mic.py` is a real four-second microphone diagnostic, not a synthetic test. |
| GUI | Jetson's system `python3-pyqt5`, working display | The historical README recommended apt rather than pip on Jetson. Do not use the PC Qt pins as a Jetson prescription. |
| Pose / fall pipeline | CUDA-capable PyTorch + compatible torchvision, Pillow, OpenCV, `trt_pose`, `torch2trt`, trusted local checkpoint | CUDA is used explicitly. The standalone SONG module imports `torch2trt.TRTModule` even when the wrapper ultimately chooses plain PyTorch weights. No CPU pose fallback is implemented. |
| Keyword model | Compatible PyTorch + torchvision; trusted `speech.pth` | Streaming features are computed with PyTorch. CPU inference is the default to reduce extra GPU contexts. |
| Optional depth | Compatible PyTorch + OpenCV + external Depth Anything V2 repository on `PYTHONPATH`, trusted Small checkpoint | Enabled only by `--depth` in real main. It loads after pose in the same GPU worker, runs on demand after a confirmed fall, and refreshes during idle periods. |
| Optional generic speech | Vosk + reviewed local model directory | Only used when HUO's model cannot initialize. No Vosk package/model was installed during the base synthetic setup. |

Before running hardware mode, obtain compatible platform packages through official sources and record exact versions for JetPack, CUDA, TensorRT, PyTorch, torchvision, `trt_pose`, `torch2trt`, and Depth Anything V2. Keep external frameworks outside this repository; use an explicitly selected, reviewed checkout on `PYTHONPATH` for depth. Installing all of the current upstream depth demo requirements (including Gradio) is not necessary for this adapter and does not establish Python 3.6 support.

The [official trt_pose instructions](https://github.com/NVIDIA-AI-IOT/trt_pose/blob/336b8cdd9c2d81ff527d9ff7e8f1749e39e6f6e4/README.md#getting-started) describe PyTorch/torchvision and torch2trt prerequisites. The [official Depth Anything V2 README](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/README.md#usage) describes its framework and checkpoint API. The inspected revisions are provenance references, **not** a hardware compatibility pin.

### Compatibility checks still required

- `modules/su_depth/__init__.py` exposes `DepthEstimator` through module-level `__getattr__`, which Python recognizes starting in 3.7. On Python 3.6 that convenience export is not available. The integration's `DepthProcessor` imports the estimator directly inside its constructor, so the limitation is specific to the package export; syntax parsing alone cannot demonstrate real-mode compatibility.
- Depth's loader requests `weights_only=True`, then falls back to ordinary `torch.load` for older builds. Historical PyTorch 1.10 cannot supply that restricted loading option. Review trusted artifact handling before loading anything.
- `torch.inference_mode`, `torch.stft(return_complex=True)`, torchvision ResNet construction, and trt_pose/torch2trt compiled extensions require compatible runtime builds. The classifier branches between modern `weights=None` and historical `pretrained=False`; that branch does not validate the rest of the stack.
- The inspected upstream Depth Anything V2 `dinov2.py` still passes `antialias` to `torch.nn.functional.interpolate`. The historical README records removing this argument for PyTorch 1.10. That external compatibility change was not applied or tested in Phase C1; review and validate any such patch separately rather than silently modifying dependencies.
- TensorRT engines depend on their build platform/runtime. Historical notes report conversion failures with JetPack 4.6 + torch2trt 0.5.0; the optional local build recipe was retained but not executed.

### Intentional hardware validation

After compatible dependencies and reviewed model files are supplied, launch from the repository root:

```bash
python3 main.py --cam 0 --mic 0
# Include depth only when its framework/checkpoint/runtime are ready:
python3 main.py --cam 0 --mic 0 --depth --depth-size 154 --depth-period 20
```

Camera/microphone indices are examples; identify actual devices first. Default real mode starts four child processes: video, audio, a combined pose/depth GPU worker, and speech. Synthetic mode starts five children with separate light gesture/depth producers. The original separate real adapters are preserved, but main selects the merged GPU worker to reduce duplicated GPU contexts on memory-constrained hardware.

Depth is relative, normalized against the current image, with ROI trimming and EMA smoothing; its FAR/MIDDLE/NEAR thresholds are historical calibration values. It provides neither metric distance nor world coordinates. Historical notes conflict between about 0.34 seconds and about 6 seconds per depth inference. These timings, GPU memory estimates, fp16 speedups, keyword calibration scores, and Nano performance claims were **not remeasured**. The default of disabling real depth and using an on-demand/20-second schedule is code behavior, not an assurance that inference meets a useful monitoring rate.

### Failure behavior and recovery

The status row separates worker activity from the last reported sensor/backend state. Every status preserves `state="alive"` and includes a `health` value and `backend`; failures and fallbacks add a fixed `reason` code. The UI displays `READY`, `SIMULATION`, `DEGRADED`, `UNAVAILABLE`, `DISABLED`, or `INITIALIZING`, with backend/reason details. Before a status arrives it displays `WAITING`; after five seconds without a received status it displays `STALE` and the last health state. Old status messages without health fields display `ALIVE (health unknown)`.

- **Camera and microphone:** `READY` requires a successful real read. Missing dependencies, open failure, or a read failure before any successful data are `UNAVAILABLE`. A read failure after successful capture is `DEGRADED`. Failed resources are released; initialization is retried after at least two seconds, and successful capture returns to `READY`. A later failed reopen reports `UNAVAILABLE`. Real mode queues no synthetic replacement frame or fabricated silent audio; simulation produces those only after an explicit `--sim`. The display can retain the last genuine frame during camera failure, alongside its health indication.
- **Pose and speech models:** initialization failure is `UNAVAILABLE` and needs dependency/model correction plus application restart. Inference failure is `DEGRADED`; the worker remains alive and attempts later inputs. Successful subsequent processing restores its normal health. HUO's optional Vosk fallback is `DEGRADED` with a fallback reason, so an active generic recognizer is distinguishable from the intended classifier. If both recognizers fail initialization, speech remains unavailable.
- **Depth:** real depth is `DISABLED` without `--depth`. With it enabled, missing pose in the merged worker is `UNAVAILABLE` with `pose_unavailable`, and depth initialization failure is `UNAVAILABLE`; failed inference or stale input is `DEGRADED`. Runtime retries occur on the next scheduled or requested depth packet. No person target is a valid no-target result, rather than a backend failure. Depth health is emitted even when inference is disabled or unavailable. See [MODELS.md](MODELS.md) for exact loading paths, including the existing incompatible-engine case that can mask plain pose weights.

Health changes emit a status immediately and unchanged health repeats nominally every two seconds. Console failure messages are limited to state transitions. Status delivery is best effort through the bounded event queue; a full queue drops a status and a later heartbeat retries it. These indicators do not assert whole-system readiness or recognition accuracy: an initialized model may be waiting for an unavailable sensor.

### Readiness limits

The UI measures heartbeat freshness from monotonic receipt time, independently of the wall-clock event timestamp. This measures receipt of worker status, not age of camera/audio observations. Long synchronous GPU inference can delay heartbeats, and a stalled device-driver read may block before the retry logic receives an error. Recovery tests use mocks and do not establish reliable unattended operation on hardware.

Depth bars update when an event arrives and do not independently expire the displayed value. Pose packets stamp `captured_at` after pose inference rather than at camera capture, so the adapter's one-second age check does not measure total camera-to-depth latency. Fusion still uses wall-clock time and can be affected by clock adjustments. Hardware validation requires observing actual devices, each backend, failure reporting, and end-to-end behavior on a Jetson; synthetic and mocked results do not prove it.
