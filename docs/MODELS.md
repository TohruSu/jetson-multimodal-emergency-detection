# Local model requirements

The supplied archive intentionally excludes model weights. Phase C1 does not download, inspect by deserialization, restore, or redistribute them. Synthetic mode and headless tests require none of these files. A `.pth` suffix does not establish whether a file is a tensor state dictionary, a pickled Python object, or a TensorRT engine.

| Local file under `models/` | Consumer and loading convention | Behavior when absent |
| --- | --- | --- |
| `resnet18_baseline_att_224x224_A_epoch_249_trt.pth` | Song integration: build `trt_pose.models.resnet18_baseline_att(18, 42)`, load a plain state dictionary with `torch.load(..., map_location="cuda")` then `load_state_dict`; attempt fp16. The filename alone is misleading. | If no separate engine exists, pose initialization fails. The real GPU worker reports `unavailable` pose health, stays idle, and cannot produce observed fall events. It also skips depth initialization. |
| `resnet18_baseline_att_224x224_A_epoch_249_trt_engine.pth` | Optional locally built engine: `TRTModule().load_state_dict(torch.load(...))`. The wrapper prefers this file whenever it exists. | The wrapper attempts plain PyTorch weights instead and reports the backend selected. An existing but incompatible engine causes initialization failure; this failure does **not** automatically retry plain weights. |
| `depth_anything_v2_vits.pth` | Su's `DepthEstimator`: `DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])`, load a state dictionary on CPU, then move the model to the selected device. It first requests `weights_only=True`; older PyTorch falls back on `TypeError` to ordinary `torch.load`. | Real `main.py` can run pose and speech with depth disabled, its default. With `--depth`, depth remains unavailable if loading fails. The merged worker reports `unavailable` backend health in that case; without `--depth`, it reports `disabled`. It does not fabricate a depth inference result. |
| `speech.pth` | Huo integration: torchvision ResNet-18 with a two-output `fc`, direct state dictionary via `torch.load(..., map_location=device)` and `load_state_dict`. Class index 0 is `hello`, 1 is `help`. | Attempts the optional Vosk recognizer using a local `models/vosk/` directory. With neither backend, real speech reports `unavailable` health, stays inactive, and emits no help events. |
| `vosk/` directory | Optional Vosk model directory consumed by `vosk.Model`, then a 16 kHz `KaldiRecognizer`. This is a separate model format, not a PyTorch checkpoint. | No Vosk fallback. It is unnecessary when the Huo checkpoint and runtime work. |

The shape counts for the pose model come from the preserved skeleton: 18 keypoints and 21 links. The original standalone `modules/song/fall_detector.py` expects a **TensorRT state dictionary** at its `--model` path, even though its default name ends in `_trt.pth`. `tools/build_trt_engine.py` instead loads plain weights on CPU and writes the separate `_trt_engine.pth` with `torch.save(model_trt.state_dict(), ...)`; it treats a dictionary containing `engine` as already converted. Do not run that tool as a file-inspection utility: it deserializes its input and may build a GPU engine.

## Origin evidence and rights

| Artifact or framework | Evidence | Limits of that evidence |
| --- | --- | --- |
| Depth Anything V2 Small | The source imports the external `depth_anything_v2.dpt` framework and uses the official Small configuration. The [official README at the inspected revision](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/README.md#pre-trained-models) links a public Hugging Face checkpoint with this filename. Its [license section](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/README.md#license) states Small is Apache-2.0, while Base/Large/Giant are CC-BY-NC-4.0. | These are upstream, model-specific statements. The missing historical file has no verified identity, source receipt, or checksum in this archive. Different variants cannot be treated as having the Small model's terms. No weights or external framework are vendored here. |
| Pose model | NVIDIA's [official trt_pose README at the inspected revision](https://github.com/NVIDIA-AI-IOT/trt_pose/blob/336b8cdd9c2d81ff527d9ff7e8f1749e39e6f6e4/README.md#step-3---run-the-example-notebook) lists `resnet18_baseline_att_224x224_A`, pretrained on MSCOCO, and links public Google Drive weights. The source uses that architecture. | This establishes an available upstream model family, not the identity or rights of the absent team checkpoint. Historical notes call the delivered `_trt.pth` plain weights; an older header calls it an engine. We could not resolve its contents without the file. The upstream software license does not alone establish all model/dataset redistribution terms. |
| `speech.pth` | The historical README attributes training to Huo, describes a two-class ResNet-18 keyword model and a small single-speaker dataset. `modules/huo/classifier.py` and `feature.py` establish the inference architecture and feature transformation. | Training code, recordings, checkpoint metadata, and consent/license records are absent. No official public source is evidenced for this project-specific checkpoint. Huo's implementation contribution does not establish permission to redistribute recordings or weights. |
| Optional TensorRT engine | The repository includes its local conversion recipe. | No engine was supplied. An engine depends on target hardware, TensorRT, and the conversion environment; it is not a portable substitute for verified source weights. Historical conversion failures and speed claims remain unverified. |
| `modules/song/human_pose.json` | Byte-identical to NVIDIA's official skeleton at the inspected revision. Its scoped MIT notice and verification hash are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). | This notice applies to that asset. It grants no blanket license to the team's code, model weights, or external frameworks. |
| Vosk fallback model | Source expects a local directory. | No particular language pack, source, version, or license is established by the archive. Select and review an official model separately if this fallback is needed. |

The official README, license, and skeleton checks used verified HTTPS and read-only Git access. Only text/source references were inspected, outside the checkout; no checkpoint link was downloaded or followed to deserialize a model. The pinned README, license, and skeleton evidence was checked again during Phase C3 on 2026-10-12. The recorded revisions identify inspected evidence, not a tested or recommended Jetson dependency combination.

## Dependency and distribution boundaries

The team's original application source is [MIT-licensed](../LICENSE) under the
owner's [confirmed collective authorization](LICENSING.md). This grant does
not relicense the bundled NVIDIA asset, separately installed dependencies or
absent models, datasets, recordings and generated TensorRT engines.
The repository supplies dependency declarations and integration source, not
copies of the PC virtual environment or Jetson runtime packages.

The tested PC packages have the following license evidence in their installed
metadata and supplied license files:

| Package pin | Package license evidence |
| --- | --- |
| NumPy 1.26.4 | BSD-3-Clause for NumPy, with additional notices for libraries bundled in its wheel. |
| pytest 8.3.5, iniconfig 2.3.1, pluggy 1.6.0 | MIT. |
| packaging 26.3 | Apache-2.0 OR BSD-2-Clause. |
| PyQt5 5.15.11 | GPL v3 for the PyPI build; the publisher also offers a separate commercial license. |
| PyQt5-Qt5 5.15.18 | LGPL v3 for this Qt runtime package. |
| PyQt5-sip 12.17.2 | BSD-2-Clause. |

PyQt and Qt have distinct terms. The publisher's [PyQt5 package
description](https://pypi.org/project/PyQt5/5.15.11/) identifies the GPL and
commercial options; the [Qt runtime package
description](https://pypi.org/project/PyQt5-Qt5/5.15.18/) identifies LGPL v3.
The current source imports PyQt5 rather than vendoring it. Riverbank's
[licensing guidance](https://www.riverbankcomputing.com/static/Docs/PyQt5/introduction.html#license)
identifies GPLv3 and commercial options. GNU lists the
[Expat/MIT license as GPL-compatible](https://www.gnu.org/licenses/license-list.html#Expat).
MIT adoption therefore removes the earlier conflict with withholding general
reuse rights, but it does not waive GPL obligations for distribution of a
combined application using GPL PyQt5 or establish commercial entitlement.
The separately installed Qt runtime has its own
[LGPL obligations](https://www.qt.io/licensing/open-source-lgpl-obligations).
A combined GUI application, executable, container or bundled environment
requires review of its actual PyQt GPL/commercial and Qt LGPL obligations;
no distribution scenario is legally certified here.

The historical Jetson dependencies are not a pinned, rights-reviewed binary
distribution. Any future packaging must review the actual versions and
notices for PyTorch, torchvision, OpenCV, Pillow, PyAudio/PortAudio, Vosk,
trt_pose, torch2trt, CUDA and TensorRT, as applicable. Framework software terms
do not establish the identity or redistribution rights of missing trained
models. Checkpoints, datasets, recordings, and generated engines remain
excluded from the source-only publication scope.

## Supplying trusted local files

1. Establish the source, applicable rights, version, and integrity of each artifact independently. Keep receipts and verified hashes locally; this repository has no verified hashes for the missing weights.
2. Place only approved, trusted files in the ignored `models/` directory using the names above. Keep Vosk's complete model directory at `models/vosk/` if used. Do not commit weights, engines, recordings, or a Git LFS pointer to publish them.
3. Review the legacy loader code before choosing real mode. Ordinary `torch.load` can execute pickled content, and the depth compatibility fallback uses it on older PyTorch. `weights_only=True` reduces that risk but does not establish provenance or permission. Do not deserialize a file just to find out what it contains.
4. Start real mode only as an intentional hardware validation step after runtime and artifact review. Run from the repository root because speech and Vosk paths are relative to the current directory. Pose/depth integration paths resolve to the repository's `models/` folder; `DepthProcessor`'s standalone default is instead `checkpoints/...`, and a direct `DepthEstimator(model_path=None)` resolves under `modules/checkpoints/`. The main integration overrides these differing defaults.

The synthetic workflow is reproducible without model files. The retained real-mode source is an integration starting point, not a verified hardware deployment or a medical/safety monitoring guarantee.
