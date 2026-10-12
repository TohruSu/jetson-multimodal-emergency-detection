# Source provenance and team contributions

## Source input and history

Project preparation used the owner-supplied `jetson_multimodal_source_only.zip`, rather
than the GitHub initialization README, as the original application source.
The archive contains one `jetson_final_project/` directory with 35 files.
Its SHA-256 is:

```text
7d6e98550e321416740bfc87fb84e73d39893d249bc3a8ecc1756613916c0ec6
```

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records every original file's
SHA-256 before changes. Those are source-input hashes, not hashes of the final
prepared versions. The original Python layout is retained at the repository
root; source files have scoped corrections and regression tests rather than
an application rewrite. The historical README is rewritten around current
source behavior; relevant attribution, architecture, model requirements, and
Jetson context are retained in the current documentation.

The source archive supplies no Git history or commit-level authorship metadata.
This independent portfolio repository starts from an export of the reviewed
source tree, without importing the development repository's Git objects,
branches, PRs or Actions history. Its clean initial commit represents a
portfolio publication snapshot, not original individual authorship of the
historical university team project. The account making that commit is not
credited as the sole author of the application. No contribution percentages
or personal names beyond the supplied team identifiers are inferred.

The original development repository retains the team's preparation and review
records. Two internal phase reports are omitted from the portfolio snapshot;
the original source inventory, technical behavior, regression evidence,
attribution and applicable rights remain documented here and in the linked
technical guides. This independent history does not erase copies, caches or
service records of any earlier repository.

## Team responsibilities

| Team identifier | Documented responsibility | Evidence and limits |
| --- | --- | --- |
| **SU** | Depth Anything V2 integration, relative depth, person ROI processing, pose-to-depth protocol and integration | Historical README; `modules/depth_su.py` owner header; `modules/su_depth/` interfaces. Ownership of this repository does not establish exclusive authorship of other components. |
| **SONG** | Pose estimation and fall detection | Historical README; `modules/gesture_song.py` owner header and integration comments; `modules/song/fall_detector.py`. NVIDIA's pose framework and topology asset remain separate third-party dependencies/content. |
| **HUO** | Speech recognition, keyword model and feature extraction | Historical README; `modules/speech_huo.py` owner header; `modules/huo/feature.py` attribution in streaming wrapper. The streaming adapter also contains team integration logic. |
| **Martin** | Fusion state machine, UI, and main application integration | Historical README and owner headers in `main.py`, `fusion/state_machine.py`, and `ui/main_window.py`; `common/messages.py` credits the message architecture decision. |
| **Team integration; individual authorship uncertain** | Merged GPU scheduling, sensor capture, queue/simulation utilities, streaming glue, diagnostics and integration tests | These support the shared application. No reliable record supports assigning their entire contents exclusively to SU or another teammate. |

These describe evidenced responsibilities, not a line-by-line authorship claim.
The owner, SU, reports that teammates agreed to GitHub publication. The
owner selected source-visible publication for viewing and technical evaluation
without a repository-wide application license. Reported publication consent is
separate from any future agreement to grant broader reuse rights;
[LICENSING.md](LICENSING.md) records that distinction and independently
applicable third-party rights. The repository remains private for review.

## Complete original file inventory

Paths below are relative to the archive's `jetson_final_project/` root.
“Preserved” means the component remains in the cleaned project, including any
explicitly documented integration corrections; it does not promise byte identity.
Original bytes can be identified using the manifest.

| Original path | Runtime role / attribution | Prepared-source disposition |
| --- | --- | --- |
| `README.md` | Historical team architecture, responsibilities, setup claims and assignment context | Rewritten; relevant context retained and unsupported claims qualified. |
| `main.py` | Main process orchestration, Qt timer, simulation controls; Martin | Preserved with scoped startup/shutdown and integration corrections. |
| `common/__init__.py` | Package marker | Preserved. |
| `common/messages.py` | Shared event/status/detection dictionary contract; Martin's architecture decision | Preserved. |
| `common/qutils.py` | Latest-item queues, event draining, child signal helper; shared integration | Preserved. |
| `common/simscript.py` | Shared synthetic clock, fall/help phases and depth score; shared integration | Preserved. |
| `capture/__init__.py` | Package marker | Preserved. |
| `capture/audio_capture.py` | Real PyAudio PCM producer or synthetic silence; shared integration | Preserved. |
| `capture/video_capture.py` | Real OpenCV camera producer or synthetic scene; shared integration | Preserved. |
| `modules/__init__.py` | Package marker | Preserved. |
| `modules/depth_su.py` | Depth process contract, synthetic levels, standalone real packet adapter; SU and integration | Preserved. |
| `modules/gesture_song.py` | SONG detector wrapper, fall events, boxes, selected person packet; SONG and team integration | Preserved. |
| `modules/gpu_worker.py` | Sequential real pose/depth in one CUDA process; team integration | Preserved. |
| `modules/speech_huo.py` | HUO keyword wrapper, optional Vosk, scripted help; HUO and team integration | Preserved. |
| `modules/huo/__init__.py` | Package marker | Preserved. |
| `modules/huo/classifier.py` | Streaming PCM windows, energy gate, ResNet classifier and cooldown; HUO model / team integration | Preserved. |
| `modules/huo/feature.py` | Torch STFT, mel filterbank and spectrogram feature extraction; HUO | Preserved. |
| `modules/song/__init__.py` | Lazy library integration note | Preserved. |
| `modules/song/fall_detector.py` | Pose features, per-person tracker/fall state machine; SONG | Preserved; real runtime remains hardware-dependent. |
| `modules/song/human_pose.json` | NVIDIA `trt_pose` keypoints and skeleton topology, not a SONG-authored model | Preserved with verified upstream origin and scoped third-party notice. |
| `modules/su_depth/__init__.py` | Depth API exports and lazy estimator access; SU/integration | Preserved. |
| `modules/su_depth/config.py` | Depth defaults and reusable CLI argument helpers; SU | Preserved; not all helpers are used by `main.py`. |
| `modules/su_depth/depth_estimator.py` | External Depth Anything V2 model wrapper, ROI statistics and depth colorization; SU | Preserved; model/framework are supplied separately. |
| `modules/su_depth/depth_processor.py` | Camera-free depth processing, bbox validation, EMA, classification and reset; SU | Preserved. |
| `modules/su_depth/integration.py` | Pose packet creation/validation, stale/no-target handling and identity reset; SU | Preserved. |
| `fusion/__init__.py` | Package marker | Preserved. |
| `fusion/state_machine.py` | L1/L2 alert fusion, contextual text and local logging; Martin | Preserved with minimal verified fusion corrections. |
| `ui/__init__.py` | Package marker | Preserved. |
| `ui/main_window.py` | Qt video/boxes, relative-depth indicator, alert and heartbeat UI; Martin | Preserved; historical “map” wording does not match the current depth-bar display. |
| `tools/build_trt_engine.py` | Optional local pose-engine conversion; team integration | Preserved; no engine built or checkpoint loaded during source preparation. |
| `tools/check_mic.py` | Local input enumeration and RMS diagnostic; team integration | Preserved; hardware-only and may display local device names. |
| `tests/__init__.py` | Package marker | Preserved. |
| `tests/test_fusion.py` | Original fusion scenario, no-fall help and reset tests; shared validation | Preserved and extended for confirmed defects. |
| `tests/smoke_sim.py` | Original five-process synthetic event-flow smoke; shared validation | Preserved and extended for reproducible validation. |
| `test_console.py` | Headless real capture/inference diagnostic; team integration | Preserved; optional hardware depth remains explicit. |

All original files were inspected, including complete fall-detector,
depth-processing, capture, UI, tools, and test implementations. Runtime roles
and limitations are detailed in [ARCHITECTURE.md](ARCHITECTURE.md), with local
validation and the original-versus-corrected fusion behavior in
[TESTING.md](TESTING.md).

## Third-party content and model provenance

The `human_pose.json` bytes match NVIDIA's official `trt_pose` asset. The
upstream revision, asset hash and applicable MIT notice are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). That notice applies to this
asset; it does not relicense the team's application or model weights.

The inspected team source imports third-party frameworks instead of vendoring
their full implementations. No additional copied third-party code was
established from the archive alone. That is a source review result, not an
independent legal clearance of every original line.

The archive intentionally omits the three historical `.pth` artifacts. Their
filenames and expected loading conventions are evidence of dependencies,
not proof of the actual weights' identity, integrity, authorship, or license.
[MODELS.md](MODELS.md) distinguishes official upstream sources from the absent
team artifacts, including unknown rights for the custom speech model and the
ambiguous pose checkpoint format. No model was fetched, restored, uploaded,
committed, or deserialized during source preparation or clean staging.

## Historical assertions and publication boundary

The old README contains conflicting depth performance claims, claims that
model files are included, dated team milestones, and obsolete stub descriptions.
The supplied archive has no weights and cannot reproduce its hardware results.
The `_trt.pth` filename is also described inconsistently as an engine and plain
PyTorch weights. These are historical statements, not measurements from source preparation.
No new inference speed, speaker accuracy, or Jetson compatibility claim is made.

The system displays/logs emergency text rather than sending SMS. Its position
context is a camera-frame region plus relative depth, not geographic coordinates
or metric distance. Synthetic tests validate software control flow and queue
communication; they do not establish reliable physical monitoring.

The owner's source-visible, no-repository-wide-license decision governs the
proposed source-only publication; selecting MIT is not a remaining requirement.
Concrete permission or provenance concerns are reported separately in
[PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md). Establish model/training rights
before any future artifact distribution, and validate the intended hardware
runtime before claiming real-world inference or performance. Keep local
checkpoints, recordings, logs, generated engines, caches, environment
directories and credentials out of Git. Clean staging does not authorize
public visibility, a release, or an application-wide license. The independent
repository remains private until the separate publication conditions and
owner authorization are satisfied.
