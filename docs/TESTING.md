# Synthetic validation

The retained team's software is tested on x86-64 Linux, CPython 3.12.14,
NumPy 1.26.4, and pytest 8.3.5. Optional GUI checks use PyQt5 5.15.11,
Qt 5.15.18, and PyQt5-sip 12.17.2. These are PC test dependencies, not a
validated Jetson runtime. See [SETUP.md](SETUP.md).

## Commands

Run from the repository root after installing `requirements-dev.txt`:

```bash
python -m pytest -q
python -m tests.test_fusion
python -m tests.smoke_sim
```

The pytest suite exercises fusion transitions and time boundaries, simulation
clock/cycle logic, queue policies, SU's packet validation and processor control
with injected estimators, blocked hardware imports, worker failures, microphone
argument routing, and cleanup after GUI startup failure. Integration coverage includes mocked
sensor failure/reconnection, model initialization/runtime failures, Vosk fallback
health, status transitions and formatting, monotonic heartbeat freshness, and
out-of-order/nonfinite fusion events. It performs no trained
inference and requires no graphical display or hardware packages.

The historical `tests.test_fusion` module remains executable and checks three
scenarios with a controlled clock. The expanded fusion regressions likewise run
with `python -m unittest tests.test_fusion_regressions`.

The 16-second smoke test executes the original five worker functions, checks
synthetic frames, all five heartbeats, event envelopes and ordering, relative
depth levels, L1 followed by L2, worker liveness, and clean zero-exit shutdown.
It deliberately uses `spawn` to exercise fresh worker imports. The main UI
retains the platform's multiprocessing default; this is a distinct startup
check rather than a claim that both use the same context.

Pure state and scenario tests use controlled clocks instead of sleeping through
the scenario. The smoke and optional GUI checks use actual process pacing and
bounded waits because they verify live queue and timer behavior.

## Optional Qt check

After installing `requirements-ui.txt`, run:

```bash
QT_QPA_PLATFORM=offscreen python -m tests.gui_sim
```

This separately checks the real window's video pixmap, relative-depth indicator,
heartbeats, L1/L2 banner, manual reset, creation of a temporary alert log, and
worker shutdown. It exits after observing the scripted sequence, with a
20-second deadline. It constructs and drives Qt widgets offscreen; it does
not test graphical interaction or real sensors. It leaves no recorded sensor
data. The headless pytest suite does not depend on Qt.

## Historical integration defects reproduced before fixes

The immutable source-only archive was inspected and executed separately before
production edits. Its original three fusion tests passed. The first complete
set of 29 new fusion regressions produced **21 failures and 8 passes**. The
same 29 cases then passed after the scoped correction; an additional injected
clock API case covers the final implementation. Three later reset-cutoff regressions reproduced two
failures in both the original and interim implementation before correction;
all three now pass. The initial fusion regression module contained 33 cases; these remain in
the current suite.

| Confirmed original behavior | Integration correction |
| --- | --- |
| Any truthy speech keyword can escalate L1, regardless of confidence. | Require a recognized help keyword and finite numeric confidence in [0.50, 1.0]. |
| A speech timestamp before the fall passes the upper-bound-only window check. | Require a nonnegative fall-to-help delay, up to and including 10 seconds. |
| Duplicate falls downgrade L2 and restart the escalation window. | Preserve severity and the first fall timestamp; refresh qualifying activity only. |
| Delayed queued speech can revive an incident that has already expired at receipt. | Expire the incident before processing event data. |
| `tick(now=0)` substitutes real time; reset retains a fall-specific region. | Distinguish zero from `None`, and clear incident region while retaining depth context. |
| Queued falls emitted before manual reset can reopen the old incident. | Reject fall/help events at or before the reset cutoff; use the inactivity deadline cutoff for automatic expiration so fresh queued falls remain valid. |
| Importing the hardware console under pytest tries to parse `-q` as seconds. | Parse console arguments only inside its entry point. |
| The original smoke reports success while its video worker remains alive. | Drain feeder queues during shutdown and assert all five zero exit codes. |
| `main.py` passes `--mic` as the Vosk model directory. | Pass the index only to audio capture; preserve the speech model-path default. |
| A Qt startup failure leaves child cleanup outside the failure path. | Put started workers in a `finally` cleanup path with bounded joins and reaping. |
| Source-only archive lacks the log directory, so app alert logging silently fails. | Create the local log directory before instantiating fusion. |

Unrelated or invalid speech no longer extends alert life. Valid help outside
the escalation window can refresh an active incident without raising severity;
duplicate confirmed falls also refresh activity. Timeout remains strictly
greater than 30 seconds. These changes are intentional and covered by
regressions, rather than a broader redesign of the team's fusion contract.

## Timing regressions

Fifteen added fusion cases exercise invalid explicit timestamps and reordered
falls. Against the earlier integration baseline they produced 13 failures and 2 passes;
after the scoped correction all 15 pass alongside the unchanged 33 initial
fusion regressions. A missing timestamp retains the receipt-time default, while
an explicit nonnumeric, boolean, `None`, NaN, or infinite fall/help timestamp is
ignored. An active incident rejects a fall older than its initiating fall, and
an older duplicate cannot replace the region reported by a newer fall.

No production clock conversion was made. Event/fusion timestamps remain wall
clock; pose/depth packet ages and heartbeat pacing/receipt ages remain
monotonic. The packet producer still assigns `captured_at` after pose inference,
so the adapter does not establish full capture-to-depth freshness. Details and
remaining limitations are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Evidence and limits

The earlier preparation suite had 65 passing cases; this is a historical
baseline, not the current suite count. The reviewed integration expanded it
to 132 cases. The independent portfolio snapshot preserves the application,
tests and dependency files byte-for-byte. Revalidation uses the commands
above, including the five-worker smoke and separate offscreen GUI check;
[PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md) records the current publication
conditions and validation scope. Internal phase reports and private
development-history links are omitted from the portfolio snapshot.
No expected-failure markers or disabled assertions are used to make the suite
pass. The pytest runner reports the actual discovered test count.

Simulation does not validate pose or speech accuracy, real camera/microphone
operation, TensorRT conversion, Depth Anything V2 inference, Jetson throughput
or memory, Python 3.6 runtime compatibility, SMS delivery, or any safety claim.
The synthetic depth worker reads shared scene state rather than consuming pose
packets; separate adapter tests cover packet control logic with injected
processors, not GPU inference.

## CI strategy

`.github/workflows/synthetic.yml` defines read-only GitHub Actions permissions,
Python 3.10/3.12 targets, pinned PC dependency installation, pytest, the retained
fusion entry point, and the five-worker smoke in the headless jobs. Those jobs
need no secrets, models, CUDA, sensors, Qt, or recorded data. A dedicated
Python 3.12 Qt job explicitly installs `requirements-ui.txt`,
sets `QT_QPA_PLATFORM=offscreen`, and runs the GUI simulation within a
five-minute job timeout. The headless jobs remain independent of Qt. Hardware
and model validation remain a separate manual phase on the actual Jetson.

The workflow pins the runner OS family to `ubuntu-24.04` and official action
revisions to full commit hashes: checkout v5.1.0 and setup-python v6.3.0.
GitHub can still update the hosted image within that OS family. These changes
retain all three jobs, their Python targets, dependencies, commands, assertions
and timeouts. Hosted results for the independent repository must correspond
to its clean commit. Runs from the original development repository are not
presented as validation runs of this independent repository.

An early hosted setup failed before executing tests because setup-python's
default cache discovery expects requirements.txt or pyproject.toml. The workflow
now supplies the actual requirements-sim.txt and requirements-dev.txt paths
explicitly. Official checkout/setup-python actions use their Node 24 versions.
This setup failure is distinguished from application test outcomes.
