"""Bounded shutdown and partial-start regressions with real child processes."""
import multiprocessing as mp
import signal
import subprocess
import sys

import pytest

import main


def test_partial_start_failure_stops_started_capture_workers():
    script = '''
import multiprocessing as mp
import main
process_type = mp.get_context().Process
class FailedStart:
    def start(self):
        raise RuntimeError("deliberate partial startup failure")
def create_process(target, name, args):
    if name == "gesture":
        return FailedStart()
    return process_type(target=target, name=name, args=args)
main.mp.Process = create_process
try:
    main.main(["--sim"])
except RuntimeError as exc:
    assert "deliberate partial startup failure" in str(exc)
else:
    raise AssertionError("Expected startup failure")
assert not mp.active_children(), "Started capture workers survived failure"
print("PARTIAL STARTUP CLEANUP: OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=main.BASE_DIR,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PARTIAL STARTUP CLEANUP: OK" in result.stdout


def _ignore_shutdown(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        signal.pause()


@pytest.mark.skipif(
    not hasattr(signal, "SIGKILL") or not hasattr(signal, "pause"),
    reason="This process-reaping regression requires POSIX signals",
)
def test_shutdown_reaps_worker_ignoring_stop_and_terminate():
    # A spawned worker imports only this test module; no sensors or models run.
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    stop = ctx.Event()
    worker = ctx.Process(target=_ignore_shutdown, args=(ready,))
    worker.start()
    try:
        assert ready.wait(timeout=5), "Child did not initialize"
        main._stop_processes([worker], stop, [], timeout=0.05)
        assert stop.is_set()
        assert not worker.is_alive()
        assert worker.exitcode == -signal.SIGKILL
    finally:
        if worker.is_alive():
            worker.kill()
        worker.join(timeout=2)
