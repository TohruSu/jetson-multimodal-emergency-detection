# -*- coding: utf-8 -*-
"""
Queue helpers (multiprocessing.Queue).
"Latest frame only" policy: overwrite the old frame if it was not
consumed, so latency never builds up on the Jetson Nano.
"""
import queue


def put_latest(q, item):
    """Replace the content of a maxsize=1 queue with item, non-blocking."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def get_latest(q, timeout=0.5):
    """Get the latest item, or None if nothing within timeout."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def drain(q, max_items=100):
    """Empty the queue and return the list of items (fusion/UI side)."""
    items = []
    for _ in range(max_items):
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            break
    return items


def ignore_sigint():
    """Child processes ignore Ctrl+C; main.py stops them via stop_event.

    Without this, SIGINT hits every process at once and each one dumps a
    KeyboardInterrupt traceback, burying the useful log lines.
    """
    import signal
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
