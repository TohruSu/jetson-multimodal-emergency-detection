# -*- coding: utf-8 -*-
"""
Shared simulation script (sim mode only).

Single source of truth for the fake scene, so the video figure, the
depth indicator and the fake events always agree — and so the demo
keys (F/H/R) drive everything at once.

Shared state is an mp.Array of 3 doubles created by main.py:
    [0] = t0          scenario start time (time.time())
    [1] = forced_fall 0.0 = follow clock, 1.0 = person forced down (F)
    [2] = forced_help 0.0 / 1.0                                    (H)

Camera view convention (top-ish view of a room):
    depth 0.0 = far  -> person high on screen, small
    depth 1.0 = near -> person low on screen,  large
Timeline: person walks toward the camera (depth 0.15 -> 0.85 over 10 s),
falls at t=10 s, calls "help" at t=13 s, cycle restarts at t=45 s.
"""
import time

CYCLE_S = 45.0
FALL_AT = 10.0
HELP_AT = 13.0

I_T0 = 0
I_FORCED_FALL = 1
I_FORCED_HELP = 2

DEPTH_START = 0.15    # far, at t=0
DEPTH_FALL = 0.85     # near, at the moment of the fall


def make_state(now=None):
    return [now if now is not None else time.time(), 0.0, 0.0]


def _t0(state):
    return state[I_T0] if state is not None else 0.0


def sim_clock(state, now=None):
    """Return scenario time/cycle; ``now`` permits deterministic tests."""
    if now is None:
        now = time.time()
    elapsed = max(0.0, now - _t0(state))
    return elapsed % CYCLE_S, int(elapsed // CYCLE_S)


def is_fallen(state, t_cycle):
    if state is not None and state[I_FORCED_FALL] >= 0.5:
        return True
    return t_cycle >= FALL_AT


def is_helping(state, t_cycle):
    if state is not None and state[I_FORCED_HELP] >= 0.5:
        return True
    return t_cycle >= HELP_AT


def phase(state, t_cycle):
    return "fallen" if is_fallen(state, t_cycle) else "walking"


def depth(state, t_cycle):
    """
    Relative closeness in [0, 1]. 0 = far, 1 = near.
    Rises steadily as the person approaches the camera, then stays high
    once on the floor. Single source of truth for screen position AND
    depth score, so the two can never disagree.
    """
    if is_fallen(state, t_cycle):
        return 0.90
    frac = min(t_cycle, FALL_AT) / FALL_AT          # 0 -> 1 over 10 s
    return DEPTH_START + (DEPTH_FALL - DEPTH_START) * frac


def screen_y(depth_value):
    """Vertical screen fraction [0,1]: far (small depth) = top."""
    return depth_value  # depth 0 -> top, depth 1 -> bottom


def near_score(state, t_cycle):
    """Depth score published by the depth module == the same depth."""
    return max(0.0, min(1.0, depth(state, t_cycle)))
