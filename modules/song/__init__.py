# -*- coding: utf-8 -*-
"""
SONG's trt_pose fall detector, used as a library (his file unchanged).
fall_detector.py imports Jetson-only packages (trt_pose, torch2trt) at
module level, so import it lazily and only in real mode.
"""
