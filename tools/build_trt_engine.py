# -*- coding: utf-8 -*-
"""
Build the TensorRT engine for SONG's pose model — RUN THIS ON THE JETSON.

Historical team notes describe the absent checkpoint
    models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth
as plain PyTorch weights rather than a torch2trt engine. Its contents and
reported tensor count cannot be verified from the source-only archive.

A TensorRT engine depends on its target platform. Run from the checkout
root only after reviewing trusted artifacts and the hardware prerequisites:

    python3 tools/build_trt_engine.py

Historical notes report conversion failures with JetPack 4.6 and
torch2trt 0.5.0. Neither conversion nor throughput is verified here.
When no engine file exists, the integrated gesture wrapper attempts plain
PyTorch weights and fp16 instead; an incompatible existing engine does not
currently retry that path. No performance comparison is claimed.

Output:
    models/resnet18_baseline_att_224x224_A_epoch_249_trt_engine.pth

This tool deserializes the input checkpoint. Use only trusted, approved
local artifacts; it is not an inspection tool for unknown model files.
"""
from __future__ import print_function

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POSE_JSON = os.path.join(ROOT, "modules", "song", "human_pose.json")
WEIGHTS = os.path.join(
    ROOT, "models", "resnet18_baseline_att_224x224_A_epoch_249_trt.pth")
ENGINE_OUT = os.path.join(
    ROOT, "models",
    "resnet18_baseline_att_224x224_A_epoch_249_trt_engine.pth")

INPUT_SIZE = 224


def main():
    if not os.path.exists(WEIGHTS):
        print("ERROR: weights not found:", WEIGHTS)
        return 1
    if not os.path.exists(POSE_JSON):
        print("ERROR: human_pose.json not found:", POSE_JSON)
        return 1

    try:
        import trt_pose.coco
        import trt_pose.models
        from torch2trt import torch2trt
    except ImportError as exc:
        print("ERROR: trt_pose / torch2trt not installed ({})".format(exc))
        print("See docs/SETUP.md and docs/MODELS.md for the prerequisites.")
        return 1

    with open(POSE_JSON, "r") as f:
        human_pose = json.load(f)
    num_parts = len(human_pose["keypoints"])
    num_links = len(human_pose["skeleton"])
    print("keypoints: {}, links: {}".format(num_parts, num_links))

    state = torch.load(WEIGHTS, map_location="cpu")
    if isinstance(state, dict) and "engine" in state:
        print("This file is ALREADY a TensorRT engine — nothing to do.")
        return 0

    print("Building resnet18_baseline_att and loading weights...")
    model = trt_pose.models.resnet18_baseline_att(
        num_parts, 2 * num_links).cuda().eval()
    model.load_state_dict(state)

    print("Converting to TensorRT (fp16), this takes a few minutes...")
    data = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE)).cuda()
    model_trt = torch2trt(model, [data], fp16_mode=True,
                          max_workspace_size=1 << 25)

    torch.save(model_trt.state_dict(), ENGINE_OUT)
    print("DONE ->", ENGINE_OUT)
    print("Now just run:  python3 main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
