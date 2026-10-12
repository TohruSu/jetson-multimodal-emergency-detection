#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose-based fall detection research prototype using NVIDIA TRT-Pose.

Historical target: Python 3.6+; runtime and throughput unverified here
Input: OpenCV VideoCapture (USB camera or a video path)
Output:
  1. OpenCV live window
  2. One JSON line per fall event on stdout
  3. One UDP JSON datagram per fall event (optional)

The detector combines:
  - fast downward head motion
  - body bounding-box width/height ratio
  - torso angle relative to the vertical direction
"""

from __future__ import print_function

import argparse
import collections
import json
import math
import os
import socket
import sys
import time

import cv2
import PIL.Image
import torch
import torchvision.transforms as transforms

import trt_pose.coco
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from torch2trt import TRTModule


STATE_NORMAL = 'NORMAL'
STATE_FALLING = 'FALLING'
STATE_FALLEN = 'FALLEN'


DEFAULT_CONFIG = {
    'head_speed_threshold': 0.80,
    'aspect_ratio_threshold': 1.15,
    'torso_angle_threshold': 60.0,
    'fast_drop_memory': 0.80,
    'fall_confirm_seconds': 0.80,
    'candidate_timeout': 1.80,
    'recovery_seconds': 1.20,
    'recovery_angle': 35.0,
    'recovery_ratio': 0.95,
    'speed_window_min': 0.12,
    'speed_window_max': 0.50,
    'track_timeout': 1.00,
    'track_match_distance': 0.18,
    'keypoint_ema_alpha': 0.45,
    'min_keypoints': 5,
    'min_bbox_height': 25.0
}


CORE_POINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'neck', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

HEAD_POINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'
]


def canonical_name(name):
    return str(name).strip().lower().replace(' ', '_').replace('-', '_')


def mean_point(points, names):
    selected = []
    for name in names:
        point = points.get(name)
        if point is not None:
            selected.append(point)
    if not selected:
        return None
    x_value = sum(point[0] for point in selected) / float(len(selected))
    y_value = sum(point[1] for point in selected) / float(len(selected))
    return (x_value, y_value)


def get_shoulder_center(points):
    left_point = points.get('left_shoulder')
    right_point = points.get('right_shoulder')
    if left_point is not None and right_point is not None:
        return ((left_point[0] + right_point[0]) * 0.5,
                (left_point[1] + right_point[1]) * 0.5)
    neck_point = points.get('neck')
    if neck_point is not None:
        return neck_point
    if left_point is not None:
        return left_point
    return right_point


def get_hip_center(points):
    return mean_point(points, ['left_hip', 'right_hip'])


def calculate_bbox(points):
    selected = []
    for name in CORE_POINT_NAMES:
        point = points.get(name)
        if point is not None:
            selected.append(point)
    if len(selected) < 2:
        return None
    x_values = [point[0] for point in selected]
    y_values = [point[1] for point in selected]
    return (min(x_values), min(y_values), max(x_values), max(y_values))


def calculate_features(points, min_bbox_height):
    bbox = calculate_bbox(points)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    bbox_width = max(1.0, x2 - x1)
    bbox_height = max(1.0, y2 - y1)
    if bbox_height < min_bbox_height:
        return None

    shoulder_center = get_shoulder_center(points)
    hip_center = get_hip_center(points)
    head_center = mean_point(points, HEAD_POINT_NAMES)

    torso_angle = None
    if shoulder_center is not None and hip_center is not None:
        dx_value = shoulder_center[0] - hip_center[0]
        dy_value = shoulder_center[1] - hip_center[1]
        if abs(dx_value) + abs(dy_value) > 1.0:
            torso_angle = math.degrees(
                math.atan2(abs(dx_value), abs(dy_value))
            )

    if shoulder_center is not None and hip_center is not None:
        body_center = ((shoulder_center[0] + hip_center[0]) * 0.5,
                       (shoulder_center[1] + hip_center[1]) * 0.5)
    else:
        body_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    return {
        'bbox': bbox,
        'bbox_width': bbox_width,
        'bbox_height': bbox_height,
        'aspect_ratio': bbox_width / bbox_height,
        'torso_angle': torso_angle,
        'head_center': head_center,
        'shoulder_center': shoulder_center,
        'hip_center': hip_center,
        'body_center': body_center,
        'keypoint_count': len(points)
    }


def camera_region(x_norm, y_norm):
    if x_norm < 1.0 / 3.0:
        horizontal = 'left'
    elif x_norm < 2.0 / 3.0:
        horizontal = 'center'
    else:
        horizontal = 'right'

    if y_norm < 1.0 / 3.0:
        vertical = 'top'
    elif y_norm < 2.0 / 3.0:
        vertical = 'middle'
    else:
        vertical = 'bottom'
    return horizontal + '-' + vertical


class UdpJsonSender(object):
    def __init__(self, host, port, enabled):
        self.host = host
        self.port = int(port)
        self.enabled = bool(enabled)
        self.sock = None
        if self.enabled:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload):
        message = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(',', ':')
        )
        print('FALL_EVENT ' + message)
        sys.stdout.flush()

        if not self.enabled:
            return
        try:
            self.sock.sendto(message.encode('utf-8'), (self.host, self.port))
        except Exception as error:
            print('UDP_SEND_ERROR: {0}'.format(error), file=sys.stderr)

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


class PosePreprocessor(object):
    def __init__(self, device):
        self.device = device
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    def __call__(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb_image)
        tensor = transforms.functional.to_tensor(pil_image).to(self.device)
        tensor.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return tensor[None, ...]


class PersonTrack(object):
    def __init__(self, track_id, config):
        self.track_id = int(track_id)
        self.config = config
        self.state = STATE_NORMAL
        self.last_seen = 0.0
        self.raw_center = None
        self.points = {}
        self.features = None
        self.head_speed = None
        self.history = collections.deque(maxlen=60)
        self.last_fast_drop_time = None
        self.trigger_head_speed = None
        self.candidate_since = None
        self.posture_since = None
        self.recovery_since = None
        self.alert_sent = False
        self.updated_this_frame = False

    def smooth_points(self, new_points):
        alpha = float(self.config['keypoint_ema_alpha'])
        smoothed = {}
        for name, point in new_points.items():
            old_point = self.points.get(name)
            if old_point is None:
                smoothed[name] = point
            else:
                smoothed[name] = (
                    alpha * point[0] + (1.0 - alpha) * old_point[0],
                    alpha * point[1] + (1.0 - alpha) * old_point[1]
                )
        self.points = smoothed

    def estimate_head_speed(self, now, head_center, bbox_height):
        if head_center is None:
            return None

        oldest_candidate = None
        for sample in self.history:
            sample_time, sample_head_y, sample_height = sample
            age = now - sample_time
            if age < self.config['speed_window_min']:
                continue
            if age > self.config['speed_window_max']:
                continue
            oldest_candidate = sample
            break

        speed = None
        if oldest_candidate is not None:
            old_time, old_head_y, old_height = oldest_candidate
            delta_time = max(0.001, now - old_time)
            reference_height = max(20.0, old_height, bbox_height)
            speed = (head_center[1] - old_head_y) / reference_height / delta_time

        self.history.append((now, head_center[1], bbox_height))
        while self.history and now - self.history[0][0] > 2.0:
            self.history.popleft()
        return speed

    def reset_to_normal(self):
        self.state = STATE_NORMAL
        self.last_fast_drop_time = None
        self.trigger_head_speed = None
        self.candidate_since = None
        self.posture_since = None
        self.recovery_since = None
        self.alert_sent = False

    def update_state(self, now, frame_width, frame_height):
        if self.features is None:
            return None

        ratio_value = self.features['aspect_ratio']
        angle_value = self.features['torso_angle']

        speed_hit = (
            self.head_speed is not None and
            self.head_speed >= self.config['head_speed_threshold']
        )
        ratio_hit = ratio_value >= self.config['aspect_ratio_threshold']
        angle_hit = (
            angle_value is not None and
            angle_value >= self.config['torso_angle_threshold']
        )
        posture_hit = ratio_hit or angle_hit

        if speed_hit:
            self.last_fast_drop_time = now
            if (
                    self.trigger_head_speed is None or
                    self.head_speed > self.trigger_head_speed):
                self.trigger_head_speed = self.head_speed

        fast_drop_recent = (
            self.last_fast_drop_time is not None and
            now - self.last_fast_drop_time <= self.config['fast_drop_memory']
        )

        event_payload = None

        if self.state == STATE_NORMAL:
            if fast_drop_recent and posture_hit:
                self.state = STATE_FALLING
                self.candidate_since = now
                self.posture_since = now
                self.recovery_since = None

        elif self.state == STATE_FALLING:
            if posture_hit:
                if self.posture_since is None:
                    self.posture_since = now
                posture_duration = now - self.posture_since
                if posture_duration >= self.config['fall_confirm_seconds']:
                    self.state = STATE_FALLEN
                    self.recovery_since = None
                    if not self.alert_sent:
                        self.alert_sent = True
                        event_payload = self.build_event_payload(
                            now,
                            frame_width,
                            frame_height,
                            True,
                            ratio_hit,
                            angle_hit
                        )
            else:
                self.posture_since = None

            if (
                    self.candidate_since is not None and
                    now - self.candidate_since > self.config['candidate_timeout'] and
                    self.state != STATE_FALLEN):
                self.reset_to_normal()

        elif self.state == STATE_FALLEN:
            angle_recovered = (
                angle_value is not None and
                angle_value <= self.config['recovery_angle']
            )
            ratio_recovered = ratio_value <= self.config['recovery_ratio']
            upright = angle_recovered and ratio_recovered

            if upright:
                if self.recovery_since is None:
                    self.recovery_since = now
                elif now - self.recovery_since >= self.config['recovery_seconds']:
                    self.reset_to_normal()
            else:
                self.recovery_since = None

        return event_payload

    def build_event_payload(self, now, frame_width, frame_height,
                            fast_drop_triggered, ratio_hit, angle_hit):
        bbox = self.features['bbox']
        center = self.features['body_center']
        head_center = self.features['head_center']

        x_norm = min(1.0, max(0.0, center[0] / float(frame_width)))
        y_norm = min(1.0, max(0.0, center[1] / float(frame_height)))

        position = {
            'center_px': {
                'x': int(round(center[0])),
                'y': int(round(center[1]))
            },
            'center_norm': {
                'x': round(x_norm, 4),
                'y': round(y_norm, 4)
            },
            'region': camera_region(x_norm, y_norm)
        }

        if head_center is not None:
            position['head_px'] = {
                'x': int(round(head_center[0])),
                'y': int(round(head_center[1]))
            }

        angle_value = self.features['torso_angle']
        payload = {
            'type': 'fall_detected',
            'timestamp_unix': round(now, 3),
            'timestamp_local': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
            'track_id': self.track_id,
            'position': position,
            'bbox_px': [
                int(round(bbox[0])),
                int(round(bbox[1])),
                int(round(bbox[2])),
                int(round(bbox[3]))
            ],
            'frame_size': {
                'width': int(frame_width),
                'height': int(frame_height)
            },
            'features': {
                'head_down_speed_body_per_sec': (
                    None if self.trigger_head_speed is None
                    else round(self.trigger_head_speed, 3)
                ),
                'aspect_ratio': round(self.features['aspect_ratio'], 3),
                'torso_angle_deg': (
                    None if angle_value is None else round(angle_value, 2)
                ),
                'head_speed_hit': bool(fast_drop_triggered),
                'aspect_ratio_hit': bool(ratio_hit),
                'torso_angle_hit': bool(angle_hit)
            }
        }
        return payload

    def update(self, detection, now, frame_width, frame_height):
        self.updated_this_frame = True
        self.last_seen = now
        self.raw_center = detection['raw_center']
        self.smooth_points(detection['points'])
        self.features = calculate_features(
            self.points,
            self.config['min_bbox_height']
        )
        if self.features is None:
            self.head_speed = None
            return None

        self.head_speed = self.estimate_head_speed(
            now,
            self.features['head_center'],
            self.features['bbox_height']
        )
        return self.update_state(now, frame_width, frame_height)


class SimpleTracker(object):
    def __init__(self, config):
        self.config = config
        self.tracks = {}
        self.next_track_id = 1

    def create_track(self):
        track = PersonTrack(self.next_track_id, self.config)
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        return track

    def update(self, detections, now, frame_width, frame_height):
        for track in self.tracks.values():
            track.updated_this_frame = False

        active_track_ids = [
            track_id for track_id, track in self.tracks.items()
            if now - track.last_seen <= self.config['track_timeout']
        ]

        frame_diagonal = math.sqrt(
            float(frame_width * frame_width + frame_height * frame_height)
        )
        max_distance = self.config['track_match_distance'] * frame_diagonal

        candidate_pairs = []
        for detection_index, detection in enumerate(detections):
            det_center = detection['raw_center']
            for track_id in active_track_ids:
                track = self.tracks[track_id]
                if track.raw_center is None:
                    continue
                dx_value = det_center[0] - track.raw_center[0]
                dy_value = det_center[1] - track.raw_center[1]
                distance = math.sqrt(dx_value * dx_value + dy_value * dy_value)
                if distance <= max_distance:
                    candidate_pairs.append((distance, track_id, detection_index))

        candidate_pairs.sort(key=lambda item: item[0])
        used_tracks = set()
        used_detections = set()
        assignments = []

        for distance, track_id, detection_index in candidate_pairs:
            if track_id in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            assignments.append((track_id, detection_index))

        for detection_index in range(len(detections)):
            if detection_index not in used_detections:
                track = self.create_track()
                assignments.append((track.track_id, detection_index))

        events = []
        updated_tracks = []
        for track_id, detection_index in assignments:
            track = self.tracks[track_id]
            event_payload = track.update(
                detections[detection_index],
                now,
                frame_width,
                frame_height
            )
            updated_tracks.append(track)
            if event_payload is not None:
                events.append(event_payload)

        expired_ids = []
        for track_id, track in self.tracks.items():
            if now - track.last_seen > self.config['track_timeout']:
                expired_ids.append(track_id)
        for track_id in expired_ids:
            del self.tracks[track_id]

        return updated_tracks, events


def extract_detections(counts, objects, peaks, keypoint_names,
                       frame_width, frame_height, min_keypoints):
    detections = []
    person_count = int(counts[0])

    for person_index in range(person_count):
        points = {}
        person_object = objects[0][person_index]

        for part_index, part_name in enumerate(keypoint_names):
            peak_index = int(person_object[part_index])
            if peak_index < 0:
                continue
            peak = peaks[0][part_index][peak_index]
            y_norm = float(peak[0])
            x_norm = float(peak[1])
            points[part_name] = (
                x_norm * float(frame_width),
                y_norm * float(frame_height)
            )

        if len(points) < min_keypoints:
            continue

        bbox = calculate_bbox(points)
        if bbox is None:
            continue
        raw_center = (
            (bbox[0] + bbox[2]) * 0.5,
            (bbox[1] + bbox[3]) * 0.5
        )
        detections.append({
            'points': points,
            'raw_center': raw_center
        })

    return detections


def draw_track(frame, track):
    if track.features is None:
        return

    if track.state == STATE_FALLEN:
        color = (0, 0, 255)
    elif track.state == STATE_FALLING:
        color = (0, 165, 255)
    else:
        color = (0, 255, 0)

    bbox = track.features['bbox']
    x1 = int(round(bbox[0]))
    y1 = int(round(bbox[1]))
    x2 = int(round(bbox[2]))
    y2 = int(round(bbox[3]))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    center = track.features['body_center']
    center_point = (int(round(center[0])), int(round(center[1])))
    cv2.circle(frame, center_point, 6, color, -1)
    cv2.line(frame, (center_point[0] - 10, center_point[1]),
             (center_point[0] + 10, center_point[1]), color, 2)
    cv2.line(frame, (center_point[0], center_point[1] - 10),
             (center_point[0], center_point[1] + 10), color, 2)

    angle_value = track.features['torso_angle']
    angle_text = 'NA' if angle_value is None else '{0:.1f}'.format(angle_value)
    speed_text = 'NA' if track.head_speed is None else '{0:.2f}'.format(track.head_speed)

    first_line = 'ID:{0} {1}'.format(track.track_id, track.state)
    second_line = 'V:{0}H/s R:{1:.2f} A:{2}'.format(
        speed_text,
        track.features['aspect_ratio'],
        angle_text
    )
    label_y = max(20, y1 - 24)
    cv2.putText(frame, first_line, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, second_line, (x1, label_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    if track.state == STATE_FALLEN:
        cv2.putText(frame, 'FALL DETECTED',
                    (x1, min(frame.shape[0] - 10, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def parse_camera_source(value):
    try:
        return int(value)
    except ValueError:
        return value


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='TRT-Pose fall detector; historical Python 3.6 target, runtime unverified'
    )
    parser.add_argument('--human-pose', default='human_pose.json')
    parser.add_argument(
        '--model',
        default='resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    )
    parser.add_argument('--camera', default='0')
    parser.add_argument('--camera-width', type=int, default=640)
    parser.add_argument('--camera-height', type=int, default=480)
    parser.add_argument('--camera-fps', type=int, default=30)
    parser.add_argument('--model-width', type=int, default=224)
    parser.add_argument('--model-height', type=int, default=224)
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--no-skeleton', action='store_true')
    parser.add_argument('--udp-host', default='127.0.0.1')
    parser.add_argument('--udp-port', type=int, default=5005)
    parser.add_argument('--no-udp', action='store_true')

    parser.add_argument('--head-speed', type=float,
                        default=DEFAULT_CONFIG['head_speed_threshold'])
    parser.add_argument('--aspect-ratio', type=float,
                        default=DEFAULT_CONFIG['aspect_ratio_threshold'])
    parser.add_argument('--torso-angle', type=float,
                        default=DEFAULT_CONFIG['torso_angle_threshold'])
    parser.add_argument('--confirm-seconds', type=float,
                        default=DEFAULT_CONFIG['fall_confirm_seconds'])
    return parser


def load_pose_components(human_pose_path, model_path):
    if not os.path.exists(human_pose_path):
        raise IOError('human_pose.json not found: {0}'.format(human_pose_path))
    if not os.path.exists(model_path):
        raise IOError('TensorRT model not found: {0}'.format(model_path))

    with open(human_pose_path, 'r') as pose_file:
        human_pose = json.load(pose_file)

    keypoint_names = [canonical_name(name) for name in human_pose['keypoints']]
    topology = trt_pose.coco.coco_category_to_topology(human_pose)

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_path))
    model_trt.cuda().eval()

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    return keypoint_names, model_trt, parse_objects, draw_objects


def main():
    args = build_argument_parser().parse_args()

    config = dict(DEFAULT_CONFIG)
    config['head_speed_threshold'] = float(args.head_speed)
    config['aspect_ratio_threshold'] = float(args.aspect_ratio)
    config['torso_angle_threshold'] = float(args.torso_angle)
    config['fall_confirm_seconds'] = float(args.confirm_seconds)

    if not torch.cuda.is_available():
        print('ERROR: CUDA is required by this TRT-Pose program.', file=sys.stderr)
        return 1

    try:
        keypoint_names, model_trt, parse_objects, draw_objects = load_pose_components(
            args.human_pose,
            args.model
        )
    except Exception as error:
        print('MODEL_LOAD_ERROR: {0}'.format(error), file=sys.stderr)
        return 1

    device = torch.device('cuda')
    preprocess = PosePreprocessor(device)
    tracker = SimpleTracker(config)
    sender = UdpJsonSender(args.udp_host, args.udp_port, not args.no_udp)

    camera_source = parse_camera_source(args.camera)
    camera = cv2.VideoCapture(camera_source)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    camera.set(cv2.CAP_PROP_FPS, args.camera_fps)

    if not camera.isOpened():
        print('ERROR: Could not open camera: {0}'.format(args.camera), file=sys.stderr)
        sender.close()
        return 1

    print('Fall detector started.')
    print('Press Q or ESC to stop.')
    if args.no_udp:
        print('UDP output: disabled; events are printed to stdout.')
    else:
        print('UDP output: {0}:{1}'.format(args.udp_host, args.udp_port))

    fps_value = 0.0
    fps_counter = 0
    fps_start = time.time()

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print('WARNING: Camera frame read failed.', file=sys.stderr)
                time.sleep(0.05)
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            frame_height, frame_width = frame.shape[:2]
            model_image = cv2.resize(
                frame,
                (args.model_width, args.model_height),
                interpolation=cv2.INTER_LINEAR
            )

            input_tensor = preprocess(model_image)
            with torch.no_grad():
                cmap, paf = model_trt(input_tensor)
            cmap = cmap.detach().cpu()
            paf = paf.detach().cpu()
            counts, objects, peaks = parse_objects(cmap, paf)

            detections = extract_detections(
                counts,
                objects,
                peaks,
                keypoint_names,
                frame_width,
                frame_height,
                config['min_keypoints']
            )

            now = time.time()
            tracks, events = tracker.update(
                detections,
                now,
                frame_width,
                frame_height
            )

            display_frame = frame.copy()
            if not args.no_skeleton:
                draw_objects(display_frame, counts, objects, peaks)

            for track in tracks:
                draw_track(display_frame, track)

            for event_payload in events:
                sender.send(event_payload)

            fps_counter += 1
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps_value = fps_counter / elapsed
                fps_counter = 0
                fps_start = now

            status_text = 'FPS:{0:.1f} Persons:{1}'.format(
                fps_value,
                len(tracks)
            )
            cv2.putText(display_frame, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow('TRT-Pose Fall Detector', display_frame)
            key_value = cv2.waitKey(1) & 0xFF
            if key_value == ord('q') or key_value == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        sender.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    sys.exit(main())
