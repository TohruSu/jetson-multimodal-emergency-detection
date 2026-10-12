# -*- coding: utf-8 -*-
"""
PyQt5 user interface (owner: Martin).
Window: video stream | relative depth indicator, alert banner, module status.

Demo keyboard shortcuts:
    F = force simulated "lay_down" (only with --sim)
    H = force simulated "help" (only with --sim)
    R = reset the alert state

Install on Jetson: sudo apt install python3-pyqt5  (NOT pip)

Golden rule: ALL UI updates happen in the main thread, via the
QTimer in main.py. Never from a child process.
"""
import time

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QHBoxLayout,
                             QVBoxLayout)

from common.messages import (SOURCE_GESTURE, SOURCE_DEPTH,
                             SOURCE_SPEECH, TYPE_STATUS, TYPE_DETECTIONS)
from common.health import format_health

HEARTBEAT_STALE_S = 5.0


class DepthIndicator(QWidget):
    """Relative depth indicator: FAR / MIDDLE / NEAR (SU\'s module).

    SU\'s Depth Anything V2 gives a RELATIVE depth level, not an (x, y)
    position and not meters. Three stacked bars light up according to
    the person\'s current level; the active one is highlighted.
    """

    LEVELS = ["NEAR", "MIDDLE", "FAR"]  # top = near (closest)

    def __init__(self, parent=None):
        super(DepthIndicator, self).__init__(parent)
        self.level = None          # "FAR"|"MIDDLE"|"NEAR"|None
        self.score = None
        self.status = None
        self.setMinimumSize(QSize(300, 300))

    def set_depth(self, level, score, status):
        self.level = level
        self.score = score
        self.status = status
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(30, 30, 35))
        p.setPen(QColor(160, 160, 170))
        p.drawText(8, 16, "Relative depth (SU) - not metric")

        colors = {"NEAR": QColor(230, 90, 60),
                  "MIDDLE": QColor(230, 190, 60),
                  "FAR": QColor(90, 170, 230)}
        n = len(self.LEVELS)
        pad = 14
        bar_h = (h - 60 - pad * (n - 1)) / float(n)
        y = 40
        for name in self.LEVELS:
            active = (name == self.level)
            base = colors[name]
            if active:
                p.setBrush(base)
                p.setPen(Qt.NoPen)
            else:
                dim = QColor(base.red() // 4 + 25, base.green() // 4 + 25,
                             base.blue() // 4 + 25)
                p.setBrush(dim)
                p.setPen(QPen(QColor(70, 70, 80)))
            p.drawRoundedRect(20, int(y), w - 40, int(bar_h), 8, 8)
            p.setPen(QColor(240, 240, 240) if active else QColor(120, 120, 130))
            label = name
            if active and self.score is not None:
                label = "{}   ({:.2f})".format(name, self.score)
            p.drawText(38, int(y + bar_h / 2 + 6), label)
            y += bar_h + pad

        if self.status in ("NO_TARGET", "STALE_PACKET", "ERROR",
                           "UNAVAILABLE"):
            p.setPen(QColor(150, 150, 160))
            p.drawText(20, h - 16, "no target" if self.status == "NO_TARGET"
                       else self.status.lower().replace("_", " "))
        p.end()


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("AI 3D Robot Sensing - Jetson Nano")

        self.video_label = QLabel("Waiting for video stream...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(QSize(480, 360))
        self.video_label.setStyleSheet(
            "background:#141418; color:#aaa; font-size:14px;")

        self.depth_widget = DepthIndicator()

        self.alert_label = QLabel("No active alert")
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setWordWrap(True)
        self._style_alert(0)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#888; font-size:12px;")
        self.status_label.setWordWrap(True)
        self._heartbeats = {SOURCE_GESTURE: 0.0, SOURCE_DEPTH: 0.0,
                            SOURCE_SPEECH: 0.0, "video": 0.0, "audio": 0.0}
        self._health_payloads = {}
        self._health_received = {}  # monotonic receipt time, separate from ts

        # events injected with the demo keys, drained by main.py
        self.manual_events = []

        # latest person boxes from the gesture module (drawn over video)
        self.tracks = []
        self.pose_fps = 0.0
        self.tracks_ts = 0.0

        top = QHBoxLayout()
        top.addWidget(self.video_label, 3)
        top.addWidget(self.depth_widget, 2)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.alert_label)
        layout.addWidget(self.status_label)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self._current_pixmap_data = None  # keep a reference (QImage)

    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        """Demo shortcuts: F/H force simulated fall/help; R resets any mode.

        These drive the shared sim scenario in main.py (so the figure
        actually falls / gets up), which then produces synthetic events.
        """
        key = event.key()
        if key == Qt.Key_F:
            self.manual_events.append({"demo_key": "F"})
        elif key == Qt.Key_H:
            self.manual_events.append({"demo_key": "H"})
        elif key == Qt.Key_R:
            self.manual_events.append({"demo_key": "R"})
        else:
            super(MainWindow, self).keyPressEvent(event)

    # ------------------------------------------------------------------
    def update_frame(self, frame_bgr):
        """frame_bgr: numpy array HxWx3 (BGR, OpenCV)."""
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1].copy()  # BGR -> RGB, contiguous
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._current_pixmap_data = rgb  # prevent garbage collection
        pix = QPixmap.fromImage(img)

        # draw the latest person boxes (full-resolution coords) before
        # scaling, so they line up with the image
        fresh = (time.time() - self.tracks_ts) < 2.0
        painter = QPainter(pix)
        colors = {"NORMAL": QColor(80, 200, 120),
                  "FALLING": QColor(240, 160, 40),
                  "FALLEN": QColor(230, 60, 60)}
        if fresh:
            for tr in self.tracks:
                x1, y1, x2, y2 = tr.get("bbox", [0, 0, 0, 0])
                col = colors.get(tr.get("state"), QColor(160, 160, 160))
                painter.setPen(QPen(col, 3))
                painter.drawRect(int(x1), int(y1),
                                 int(x2 - x1), int(y2 - y1))
                painter.drawText(int(x1), max(14, int(y1) - 6),
                                 "{} #{}".format(tr.get("state", "?"),
                                                 tr.get("track_id", "?")))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(10, 20, "pose {:.1f} fps | persons: {}".format(
            self.pose_fps, len(self.tracks) if fresh else 0))
        painter.end()

        pix = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    # ------------------------------------------------------------------
    def handle_message(self, msg):
        """Update heartbeats, detections, and relative depth from a message."""
        source = msg.get("source")
        payload = msg.get("payload", {})

        if msg.get("type") == TYPE_DETECTIONS:
            self.tracks = payload.get("tracks", [])
            self.pose_fps = payload.get("fps", 0.0)
            self.tracks_ts = msg.get("ts", time.time())
            self._heartbeats[SOURCE_GESTURE] = self.tracks_ts
            return

        if msg.get("type") == TYPE_STATUS:
            key = payload.get("module", source)
            self._heartbeats[key] = msg.get("ts", time.time())
            self._health_payloads[key] = dict(payload)
            self._health_received[key] = time.monotonic()
            return

        if source == SOURCE_DEPTH and "depth_level" in payload:
            self.depth_widget.set_depth(payload.get("depth_level"),
                                        payload.get("near_score"),
                                        payload.get("status"))

    # ------------------------------------------------------------------
    def show_alert(self, alert):
        self.alert_label.setText(alert["sentence"])
        self._style_alert(alert["level"])

    def clear_alert(self):
        self.alert_label.setText("No active alert")
        self._style_alert(0)

    def _style_alert(self, level):
        if level >= 2:
            css = "background:#8c1d1d; color:white;"
        elif level == 1:
            css = "background:#8a5a12; color:white;"
        else:
            css = "background:#20242a; color:#9a9a9a;"
        self.alert_label.setStyleSheet(
            css + " font-size:16px; padding:12px; border-radius:4px;")

    # ------------------------------------------------------------------
    def refresh_status(self):
        now = time.monotonic()
        parts = []
        for name in ("video", "audio", SOURCE_GESTURE, SOURCE_DEPTH,
                     SOURCE_SPEECH):
            received = self._health_received.get(name)
            if received is None:
                description = "WAITING"
            else:
                fresh = (now - received) < HEARTBEAT_STALE_S
                description = format_health(self._health_payloads[name], fresh)
            parts.append("{} {}".format(name, description))
        self.status_label.setText(
            "Modules:   " + "    ".join(parts)
            + "        Keys: F/H = simulation only   R = reset")
