"""合成手势视频源（拼图项目）：无摄像头时的离线测试。

接口兼容 cv2.VideoCapture（read / isOpened / set / get）。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class _HandSpec:
    label: str
    center: Tuple[float, float]
    amplitude: Tuple[float, float]
    phase: float
    freq_hz: float
    pinch_phase: float
    pinch_freq: float


class SyntheticHandSource:
    """生成单手动画（拼图模式只需要一只手）。"""

    def __init__(self, width: int = 1280, height: int = 720,
                 fps: int = 30, seed: int = 0):
        self.width = width
        self.height = height
        self.fps = fps
        self.t0 = time.time()
        self.frame_idx = 0
        rng = np.random.default_rng(seed)
        self._spec = _HandSpec(
            label="Right",
            center=(width * 0.5, height * 0.55),
            amplitude=(220, 110),
            phase=0.0,
            freq_hz=0.4,
            pinch_phase=0.0,
            pinch_freq=0.3,
        )
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        t = self.frame_idx / self.fps
        self.frame_idx += 1
        img = self._render(t)
        return True, img

    def release(self):
        self._opened = False

    def set(self, prop, value):
        return True

    def get(self, prop):
        mapping = {
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FPS: self.fps,
        }
        return mapping.get(prop, 0)

    def _render(self, t: float) -> np.ndarray:
        h, w = self.height, self.width
        img = np.zeros((h, w, 3), dtype=np.uint8)
        grad = np.linspace(40, 80, h, dtype=np.float32)[:, None]
        img[:] = np.dstack([grad * 0.5, grad * 0.7, grad])
        for y in range(0, h, 80):
            cv2.line(img, (0, y), (w, y), (60, 60, 90), 1, cv2.LINE_AA)
        for x in range(0, w, 80):
            cv2.line(img, (x, 0), (x, h), (60, 60, 90), 1, cv2.LINE_AA)
        self._draw_hand(img, self._spec, t)
        cv2.putText(img, "SYNTHETIC HAND (puzzle_game)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (180, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"frame={self.frame_idx}  t={t:.2f}s",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 220, 255), 1, cv2.LINE_AA)
        return img

    def _draw_hand(self, img, spec, t):
        cx = spec.center[0] + spec.amplitude[0] * math.sin(spec.freq_hz * t + spec.phase)
        cy = spec.center[1] + spec.amplitude[1] * math.sin(spec.freq_hz * t * 0.7 + spec.phase + 1.0)
        index_len = 110 + 30 * math.sin(spec.freq_hz * 1.7 * t + spec.phase)
        index_tip = (cx, cy - index_len)
        thumb_len = 75 + 20 * math.sin(spec.freq_hz * 1.3 * t + spec.phase + 0.7)
        thumb_tip = (cx - thumb_len * 0.85, cy + thumb_len * 0.45)
        pinch_t = (math.sin(spec.pinch_freq * t + spec.pinch_phase) + 1) * 0.5
        if pinch_t > 0.78:
            pull = (pinch_t - 0.78) / 0.22 * 60
            index_tip = (
                index_tip[0] + (thumb_tip[0] - index_tip[0]) * (pull / max(1, index_len + 60)),
                index_tip[1] + (thumb_tip[1] - index_tip[1]) * (pull / max(1, index_len + 60)),
            )
        wrist = (cx, cy + 40)
        pts = [wrist]
        pts += [(cx - 20, cy + 30), (cx - 38, cy + 15), (cx - 55, cy - 5), thumb_tip]
        pts += [(cx - 18, cy - 5), (cx - 14, cy - 40), (cx - 10, cy - 75), index_tip]
        mtip = (cx + 8, cy - index_len - 15)
        pts += [(cx + 5, cy - 5), (cx + 4, cy - 50), (cx + 6, cy - 90), mtip]
        rt = (cx + 32, cy - index_len + 5)
        pts += [(cx + 28, cy - 5), (cx + 28, cy - 50), (cx + 30, cy - 85), rt]
        pt = (cx + 55, cy - index_len + 25)
        pts += [(cx + 50, cy - 5), (cx + 50, cy - 45), (cx + 52, cy - 75), pt]
        pts_arr = np.array(pts, dtype=np.int32)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        palm_color = (90, 160, 220)
        palm_pts = np.array([pts_arr[i] for i in (0, 5, 9, 13, 17, 1)], dtype=np.int32)
        cv2.fillPoly(img, [palm_pts], palm_color)
        finger_chains = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [5, 9, 10, 11, 12],
            [9, 13, 14, 15, 16],
            [13, 17, 18, 19, 20],
        ]
        for chain in finger_chains:
            cv2.polylines(img, [np.array([pts_arr[i] for i in chain], dtype=np.int32)],
                          False, palm_color, 18, cv2.LINE_AA)
        for a, b in connections:
            cv2.line(img, tuple(pts_arr[a]), tuple(pts_arr[b]), (40, 30, 20), 1, cv2.LINE_AA)
        for i, p in enumerate(pts_arr):
            c = (255, 255, 80) if i in (4, 8) else (180, 230, 255)
            r = 5 if i in (4, 8) else 3
            cv2.circle(img, tuple(p), r, c, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(pts_arr[0]), 8, (255, 255, 255), 2, cv2.LINE_AA)