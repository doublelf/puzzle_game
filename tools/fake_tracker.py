"""拼图项目的合成 HandTracker，绕过 MediaPipe 直接产出 HandState。"""
from __future__ import annotations

import math
import time

import numpy as np

from gesture import HandState, INDEX_MCP, PINKY_MCP, WRIST


class SyntheticHandTracker:
    """接口与 gesture.HandTracker 完全一致；返回合成的 HandState。"""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_idx = 0
        self._states: dict[str, HandState] = {}

    def is_available(self) -> bool:
        return True

    def close(self):
        pass

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> dict:
        t = self.frame_idx / 30.0
        self.frame_idx += 1
        cx = self.width * 0.5 + 220 * math.sin(0.4 * t)
        cy = self.height * 0.55 + 110 * math.sin(0.4 * t * 0.7 + 1.0)
        index_len = 110 + 30 * math.sin(0.4 * 1.7 * t)
        index_tip = (cx, cy - index_len)
        thumb_len = 75 + 20 * math.sin(0.4 * 1.3 * t + 0.7)
        thumb_tip = (cx - thumb_len * 0.85, cy + thumb_len * 0.45)
        pinch_t = (math.sin(0.3 * t) + 1) * 0.5
        pinching = pinch_t > 0.78
        if pinching:
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
        pts_arr = np.array(pts, dtype=np.float32)
        prev = self._states.get("Right")
        was_active = prev.pinch_active if prev is not None else False
        state = HandState(
            label="Right",
            landmarks_px=pts_arr,
            wrist_px=pts_arr[WRIST],
            thumb_tip=tuple(pts_arr[4].astype(int)),
            index_tip=tuple(pts_arr[8].astype(int)),
        )
        state.pinching = pinching
        state.pinch_active = pinching
        state.pinch_anchor = state.index_tip if pinching else None
        state.pinch_begin_edge = pinching and not was_active
        state.pinch_end_edge = (not pinching) and was_active
        self._states = {"Right": state}
        return {"left": None, "right": state}