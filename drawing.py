"""单手捏合绘制矩形工具。

用食指指尖作为"笔"，捏合落下开始绘制 / 松开结束绘制。
完成两次后取并集作为最终 ROI。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import Rect, normalize_rect, rect_area, smooth_point, fill_rect_translucent, put_text


@dataclass
class RectDrawer:
    max_rects: int = 2
    min_side: int = 40
    rects: List[Rect] = field(default_factory=list)
    _drawing: bool = False
    _anchor: Optional[Tuple[int, int]] = None
    _smoothed: Optional[Tuple[int, int]] = None

    def reset(self):
        self.rects.clear()
        self._drawing = False
        self._anchor = None
        self._smoothed = None

    @property
    def is_full(self) -> bool:
        return len(self.rects) >= self.max_rects

    @property
    def final_rect(self) -> Optional[Rect]:
        if not self.rects:
            return None
        r = self.rects[0]
        for nxt in self.rects[1:]:
            r = (
                min(r[0], nxt[0]),
                min(r[1], nxt[1]),
                max(r[2], nxt[2]),
                max(r[3], nxt[3]),
            )
        return r

    def update(self, hand_state, frame_shape) -> Optional[Rect]:
        """每帧调用, hand_state 是 HandState|None。返回新完成的那个矩形 (刚刚 added)。"""
        h, w = frame_shape[:2]
        if hand_state is None or self.is_full:
            self._drawing = False
            self._anchor = None
            self._smoothed = None
            return None

        tip = hand_state.index_tip
        self._smoothed = smooth_point(self._smoothed, tip, alpha=0.4)

        pinching = hand_state.pinching
        just_completed: Optional[Rect] = None

        if not self._drawing:
            if pinching:
                # 开始绘制
                self._drawing = True
                self._anchor = self._smoothed
        else:
            if not pinching:
                # 松开 → 收尾
                r = normalize_rect(self._anchor[0], self._anchor[1],
                                   self._smoothed[0], self._smoothed[1])
                if rect_area(r) >= self.min_side * self.min_side:
                    self.rects.append(r)
                    just_completed = r
                self._drawing = False
                self._anchor = None
                self._smoothed = None

        return just_completed

    def draw(self, img: np.ndarray) -> None:
        # 已有矩形（半透明）
        for idx, r in enumerate(self.rects):
            color = (180, 240, 120) if idx == 0 else (120, 200, 240)
            fill_rect_translucent(img, r, color, alpha=0.18)
            cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), color, 2)

        # 进行中的矩形
        if self._drawing and self._anchor and self._smoothed:
            cur = normalize_rect(
                self._anchor[0], self._anchor[1],
                self._smoothed[0], self._smoothed[1],
            )
            cv2.rectangle(img, (cur[0], cur[1]), (cur[2], cur[3]),
                          (0, 200, 255), 2)
            cv2.line(img, self._anchor, self._smoothed, (0, 200, 255), 1, cv2.LINE_AA)

        put_text(img,
                 f"Draw rectangles: {len(self.rects)}/{self.max_rects}",
                 (20, 40), size=0.8, color=(255, 255, 255),
                 bg=(0, 0, 0), bg_alpha=0.5)
        if self.is_full:
            put_text(img, "READY - generating puzzle...",
                     (20, 80), size=0.8, color=(120, 255, 120),
                     bg=(0, 0, 0), bg_alpha=0.5)