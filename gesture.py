"""手势识别模块：基于 MediaPipe Hand Landmarker (Tasks API)。

提供:
- HandTracker: 每帧处理摄像头画面，返回双手状态。
- HandState: 单只手的标准化数据。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    MP_TASKS_AVAILABLE = True
except Exception:
    MP_TASKS_AVAILABLE = False


# MediaPipe Hands 的关键点索引
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
PINKY_MCP = 17

# MediaPipe 模型下载 URL（首次运行时下载）
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"


def _resolve_model_path() -> str:
    """优先使用仓库根目录下的模型文件，否则从工作目录/用户缓存加载。"""
    candidates = [
        os.path.join(os.getcwd(), MODEL_FILENAME),
        os.path.join(os.path.dirname(__file__), MODEL_FILENAME),
        os.path.expanduser(os.path.join("~", ".cache", "mediapipe", MODEL_FILENAME)),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]


def _download_model(url: str, dst: str) -> bool:
    try:
        import urllib.request
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"[gesture] downloading model from {url} ...")
        urllib.request.urlretrieve(url, dst)
        print(f"[gesture] saved to {dst}")
        return True
    except Exception as e:
        print(f"[gesture] download failed: {e}")
        return False


@dataclass
class HandState:
    label: str  # 'Left' | 'Right'
    landmarks_px: np.ndarray  # (21, 2) 像素坐标
    wrist_px: np.ndarray       # 单独存一份方便
    thumb_tip: tuple          # (x, y) 像素
    index_tip: tuple
    pinching: bool = False
    pinch_anchor: Optional[tuple] = None  # 像素坐标，捏合开始时锁定
    pinch_active: bool = False            # 当前帧是否在捏合状态
    pinch_begin_edge: bool = False        # 上升沿：False → True
    pinch_end_edge: bool = False          # 下降沿：True → False
    missing_frames: int = 0

    def palm_center(self) -> tuple:
        # 使用 INDEX_MCP/PINKY_MCP/WRIST 的中点近似掌心
        pts = self.landmarks_px[[INDEX_MCP, PINKY_MCP, WRIST]]
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        return (cx, cy)

    def palm_size(self) -> float:
        """用 INDEX_MCP -> PINKY_MCP 的距离作为手掌参考尺寸，用于距离归一化。"""
        a = self.landmarks_px[INDEX_MCP]
        b = self.landmarks_px[PINKY_MCP]
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))


class HandTracker:
    """统一接口的双手追踪器。"""

    def __init__(self, num_hands: int = 2, pinch_threshold: float = 0.32,
                 smoothing: float = 0.55):
        self.num_hands = num_hands
        self.pinch_threshold = pinch_threshold
        self.smoothing = smoothing
        self._states: dict[str, HandState] = {}
        self._last_pts: dict[str, dict[int, tuple]] = {}
        self._landmarker = None
        if MP_TASKS_AVAILABLE:
            self._landmarker = self._init_landmarker()

    def _init_landmarker(self):
        model_path = _resolve_model_path()
        if not os.path.isfile(model_path):
            if not _download_model(MODEL_URL, model_path):
                return None
        base = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=self.num_hands,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    def close(self):
        try:
            if self._landmarker is not None:
                self._landmarker.close()
        except Exception:
            pass

    def is_available(self) -> bool:
        return self._landmarker is not None

    def _smooth(self, label: str, idx: int, pt: tuple) -> tuple:
        prev = self._last_pts.get(label, {}).get(idx)
        if prev is None:
            self._last_pts.setdefault(label, {})[idx] = pt
            return pt
        a = self.smoothing
        sx = int(a * prev[0] + (1 - a) * pt[0])
        sy = int(a * prev[1] + (1 - a) * pt[1])
        self._last_pts[label][idx] = (sx, sy)
        return (sx, sy)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> dict:
        """输入 BGR 帧，返回 {'left': HandState|None, 'right': HandState|None}"""
        if self._landmarker is None:
            return {"left": None, "right": None}

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        detected = {}
        if result and result.hand_landmarks and result.handedness:
            for lm_list, handed in zip(result.hand_landmarks, result.handedness):
                label = handed[0].category_name  # 'Left' or 'Right'
                # MediaPipe 在镜像摄像头下会把左右反过来；这里用 mirror=False 处理,
                # 但 cv2 显示时是镜像的，所以这里 raw 输出到帧坐标即可
                pts_px = []
                last = {}
                for i, lm in enumerate(lm_list):
                    px = self._smooth(label, i, (lm.x * w, lm.y * h))
                    pts_px.append(px)
                    last[i] = px
                pts_px = np.array(pts_px, dtype=np.float32)
                state = HandState(
                    label=label,
                    landmarks_px=pts_px,
                    wrist_px=pts_px[WRIST],
                    thumb_tip=tuple(pts_px[THUMB_TIP].astype(int)),
                    index_tip=tuple(pts_px[INDEX_TIP].astype(int)),
                )
                # 捏合距离用拇指尖-食指尖归一化（除以手掌参考尺寸）
                psize = state.palm_size() + 1e-6
                raw_dist = math.hypot(
                    state.thumb_tip[0] - state.index_tip[0],
                    state.thumb_tip[1] - state.index_tip[1],
                )
                norm_dist = raw_dist / psize
                prev = self._states.get(label)
                was_active = prev.pinch_active if prev is not None else False
                pinching_now = norm_dist < self.pinch_threshold
                if prev is not None and prev.pinch_active and pinching_now:
                    state.pinching = True
                    state.pinch_anchor = prev.pinch_anchor
                elif pinching_now:
                    state.pinching = True
                    state.pinch_anchor = state.index_tip
                else:
                    state.pinching = False
                    state.pinch_anchor = None
                state.pinch_active = pinching_now
                # 边沿事件
                state.pinch_begin_edge = pinching_now and not was_active
                state.pinch_end_edge = (not pinching_now) and was_active
                detected[label] = state

        # 更新持久状态（缺失帧统计）
        new_states = {}
        for label in ("Left", "Right"):
            if label in detected:
                new_states[label] = detected[label]
            else:
                prev = self._states.get(label)
                if prev is not None:
                    prev.missing_frames += 1
                    # 丢失超过 5 帧后清空
                    if prev.missing_frames > 5:
                        prev.pinching = False
                        prev.pinch_active = False
                        prev.pinch_anchor = None
                        self._last_pts.pop(label, None)
                    else:
                        new_states[label] = prev
        self._states = new_states

        return {
            "left": self._states.get("Left"),
            "right": self._states.get("Right"),
        }

    # 注: 关键点绘制已迁出至 overlay.HandOverlay (默认开启，效果更丰富)。