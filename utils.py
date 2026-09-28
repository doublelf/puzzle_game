"""几何与图像处理工具函数。"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]  # x1, y1, x2, y2


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dist(a: Point | Sequence[float], b: Point | Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize_rect(x1: int, y1: int, x2: int, y2: int) -> Rect:
    return (
        int(min(x1, x2)),
        int(min(y1, y2)),
        int(max(x1, x2)),
        int(max(y1, y2)),
    )


def rect_area(r: Rect) -> int:
    return max(0, r[2] - r[0]) * max(0, r[3] - r[1])


def union_rect(a: Rect, b: Rect) -> Rect:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def point_in_rect(p: Point, r: Rect) -> bool:
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def draw_rect_outline(img: np.ndarray, r: Rect, color, thickness: int = 2, alpha: float = 1.0) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (r[0], r[1]), (r[2], r[3]), color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def fill_rect_translucent(img: np.ndarray, r: Rect, color, alpha: float = 0.25) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (r[0], r[1]), (r[2], r[3]), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text(img: np.ndarray, text: str, org: Point, size: float = 0.7,
             color=(255, 255, 255), thickness: int = 2, bg: tuple | None = (0, 0, 0),
             bg_alpha: float = 0.55) -> None:
    """在 img 上画带可选半透明背景的文字。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg is not None:
        (tw, th), baseline = cv2.getTextSize(text, font, size, thickness)
        x, y = org
        overlay = img.copy()
        cv2.rectangle(overlay, (x - 4, y - th - 6), (x + tw + 6, y + baseline + 4), bg, -1)
        cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    cv2.putText(img, text, org, font, size, color, thickness, cv2.LINE_AA)


def convex_quad_order(pts: Sequence[Point]) -> np.ndarray:
    """把 4 个点按凸包顺序排列，便于 cv2.fillConvexPoly / perspective."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) != 4:
        return pts.reshape(-1, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    return pts[order]


def quad_bounding_rect(quad: Sequence[Point]) -> Rect:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def mask_polygon(shape_hw: Tuple[int, int], quad: Sequence[Point]) -> np.ndarray:
    """返回与 shape (h, w) 同形状的 uint8 mask，多边形内部为 255。"""
    mask = np.zeros(shape_hw, dtype=np.uint8)
    poly = convex_quad_order(quad).astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask


def smooth_point(prev: Point | None, cur: Point, alpha: float = 0.6) -> Point:
    """指数滑动平均。alpha 越大越信任上一帧。"""
    if prev is None:
        return cur
    return (
        int(alpha * prev[0] + (1 - alpha) * cur[0]),
        int(alpha * prev[1] + (1 - alpha) * cur[1]),
    )


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def hsv_in_range(h: float, s_lo: float, s_hi: float) -> bool:
    return s_lo <= h <= s_hi


def add_alpha_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   color, alpha: float = 0.4) -> None:
    """画带 alpha 的矩形（在原图上叠加）。"""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def add_alpha_outline(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                      color, thickness: int = 3, alpha: float = 1.0) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
    if alpha >= 1.0:
        img[:] = overlay
    else:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_arrow(img: np.ndarray, x1, y1, x2, y2, color,
               thickness: int = 2, tip_size: int = 14) -> None:
    """画一条带箭头的直线。"""
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    # 计算箭头三角
    import math as _m
    ang = _m.atan2(y2 - y1, x2 - x1)
    tipx, tipy = int(x2), int(y2)
    bx = int(tipx - tip_size * _m.cos(ang - _m.pi / 6))
    by = int(tipy - tip_size * _m.sin(ang - _m.pi / 6))
    cx = int(tipx - tip_size * _m.cos(ang + _m.pi / 6))
    cy = int(tipy - tip_size * _m.sin(ang + _m.pi / 6))
    cv2.fillConvexPoly(img, np.array([[tipx, tipy], [bx, by], [cx, cy]], dtype=np.int32), color)


def draw_centered_text(img: np.ndarray, text: str, cy: int,
                       size: float = 1.0, color=(255, 255, 255),
                       thickness: int = 2,
                       bg: tuple | None = (0, 0, 0),
                       bg_alpha: float = 0.6, padding: int = 14) -> None:
    """画居中（水平方向）的文字，cy 为文字基线 y。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, size, thickness)
    h, w = img.shape[:2]
    x = max(padding, (w - tw) // 2)
    y = cy
    if bg is not None:
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (x - padding, y - th - padding // 2),
                      (x + tw + padding, y + baseline + padding // 2),
                      bg, -1)
        cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    cv2.putText(img, text, (x, y), font, size, color, thickness, cv2.LINE_AA)


def hsv_rainbow(t: float) -> tuple:
    """根据 t in [0,1) 返回 BGR 颜色。"""
    import colorsys
    h = t % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def lerp_color(c1, c2, t: float):
    """两个 BGR 元组线性插值，t in [0,1]。"""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def draw_hud_chip(img: np.ndarray, x: int, y: int, label: str, active: bool,
                   active_color=(60, 230, 255), inactive_color=(120, 120, 120)) -> None:
    """左上/右上角的小型状态徽章。"""
    color = active_color if active else inactive_color
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, 0.55, 2)
    pad = 6
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  color, 1)
    cv2.putText(img, label, (x, y), font, 0.55, color, 1, cv2.LINE_AA)