"""反馈层：拖尾 / 扩散环 / 呼吸光晕 / 矩形闪光 / 粒子爆裂 / Toast / HandOverlay。"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    add_alpha_outline,
    add_alpha_rect,
    draw_arrow,
    draw_centered_text,
    draw_hud_chip,
    hsv_rainbow,
    lerp_color,
    put_text,
)


# ---------------- 事件 ----------------

@dataclass
class GameEvent:
    type: str           # pinch_begin / pinch_end / both_connected / grab /
                        # swap / error / solved / filter_switch / swipe
    payload: dict = field(default_factory=dict)


# ---------------- Trail ----------------

@dataclass
class TrailPoint:
    x: float
    y: float
    t: float
    color: tuple


class Trail:
    """手势拖尾。每个手 (label) 一条轨迹。"""

    def __init__(self, max_len: int = 14, lifetime: float = 0.45):
        self.max_len = max_len
        self.lifetime = lifetime
        self.points: Deque[TrailPoint] = deque(maxlen=max_len)

    def push(self, x, y, t, color=(255, 255, 255)):
        self.points.append(TrailPoint(float(x), float(y), float(t), color))

    def draw(self, img: np.ndarray, now: float):
        pts = list(self.points)
        if len(pts) < 2:
            return
        for i in range(1, len(pts)):
            a = pts[i - 1]
            b = pts[i]
            age_a = now - a.t
            age_b = now - b.t
            if age_a > self.lifetime or age_b > self.lifetime:
                continue
            # 越新越亮
            alpha_b = max(0.0, 1.0 - age_b / self.lifetime)
            alpha_a = max(0.0, 1.0 - age_a / self.lifetime)
            thickness = max(1, int(6 * (alpha_b + alpha_a) / 2))
            overlay = img.copy()
            cv2.line(overlay,
                     (int(a.x), int(a.y)), (int(b.x), int(b.y)),
                     b.color, thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha_b, img, 1 - alpha_b, 0, img)
        # 末端亮点
        last = pts[-1]
        if now - last.t <= self.lifetime:
            cv2.circle(img, (int(last.x), int(last.y)), 5, last.color, -1,
                       cv2.LINE_AA)


# ---------------- PinchPulse (扩散环) ----------------

@dataclass
class Pulse:
    x: float
    y: float
    t0: float
    duration: float
    max_radius: float
    color: tuple
    width: int = 3


class PinchPulse:
    def __init__(self):
        self.pulses: List[Pulse] = []

    def trigger(self, x, y, now, color=(255, 220, 80),
                duration: float = 0.32, max_radius: float = 48.0):
        self.pulses.append(Pulse(float(x), float(y), now, duration,
                                 max_radius, color))

    def draw(self, img, now):
        alive = []
        for p in self.pulses:
            t = (now - p.t0) / p.duration
            if t >= 1.0:
                continue
            alive.append(p)
            r = p.max_radius * t
            alpha = max(0.0, 1.0 - t)
            overlay = img.copy()
            cv2.circle(overlay, (int(p.x), int(p.y)), max(1, int(r)),
                       p.color, p.width, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            # 第二圈 (反向) 增加层次感
            r2 = p.max_radius * (1 - t) * 0.6
            if r2 > 1:
                cv2.circle(img, (int(p.x), int(p.y)), int(r2),
                           lerp_color(p.color, (255, 255, 255), 0.5),
                           1, cv2.LINE_AA)
        self.pulses = alive


# ---------------- BreathHalo (呼吸光晕) ----------------

class BreathHalo:
    """持续呼吸光晕 (跟随指尖)。"""

    def __init__(self, base_color=(255, 220, 80)):
        self.color = base_color

    def draw(self, img, x, y, now, strength: float = 1.0,
             base_radius: int = 10):
        if strength <= 0.0:
            return
        # 正弦呼吸
        s = (math.sin(now * 6.0) + 1.0) / 2.0  # 0..1
        radius = base_radius + int(8 * s * strength)
        alpha = 0.35 + 0.35 * s * strength
        overlay = img.copy()
        cv2.circle(overlay, (int(x), int(y)), radius, self.color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        # 内圈高亮
        cv2.circle(img, (int(x), int(y)), base_radius - 2,
                   (255, 255, 255), 1, cv2.LINE_AA)


# ---------------- RectFlash (矩形闪光) ----------------

@dataclass
class RectFlashItem:
    x1: int
    y1: int
    x2: int
    y2: int
    t0: float
    duration: float
    color: tuple
    thickness: int = 4


class RectFlash:
    def __init__(self):
        self.items: List[RectFlashItem] = []

    def flash(self, rect, now, color=(80, 255, 120),
              duration: float = 0.35, thickness: int = 5):
        x1, y1, x2, y2 = rect
        self.items.append(RectFlashItem(int(x1), int(y1), int(x2), int(y2),
                                        now, duration, color, thickness))

    def draw(self, img, now):
        alive = []
        for it in self.items:
            t = (now - it.t0) / it.duration
            if t >= 1.0:
                continue
            alive.append(it)
            # 透明 alpha 衰减 + 边框渐变白色
            alpha = max(0.0, 1.0 - t)
            color = lerp_color(it.color, (255, 255, 255), t * 0.6)
            overlay = img.copy()
            cv2.rectangle(overlay, (it.x1, it.y1), (it.x2, it.y2),
                          color, it.thickness)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        self.items = alive


# ---------------- ParticleBurst ----------------

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    t0: float
    life: float
    color: tuple
    size: float
    gravity: float = 380.0


class ParticleBurst:
    def __init__(self):
        self.particles: List[Particle] = []

    def burst(self, x, y, now, n: int = 60, life: float = 1.6,
              speed_range: Tuple[float, float] = (180, 420),
              size_range: Tuple[float, float] = (3, 8),
              palette: str = "rainbow"):
        for i in range(n):
            ang = (i / max(1, n)) * math.tau + random.uniform(-0.15, 0.15)
            sp = random.uniform(*speed_range)
            color = hsv_rainbow(random.random()) if palette == "rainbow" else (
                random.randint(120, 255), random.randint(180, 255),
                random.randint(120, 255))
            self.particles.append(Particle(
                x=float(x), y=float(y),
                vx=math.cos(ang) * sp, vy=math.sin(ang) * sp - 120,
                t0=now, life=life, color=color,
                size=random.uniform(*size_range),
            ))

    def draw(self, img, now):
        h, w = img.shape[:2]
        alive = []
        dt = 1 / 30.0
        for p in self.particles:
            age = now - p.t0
            if age >= p.life:
                continue
            alive.append(p)
            # 更新位置
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy += p.gravity * dt
            alpha = max(0.0, 1.0 - age / p.life)
            ix, iy = int(p.x), int(p.y)
            if 0 <= ix < w and 0 <= iy < h:
                overlay = img.copy()
                cv2.circle(overlay, (ix, iy), max(1, int(p.size)),
                           p.color, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        self.particles = alive


# ---------------- SwipeIndicator (横滑闪光) ----------------

@dataclass
class SwipeFlash:
    direction: int   # +1 右 / -1 左
    t0: float
    duration: float
    frame_w: int


class SwipeIndicator:
    def __init__(self):
        self.flashes: List[SwipeFlash] = []

    def flash(self, direction: int, now, frame_w: int, duration: float = 0.55):
        self.flashes.append(SwipeFlash(direction, now, duration, frame_w))

    def draw(self, img, now):
        alive = []
        for f in self.flashes:
            t = (now - f.t0) / f.duration
            if t >= 1.0:
                continue
            alive.append(f)
            w = img.shape[1]
            thickness = max(40, int(w * 0.18))
            x_center = w - 30 if f.direction > 0 else 30
            color = (60, 230, 255) if f.direction > 0 else (255, 180, 80)
            # 透明渐变带
            overlay = img.copy()
            cv2.rectangle(overlay,
                          (x_center - thickness // 2, 0),
                          (x_center + thickness // 2, img.shape[0]),
                          color, -1)
            alpha = max(0.0, 0.55 * (1.0 - t))
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            # 箭头 + 文字
            cx = x_center + (30 if f.direction > 0 else -30)
            draw_arrow(img, cx - 60 * f.direction, img.shape[0] // 2,
                       cx + 10 * f.direction, img.shape[0] // 2,
                       color, thickness=4, tip_size=20)
        self.flashes = alive


# ---------------- Toast ----------------

@dataclass
class ToastItem:
    text: str
    t0: float
    duration: float
    color: tuple
    sub: str = ""
    big: bool = False


class Toasts:
    def __init__(self):
        self.items: List[ToastItem] = []

    def push(self, text: str, now, duration: float = 0.7,
             color=(255, 255, 255), sub: str = "", big: bool = False):
        self.items.append(ToastItem(text, now, duration, color, sub, big))

    def draw(self, img, now):
        alive = []
        for it in self.items:
            t = (now - it.t0) / it.duration
            if t >= 1.0:
                continue
            alive.append(it)
            # 滑入 / 滑出
            enter_t = min(1.0, t / 0.12)
            exit_t = max(0.0, 1.0 - max(0.0, (t - 0.85) / 0.15))
            y_offset = int((1 - enter_t) * 40 - (1 - exit_t) * 20)
            alpha = min(1.0, enter_t) * min(1.0, exit_t)
            size = 1.2 if it.big else 0.75
            h, w = img.shape[:2]
            (tw, th), baseline = cv2.getTextSize(it.text,
                                                 cv2.FONT_HERSHEY_SIMPLEX,
                                                 size, 2)
            x = (w - tw) // 2
            cy = 70 + y_offset
            overlay = img.copy()
            cv2.rectangle(overlay,
                          (x - 18, cy - th - 14),
                          (x + tw + 18, cy + baseline + 14),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55 * alpha, img,
                            1 - 0.55 * alpha, 0, img)
            color = lerp_color((180, 180, 180), it.color, alpha)
            cv2.putText(img, it.text, (x, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)
            if it.sub:
                put_text(img, it.sub, (x + 8, cy + 30),
                         size=0.55, color=(200, 200, 200), bg=(0, 0, 0))
        self.items = alive


# ---------------- HandOverlay (聚合) ----------------

class HandOverlay:
    """所有反馈的聚合入口。"""

    def __init__(self):
        self.trails: dict[str, Trail] = {}
        self.pulse = PinchPulse()
        self.rect_flash = RectFlash()
        self.burst = ParticleBurst()
        self.swipe = SwipeIndicator()
        self.toasts = Toasts()
        # 缓存上一次两个 tip 的坐标 (用于画"捏合连接"过渡)
        self._last_tip_l: Optional[tuple] = None
        self._last_tip_r: Optional[tuple] = None
        self._last_both_state: bool = False

    def trail_for(self, label: str) -> Trail:
        if label not in self.trails:
            self.trails[label] = Trail()
        return self.trails[label]

    def add_event(self, evt: GameEvent, now: float):
        et = evt.type
        p = evt.payload
        if et == "pinch_begin":
            x, y = p["x"], p["y"]
            color = p.get("color", (255, 220, 80))
            self.pulse.trigger(x, y, now, color=color, duration=0.35,
                               max_radius=46)
        elif et == "pinch_end":
            x, y = p["x"], p["y"]
            self.pulse.trigger(x, y, now, color=(180, 180, 220),
                               duration=0.28, max_radius=30)
        elif et == "both_connected":
            x, y = p["x"], p["y"]
            self.pulse.trigger(x, y, now, color=(120, 255, 180),
                               duration=0.55, max_radius=70)
            self.toasts.push("CONNECTED", now, duration=0.55,
                             color=(120, 255, 180))
        elif et == "grab":
            self.rect_flash.flash(p["rect"], now, color=(255, 220, 80),
                                  duration=0.4, thickness=5)
            self.toasts.push(f"Tile #{p['label']} selected", now,
                             duration=0.55, color=(255, 220, 80))
        elif et == "swap":
            for rect in p["rects"]:
                self.rect_flash.flash(rect, now, color=(80, 255, 120),
                                      duration=0.4, thickness=6)
            self.toasts.push("SWAP", now, duration=0.35, color=(80, 255, 120),
                             big=True)
        elif et == "error":
            self.rect_flash.flash(p["rect"], now, color=(80, 80, 255),
                                  duration=0.3, thickness=4)
        elif et == "solved":
            w, h = p["w"], p["h"]
            self.burst.burst(w // 2, h // 2, now, n=80, life=2.0)
            self.toasts.push("SOLVED!", now, duration=1.8,
                             color=(255, 230, 80), big=True,
                             sub="press R to restart")
        elif et == "filter_switch":
            self.toasts.push(f"Filter {p['idx']+1}/{p['total']}  {p['name']}",
                             now, duration=0.8, color=p.get("color", (255, 255, 255)))
        elif et == "swipe":
            self.swipe.flash(p["direction"], now, p.get("frame_w", 1280))

    # --------- 每帧主渲染 ---------

    def update_and_render(self, img: np.ndarray, hands: dict,
                          mode: str, now: float, options: dict | None = None):
        """options: dict { 'show_landmarks': True, 'show_trails': True, ... }"""
        opts = options or {}
        show_lm = opts.get("show_landmarks", True)
        show_trail = opts.get("show_trails", True)

        # 1. 关键点层（始终画，食指尖 + 拇指尖加重）
        if show_lm:
            for hand in (hands.get("left"), hands.get("right")):
                self._draw_hand(img, hand, now)

        # 2. 拖尾
        if show_trail:
            for hand in (hands.get("left"), hands.get("right")):
                if hand is None:
                    continue
                color = (0, 255, 180) if hand.label == "Left" else (255, 200, 80)
                tr = self.trail_for(hand.label)
                tr.push(hand.index_tip[0], hand.index_tip[1], now, color=color)
            for tr in self.trails.values():
                tr.draw(img, now)

        # 3. 呼吸光晕（仅在捏合时）
        for hand in (hands.get("left"), hands.get("right")):
            if hand is None:
                continue
            if hand.pinching:
                breath = BreathHalo((255, 220, 80))
                breath.draw(img, hand.index_tip[0], hand.index_tip[1],
                            now, strength=1.0, base_radius=14)

        # 4. 扩散环 / 矩形闪光 / 横滑闪光 / 粒子 / toast
        self.pulse.draw(img, now)
        self.rect_flash.draw(img, now)
        self.swipe.draw(img, now)
        self.burst.draw(img, now)
        self.toasts.draw(img, now)

        # 5. HUD 徽章（左上 = 左手、右上 = 右手）
        self._draw_hud(img, hands, mode, now)

    def _draw_hand(self, img, hand, now):
        if hand is None:
            return
        pts = hand.landmarks_px.astype(int)
        # 骨架
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        for a, b in connections:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]),
                     (160, 220, 255), 1, cv2.LINE_AA)
        # 所有点
        for i, p in enumerate(pts):
            color = (120, 230, 160) if i not in (4, 8) else (50, 220, 255)
            radius = 3 if i not in (4, 8) else 5
            cv2.circle(img, tuple(p), radius, color, -1, cv2.LINE_AA)
            # 高亮指尖再加外圈
            if i in (4, 8):
                cv2.circle(img, tuple(p), radius + 4, (255, 255, 255), 1,
                           cv2.LINE_AA)
        # 掌心十字
        from gesture import INDEX_MCP, PINKY_MCP, WRIST
        for i in (INDEX_MCP, PINKY_MCP, WRIST):
            cv2.circle(img, tuple(pts[i]), 2, (255, 255, 255), -1, cv2.LINE_AA)
        # 拇指食指连线（粗细+颜色根据 pinch）
        col = (0, 200, 255) if hand.pinching else (200, 200, 200)
        thick = 4 if hand.pinching else 1
        cv2.line(img, hand.thumb_tip, hand.index_tip, col, thick, cv2.LINE_AA)

    def _draw_hud(self, img, hands, mode, now):
        h, w = img.shape[:2]
        # 左上角徽章：左手
        l = hands.get("left")
        r = hands.get("right")
        draw_hud_chip(img, 14, 32, f"L {'● PINCH' if l and l.pinching else 'o ready'}",
                      active=(l is not None and l.pinching))
        draw_hud_chip(img, 14, 70, f"R {'● PINCH' if r and r.pinching else 'o ready'}",
                      active=(r is not None and r.pinching))
        # 右下角 mode tag
        label = f"mode: {mode.upper()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, 0.55, 1)
        x = w - tw - 20
        y = h - 16
        overlay = img.copy()
        cv2.rectangle(overlay, (x - 8, y - th - 6),
                      (x + tw + 8, y + baseline + 4), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        cv2.putText(img, label, (x, y), font, 0.55, (180, 220, 255),
                    1, cv2.LINE_AA)