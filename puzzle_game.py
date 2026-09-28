"""3x3 拼图游戏。

阶段: DRAW -> SCRAMBLE -> PLAY -> SOLVED

输入最终 ROI 后截取快照，按 3x3 切片打乱。
玩家通过单手捏合吸附最近块 + 移动至空白邻位来交换方块。"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import Rect, dist, put_text, normalize_rect, quad_bounding_rect
from overlay import GameEvent


GRID = 3
TILES = GRID * GRID
BLANK = TILES - 1
NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _solved_board():
    return list(range(TILES))


def _swap(board, i, j):
    board[i], board[j] = board[j], board[i]


def _inversions(board) -> int:
    inv = 0
    arr = [x for x in board if x != BLANK]
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv


def _is_solvable(board, grid_w=GRID) -> bool:
    inv = _inversions(board)
    if grid_w % 2 == 1:
        return inv % 2 == 0
    # 3x3 行宽，BLANK 所在行
    blank_row_from_bottom = (TILES - 1 - board.index(BLANK)) // grid_w
    row_from_bottom = blank_row_from_bottom
    if row_from_bottom % 2 == 0:
        return inv % 2 == 1
    return inv % 2 == 0


def shuffle_solvable(steps: int = 200, rng=None) -> List[int]:
    """从解出状态出发，随机走 steps 步，得到一定可达的乱序。"""
    rng = rng or random.Random()
    board = _solved_board()
    blank_idx = board.index(BLANK)
    last_blank = -1
    for _ in range(steps):
        neighbors = []
        for dr, dc in NEIGHBOR_OFFSETS:
            r, c = divmod(blank_idx, GRID)
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID and 0 <= nc < GRID:
                nidx = nr * GRID + nc
                if nidx != last_blank:
                    neighbors.append(nidx)
        if not neighbors:
            continue
        target = rng.choice(neighbors)
        last_blank = blank_idx
        _swap(board, blank_idx, target)
        blank_idx = target
    return board


@dataclass
class PuzzleGame:
    """3x3 拼图游戏状态机。"""
    state: str = "DRAW"          # DRAW | SCRAMBLE | PLAY | SOLVED
    rect: Optional[Rect] = None
    snapshot: Optional[np.ndarray] = None  # ROI 截图 (彩色)
    board: List[int] = field(default_factory=_solved_board)
    tile_w: int = 0
    tile_h: int = 0
    display_origin: Tuple[int, int] = (0, 0)  # 渲染的原点在画面上的位置
    scale: float = 1.0          # ROI → 渲染的缩放（可能等比缩放以适配屏幕）
    grab: Optional[int] = None   # 当前被吸住的拼图块编号（tile id）
    grab_offset: Tuple[int, int] = (0, 0)
    grab_origin: Tuple[int, int] = (0, 0)   # 抓取时块在画面上的中心
    grab_pinch_start: Tuple[int, int] = (0, 0)
    grab_drag_distance: float = 0.0
    move_cooldown: float = 0.0   # 防止连续多次 swap
    solved_at: float = 0.0
    moves: int = 0
    events: List[GameEvent] = field(default_factory=list)

    def reset(self):
        self.__init__()

    def set_rect(self, r: Rect):
        self.rect = normalize_rect(*r)
        self._capture()

    def _capture(self):
        # 由 main 调用时传入 frame; 这里仅占位。实际图像由 set_snapshot 设置。
        pass

    def set_snapshot(self, frame_bgr: np.ndarray):
        if self.rect is None:
            return
        x1, y1, x2, y2 = self.rect
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame_bgr.shape[1], x2); y2 = min(frame_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return
        self.snapshot = frame_bgr[y1:y2, x1:x2].copy()
        h, w = self.snapshot.shape[:2]
        # 缩放到合理尺寸（最长边 ≤ 480）
        target = 480
        m = max(h, w)
        if m > target:
            self.scale = target / m
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            self.snapshot = cv2.resize(self.snapshot, (new_w, new_h))
            self.tile_w = self.snapshot.shape[1] // GRID
            self.tile_h = self.snapshot.shape[0] // GRID
            self.snapshot = self.snapshot[: self.tile_h * GRID, : self.tile_w * GRID]
        else:
            self.scale = 1.0
            self.tile_w = w // GRID
            self.tile_h = h // GRID
            self.snapshot = self.snapshot[: self.tile_h * GRID, : self.tile_w * GRID]
        # 显示在画面右上角
        self.display_origin = (self.rect[0] + self.rect[2]) // 2 - self.snapshot.shape[1] // 2, \
                              max(20, self.rect[1] - self.snapshot.shape[0] - 20)

    def start_scramble(self):
        """开始打乱 + 切到 PLAY 阶段。打乱前先确保 snapshot 已准备好。"""
        if self.snapshot is None:
            return
        self.board = shuffle_solvable(steps=240)
        self.state = "PLAY"
        self.moves = 0
        self.move_cooldown = 0.0
        self.grab = None

    def _tile_pixel(self, board_idx: int) -> Tuple[int, int]:
        """board_idx 位置对应的 tile 中心在快照坐标里的位置。"""
        r, c = divmod(board_idx, GRID)
        return (c * self.tile_w + self.tile_w // 2,
                r * self.tile_h + self.tile_h // 2)

    def _board_pos_from_pixel(self, px: int, py: int) -> Tuple[int, int]:
        """把屏幕坐标转换为 board_idx。"""
        ox, oy = self.display_origin
        x = (px - ox) / max(1, self.tile_w)
        y = (py - oy) / max(1, self.tile_h)
        c = int(x); r = int(y)
        if r < 0 or r >= GRID or c < 0 or c >= GRID:
            return (-1, -1)
        return r, c

    def handle_pinch(self, hand_state, dt: float = 1 / 30):
        if self.state != "PLAY" or self.snapshot is None or hand_state is None:
            self.grab = None
            return
        self.move_cooldown = max(0.0, self.move_cooldown - dt)
        tip = hand_state.index_tip
        pinching = hand_state.pinching

        if not pinching:
            # 松开 → 检查是否与空白邻位发生交换
            if self.grab is not None:
                self._try_complete_drag()
            self.grab = None
            return

        # 取出吸附的 tile
        if self.grab is None:
            # 找离 pinch 点最近的 tile 中心
            best_pos = None
            best_dist = float("inf")
            for pos in range(TILES):
                if pos == BLANK:
                    continue
                bx, by = self._tile_pixel(pos)
                cx_screen = self.display_origin[0] + bx
                cy_screen = self.display_origin[1] + by
                d = dist(tip, (cx_screen, cy_screen))
                if d < best_dist:
                    best_dist = d
                    best_pos = pos
            if best_pos is None or best_dist > max(self.tile_w, self.tile_h) * 1.6:
                return
            # best_pos 是当前位置索引; tile_id 是该位置的拼图块 ID
            tile_id = self.board[best_pos]
            self.grab = tile_id
            self.grab_pinch_start = tip
            src_pos = best_pos
            # 块在画面上的中心
            tile_px = self._tile_pixel(src_pos)
            self.grab_origin = (self.display_origin[0] + tile_px[0],
                                self.display_origin[1] + tile_px[1])
            # 触发 GRAB 事件 (金色脉冲 + toast)
            tile_rect_screen = self._tile_screen_rect(src_pos)
            self.events.append(GameEvent("grab", {
                "rect": tile_rect_screen,
                "label": tile_id + 1,
            }))

        # 拖动方向：tip 相对 grab_pinch_start
        dx = tip[0] - self.grab_pinch_start[0]
        dy = tip[1] - self.grab_pinch_start[1]
        self.grab_drag_distance = dist((0, 0), (dx, dy))
        # 如果拖到另一个 tile 中心附近，且该 tile 是空白格 → 触发交换
        # 这里采用"块中心进入邻接空白 tile 的中心圆范围"时立即交换
        tile_id = self.grab
        src_pos = self.board.index(tile_id)
        # 找出每个邻接格
        swapped = False
        for off_r, off_c in NEIGHBOR_OFFSETS:
            r, c = divmod(src_pos, GRID)
            nr, nc = r + off_r, c + off_c
            if 0 <= nr < GRID and 0 <= nc < GRID:
                nidx = nr * GRID + nc
                if self.board[nidx] == BLANK:
                    # 检查 pinch tip 是否进入了空白格的中心阈值
                    blank_cx = self.display_origin[0] + nc * self.tile_w + self.tile_w // 2
                    blank_cy = self.display_origin[1] + nr * self.tile_h + self.tile_h // 2
                    if self.move_cooldown == 0 and dist(tip, (blank_cx, blank_cy)) < min(self.tile_w, self.tile_h) * 0.55:
                        old_rect = self._tile_screen_rect(src_pos)
                        new_rect = self._tile_screen_rect(nidx)
                        _swap(self.board, src_pos, nidx)
                        self.moves += 1
                        self.move_cooldown = 0.18
                        # 重置 grab 锚点
                        self.grab_pinch_start = (blank_cx, blank_cy)
                        # 触发 SWAP 事件 (双矩形闪光 + toast)
                        self.events.append(GameEvent("swap", {
                            "rects": [old_rect, new_rect],
                        }))
                        swapped = True
                        break

        # 错误拖动检测: 当拖动距离 > 半格 且 拖动方向无空白邻位 → 红色闪光
        if not swapped and self.move_cooldown == 0:
            min_side = min(self.tile_w, self.tile_h)
            if self.grab_drag_distance > min_side * 0.6:
                # 找到主拖动方向上的邻位 (dx/dy 主轴)
                if abs(dx) > abs(dy):
                    off_r, off_c = 0, (1 if dx > 0 else -1)
                else:
                    off_r, off_c = (1 if dy > 0 else -1), 0
                r, c = divmod(src_pos, GRID)
                nr, nc = r + off_r, c + off_c
                if 0 <= nr < GRID and 0 <= nc < GRID:
                    nidx = nr * GRID + nc
                    if self.board[nidx] != BLANK:
                        err_rect = self._tile_screen_rect(nidx)
                        self.events.append(GameEvent("error", {"rect": err_rect}))
                        self.move_cooldown = 0.18

        if self.board == _solved_board():
            self.state = "SOLVED"
            self.solved_at = time.time()
            self.grab = None
            # 触发 SOLVED 事件 (粒子爆裂)
            self.events.append(GameEvent("solved", {
                "w": 0,  # 由 main 填实际尺寸
                "h": 0,
            }))

    def _tile_screen_rect(self, board_idx: int):
        """返回 tile 在屏幕坐标系下的矩形 (x1, y1, x2, y2)。"""
        if board_idx < 0 or board_idx >= TILES:
            return (0, 0, 0, 0)
        r, c = divmod(board_idx, GRID)
        ox, oy = self.display_origin
        return (ox + c * self.tile_w,
                oy + r * self.tile_h,
                ox + (c + 1) * self.tile_w,
                oy + (r + 1) * self.tile_h)

    def _try_complete_drag(self):
        # 松开时若未完成交换，松手只是放下，不需要额外操作
        pass

    def render(self, frame: np.ndarray, hand_state=None) -> np.ndarray:
        if self.snapshot is None:
            return frame
        out = frame
        ox, oy = self.display_origin
        snap_h, snap_w = self.snapshot.shape[:2]
        # 把帧粘贴到 background 上（但只显示当前 board 的合成图）
        composite = self._compose_board()
        # 边界检查 display_origin
        x1 = max(0, ox); y1 = max(0, oy)
        x2 = min(frame.shape[1], ox + snap_w)
        y2 = min(frame.shape[0], oy + snap_h)
        cw = x2 - x1; ch = y2 - y1
        if cw > 0 and ch > 0:
            sub = composite[y1 - oy:y1 - oy + ch, x1 - ox:x1 - ox + cw]
            blend = cv2.addWeighted(out[y1:y2, x1:x2], 0.4, sub, 0.7, 0)
            out[y1:y2, x1:x2] = blend

        # 边框 + 标题
        cv2.rectangle(out, (ox, oy), (ox + snap_w - 1, oy + snap_h - 1),
                      (255, 255, 255), 2)
        put_text(out, f"3x3 Puzzle  moves={self.moves}",
                 (ox, oy - 10), size=0.65, color=(255, 255, 255),
                 bg=(0, 0, 0), bg_alpha=0.5)

        # 选中提示
        if self.grab is not None and hand_state is not None and hand_state.pinching:
            cv2.circle(out, hand_state.index_tip, 12, (0, 255, 255), 2)
            put_text(out, "drag toward blank tile",
                     (hand_state.index_tip[0] + 14, hand_state.index_tip[1] - 10),
                     size=0.55, color=(0, 255, 255))

        if self.state == "SOLVED":
            msg = "SOLVED!"
            put_text(out, msg,
                     (out.shape[1] // 2 - 130, out.shape[0] // 2),
                     size=1.8, color=(120, 255, 120), thickness=3,
                     bg=(0, 0, 0), bg_alpha=0.6)
        return out

    def _compose_board(self) -> np.ndarray:
        """按当前 board 拼出快照。"""
        snap = np.zeros_like(self.snapshot)
        for pos in range(TILES):
            tile_id = self.board[pos]
            if tile_id == BLANK:
                continue
            tr, tc = divmod(tile_id, GRID)
            sub = self.snapshot[tr * self.tile_h:(tr + 1) * self.tile_h,
                                tc * self.tile_w:(tc + 1) * self.tile_w]
            r, c = divmod(pos, GRID)
            snap[r * self.tile_h:(r + 1) * self.tile_h,
                 c * self.tile_w:(c + 1) * self.tile_w] = sub
            # 数字标签
            cx = c * self.tile_w + self.tile_w // 2
            cy = r * self.tile_h + self.tile_h // 2
            cv2.putText(snap, str(tile_id + 1),
                        (cx - 14, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(snap, str(tile_id + 1),
                        (cx - 14, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
        return snap

    @property
    def is_solved(self) -> bool:
        return self.state == "SOLVED"