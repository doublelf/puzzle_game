"""3x3 手势拼图游戏：单手捏合绘制两个矩形 → 自动切片打乱 → 捏合吸附 + 拖向空白邻位完成交换。"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from gesture import HandTracker
from drawing import RectDrawer
from puzzle_game import PuzzleGame
from overlay import HandOverlay, GameEvent
from utils import put_text


SCREEN_W, SCREEN_H = 1280, 720


def parse_args():
    p = argparse.ArgumentParser(description="Pinch-Puzzle: 单手捏合拼图游戏")
    p.add_argument("--cam", type=int, default=0, help="摄像头编号 (默认 0)")
    p.add_argument("--no-flip", action="store_true",
                   help="不镜像摄像头画面 (默认镜像便于交互)")
    p.add_argument("--no-landmarks", action="store_true",
                   help="关闭默认的关键点叠加层")
    p.add_argument("--hide-trails", action="store_true",
                   help="关闭手势拖尾")
    return p.parse_args()


def draw_menu(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    put_text(frame, "Pinch-Puzzle",
             (w // 2 - 170, h // 2 - 120),
             size=1.4, color=(255, 255, 255), thickness=3, bg=(0, 0, 0))
    put_text(frame, "single-hand pinch  -  draw + slide tiles",
             (w // 2 - 250, h // 2 - 40),
             size=0.8, color=(160, 255, 180), bg=(0, 0, 0))
    put_text(frame,
             "Pinch with thumb+index to draw two rectangles,",
             (w // 2 - 280, h // 2 + 10),
             size=0.7, color=(220, 220, 220), bg=(0, 0, 0))
    put_text(frame,
             "then pinch-drag tiles into the blank slot to solve the 3x3.",
             (w // 2 - 290, h // 2 + 40),
             size=0.7, color=(220, 220, 220), bg=(0, 0, 0))

    put_text(frame, "press any key to start    press ESC to quit",
             (w // 2 - 230, h // 2 + 110),
             size=0.7, color=(220, 220, 220), bg=(0, 0, 0))


def draw_status_bar(frame: np.ndarray, state: str, moves: int) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(frame, f"Mode: PUZZLE  state={state}  moves={moves}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "R:Reset   M:Menu   ESC:Quit",
                (w - 280, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)


def pick_primary_hand(hands: dict):
    return hands.get("left") or hands.get("right")


def emit_pinch_events(hands: dict, overlay: HandOverlay, now: float):
    for h in (hands.get("left"), hands.get("right")):
        if h is None:
            continue
        if h.pinch_begin_edge:
            x, y = h.index_tip
            overlay.add_event(GameEvent("pinch_begin", {
                "x": x, "y": y, "label": h.label,
                "color": (255, 220, 80),
            }), now)
        if h.pinch_end_edge:
            x, y = h.index_tip
            overlay.add_event(GameEvent("pinch_end", {
                "x": x, "y": y, "label": h.label,
            }), now)


def run_puzzle(cap, tracker: HandTracker, overlay: HandOverlay,
               flip: bool, show_lm: bool, show_trail: bool) -> str:
    drawer = RectDrawer()
    puzzle = PuzzleGame()
    puzzle.state = "DRAW"
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (SCREEN_W, SCREEN_H))
        ts_ms = int((time.time() - t0) * 1000)
        hands = tracker.process(frame, ts_ms)
        now = time.time()

        emit_pinch_events(hands, overlay, now)
        primary = pick_primary_hand(hands)

        if puzzle.state == "DRAW":
            drawer.update(primary, frame.shape)
            if drawer.is_full:
                puzzle.set_rect(drawer.final_rect)
                puzzle.set_snapshot(frame)
                if puzzle.snapshot is not None:
                    puzzle.start_scramble()
            drawer.draw(frame)
            put_text(frame,
                     "Pinch with thumb+index to drag two rectangles.",
                     (20, SCREEN_H - 60),
                     size=0.65, color=(200, 255, 200))
            put_text(frame,
                     "Pinch again to start over (auto-clears).",
                     (20, SCREEN_H - 30),
                     size=0.65, color=(200, 255, 200))

        elif puzzle.state in ("PLAY", "SCRAMBLE"):
            if primary is not None:
                puzzle.handle_pinch(primary)
            if puzzle.snapshot is not None:
                puzzle.render(frame, primary)

        # 应用 puzzle 发出的事件 (grab/swap/error/solved) 到 overlay
        for evt in puzzle.events:
            if evt.type == "solved":
                evt.payload["w"] = frame.shape[1]
                evt.payload["h"] = frame.shape[0]
            overlay.add_event(evt, now)
        puzzle.events.clear()

        overlay.update_and_render(
            frame, hands, mode="puzzle", now=now,
            options={"show_landmarks": show_lm, "show_trails": show_trail},
        )

        draw_status_bar(frame, puzzle.state, puzzle.moves)
        cv2.imshow("Pinch-Puzzle", frame)
        key = cv2.waitKey(15) & 0xFF
        if key == 27:
            return "QUIT"
        if key in (ord('m'), ord('M')):
            return "MENU"
        if key in (ord('r'), ord('R')):
            return "RESET"

    return "QUIT"


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)
    if not cap.isOpened():
        print("ERROR: cannot open camera", args.cam, file=sys.stderr)
        sys.exit(1)

    tracker = HandTracker(num_hands=2)
    if not tracker.is_available():
        print("[main] MediaPipe Hand Landmarker 不可用, 请检查网络/模型。",
              file=sys.stderr)

    cv2.namedWindow("Pinch-Puzzle", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pinch-Puzzle", SCREEN_W, SCREEN_H)

    overlay = HandOverlay()
    show_lm = not args.no_landmarks
    show_trail = not args.hide_trails

    mode = "MENU"
    t0 = time.time()
    try:
        while True:
            if mode == "MENU":
                ret, frame = cap.read()
                if not ret:
                    break
                if not args.no_flip:
                    frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (SCREEN_W, SCREEN_H))
                draw_menu(frame)
                cv2.imshow("Pinch-Puzzle", frame)
                key = cv2.waitKey(20) & 0xFF
                if key == 27:
                    break
                if key != 255:  # any key starts the game
                    mode = "PUZZLE"
            elif mode == "PUZZLE":
                nxt = run_puzzle(cap, tracker, overlay,
                                 not args.no_flip, show_lm, show_trail)
                if nxt in ("QUIT", "MENU"):
                    if nxt == "MENU":
                        mode = "MENU"
                    else:
                        break
                elif nxt == "RESET":
                    mode = "PUZZLE"
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()