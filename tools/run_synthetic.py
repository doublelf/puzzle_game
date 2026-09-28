"""拼图项目 — 无摄像头/无 GUI 时的端到端运行脚本。

驱动 PuzzleGame + HandOverlay，每帧保存到 frames/ 目录下的 PNG 文件。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT)

from gesture import HandState
from drawing import RectDrawer
from puzzle_game import PuzzleGame
from overlay import HandOverlay, GameEvent

from tools.synthetic_source import SyntheticHandSource
from tools.fake_tracker import SyntheticHandTracker


SCREEN_W, SCREEN_H = 1280, 720


def emit_pinch_events(hands, overlay, now):
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


def run(n_frames: int, out_dir: str):
    src = SyntheticHandSource(width=SCREEN_W, height=SCREEN_H)
    tr = SyntheticHandTracker(width=SCREEN_W, height=SCREEN_H)
    overlay = HandOverlay()
    drawer = RectDrawer()
    puzzle = PuzzleGame()
    puzzle.state = "DRAW"
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    events_log = []
    for i in range(n_frames):
        ok, frame = src.read()
        if not ok:
            break
        ts_ms = int((time.time() - t0) * 1000)
        hands = tr.process(frame, ts_ms)
        now = time.time()
        emit_pinch_events(hands, overlay, now)
        primary = hands.get("left") or hands.get("right")
        if puzzle.state == "DRAW":
            drawer.update(primary, frame.shape)
            if drawer.is_full:
                puzzle.set_rect(drawer.final_rect)
                puzzle.set_snapshot(frame)
                if puzzle.snapshot is not None:
                    puzzle.start_scramble()
            drawer.draw(frame)
        elif puzzle.state in ("PLAY", "SCRAMBLE"):
            if primary is not None:
                puzzle.handle_pinch(primary)
            if puzzle.snapshot is not None:
                puzzle.render(frame, primary)
        for evt in puzzle.events:
            events_log.append((i, evt.type, evt.payload))
            if evt.type == "solved":
                evt.payload["w"] = frame.shape[1]
                evt.payload["h"] = frame.shape[0]
            overlay.add_event(evt, now)
        puzzle.events.clear()
        overlay.update_and_render(
            frame, hands, mode="puzzle", now=now,
            options={"show_landmarks": True, "show_trails": True},
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (30, 30, 30), -1)
        cv2.putText(frame, f"PUZZLE state={puzzle.state} moves={puzzle.moves} frame={i}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"puzzle_{i:04d}.png"), frame)
    return events_log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--out", default="/tmp/puzzle_synthetic")
    args = p.parse_args()
    print(f"== puzzle_game synthetic runner == frames={args.frames} out={args.out}")
    events = run(args.frames, args.out)
    types = {}
    for _, t, _ in events:
        types[t] = types.get(t, 0) + 1
    print(f"events total={len(events)} by_type={types}")
    print(f"frames written to {args.out}")


if __name__ == "__main__":
    main()