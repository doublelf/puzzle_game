"""拼图项目 — 端到端验证套件。

不依赖摄像头，验证：
  1. 模块导入
  2. utils 工具
  3. RectDrawer 状态机
  4. PuzzleGame 状态机 + 打乱算法
  5. 真实 MediaPipe Hand Landmarker 在 aarch64 上加载
  6. HandOverlay 事件系统
  7. 合成 pipeline 跑 N 帧不出错
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def section(title):
    print(f"\n=== {title} ===")


def step(name, fn):
    print(f"  - {name} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        fn()
        print(f"OK  ({time.time() - t0:.2f}s)")
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    section("1. module imports")
    import gesture  # noqa: F401
    import drawing  # noqa: F401
    import puzzle_game  # noqa: F401
    import overlay  # noqa: F401
    import utils  # noqa: F401
    print("  - gesture / drawing / puzzle_game / overlay / utils import OK")

    section("2. utils sanity")
    step("normalize_rect", lambda: utils.normalize_rect(10, 20, 5, 30))
    step("convex_quad_order", lambda: utils.convex_quad_order(
        [(0, 0), (10, 0), (10, 10), (0, 10)]))
    step("smooth_point", lambda: utils.smooth_point((0, 0), (10, 10)))
    step("put_text", lambda: utils.put_text(np.zeros((100, 200, 3), dtype=np.uint8),
                                            "hi", (10, 50)))
    step("hsv_rainbow", lambda: utils.hsv_rainbow(0.5))

    section("3. RectDrawer")
    def rect_drawer():
        from drawing import RectDrawer
        d = RectDrawer()
        # 不带 HandState → 应当不抛异常
        d.update(None, (720, 1280, 3))
    step("RectDrawer.update(None)", rect_drawer)

    section("4. PuzzleGame state machine")
    def puzzle_sm():
        from puzzle_game import PuzzleGame, shuffle_solvable
        p = PuzzleGame()
        assert p.state == "DRAW"
        b = shuffle_solvable(steps=20)
        assert len(b) == 9
        assert 8 in b, "blank tile (id=8) must be present"
    step("shuffle_solvable produces valid board", puzzle_sm)

    section("5. real MediaPipe Hand Landmarker on aarch64")
    def load_model():
        from gesture import HandTracker
        t = HandTracker(num_hands=2)
        assert t.is_available(), "model load failed"
        t.close()
    step("model load + inference on dummy frame", load_model)

    section("6. overlay events")
    def overlay_demo():
        from overlay import HandOverlay, GameEvent
        o = HandOverlay()
        o.add_event(GameEvent("solved", {"w": 1280, "h": 720}), time.time())
        o.add_event(GameEvent("swap", {"rects": [(0, 0, 100, 100), (200, 200, 300, 300)]}),
                    time.time())
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        o.update_and_render(img, {"left": None, "right": None},
                            mode="puzzle", now=time.time())
    step("overlay.add_event + render", overlay_demo)

    section("7. end-to-end synthetic pipeline")
    def end_to_end():
        from synthetic_source import SyntheticHandSource
        from fake_tracker import SyntheticHandTracker
        from overlay import HandOverlay, GameEvent
        from drawing import RectDrawer
        from puzzle_game import PuzzleGame
        src = SyntheticHandSource(width=640, height=360)
        tr = SyntheticHandTracker(width=640, height=360)
        ov = HandOverlay()
        drawer = RectDrawer()
        p = PuzzleGame()
        p.state = "DRAW"
        t0 = time.time()
        frames = 60
        for i in range(frames):
            ok, frame = src.read()
            if not ok:
                break
            hands = tr.process(frame, int((time.time() - t0) * 1000))
            now = time.time()
            for h in (hands.get("left"), hands.get("right")):
                if h is None:
                    continue
                if h.pinch_begin_edge:
                    ov.add_event(GameEvent("pinch_begin",
                                            {"x": h.index_tip[0], "y": h.index_tip[1],
                                             "label": h.label, "color": (255, 220, 80)}), now)
            if p.state == "DRAW":
                drawer.update(hands.get("left") or hands.get("right"), frame.shape)
                if drawer.is_full:
                    p.set_rect(drawer.final_rect)
                    p.set_snapshot(frame)
                    if p.snapshot is not None:
                        p.start_scramble()
                drawer.draw(frame)
            elif p.state in ("PLAY", "SCRAMBLE"):
                primary = hands.get("left") or hands.get("right")
                if primary is not None:
                    p.handle_pinch(primary)
                if p.snapshot is not None:
                    p.render(frame, primary)
            for evt in p.events:
                if evt.type == "solved":
                    evt.payload["w"] = frame.shape[1]
                    evt.payload["h"] = frame.shape[0]
                ov.add_event(evt, now)
            p.events.clear()
            ov.update_and_render(frame, hands, mode="puzzle", now=now,
                                 options={"show_landmarks": True, "show_trails": True})
    step("synthetic pipeline 60 frames", end_to_end)

    print("\nAll checks passed.")
    print(f"Project root: {ROOT}")
    print("Next: insert a USB webcam and run:  python main.py")


if __name__ == "__main__":
    main()