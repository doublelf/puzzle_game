# Pinch-Puzzle (拼图小游戏)

基于 MediaPipe Tasks Hand Landmarker + OpenCV 的单手捏合 3×3 拼图游戏。

## 安装

```bash
pip install -r requirements.txt
```

> MediaPipe Tasks 版模型 `hand_landmarker.task`（约 8 MB）已包含在仓库根目录。
> 若运行时提示找不到模型，请手动从
> https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
> 下载并放到本目录。

## 启动

```bash
python main.py                  # 默认摄像头 0
python main.py --cam 1          # 使用第 2 个摄像头
python main.py --no-flip        # 关闭水平镜像
python main.py --no-landmarks   # 关闭关键点叠加层（默认开启）
python main.py --hide-trails    # 关闭手势拖尾（默认开启）
```

按 `M` 返回菜单、`R` 重置当前拼图、`ESC` 退出。

## 玩法

1. 单手用拇指 + 食指**捏合**，食指按住拖动形成第一个矩形
2. 松开；再次捏合拖动形成第二个矩形
3. 程序对两个矩形取**并集**，按 3×3 切片打乱生成拼图
4. 玩拼图：
   - 捏合时，自动吸附**最近**的拼图块
   - 沿 4 邻接方向拖动指尖至**空白格**附近，块会自动与空白交换
   - 松开即锁定一次操作
5. 9 块全部归位 → 胜利

任何时候按 `R` 可重置当前拼图。

## 视觉反馈（默认全部开启）

| 触发 | 反馈效果 |
|---|---|
| 检测到手部 | 21 个关键点 + 骨架线，拇指/食指指尖金色圆点 + 白光环 |
| 顶部 HUD 徽章 | 左上角实时显示 `L ● PINCH` / `R ○ ready` 状态 |
| 捏合**开始** | 食指指尖扩散环动画（金色，0.35s） |
| 捏合**持续** | 食指指尖呼吸光晕（正弦脉动） |
| 捏合**结束** | 食指指尖淡灰扩散环 |
| 任意手指移动 | 拖尾轨迹（每只手最多 12 段，渐变衰减） |
| 拼图**抓取** | 选中块金色脉冲边框 + toast `Tile #N selected` |
| 拼图**交换** | 两块同时绿色闪光 + toast `SWAP!` |
| 拼图**拖错** | 邻接块红色闪光（仅当拖向非空白格） |
| 拼图**解出** | 80 颗彩虹粒子从中心爆裂 + 大字 toast `SOLVED!` |

## 文件结构

```
puzzle_game/
├── main.py            # 入口 / 菜单 / 主循环
├── gesture.py         # MediaPipe Tasks 双手追踪 + 捏合判定（含 pinch 边沿事件）
├── drawing.py         # 单手捏合绘制矩形
├── puzzle_game.py     # 3×3 拼图状态机（含事件流）
├── overlay.py         # 反馈层：拖尾 / 扩散环 / 呼吸光晕 / 闪光 / 粒子 / Toast
├── utils.py           # 几何与文本绘制工具
├── hand_landmarker.task
├── tools/             # 离线验证工具（可选）
│   ├── verify_all.py  # 端到端验证套件
│   ├── synthetic_source.py  # 无摄像头合成手源
│   ├── fake_tracker.py      # 合成 HandState 提供者
│   └── run_synthetic.py     # 无摄像头跑完整游戏循环
├── requirements.txt
└── README.md
```

## 无摄像头测试

```bash
python3 -m tools.verify_all              # 8 步端到端验证
python3 -m tools.run_synthetic --frames 120 --out /tmp/puzzle_frames
```

## 调参

- `gesture.py` 中 `pinch_threshold`（默认 0.32）控制捏合灵敏度，越小越严格
- `gesture.py` 中 `smoothing`（默认 0.55）控制坐标抖动平滑

## 已知局限

- 依赖单一摄像头视野
- 拼图打乱使用"反向随机走 N 步"保证可解性