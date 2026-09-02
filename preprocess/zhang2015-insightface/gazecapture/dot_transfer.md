# GazeCapture dot → 相机系 gaze point 最终定稿（唯一实现见 preprocessor.py）

`_dot_to_ccs_mm(ori, xcam, ycam)` + `_gaze_point_cam(ori, ccs_x, ccs_y)`，
两步链路，所有预处理管线（v1/v2 + 可视化脚本）统一调用这两个函数。

## 官方标注

`dotInfo.json` 的 `XCam/YCam`：dot 在**当前朝向逻辑屏坐标系**中的位置（cm）。
坐标轴随设备朝向旋转（绑定逻辑屏而非传感器物理方向），四种朝向符号模式不同。

## 第一步：dot → CCS（mm）

统一到前摄传感器固定坐标系（不随朝向旋转），单位 mm：

| Ori | 朝向 | CCS_x | CCS_y |
|---|---|---|---|
| 1 | Portrait | `-xcam × 10` | `-ycam × 10` |
| 2 | PortraitUpsideDown | `+xcam × 10` | `+ycam × 10` |
| 3 | LandscapeRight | `-ycam × 10` | `+xcam × 10` |
| 4 | LandscapeLeft | `+ycam × 10` | `-xcam × 10` |

验证不变量：`ccs_y > 0`（屏幕从相机向 home 方向延伸，全正）；`ccs_x` 零均值对称。
违例帧（`ccs_y <= 0`，~0.04%，朝向过渡噪声）→ 预处理记 `invalid_dot` 跳过。

## 第二步：CCS → 相机系 gaze point（mm）

与该帧 PnP 同一坐标系（= 前摄传感器系），供 `normalizeData_face` 直接使用：

```python
if ori in (1, 2):    # 竖屏
    p = (ccs_x, ccs_y, 0.0)
else:                # 横屏
    p = (-ccs_y, ccs_x, 0.0)
if ori in (2, 4):    # 180° 旋转存储帧
    p = (-p[0], -p[1], 0.0)
```

## 修正历史

- **2026-08-27**：补 ori 2/4 的 180° 旋转翻转（此前遗漏，导致 ~36% 标签
  pitch/yaw 双取反）
- **2026-08-28**：修正 ori 1/2 的 x 方向（原 `(-ccs_x, ...)` 取反错误，
  导致全量 yaw 取反）；横屏 x 方向同步修正

推导依据：300 session 逐帧抽样 9485 帧的符号模式实证（各朝向 XCam/YCam
范围与符号规律，已删——结论固化于上表）；前摄为非镜像摄像头（逻辑屏 x
与传感器 x 同向）。

已排除的疑点（勿回头重查）：
- 存储帧一律是界面视角（人脸全部正立），"ori2 脸倒置/横屏是传感器原生"不成立
- 00028 的 ori3 中位角 180° 是 5 帧朝向过渡噪声，非系统 bug
- 内参主点全部居中（30 个 xml 零偏移），非残差来源
- 帧尺寸与朝向号不严格对应（如 iPad Air 2 的 ori4 帧实为 480×640），
  内参必须按实际帧尺寸查表
