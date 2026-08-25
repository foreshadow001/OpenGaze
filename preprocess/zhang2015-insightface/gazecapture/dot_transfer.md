# GazeCapture dot 坐标（XCam/YCam）→ 统一相机坐标系（CCS）转换

## 1. 官方定义（论文 Eye Tracking for Everyone, CVPR 2016）

> ... we predict the dot location relative to the camera (in centimeters in the
> x and y direction). We obtain this through precise measurements of device
> screen sizes and camera placement.

`dotInfo.json` 的 `XCam/YCam`：注视点（dot）在前摄相机坐标系中的位置，**单位 cm**，
z 平面为屏幕平面（dot 在相机所在平面上，z ≈ 0）。

**关键事实（实证确认）**：官方 XCam/YCam 的坐标轴**随设备朝向旋转**——绑定的是
当前 interface orientation 的逻辑屏幕轴（原点在相机、轴沿当前逻辑屏的宽高方向），
而非绑定相机传感器的物理方向。因此四种朝向下 dot 的符号模式不同（见 §2），
直接混用四种朝向的数据会导致坐标系不一致。

**与 CCS 的 x 方向关系（实测）**：Portrait 朝向下，设备物理**右侧**的 dot 其
XCam 为正——与 CCS 的 +x（从右指向左）**相反**。

## 2. 实证统计（2026-08-24，300 session 逐帧抽样 9485 帧）

`screen.json` 的 `Orientation` 为 iOS interface orientation 编码：

| Ori | 朝向（设备旋转） | 相机物理位置 | XPts/YPts 范围 | XCam (cm) | YCam (cm) | 符号模式 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Portrait | 顶部 | [40,728] / [40,984] | [-6.89, +6.88] | [-25.11, -1.43] | x 对称，**y 全负** |
| 2 | PortraitUpsideDown（倒立） | 底部 | [40,728] / [40,984] | [-6.88, +6.89] | [-4.58, +20.25] | x 对称，**y 全正** |
| 3 | LandscapeRight（home 键在右，逆时针 90°） | 左侧 | [40,984] / [40,728] | [+1.43, +26.37] | [-6.89, +6.88] | **x 全正**，y 对称 |
| 4 | LandscapeLeft（home 键在左，顺时针 90°） | 右侧 | [40,984] / [40,728] | [-20.25, +4.58] | [-6.88, +6.88] | **x 全负**，y 对称 |

规律：dot 永远出现在相机的一侧（屏幕从相机向某一方向延伸）——
- 主延伸轴（长边方向）：Ori 1/2 为 YCam，Ori 3/4 为 XCam，符号由相机物理位置决定；
- 对称轴（短边方向，dot 遍历全屏宽）：区间 ≈ ±半屏宽 cm，四种朝向一致。
- XPts/YPts 同样随朝向交换（[40,984] 在竖屏时出现在 YPts、横屏时出现在 XPts），
  证实坐标轴绑定逻辑屏幕方向。

## 3. 统一相机坐标系（CCS）定义

以**前摄传感器**为基准的固定相机坐标系（不随设备朝向旋转），与通用相机坐标系
（OpenCV 约定）对齐的右手系：

- 原点：前摄中心（在屏幕平面上，位于设备 portrait 姿态的顶部边框）
- +x：沿设备物理短边，**portrait 姿态下从右指向左**（设备物理左侧为 +x，
  即前摄相机自身视角的右方向）
- +y：沿设备物理长边，**从摄像头指向 home 键方向**（即前摄相机自身视角的下方）
- z：右手法则自然导出——沿相机光轴指向被摄者（出屏朝用户）
- **单位：m**

该系下任何朝向的 dot 都满足：**y > 0 全正**（屏幕永远从相机向 home 方向延伸），
**x 对称有正有负**（dot 遍历全屏宽）。

## 4. 各朝向转换公式（待审核）

官方 XCam/YCam 单位为 cm，CCS 单位为 m，转换含 `/100`。

| Orientation | 朝向 | CCS_x (m) | CCS_y (m) | 推导 |
| --- | --- | --- | --- | --- |
| 1 | Portrait | `-XCam/100` | `-YCam/100` | x：官方与 CCS 的 +x 相反（§1 实测），取负；y：dot 在相机 home 侧（官方数据 YCam 全负），与 CCS +y 同向，取负变正 |
| 2 | PortraitUpsideDown | `+XCam/100` | `+YCam/100` | 倒立 180°：官方轴 = Ori 1 轴整体旋转 180°，物理点坐标在两系中互为相反数，叠加 Ori 1 的符号修正 |
| 3 | LandscapeRight | `-YCam/100` | `+XCam/100` | 逆时针 90°：官方 x 轴（数据全正，沿物理长边）转到 CCS +y 后全正；官方 y 轴（沿物理短边）转到 CCS -x |
| 4 | LandscapeLeft | `+YCam/100` | `-XCam/100` | 顺时针 90°：官方 x 轴（数据全负，沿物理长边）转到 CCS -y 后仍全正；官方 y 轴转到 CCS +x |

验证条件（转换后应满足）：

```python
assert (ccs_y > 0).all()                       # y 全正（m）
assert abs(ccs_x.mean()) < 0.5 * ccs_x.std()   # x 近似零均值对称
```

已知数据噪声：约 0.04%（4/9485 抽样帧）违例 `ccs_y <= 0`，均为 session 00258
的**朝向切换过渡帧**（Orientation 字段与 dot 坐标更新不同步，DotNum 有效）。
预处理时对 `ccs_y <= 0` 的帧记 `invalid_dot` 并跳过。

## 5. 与视线标签（TODO-2）的衔接

CCS 只解决 dot 位置的统一（单位 m）；3D 视线向量还需：

```
gaze_vec = (CCS_x, CCS_y, 0) - eye_center_ccs
```

`eye_center_ccs` 由 insightface 关键点 + PnP 头姿 + face model 眼点推出
（注意单位对齐：face model 的尺度决定 tvec 单位，需换算到 m 与 CCS 同尺度）。
之后送入 normalizeData_face 归一化 → (theta, phi)。
