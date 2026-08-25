# GazeCapture 前置相机内参获取（目标：与 ETH-XGaze cam00.xml 同构）

按步骤组织：目标格式（§1）→ 缺口盘点（§2）→ 原理（§3）→ fx 的三级获取
（§4 假设值 → §5 EXIF 锚点 → §6 PnP 校准）→ 生成内参文件（§7）。

---

## 第 1 步：目标格式

与 ETH-XGaze 官方标定文件（`example/input/cam00.xml`，OpenCV FileStorage 格式）
完全一致，含两个矩阵：

```
Camera_Matrix (3×3)                Distortion_Coefficients (1×5)
| fx   0   cx |                    [ k1  k2  p1  p2  k3 ]
|  0  fy   cy |
|  0   0    1 |
```

对应 OpenCV 读取方式与 `XGazePreprocessor._load_camera_calibration` 完全兼容：
`cv2.FileStorage` → `Camera_Matrix` / `Distortion_Coefficients` 两个节点。

## 第 2 步：缺口盘点——只缺 1 个数

| 目标量 | GazeCapture 现状 | 结论 |
| --- | --- | --- |
| fx（=fy） | 无标定 | **唯一实质缺口**，3 个硬件组各需一个值（或统一值） |
| cx, cy | 无标定 | 不缺：480×640 图像中心 (240, 320)。主点居中假设由 xgaze
|         |                | 实测支持（其 6000×4000 标定的主点 (3000,2000) 恰为中心） |
| skew s   | 无标定 | 不缺：0（xgaze 实测亦为 0，方形像素 fx≈fy 差 0.06%） |
| k1~k3, p1, p2 | 无采集 | 不缺数据、缺决策：**全部置 0**，量化论证见下 |

**畸变置 0 的量化论证**：
- 手机前摄广角定焦典型 k1 ≈ -0.05~-0.15；自拍人脸 6 个 PnP 点位于画面中心
  r ≈ 0.2~0.4，忽略畸变的残余位移 δ = |k1|·r³·fx ≈ **1.5~5.8px**（与关键点
  噪声 ~1.5px 同量级或略大）
- 但误差去向良性：6 点径向外扩近乎一致 → 被 PnP 的 tvec（距离）吸收，
  rvec（头姿）仅二阶小量，而归一化管线敏感的只有 rvec
- 不可自标定：中心区小半径点上 k1·r³ 与 1/fx 共线（§6 谷底漂移的病因
  之一），可观性需要 r>0.7 的画面边缘点，人脸点没有
- 不设文献典型值：各组真实 k1 未知，量级设错比 0 引入更难预测的误差模式
- 影响面：within-dataset 几乎无影响（同组帧同一畸变模式，训练映射一致）；
  cross-dataset 有 <1° 量级系统小偏，与 fx 残余误差同量级，暂接受

## 第 3 步：原理——为什么 fx 是唯一需要"获取"的量

完整针孔内参矩阵 5 个自由度（fx, fy, s, cx, cy）中，4 个已由标准假设锁定
（§2），只剩焦距标量。而**等效焦距 f₃₅（35mm equivalent）本身编码了传感器
尺寸**：f₃₅ = f_real × crop，crop = 43.267mm / 传感器对角线——因此无需单独
知道传感器尺寸，f₃₅ 即可唯一确定 FOV 与像素焦距：

```
f_px = 800 × f₃₅ / 43.267 ≈ 18.49 × f₃₅        [800 = √(480²+640²) 对角像素]
```

**适用前提（已论证）**：GazeCapture 视频帧为 4:3，与前摄传感器原生 4:3 一致
且为全幅下采样（App 用 640×480 preset，非 720p 的 16:9 裁切）——拍照模式的
等效焦距可直接用于视频帧。

**帧分辨率是逐帧属性（2026-08-24 全量核验）**：1409/1474 个 session 内混合
两种分辨率，且与朝向逐帧对应（3 session / 151 抽样帧无一例外）：

| Orientation | 握持 | 帧尺寸（宽×高） | 说明 |
| --- | --- | --- | --- |
| 1 / 2 | 竖握 | 480×640 | app 将传感器横帧旋转 90° 输出 |
| 3 / 4 | 横握 | 640×480 | 传感器原生方向直出 |

同一传感器方形像素 → 两种帧 **fx 相同**，仅主点互换（竖屏 cx=240,cy=320；
横屏 cx=320,cy=240）。预处理实现须逐帧按实际尺寸构造 K；Boinx 换算用
`fx = 320/tan(HFOV/2)`（横构图 640 长边）对两种帧同时成立。

## 第 4 步：fx 初值——假设值（零成本，先跑通管线）

按学术惯例（Zhang et al. 2018 归一化方案：缺真实内参时用名义焦距）取对角
FOV 假设，初值已写入 front_cameras.yaml 的 intrinsics_480x640 段：

| 组 | FOV 假设 | fx 初值 | 对应 f₃₅ |
| --- | --- | --- | --- |
| group_vga（iPhone 4S, iPad 2/3/4） | 55° | 769 | 41.6mm |
| group_12mp_a（iPhone 5/5C/5S, iPad Mini/Air） | 58° | 722 | 39.1mm |
| group_12mp_b（iPhone 6/6+/6s/6s+, iPad Air 2/Pro） | 63° | 654 | 35.4mm |
| 统一兜底 | 60° | 693 | 37.5mm |

## 第 5 步：外部资料锚点（已逐源核验，2026-08-24）

**途径 1（自拍照 EXIF 直读等效焦距）——已核查降级**：iOS 相机 app 的 EXIF
只写实际焦距（FocalLength/2.65mm 等），**不写** FocalLengthIn35mmFormat
（该字段安卓常写、iOS 不写）。

**途径 2（外部 FOV 实测表）——主锚点，已核验一手来源**：

- **Boinx 实验室实测表（2013-03，激光测距仪实测，一手数据）**
  [Wayback 存档](http://web.archive.org/web/20190105053437/https://boinx.com/chronicles/2013/3/22/field-of-view-fov-of-cameras-in-ios-devices/)，
  即 [SO 求助帖](https://stackoverflow.com/questions/49260945/iphone-ipad-front-camera-fov)
  所引 "measured results" 的原始出处（该帖经官方 API 核验为求助帖、无数据；
  帖内另一引用 caramba 表经核验**仅后置**、仅到 2012，不可用）：

  | 设备（前置） | Boinx 实测 HFOV | fx = 320/tan(HFOV/2) |
  | --- | --- | --- |
  | iPhone 4S | 55.0° | 615 |
  | iPhone 5 | 54.2° | 625 |
  | iPad 2 / 3 / 4 / mini | 54.4~54.8° | 618~622 |

  **换算注意**：Boinx 的水平 FOV 为横构图（沿传感器 640 长边）；方形像素下
  `fx = 320/tan(HFOV/2)`（等价于用 480 方向的更窄角算 `240/tan`，结果相同）。
  覆盖范围：iPhone 5 及更早 / iPad 4 及更早（无 5C/5S/6/6s、iPad Air 系）。

- Reddit "iPhone 6 前置 ~31mm 等效"（403 未核验原文，二手换算）→ fx ≈ 573，
  作为 group_12mp_b 的弱参考。

已核验**不可用**的来源：pixelcraft 博客（仅 iPhone 14 后置）、caramba 表
（仅后置）、搜索引擎综合转述（无主）。

**途径 3（物理焦距 × 传感器尺寸）**：f₃₅ = f_real × 43.267/d_sensor，
传感器尺寸查拆解报告（口径杂，仅旁证）。

**收敛结论**（2026-08-24 方案 B 成功后，三组 fx 全部一手闭环）：

| 组 | 一手锚点 | fx |
| --- | --- | --- |
| group_vga（iPhone 4S, iPad 2/3/4） | Boinx 激光实测 615~622 | **618** |
| group_12mp_a（iPhone 5/5C/5S, iPad Mini/Air） | **双路线互证**：SystemPlus 像元（Sony 1.75µm + EXIF 焦距）→ 614~623；Boinx 实测 → 622~625（差 0.3%） | **618** |
| group_12mp_b（iPhone 6/6+, iPad Air 2/Pro） | SystemPlus 像元（Sony 2.2µm + EXIF 2.65mm + 2×2 binning）→ 602，对角 FOV 67.2°、等效 32.6mm | **602** |

group_12mp_b 的求值过程（方案 B，传感器规格路线）：
[SystemPlus 逆向报告通稿](https://www.prnewswire.com/news-releases/reverse-costing-analysis-of-apples-iphone-6--6-plus-front-facing-camera-module-300038405.html)
（2015）给出 iPhone 6 前置 Sony Exmor-RS **像元从上代 1.75µm 提升到 2.2µm**——
一句话同时提供两代一手像元。换算链：

```
fx = f_real / dx_eff = 2.65mm / (2.2µm × 2) = 602 px    [2× 为 1280×960 → 640×480 binning]
```

Reddit 二手值（573）被一手否定（差 5%），弃用。

**6s/6s+（5MP 前置，报告未覆盖）的补充实证**（2026-08-24，bbox 宽度分布，
每设备 25 session）：

| 设备 | bbox 宽中位数 | 相对 iPhone 6 |
| --- | --- | --- |
| iPhone 6 | 294.5 | 1.000 |
| iPhone 6 Plus（与 6 同模组） | 263.5 | 1.117 |
| iPhone 6s | 262.0 | 1.124 |
| iPad Air 2 | 228.2 | 1.290 |

- 排除全幅缩放路径：若 6s 为 fx≈437，bbox 应小 38%（≈213px），实测 262px
- 揭示拍摄距离混杂：同模组的 6 Plus 也小 11.7%，且随屏幕尺寸单调（大屏
  持机更远）——bbox 比不能直接当 fx 比
- 结论：fx(6s) ∈ [536, 602]（下界=距离无差异假设，上界=全为距离混杂），
  **取设备级 fx = 560**（区间中值），写入 front_cameras.yaml 的 fx_by_device

## 第 6 步：PnP 重投影校准——已实验，判别力不足（2026-08-24）

原理：insightface 关键点与 face model 的 3D-2D 对应是天然的标定物——扫描
fx 取重投影误差最小值。实现见 calibrate_fx.py（按设备扫描，已跑）。

**实验结果（负结果，如实记录）**：

| 设备 | 扫描结果 | 判读 |
| --- | --- | --- |
| iPhone 6 | 谷底 550（曲线完整） | 与 SystemPlus 一手 602 差 9% |
| iPhone 6s | 谷底 530~540（平底谷） | 无外部锚点可对照 |
| iPhone 6 Plus | 450→390 谷底随扫描范围持续下移 | 边界伪影，未收敛 |
| iPad Air 2 | 450→420 同样下移 | 同上 |

**失败原因分析**：
1. 判别力不足——谷深仅 0.1~0.2px，而 RMS 本身 1.1~1.5px（关键点噪声主导）
2. iPhone 6 与 6 Plus 前置模组相同却给出 550 vs <390，结果自相矛盾
3. 系统性偏低——前置广角镜头的桶形畸变被置零（dist=0），PnP 以更小 fx
   补偿边缘外扩，谷底持续向更广视场逃逸

**定稿结论**：fx 以 §5 外部一手锚点为准（602 / 618 / 618，SystemPlus 硅级
像元测量）；本校准降级为管线内 sanity check（实现时监控重投影 RMS 在
~1.5px 量级即健康），不作为 fx 来源。若将来需要数据自标定，须升级为
`cv2.calibrateCamera` 联合估计 fx/cx/cy/k1k2 且每设备 500+ 帧（人脸形状
模型误差与内参强耦合，可观性存疑，暂不做）。

## 第 7 步：生成内参文件（纯机械）

fx 定稿后，按组生成与 cam00.xml 同构的文件（3 组 3 个文件，或直接在管线内
构造矩阵，二选一）。注意帧分辨率是逐帧属性（§3）：**竖屏 480×640 与
横屏 640×480 两种内参，同一 fx、主点互换**：

```python
import cv2, numpy as np

def build_K(fx, w):          # w = 帧宽（480 竖屏 / 640 横屏），主点=半宽半高
    cx, cy = w / 2, (640 if w == 480 else 480) / 2
    return np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)

# 管线内逐帧：K = build_K(fx_by_group[group], img.shape[1])
# 落盘为 cam00.xml 同构文件（每组一个，两种分辨率各一套或运行时构造）：
dist = np.zeros(5)
for w in (480, 640):
    fs = cv2.FileStorage(f'group_12mp_b_{w}x{640 if w==480 else 480}.xml',
                         cv2.FileStorage_WRITE)
    fs.write('Camera_Matrix', build_K(596, w))
    fs.write('Distortion_Coefficients', dist)
    fs.release()
```

## 附：等效焦距换算速查表（480×640）

| f₃₅ | 对角 FOV | f_px |
| --- | --- | --- |
| 28mm | 75.4° | 518 |
| 31mm | 69.8° | 573 |
| 33mm | 66.5° | 610 |
| 35mm | 63.3° | 647 |
| 38mm | 59.0° | 703 |
| 40mm | 56.4° | 740 |
| 43mm | 53.1° | 795 |
