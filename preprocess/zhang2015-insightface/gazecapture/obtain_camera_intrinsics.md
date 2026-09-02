# GazeCapture 前置相机内参（定稿，生成脚本 generate_calibration.py）

## 构建方式

与 ETH-XGaze cam00.xml 同构，`generate_calibration.py` 生成 → `calibration/<slug>_<w>x<h>.xml`。
5 个自由度中 4 个由标准假设锁定：

| 参数 | 值 | 依据 |
|---|---|---|
| cx, cy | 帧中心 | 主点居中（xgaze 官方 6000×4000 标定实测主点恰为中心） |
| skew | 0 | 方形像素（xgaze 官方 fx/fy 差 0.06%） |
| 畸变 k1~k3, p1~p2 | 0 | 见下文论证 |
| **fx（=fy）** | **逐组查表** | 外部一手锚点（见下） |

### 畸变置 0 论证

手机前摄 k1 ≈ −0.05~−0.15，忽略后残余位移 1.5~5.8px（与关键点噪声同量级）。
误差被 PnP 的 tvec 吸收（6 点径向外扩近乎一致），rvec 仅二阶小量——归一化
管线敏感的只有 rvec。无法自标定（中心区小半径点上 k1·r³ 与 1/fx 共线），
设文献典型值不如 0（量级未知，设错更糟）。

### 帧分辨率（逐帧属性）

| 朝向 | 帧尺寸 | 主点 |
|---|---|---|
| 竖屏（1/2） | 480×640 | (240, 320) |
| 横屏（3/4） | 640×480 | (320, 240) |

同一传感器方形像素 → 两种帧 **fx 相同**，仅主点互换。每设备生成两份 xml。

## fx 定稿值与数据来源

三个硬件组，各组一个 fx（480×640 对角像素基准）：

| 组 | 设备 | fx | 来源 |
|---|---|---|---|
| group_vga | iPhone 4S, iPad 2/3/4 | **618** | Boinx 激光实测 HFOV 55° → `320/tan(27.5°)` |
| group_12mp_a | iPhone 5/5C/5S, iPad Mini/Air | **618** | 双路线互证：Boinx 622~625 + SystemPlus 像元 614~623（差 0.3%） |
| group_12mp_b | iPhone 6/6+/6s/6s+, iPad Air 2/Pro | **602**（6s 单独 560） | SystemPlus 像元：2.65mm / (2.2µm × 2 binning) |

### 溯源链接

1. **Boinx 实验室 iOS 前置 FOV 实测表**（2013，激光测距仪一手数据）：
   http://web.archive.org/web/20190105053437/https://boinx.com/chronicles/2013/3/22/field-of-view-fov-of-cameras-in-ios-devices/
   覆盖 iPhone 4S~5、iPad 2~mini 的前置 HFOV。

2. **SystemPlus iPhone 6 前置模组逆向报告通稿**（2015，硅级像元测量）：
   https://www.prnewswire.com/news-releases/reverse-costing-analysis-of-apples-iphone-6--6-plus-front-facing-camera-module-300038405.html
   给出像元从 1.75µm → 2.2µm（同时覆盖两代）。换算：`fx = 2.65 / (2.2e-3 × 2) = 602`。

3. **iPhone 6s/6s+ 补充实证**（无报告覆盖，用 bbox 宽度分布排除全幅缩放）：
   fx(6s) ∈ [536, 602]，取中值 560（写入 `front_cameras.yaml` 的 `fx_by_device`）。

### 已否决的途径

- **EXIF 直读**：iOS 不写 FocalLengthIn35mmFormat（安卓才写）
- **PnP 重投影自标定**：谷深仅 0.1~0.2px（噪声主导），同模组设备给出矛盾
  结果，且系统性偏低（畸变置零导致 fx 补偿性逃逸）。实验已删（calibrate_fx.py）。
