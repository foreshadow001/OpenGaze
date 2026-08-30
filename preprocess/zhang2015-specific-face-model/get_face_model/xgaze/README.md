# XGaze 人脸建模（第二版预处理重跑）

## 人脸模型标准坐标系（解剖轴定义，2026-08-29 定稿）

**零位标准**：

1. **roll = 0**：两眼中心连线 ∥ x 轴
2. **yaw = 0**：眼心—鼻心连线 ⊥ x 轴（无 x 分量）
3. **pitch = 0**：眼心—鼻心连线 ∥ y 轴

**构造**（唯一实现 [utils/normalization.py](../../../../utils/normalization.py) 的
`canonicalize_face_model()`，返回 `(P6_std, R, origin)`；各脚本一律导入）：

```
eye_L = (eye_out_L + eye_in_L)/2      eye_R = (eye_out_R + eye_in_R)/2
eye_c = (eye_L + eye_R)/2             nose_c = (nose_L + nose_R)/2
x̂ = normalize(eye_R − eye_L)                    # 眼心连线
ŷ = normalize((nose_c − eye_c) − 投影到 x̂ 的分量)   # 眼→鼻去 x 分量
ẑ = x̂ × ŷ                                       # 右手系，指向头内（离开观察者）
原点 = eye_c；P_canonical = [x̂; ŷ; ẑ] @ (P − eye_c)
```

参考数字：gen6 自身坐标系相对该标准偏 **13.46°**（x 轴 2.6°/y 轴 13.5°/z 轴 13.2°，
几何来源：gen6 眼角 z≈29~37、鼻点 z≈21.7，眼→鼻线带 ~10mm 的 −z 分量）。

**模型标准化三步管线**（所有个性化/通用模型交付前必经，防坐标系漂移）：

1. **粗对齐**：Kabsch 刚体对齐到 gen6（无缩放）——只消头运动/初始朝向，
   不改变模型几何与尺度；
2. **欧拉角归零**：`canonicalize_face_model()` 把三个欧拉角归到上述解剖零位；
3. **中心化**：原点平移到眼心 eye_c（已并入第 2 步输出：P_std = R @ (P − eye_c)）。

交付产物中 `true6.txt` = 第 1 步结果（gen6 对齐系），`true6_canonical.txt` = 第 2+3 步
结果（标准系）；下游归一化统一用标准系版本。

## 每人真实 6 点模型（严格三角化）

**数据用量**：每人全部帧（200~611，中位 560）× 18 台相机（每相机每帧 1 张图）
= 3532~10993 张（中位 10053）。

**方法**（全部官方数据，无任何自标/替代——见 CLAUDE.md 约定 0）：

1. 每帧：18 相机 insightface 6 点（翻转/正立坐标系 h5）→ 官方 K/dist 去畸变
   → 官方外参 P=[R|t] 逐点 DLT → 该帧 6 点世界坐标；
2. 逐帧 Kabsch（刚体无缩放）对齐到 gen6（仅消头运动）→ 帧间中位 = true6；
3. `canonicalize()` → 标准系模型 true6_canonical。

**输出**（`/media/yanglinxuan/sfm/xgaze_specific_face_model/face_models/`）：

- `<subject>/true6.txt` —— gen6 对齐系（6,3，mm）
- `<subject>/true6_canonical.txt` —— 标准系（6,3，mm）
- `canonical_mean6.txt` —— 80 人标准系均值（标准模型）

**统计**（真实 vs gen6）：IOD 86.1±3.8 vs 91.3 ｜ 鼻宽 26.7±1.5 vs 21.9 ｜
眼心鼻心 44.8 vs 48.7（mm；逐人明细可由 gen6_vs_dlt.py 重算）。

## 相关实验与可视化

- [metrics/eye_nose_features/](metrics/eye_nose_features/)：模型几何特征——
  `gen6_vs_dlt.py`（gen6 vs DLT 真值 6 指标逐人对比 → 本目录 `gen6_vs_dlt.png`，
  汇总进日志；实证 gen6 鼻宽 21.9 vs 真值 26.7 的缺陷）
- [metrics/frame_consistency/](metrics/frame_consistency/)：跨相机一致性证据——
  - `frame_consistency.py`：Gen6+PnP vs 严格三角化全对口径
    （3D 位置 53.60 vs 4.99 mm；产物 `aggregate.md` / `per_camera.csv`）
  - `pos_hcs_consistency.py`：双臂 + 固定参考 cam00（所有指标 = vs cam00），
    80 被试 × 120 帧 → `consistency_overall.csv` / `consistency_per_camera.csv`
    （末行 AVG）：

    | 臂 | 3D 位置 (mm) | HCS 头姿 (°) |
    |---|---|---|
    | gen6（逐相机 PnP，v1 形态） | 35.6 | 7.2 |
    | true6（5+5 组独立 DLT→Kabsch，v2 形态） | 5.0 | 0.37 |
    | true6_full（10 台一组共享头姿，v2 部署形态） | — | 0.00（机器精度） |
- [viz/viz_canonical_models.py](viz/viz_canonical_models.py)：标准系模型 3D 交互
  可视化（`viz/canonical_standard_3d.html`，浏览器打开；标准模型 + 标准化 gen6
  + 解剖轴骨架 + gen6 坐标系偏差报告）
