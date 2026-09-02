# preprocess/zhang2015-insightface/viz/ 目录结构

参照 v2（`zhang2015-specific-face-model/viz/`）组织：一个功能目录 = 脚本 + 产物。

```
viz/
├── README.md                                # 本文件
├── all_datasets_normalized.png             # 20 样本归一化图
├── cross_dataset_analysis.md                # 跨数据集分析文档
├── gaze_distribution/                       # 四数据集 gaze/head 分布图（turbo 热力图）
│   ├── gaze_distribution.py                 # 脚本
│   ├── gaze_distribution_insightface_224.png
│   └── gaze_distribution_stats.csv
└── error_distribution/                      # 训练误差 vs 角度分析
    ├── error_by_angle_array.png             # 旧版（v1 产物，脚本已遗失）
    └── gc_fixed_gaze_dist_audit_stats.csv   # GC 修复审计统计（旧产物归档）
```

## error_distribution/（新建）

用于训练误差 vs 头姿/视线角度的分析热力图，风格与 v2 的
`viz/distribution/gaze_distribution_specific.png` 一致（turbo 色带、深蓝底、
白色 30°/15° 网格、独立对数色柱、±120° 显示范围）。脚本待实现
（`error_by_angle.py`）：读 exp 的 test_result + 预处理产物，按头姿/视线
角度 bin 画误差热力图。
