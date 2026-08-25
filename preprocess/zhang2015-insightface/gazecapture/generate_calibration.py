"""生成 GazeCapture 各设备相机内参文件（与 ETH-XGaze cam00.xml 同构）

每设备两个分辨率各一份（帧分辨率是逐帧属性，见 front_cameras.yaml）：
    <slug>_480x640.xml   竖屏帧（Orientation 1/2）：cx=240, cy=320
    <slug>_640x480.xml   横屏帧（Orientation 3/4）：cx=320, cy=240
fx 按设备所属组的 fx_by_group 取值；畸变 5 参数全 0（论证见
obtain_camera_intrinsics.md §2）。

用法（仓库根目录）：
    python preprocess/zhang2015-insightface/gazecapture/generate_calibration.py \
        [--output-dir /media/hitsz/ylx/GazeCapture/calibration]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
FRONT_CAMERAS = HERE / 'front_cameras.yaml'
DEFAULT_OUTPUT = Path('/media/hitsz/ylx/GazeCapture/calibration')


def slugify(device_name):
    return device_name.lower().replace(' ', '-')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    spec = yaml.safe_load(open(FRONT_CAMERAS))
    fx_by_group = spec['intrinsics']['fx_by_group']
    dist = np.zeros((1, 5))   # (1,5) 二维 -> OpenCV 写成 rows=1/cols=5 的 opencv-matrix，与 cam00.xml 完全同构

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    print(f'{"设备":<18}{"组":<14}{"fx":>5}  文件')
    for device, info in spec['devices'].items():
        fx = fx_by_group[info['group']]
        for w, h in [(480, 640), (640, 480)]:     # 竖屏帧 / 横屏帧，主点=半宽半高
            K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
            path = args.output_dir / f'{slugify(device)}_{w}x{h}.xml'
            fs = cv2.FileStorage(str(path), cv2.FileStorage_WRITE)
            fs.write('Camera_Matrix', K)
            fs.write('Distortion_Coefficients', dist)
            fs.release()
            n += 1
        print(f'{device:<18}{info["group"]:<14}{fx:>5}  {slugify(device)}_*.xml')

    # 自校验：重读全部文件，与预期矩阵逐项比对（节点存在性由读取成功保证，
    # 与 cam00.xml 的同构加载方式一致：FileStorage + getNode 两个同名节点）
    ok = 0
    for device, info in spec['devices'].items():
        fx = fx_by_group[info['group']]
        for w, h in [(480, 640), (640, 480)]:
            path = args.output_dir / f'{slugify(device)}_{w}x{h}.xml'
            fs = cv2.FileStorage(str(path), cv2.FileStorage_READ)
            K = fs.getNode('Camera_Matrix').mat()
            D = fs.getNode('Distortion_Coefficients').mat()
            fs.release()
            assert K is not None and D is not None, f'{path} 节点缺失'
            assert np.allclose(K, [[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]])
            assert D.shape == (1, 5) and not D.any()
            ok += 1
    print(f'\n生成 {n} 个文件到 {args.output_dir}，{ok} 个全部通过自校验'
          f'（节点可按 cam00.xml 同款方式加载、矩阵值正确）')


if __name__ == '__main__':
    main()
