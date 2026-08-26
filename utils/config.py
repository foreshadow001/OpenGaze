"""yaml 配置加载与合并

配置沿「数据集 × 方法」两个正交轴组织：
    configs/datasets/<pipeline>/<name>.yaml   数据位置、划分方式、加载参数（按预处理管线分文件夹）
    configs/methods/<name>.yaml    模型结构 + 训练策略
运行时合并为一个 namespace 配置，支持 --set key=value 点路径覆盖。
"""
import os
from types import SimpleNamespace

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ROOT = os.path.join(PROJECT_ROOT, 'configs')


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def yaml_to_ns(obj):
    """dict → SimpleNamespace，支持 config.dataset.name 形式递归访问"""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: yaml_to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [yaml_to_ns(v) for v in obj]
    return obj


def ns_to_dict(obj):
    """SimpleNamespace → dict（配置快照落盘用）"""
    if isinstance(obj, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [ns_to_dict(v) for v in obj]
    return obj


def parse_value(raw):
    """--set 的值解析：先按 yaml 标量，失败或仍是字符串时尝试 float"""
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    if isinstance(value, str):
        # yaml 1.1 不识别 1e-4 这类科学计数，补一次 float 解析
        try:
            value = float(raw)
        except ValueError:
            pass
    return value


def apply_overrides(config, overrides):
    """把 ['key=value', ...] 列表应用到已合并的配置上"""
    for item in overrides:
        path, sep, raw = item.partition('=')
        if not sep or not path:
            raise ValueError(f'--set 格式应为 key=value，收到: {item}')
        set_by_path(config, path.strip(), parse_value(raw.strip()))


def set_by_path(config, path, value):
    """点路径赋值：set_by_path(config, 'method.train.epochs', 2)"""
    keys = path.split('.')
    obj = config
    for k in keys[:-1]:
        if not hasattr(obj, k):
            raise AttributeError(f'--set {path}: 配置中不存在 "{k}"')
        obj = getattr(obj, k)
    if not hasattr(obj, keys[-1]):
        raise AttributeError(f'--set {path}: 配置中不存在该字段')
    setattr(obj, keys[-1], value)


def load_config(dataset_name, method_name, overrides=None, config_root=CONFIG_ROOT):
    """加载并合并数据集与方法配置。

    dataset_name 支持子路径（配置按预处理管线分文件夹，如
    'zhang2015-insightface/xgaze' 对应 configs/datasets/zhang2015-insightface/xgaze.yaml）；
    配置内 name: 字段（loader 注册名）不含管线前缀。

    Returns:
        (config, dataset_yaml_path, method_yaml_path)
    """
    dataset_path = os.path.join(config_root, 'datasets', f'{dataset_name}.yaml')
    method_path = os.path.join(config_root, 'methods', f'{method_name}.yaml')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'数据集配置不存在: {dataset_path}')
    if not os.path.exists(method_path):
        raise FileNotFoundError(f'方法配置不存在: {method_path}')

    merged = {
        'experiment': f'{dataset_name}_{method_name}',
        # 来源配置名（区别于 dataset.name 注册名，冒烟配置如 xgaze_smoke 的 name 仍是 xgaze）
        'dataset_config': dataset_name,
        'method_config': method_name,
        'dataset': load_yaml(dataset_path),
        'method': load_yaml(method_path),
    }
    config = yaml_to_ns(merged)
    if overrides:
        apply_overrides(config, overrides)

    return config, dataset_path, method_path


def load_dataset_config(dataset_name, config_root=CONFIG_ROOT):
    """单独加载数据集配置（跨数据集测试时替换 dataset 段用）；
    dataset_name 同样支持 '管线/数据集' 子路径"""
    path = os.path.join(config_root, 'datasets', f'{dataset_name}.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f'数据集配置不存在: {path}')
    return yaml_to_ns(load_yaml(path)), path
