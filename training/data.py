"""按清洗主表读取灰度图；训练只使用原有划分，不重新切分。"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

FAMILIES = ['Ramnit', 'Lollipop', 'Kelihos_ver3', 'Vundo', 'Simda',
            'Tracur', 'Kelihos_ver1', 'Obfuscator.ACY', 'Gatak']
PREPROCESS = 'full_bytes_1024_bins_known_mean_round_half_up_v1'


def image_tensor(path):
    with Image.open(path) as image:
        if image.size != (32, 32) or image.mode != 'L':
            raise ValueError(f'需要32x32单通道灰度图: {path}')
        pixels = np.array(image, dtype=np.uint8)
    return torch.from_numpy(pixels.copy()).float().unsqueeze(0) / 255


def read_manifest(root):
    root = Path(root).resolve()
    frame = pd.read_csv(root / 'dataset.csv', dtype={'Id': str})
    if not {'Id', 'Class', 'split', 'image'}.issubset(frame):
        raise ValueError('dataset.csv缺少Id/Class/split/image')
    if frame[['Id', 'Class', 'split', 'image']].isna().any().any():
        raise ValueError('主表含缺失值')
    if frame.Id.duplicated().any() or frame.Id.str.strip().eq('').any():
        raise ValueError('样本Id重复或为空')
    if not frame.Class.isin(range(1, 10)).all():
        raise ValueError('Class必须为1至9，内部转换为0至8')
    if set(frame.split) != {'train', 'val', 'test'}:
        raise ValueError('需要非空train/val/test划分')
    if set(frame.loc[frame.split.eq('train'), 'Class']) != set(range(1, 10)):
        raise ValueError('训练集缺少类别')
    frame = frame.sort_values('Id').reset_index(drop=True)
    hashes, paths = [], []
    for row in frame.itertuples():
        path = (root / row.image.replace('\\', '/')).resolve()
        if not path.is_relative_to(root):
            raise ValueError('图像路径超出数据目录')
        tensor = image_tensor(path)
        hashes.append(hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
        paths.append(path)
    frame['pixel_sha256'] = hashes
    frame['path'] = paths
    if frame.groupby('pixel_sha256').split.nunique().gt(1).any():
        raise ValueError('相同图像跨数据划分，需先检查')
    if frame.groupby('pixel_sha256').Class.nunique().gt(1).any():
        raise ValueError('相同图像具有冲突标签')
    return frame


class ImageDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return image_tensor(row.path), int(row.Class) - 1


def bytes_tensor(path):
    """与原make_images.aggregate一致：全序列1024等分，已知字节均值。
    不执行文件。未知字节保留位置但不参加均值；无有效字节则拒绝。
    """
    import re
    values, known = bytearray(), bytearray()
    with Path(path).open('rb') as handle:
        for line in handle:
            tokens = line.split()
            if tokens and re.fullmatch(rb'[0-9A-Fa-f]{8}', tokens[0]):
                tokens = tokens[1:]
            for token in tokens:
                if token == b'??':
                    values.append(0); known.append(0)
                elif re.fullmatch(rb'[0-9A-Fa-f]{2}', token):
                    values.append(int(token, 16)); known.append(1)
                else:
                    raise ValueError('无效token，需要.bytes文本，不能直接输入exe')
    if not any(known):
        raise ValueError('没有有效字节')
    values = np.frombuffer(values, dtype=np.uint8).astype(np.int64)
    known = np.frombuffer(known, dtype=np.uint8).astype(np.int64)
    edges = np.arange(1025, dtype=np.int64) * len(values) // 1024
    counts = np.diff(np.r_[0, np.cumsum(known)][edges])
    sums = np.diff(np.r_[0, np.cumsum(values * known)][edges])
    pixels = np.zeros(1024, dtype=np.uint8)
    mask = counts > 0
    pixels[mask] = (sums[mask] + counts[mask] // 2) // counts[mask]
    return torch.from_numpy(pixels.reshape(1, 32, 32)).float() / 255
