import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EMO_COLS = [
    "Admiration",
    "Amusement",
    "Determination",
    "Empathic Pain",
    "Excitement",
    "Joy",
]

def norm_id(sample_id, width: int = 5) -> str:
    s = str(sample_id).strip()
    if s.isdigit():
        return s.zfill(width)
    try:
        return str(int(float(s))).zfill(width)
    except Exception:
        return s

class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def load_pkl_array(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
        except Exception:
            f.seek(0)
            obj = CompatUnpickler(f).load()
    if isinstance(obj, np.ndarray):
        arr = obj
    elif hasattr(obj, "detach"):
        arr = obj.detach().cpu().numpy()
    else:
        arr = np.asarray(obj)
    # 处理 NaN/Inf，确保下游数值稳定
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # 强制 2D: [T, D]
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 0:
        arr = arr[None, None]
    return arr

def tsn_sample(seq: np.ndarray, k: int, is_train: bool) -> np.ndarray:
    """
    Temporal Segment Network (TSN) 风格采样。
    seq: [T, D] 的输入特征。
    k: 要采样的固定帧数。
    is_train: 训练时段内随机采样，验证/测试时取段中点。
    返回: [K, D] 的特征。
    """
    T, D = seq.shape
    if T == 0:
        return np.zeros((k, D), dtype=np.float32)
        
    # 将时间轴均分为 K 个区间（首尾索引）
    indices = np.linspace(0, T, k + 1, dtype=int)
    sampled_indices = []
    
    for i in range(k):
        start = indices[i]
        end = indices[i + 1]
        
        # 当 T < K 时，某些区间的 start == end，防止报错直接取 start 并限制在合法范围内
        if start >= end:
            idx = min(start, T - 1)
        else:
            if is_train:
                idx = random.randint(start, end - 1)
            else:
                idx = (start + end - 1) // 2
        sampled_indices.append(idx)
        
    return seq[sampled_indices]

class EMIMMTAFeatureDatasetPaper(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vit_dir: str,
        audio_dir: str,
        text_dir: str,
        tsn_k: int = 16,
        is_train: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.vit_dir = vit_dir
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.tsn_k = tsn_k
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = norm_id(row["Filename"], 5)
        
        vit = load_pkl_array(os.path.join(self.vit_dir, f"{sid}.pkl"))
        aud = load_pkl_array(os.path.join(self.audio_dir, f"{sid}.pkl"))
        txt = load_pkl_array(os.path.join(self.text_dir, f"{sid}.pkl"))

        # TSN 采样：统一到长度 K
        vit = tsn_sample(vit, self.tsn_k, self.is_train)
        aud = tsn_sample(aud, self.tsn_k, self.is_train)
        txt = tsn_sample(txt, self.tsn_k, self.is_train)

        # 转换为 Tensor
        vit = torch.from_numpy(vit)
        aud = torch.from_numpy(aud)
        txt = torch.from_numpy(txt)
        y = torch.tensor(row[EMO_COLS].to_numpy(dtype=np.float32))
        
        return idx, sid, vit, aud, txt, y

def collate_fn(batch):
    idxs, sids, vit, aud, txt, y = zip(*batch)
    return (
        torch.tensor(idxs, dtype=torch.long),
        list(sids),
        torch.stack(vit, 0),
        torch.stack(aud, 0),
        torch.stack(txt, 0),
        torch.stack(y, 0)
    )
