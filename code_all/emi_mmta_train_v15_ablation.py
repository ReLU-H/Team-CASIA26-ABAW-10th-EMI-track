import argparse
import os
import pickle
import random
from contextlib import contextmanager

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


EMO_COLS = [
    "Admiration",
    "Amusement",
    "Determination",
    "Empathic Pain",
    "Excitement",
    "Joy",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 0:
        arr = arr[None, None]
    return arr


def tsn_sample(seq: np.ndarray, k: int, is_train: bool) -> np.ndarray:
    t, d = seq.shape
    if t == 0:
        return np.zeros((k, d), dtype=np.float32)

    bounds = np.linspace(0, t, k + 1, dtype=int)
    out_idx = []
    for i in range(k):
        st, ed = bounds[i], bounds[i + 1]
        if st >= ed:
            idx = min(st, t - 1)
        else:
            idx = random.randint(st, ed - 1) if is_train else (st + ed - 1) // 2
        out_idx.append(idx)
    return seq[out_idx]


# =========================== dataset ===========================

class EMIMMTAFeatureDatasetV15(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vit_dir: str,
        aud_dir: str,
        txt_dir: str,
        seq_len: int = 128,
        use_tsn: int = 0,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.vit_dir = vit_dir
        self.aud_dir = aud_dir
        self.txt_dir = txt_dir
        self.seq_len = seq_len
        self.use_tsn = bool(use_tsn)
        self.is_train = is_train

    @staticmethod
    def adaptive_pool_seq(x: np.ndarray, target_len: int) -> np.ndarray:
        tx = torch.from_numpy(x).transpose(0, 1).unsqueeze(0)
        tx = F.adaptive_avg_pool1d(tx, target_len)
        return tx.squeeze(0).transpose(0, 1).numpy().astype(np.float32, copy=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = norm_id(row["Filename"], 5)

        vit = load_pkl_array(os.path.join(self.vit_dir, f"{sid}.pkl"))
        aud = load_pkl_array(os.path.join(self.aud_dir, f"{sid}.pkl"))
        txt_raw = load_pkl_array(os.path.join(self.txt_dir, f"{sid}.pkl"))

        txt_gap = txt_raw.mean(axis=0)

        if self.use_tsn:
            vit = tsn_sample(vit, self.seq_len, self.is_train)
            aud = tsn_sample(aud, self.seq_len, self.is_train)
            txt = tsn_sample(txt_raw, self.seq_len, self.is_train)
        else:
            vit = self.adaptive_pool_seq(vit, self.seq_len)
            aud = self.adaptive_pool_seq(aud, self.seq_len)
            txt = self.adaptive_pool_seq(txt_raw, self.seq_len)

        y = torch.tensor(row[EMO_COLS].to_numpy(dtype=np.float32))
        return (
            sid,
            torch.from_numpy(vit),
            torch.from_numpy(aud),
            torch.from_numpy(txt),
            torch.from_numpy(txt_gap),
            y,
        )


def collate_fn(batch):
    sids, vit, aud, txt, txt_gap, y = zip(*batch)
    return (
        list(sids),
        torch.stack(vit, 0),
        torch.stack(aud, 0),
        torch.stack(txt, 0),
        torch.stack(txt_gap, 0),
        torch.stack(y, 0),
    )


# =========================== metrics / losses ===========================

def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return 0.0
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    x = (x - x.mean()) / sx
    y = (y - y.mean()) / sy
    return float(np.mean(x * y))


def avg_pearson(y_true: np.ndarray, y_pred: np.ndarray):
    per_dim = {}
    for i, c in enumerate(EMO_COLS):
        per_dim[c] = pearsonr_safe(y_true[:, i], y_pred[:, i])
    return per_dim, sum(per_dim.values()) / len(EMO_COLS)


def batch_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    if pred.size(0) < 2:
        return pred.new_tensor(0.0)
    pred = pred.float()
    target = target.float()
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    targ_c = target - target.mean(dim=0, keepdim=True)
    pred_std = pred_c.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    targ_std = targ_c.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    corr = ((pred_c / pred_std) * (targ_c / targ_std)).mean(dim=0)
    return 1.0 - corr.mean()


def text_contrastive_triplet_loss(
    fused_feat: torch.Tensor,
    txt_gap: torch.Tensor,
    n_triplets: int = 32,
    margin: float = 0.1,
) -> torch.Tensor:
    B = fused_feat.size(0)
    if B < 3:
        return fused_feat.new_tensor(0.0)

    actual_triplets = min(n_triplets, B)
    losses = []

    for _ in range(actual_triplets):
        idx = torch.randperm(B, device=fused_feat.device)[:3]
        i, j, k = int(idx[0].item()), int(idx[1].item()), int(idx[2].item())

        with torch.no_grad():
            dij = torch.sum((txt_gap[i] - txt_gap[j]) ** 2)
            dik = torch.sum((txt_gap[i] - txt_gap[k]) ** 2)
            djk = torch.sum((txt_gap[j] - txt_gap[k]) ** 2)

        if dij <= dik and dij <= djk:
            anc, pos, neg = i, j, k
        elif dik <= dij and dik <= djk:
            anc, pos, neg = i, k, j
        else:
            anc, pos, neg = j, k, i

        f_anc = fused_feat[anc]
        f_pos = fused_feat[pos]
        f_neg = fused_feat[neg]

        d_ap2 = torch.sum((f_anc - f_pos) ** 2)
        d_an2 = torch.sum((f_anc - f_neg) ** 2)
        d_pn2 = torch.sum((f_pos - f_neg) ** 2)

        l1 = F.relu(d_ap2 - d_an2 + margin)
        l2 = F.relu(d_ap2 - d_pn2 + margin)
        losses.append(l1 + l2)

    if not losses:
        return fused_feat.new_tensor(0.0)
    return torch.stack(losses).mean()


# =========================== EMA ===========================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        src_model = model.module if isinstance(model, nn.DataParallel) else model
        for name, param in src_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()
        for name, buf in src_model.named_buffers():
            self.shadow[f"__buf__{name}"] = buf.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        src_model = model.module if isinstance(model, nn.DataParallel) else model

        for name, param in src_model.named_parameters():
            if not param.requires_grad:
                continue
            new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()
            self.shadow[name] = new_avg.clone()

        for name, buf in src_model.named_buffers():
            self.shadow[f"__buf__{name}"] = buf.detach().clone()

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        src_model = model.module if isinstance(model, nn.DataParallel) else model
        self.backup = {}

        for name, param in src_model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].data)

        for name, buf in src_model.named_buffers():
            key = f"__buf__{name}"
            if key in self.shadow:
                self.backup[key] = buf.detach().clone()
                buf.data.copy_(self.shadow[key].data)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        src_model = model.module if isinstance(model, nn.DataParallel) else model

        for name, param in src_model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.data.copy_(self.backup[name].data)

        for name, buf in src_model.named_buffers():
            key = f"__buf__{name}"
            if key in self.backup:
                buf.data.copy_(self.backup[key].data)

        self.backup = {}

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


@contextmanager
def ema_scope(model: nn.Module, ema: ModelEMA):
    ema.apply_shadow(model)
    try:
        yield
    finally:
        ema.restore(model)


# =========================== model blocks ===========================

class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        z = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + z)


class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        a = torch.softmax(self.fc(x).squeeze(-1), dim=1)
        return torch.sum(x * a.unsqueeze(-1), dim=1)


class MeanPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class TemporalAugmentBiGRU(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gru1 = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru2 = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z, _ = self.gru1(x)
        z = self.drop(z)
        z, _ = self.gru2(z)
        z = self.norm(z)
        z = self.fc(z)
        return z


# =========================== branches ===========================

class VisualBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        feat_dim: int,
        dropout: float,
        use_tcn: bool = True,
        use_bigru: bool = True,
        use_attn_pool: bool = True,
    ):
        super().__init__()
        self.use_tcn = use_tcn
        self.use_bigru = use_bigru

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tcn = TemporalConvBlock(hidden, dropout) if use_tcn else nn.Identity()
        self.temporal_aug = TemporalAugmentBiGRU(hidden, dropout=dropout) if use_bigru else nn.Identity()
        self.pool = AttnPool(hidden) if use_attn_pool else MeanPool()

        self.feat_fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, feat_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
            nn.Sigmoid(),
        )

    def forward(self, x):
        seq = self.proj(x)
        seq = self.tcn(seq)
        seq = self.temporal_aug(seq)
        z = self.feat_fc(self.pool(seq))
        p = self.head(z)
        return p, z


class TextBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        feat_dim: int,
        dropout: float,
        use_tcn: bool = True,
        use_bigru: bool = True,
        use_attn_pool: bool = True,
    ):
        super().__init__()
        self.use_tcn = use_tcn
        self.use_bigru = use_bigru

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tcn = TemporalConvBlock(hidden, dropout) if use_tcn else nn.Identity()
        self.temporal_aug = TemporalAugmentBiGRU(hidden, dropout=dropout) if use_bigru else nn.Identity()
        self.pool = AttnPool(hidden) if use_attn_pool else MeanPool()

        self.feat_fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, feat_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
            nn.Sigmoid(),
        )

    def forward(self, x):
        seq = self.proj(x)
        seq = self.tcn(seq)
        seq = self.temporal_aug(seq)
        z = self.feat_fc(self.pool(seq))
        p = self.head(z)
        return p, z


class AudioVADBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        feat_dim: int,
        dropout: float,
        rnn_dropout: float = 0.1,
        use_vad_module: bool = True,
        use_bigru: bool = True,
        use_attn_pool: bool = True,
    ):
        super().__init__()
        self.use_vad_module = use_vad_module

        self.audio_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if use_vad_module:
            self.vad_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 3),
                nn.Sigmoid(),
            )
            self.fuse_proj = nn.Sequential(
                nn.LayerNorm(hidden + 3),
                nn.Linear(hidden + 3, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.vad_head = None
            self.fuse_proj = None

        self.temporal_aug = TemporalAugmentBiGRU(hidden, dropout=max(dropout, rnn_dropout)) if use_bigru else nn.Identity()
        self.pool = AttnPool(hidden) if use_attn_pool else MeanPool()

        self.feat_fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, feat_dim),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
            nn.Sigmoid(),
        )

    def forward(self, x):
        a = self.audio_proj(x)

        if self.use_vad_module:
            a_mean = a.mean(dim=1)
            vad = self.vad_head(a_mean)
            vad_rep = vad.unsqueeze(1).expand(-1, a.size(1), -1)
            a = torch.cat([a, vad_rep], dim=-1)
            a = self.fuse_proj(a)
        else:
            vad = a.new_full((a.size(0), 3), 0.5)

        seq = self.temporal_aug(a)
        z = self.feat_fc(self.pool(seq))
        p = self.reg_head(z)
        return p, z, vad


# =========================== full model ===========================

class EMI_MMTA_V15_Ablation(nn.Module):
    def __init__(
        self,
        vit_dim: int,
        aud_dim: int,
        txt_dim: int,
        hidden: int = 256,
        feat_dim: int = 256,
        dropout: float = 0.2,
        rnn_dropout: float = 0.1,
        use_visual_tcn: bool = True,
        use_visual_bigru: bool = True,
        use_text_tcn: bool = True,
        use_text_bigru: bool = True,
        use_audio_bigru: bool = True,
        use_attn_pool: bool = True,
        use_vad_module: bool = True,
        fusion_type: str = "avg",
    ):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        assert self.fusion_type in ["avg", "concat"]

        self.v_branch = VisualBranch(
            vit_dim, hidden, feat_dim, dropout,
            use_tcn=use_visual_tcn,
            use_bigru=use_visual_bigru,
            use_attn_pool=use_attn_pool,
        )
        self.a_branch = AudioVADBranch(
            aud_dim, hidden, feat_dim, dropout,
            rnn_dropout=rnn_dropout,
            use_vad_module=use_vad_module,
            use_bigru=use_audio_bigru,
            use_attn_pool=use_attn_pool,
        )
        self.t_branch = TextBranch(
            txt_dim, hidden, feat_dim, dropout,
            use_tcn=use_text_tcn,
            use_bigru=use_text_bigru,
            use_attn_pool=use_attn_pool,
        )

        if self.fusion_type == "avg":
            fusion_in = feat_dim
        else:
            fusion_in = feat_dim * 3

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
            nn.Sigmoid(),
        )

    def forward(self, vit, aud, txt):
        pv, zv = self.v_branch(vit)
        pa, za, vad = self.a_branch(aud)
        pt, zt = self.t_branch(txt)

        if self.fusion_type == "avg":
            fused_feat = (zv + za + zt) / 3.0
            pred = self.fusion_head(fused_feat).clamp(0.0, 1.0)
            w = pred.new_full((pred.size(0), 3), 1.0 / 3.0)
        else:
            fused_feat = torch.cat([zv, za, zt], dim=-1)
            pred = self.fusion_head(fused_feat).clamp(0.0, 1.0)
            w = pred.new_zeros((pred.size(0), 3))

        return pred, (pv, pa, pt), vad, w, fused_feat


# =========================== training utilities ===========================

def build_optimizer(name: str, params, lr: float, wd: float):
    n = name.lower()
    if n == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if n == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    raise ValueError(n)


@torch.no_grad()
def predict(model, loader, device, use_amp: bool):
    model.eval()
    all_sids, all_pred, all_gt = [], [], []
    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for sids, vit, aud, txt, _txt_gap, y in tqdm(loader, desc="Infer", ncols=100):
        vit = vit.to(device, non_blocking=True)
        aud = aud.to(device, non_blocking=True)
        txt = txt.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            pred, _, _, _, _ = model(vit, aud, txt)

        pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        all_sids.extend(sids)
        all_pred.append(pred.cpu().numpy())
        all_gt.append(y.cpu().numpy())

    return all_sids, np.concatenate(all_pred, axis=0), np.concatenate(all_gt, axis=0)


def load_ckpt_to_model(model, ckpt_path: str, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(sd, strict=True)
    else:
        model.load_state_dict(sd, strict=True)


# =========================== main ===========================

def main():
    ap = argparse.ArgumentParser()

    # data / runtime
    ap.add_argument("--train_split", default="train_split.csv")
    ap.add_argument("--valid_split", default="valid_split.csv")
    ap.add_argument("--audio_dir", default="audio_feats_wavlm_large")
    ap.add_argument("--face_image_dir", default="images_feats_EmoVit")
    ap.add_argument("--wavlm_dir", default="text_feats_ChatGLM3")
    ap.add_argument("--output_dir", default="outputs_emi_mmta_v15_ablation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--eval", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--use_amp", type=int, default=1)

    # optimization
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--optimizer", default="adamw")

    # model
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--rnn_dropout", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--feat_dim", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--use_tsn", type=int, default=0)

    # losses
    ap.add_argument("--lam_corr", type=float, default=0.10)
    ap.add_argument("--lam_aux", type=float, default=0.03)
    ap.add_argument("--lam_triplet", type=float, default=0.05)
    ap.add_argument("--triplet_margin", type=float, default=0.1)
    ap.add_argument("--triplet_n", type=int, default=32)
    ap.add_argument("--triplet_warmup_epochs", type=int, default=3)
    ap.add_argument("--corr_warmup_epochs", type=int, default=8)

    # ema
    ap.add_argument("--use_ema", type=int, default=1)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--ema_start_epoch", type=int, default=1)

    # ================= ablation switches =================
    # Module 1: temporal augmentation
    ap.add_argument("--use_visual_tcn", type=int, default=1)
    ap.add_argument("--use_visual_bigru", type=int, default=1)
    ap.add_argument("--use_text_tcn", type=int, default=1)
    ap.add_argument("--use_text_bigru", type=int, default=1)
    ap.add_argument("--use_audio_bigru", type=int, default=1)
    ap.add_argument("--use_attn_pool", type=int, default=1)

    # Module 2: audio VAD-aware modeling
    ap.add_argument("--use_vad_module", type=int, default=1)
    ap.add_argument("--use_vad_reg", type=int, default=1)

    # Module 3: text-guided triplet
    ap.add_argument("--use_triplet_module", type=int, default=1)

    # Module 4: late fusion + multi-objective
    ap.add_argument("--fusion_type", default="avg", choices=["avg", "concat"])
    ap.add_argument("--use_corr_loss", type=int, default=1)
    ap.add_argument("--use_aux_loss", type=int, default=1)

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    print("Device:", device)

    print("========== Ablation Config ==========")
    print(f"use_visual_tcn     = {args.use_visual_tcn}")
    print(f"use_visual_bigru   = {args.use_visual_bigru}")
    print(f"use_text_tcn       = {args.use_text_tcn}")
    print(f"use_text_bigru     = {args.use_text_bigru}")
    print(f"use_audio_bigru    = {args.use_audio_bigru}")
    print(f"use_attn_pool      = {args.use_attn_pool}")
    print(f"use_vad_module     = {args.use_vad_module}")
    print(f"use_vad_reg        = {args.use_vad_reg}")
    print(f"use_triplet_module = {args.use_triplet_module}")
    print(f"fusion_type        = {args.fusion_type}")
    print(f"use_corr_loss      = {args.use_corr_loss}")
    print(f"use_aux_loss       = {args.use_aux_loss}")
    print(f"use_ema            = {args.use_ema}")
    print("=====================================")

    train_df = pd.read_csv(args.train_split)
    valid_df = pd.read_csv(args.valid_split)

    ds_train = EMIMMTAFeatureDatasetV15(
        train_df,
        args.face_image_dir,
        args.audio_dir,
        args.wavlm_dir,
        seq_len=args.seq_len,
        use_tsn=args.use_tsn,
        is_train=True,
    )
    ds_valid = EMIMMTAFeatureDatasetV15(
        valid_df,
        args.face_image_dir,
        args.audio_dir,
        args.wavlm_dir,
        seq_len=args.seq_len,
        use_tsn=args.use_tsn,
        is_train=False,
    )

    _, vit0, a0, t0, tg0, _ = ds_train[0]
    print(f"dims: vit={vit0.shape[-1]}, aud={a0.shape[-1]}, txt={t0.shape[-1]}, txt_gap={tg0.shape}")
    print(f"seq_len={args.seq_len}")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )

    model = EMI_MMTA_V15_Ablation(
        vit_dim=vit0.shape[-1],
        aud_dim=a0.shape[-1],
        txt_dim=t0.shape[-1],
        hidden=args.hidden,
        feat_dim=args.feat_dim,
        dropout=args.dropout,
        rnn_dropout=args.rnn_dropout,
        use_visual_tcn=bool(args.use_visual_tcn),
        use_visual_bigru=bool(args.use_visual_bigru),
        use_text_tcn=bool(args.use_text_tcn),
        use_text_bigru=bool(args.use_text_bigru),
        use_audio_bigru=bool(args.use_audio_bigru),
        use_attn_pool=bool(args.use_attn_pool),
        use_vad_module=bool(args.use_vad_module),
        fusion_type=args.fusion_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    opt = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    use_amp = bool(args.use_amp) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"

    use_ema = bool(args.use_ema)
    ema = ModelEMA(model, decay=args.ema_decay) if use_ema else None

    best = -1e9
    bad = 0
    best_path = os.path.join(args.output_dir, "best.pt")
    best_ema_path = os.path.join(args.output_dir, "best_ema.pt")
    pred_csv = os.path.join(args.output_dir, "preds_valid.csv")

    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Train {ep}/{args.epochs}", ncols=140)
        running = 0.0

        lam_corr = args.lam_corr * min(1.0, ep / max(1, args.corr_warmup_epochs))
        lam_tri = args.lam_triplet * min(1.0, ep / max(1, args.triplet_warmup_epochs))

        if not bool(args.use_corr_loss):
            lam_corr = 0.0
        if not bool(args.use_triplet_module):
            lam_tri = 0.0

        for _, vit, aud, txt, txt_gap, y in pbar:
            vit = vit.to(device, non_blocking=True)
            aud = aud.to(device, non_blocking=True)
            txt = txt.to(device, non_blocking=True)
            txt_gap = txt_gap.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                pred, aux, vad, w, fused_feat = model(vit, aud, txt)
                pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                l_mse = F.mse_loss(pred, y)

                l_corr = batch_pearson_loss(pred, y) if bool(args.use_corr_loss) else pred.new_tensor(0.0)

                if bool(args.use_aux_loss):
                    l_aux = (
                        0.20 * F.smooth_l1_loss(aux[0], y)
                        + 0.35 * F.smooth_l1_loss(aux[1], y)
                        + 0.45 * F.smooth_l1_loss(aux[2], y)
                    )
                else:
                    l_aux = pred.new_tensor(0.0)

                if bool(args.use_vad_module) and bool(args.use_vad_reg):
                    l_vad_reg = ((vad - 0.5) ** 2).mean()
                else:
                    l_vad_reg = pred.new_tensor(0.0)

                if bool(args.use_triplet_module):
                    # avg fusion has feat_dim; concat fusion has 3*feat_dim, both are valid
                    l_triplet = text_contrastive_triplet_loss(
                        fused_feat=fused_feat,
                        txt_gap=txt_gap,
                        n_triplets=args.triplet_n,
                        margin=args.triplet_margin,
                    )
                else:
                    l_triplet = pred.new_tensor(0.0)

                loss = (
                    l_mse
                    + lam_corr * l_corr
                    + args.lam_aux * l_aux
                    + 0.01 * l_vad_reg
                    + lam_tri * l_triplet
                )

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if use_ema and ep >= args.ema_start_epoch:
                ema.update(model)

            running += float(loss.item())
            w_m = w.mean(dim=0).detach()

            pbar.set_postfix(
                loss=running / max(1, pbar.n),
                mse=float(l_mse.item()),
                pcc=float(l_corr.item()),
                aux=float(l_aux.item()),
                tri=float(l_triplet.item()),
                wv=float(w_m[0].item()) if w.size(1) == 3 else 0.0,
                wa=float(w_m[1].item()) if w.size(1) == 3 else 0.0,
                wt=float(w_m[2].item()) if w.size(1) == 3 else 0.0,
                lr=opt.param_groups[0]["lr"],
            )

        sched.step()

        if args.eval:
            if use_ema and ep >= args.ema_start_epoch:
                with ema_scope(model, ema):
                    sids_v, pred_v, gt_v = predict(model, dl_valid, device, use_amp)
            else:
                sids_v, pred_v, gt_v = predict(model, dl_valid, device, use_amp)

            per_dim, avg = avg_pearson(gt_v, pred_v)

            print("\n[VALID] Pearson per dim:")
            for k in EMO_COLS:
                print(f"  {k:15s}: {per_dim[k]:.6f}")
            print(f"[VALID] Average Pearson: {avg:.6f}\n")

            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            if avg > best + 1e-4:
                best = avg
                bad = 0
                torch.save(state, best_path)
                if use_ema and ep >= args.ema_start_epoch:
                    torch.save(ema.state_dict(), best_ema_path)
                print(f"[SAVE] {best_path} (avg pearson={best:.6f})")
                if use_ema and ep >= args.ema_start_epoch:
                    print(f"[SAVE] {best_ema_path}")
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"[EARLY STOP] no improvement for {args.patience} evals.")
                    break

    if args.eval and os.path.isfile(best_path):
        load_ckpt_to_model(model, best_path, device)

        if use_ema and os.path.isfile(best_ema_path):
            ema_sd = torch.load(best_ema_path, map_location="cpu", weights_only=False)
            ema.load_state_dict(ema_sd)
            with ema_scope(model, ema):
                sids_v, pred_v, gt_v = predict(model, dl_valid, device, use_amp)
        else:
            sids_v, pred_v, gt_v = predict(model, dl_valid, device, use_amp)

        per_dim, avg = avg_pearson(gt_v, pred_v)
        print("\n[FINAL VALID] Pearson per dim:")
        for k in EMO_COLS:
            print(f"  {k:15s}: {per_dim[k]:.6f}")
        print(f"[FINAL VALID] Average Pearson: {avg:.6f}")

        out = pd.DataFrame({"Filename": sids_v})
        for i, c in enumerate(EMO_COLS):
            out[c] = pred_v[:, i]
        out.to_csv(pred_csv, index=False)
        print("Saved:", pred_csv)

        out.to_csv(os.path.join(args.output_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    main()