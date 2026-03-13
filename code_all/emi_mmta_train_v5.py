import argparse
import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


EMO_COLS = [
    "Admiration", "Amusement", "Determination",
    "Empathic Pain", "Excitement", "Joy",
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
    return arr


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


def batch_pearson_loss_weighted(pred: torch.Tensor, target: torch.Tensor, dim_weights: torch.Tensor, eps: float = 1e-8):
    if pred.size(0) < 2:
        return pred.new_tensor(0.0)
    pred = pred.float()
    target = target.float()
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    targ_c = target - target.mean(dim=0, keepdim=True)
    pred_std = pred_c.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    targ_std = targ_c.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    corr = ((pred_c / pred_std) * (targ_c / targ_std)).mean(dim=0)  # [6]
    w = dim_weights.float() / dim_weights.float().sum().clamp_min(eps)
    return 1.0 - torch.sum(corr * w)


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, dim_weights: torch.Tensor):
    # per-dimension SmoothL1 then weighted average
    per_dim = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=0)  # [6]
    w = dim_weights.float() / dim_weights.float().sum().clamp_min(1e-8)
    return torch.sum(per_dim * w)


def adaptive_pool_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
    x = x.transpose(0, 1).unsqueeze(0)
    x = F.adaptive_avg_pool1d(x, target_len)
    return x.squeeze(0).transpose(0, 1)


class EMIMMTAFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vit_dir: str, w2v_dir: str, wavlm_dir: str, seq_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.vit_dir = vit_dir
        self.w2v_dir = w2v_dir
        self.wavlm_dir = wavlm_dir
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = norm_id(row["Filename"], 5)
        vit = torch.from_numpy(load_pkl_array(os.path.join(self.vit_dir, f"{sid}.pkl")))
        w2v = torch.from_numpy(load_pkl_array(os.path.join(self.w2v_dir, f"{sid}.pkl")))
        wavlm = torch.from_numpy(load_pkl_array(os.path.join(self.wavlm_dir, f"{sid}.pkl")))

        vit = adaptive_pool_seq(vit, self.seq_len)
        w2v = adaptive_pool_seq(w2v, self.seq_len)
        wavlm = adaptive_pool_seq(wavlm, self.seq_len)
        y = torch.tensor(row[EMO_COLS].to_numpy(dtype=np.float32))
        return sid, vit, w2v, wavlm, y


def collate_fn(batch):
    sids, vit, w2v, wavlm, y = zip(*batch)
    return list(sids), torch.stack(vit, 0), torch.stack(w2v, 0), torch.stack(wavlm, 0), torch.stack(y, 0)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
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


class UnimodalBranch(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tcn = TemporalConvBlock(hidden, dropout)
        self.pool = AttnPool(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.proj(x)
        z = self.tcn(z)
        z = self.pool(z)
        return self.head(z), z


class EMI_MMTA_V5_LateFusion(nn.Module):
    def __init__(
        self,
        vit_dim: int,
        w2v_dim: int,
        wavlm_dim: int,
        hidden: int = 256,
        dropout: float = 0.2,
        fusion_mode: str = "sample",
    ):
        super().__init__()
        self.v_branch = UnimodalBranch(vit_dim, hidden, dropout)
        self.a_branch = UnimodalBranch(w2v_dim, hidden, dropout)
        self.w_branch = UnimodalBranch(wavlm_dim, hidden, dropout)
        self.fusion_mode = fusion_mode

        if fusion_mode == "global":
            self.fusion_logits = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        elif fusion_mode == "sample":
            self.gate = nn.Sequential(
                nn.LayerNorm(hidden * 3),
                nn.Linear(hidden * 3, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 3),
            )
        else:
            raise ValueError(f"unknown fusion_mode: {fusion_mode}")

    def forward(self, vit, w2v, wavlm):
        pv, zv = self.v_branch(vit)
        pa, za = self.a_branch(w2v)
        pw, zw = self.w_branch(wavlm)
        pred_stack = torch.stack([pv, pa, pw], dim=1)  # (B, 3, 6)

        if self.fusion_mode == "global":
            w = torch.softmax(self.fusion_logits, dim=0).view(1, 3, 1)
        else:
            gate_in = torch.cat([zv, za, zw], dim=1)
            w = torch.softmax(self.gate(gate_in), dim=1).unsqueeze(-1)  # (B, 3, 1)
        return torch.sum(pred_stack * w, dim=1)


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
    for sids, vit, w2v, wavlm, y in tqdm(loader, desc="Infer", ncols=100):
        vit = vit.to(device, non_blocking=True)
        w2v = w2v.to(device, non_blocking=True)
        wavlm = wavlm.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            pred = model(vit, w2v, wavlm)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_split", default="train_split.csv")
    ap.add_argument("--valid_split", default="valid_split.csv")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--audio_dir", default="wav2vec2")
    ap.add_argument("--face_image_dir", default="vit")
    ap.add_argument("--wavlm_dir", default="wavlm_feats")
    ap.add_argument("--output_dir", default="outputs_emi_mmta_v5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--eval", type=int, default=1)
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lam_corr", type=float, default=0.3)
    ap.add_argument("--corr_warmup_epochs", type=int, default=5)
    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--topk_checkpoints", type=int, default=5)
    ap.add_argument("--fusion_mode", choices=["sample", "global"], default="sample")
    ap.add_argument("--joy_weight", type=float, default=1.0)
    args = ap.parse_args()
    if args.epoch is not None:
        args.epochs = args.epoch

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    print("Device:", device)

    train_df = pd.read_csv(args.train_split)
    valid_df = pd.read_csv(args.valid_split)
    ds_train = EMIMMTAFeatureDataset(train_df, args.face_image_dir, args.audio_dir, args.wavlm_dir, args.seq_len)
    ds_valid = EMIMMTAFeatureDataset(valid_df, args.face_image_dir, args.audio_dir, args.wavlm_dir, args.seq_len)
    _, vit0, w20, wlm0, _ = ds_train[0]
    print(f"dims: vit={vit0.shape[-1]}, w2v={w20.shape[-1]}, wavlm={wlm0.shape[-1]} seq_len={args.seq_len}")

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn, persistent_workers=(args.num_workers > 0),
    )
    dl_valid = DataLoader(
        ds_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn, persistent_workers=(args.num_workers > 0),
    )

    model = EMI_MMTA_V5_LateFusion(
        vit0.shape[-1], w20.shape[-1], wlm0.shape[-1],
        hidden=args.hidden, dropout=args.dropout, fusion_mode=args.fusion_mode,
    ).to(device)
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    opt = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    use_amp = bool(args.use_amp) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
    dim_weights = torch.ones(len(EMO_COLS), dtype=torch.float32, device=device)
    joy_idx = EMO_COLS.index("Joy")
    dim_weights[joy_idx] = max(1.0, float(args.joy_weight))

    best = -1e9
    bad = 0
    best_path = os.path.join(args.output_dir, "best.pt")
    pred_csv = os.path.join(args.output_dir, "preds_valid.csv")
    pred_ens_csv = os.path.join(args.output_dir, "preds_valid_ens.csv")
    pred_ens_w_csv = os.path.join(args.output_dir, "preds_valid_ens_weighted.csv")
    topk: List[Tuple[float, str]] = []

    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Train {ep}/{args.epochs}", ncols=100)
        running = 0.0
        lam = args.lam_corr * min(1.0, ep / max(1, args.corr_warmup_epochs))

        for _, vit, w2v, wavlm, y in pbar:
            vit = vit.to(device, non_blocking=True)
            w2v = w2v.to(device, non_blocking=True)
            wavlm = wavlm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                pred = model(vit, w2v, wavlm)
                pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                mse = weighted_smooth_l1(pred, y, dim_weights)
                corr = batch_pearson_loss_weighted(pred, y, dim_weights)
                loss = mse + lam * corr
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            pbar.set_postfix(loss=running / max(1, pbar.n), mse=float(mse.item()), corr=float(corr.item()), lam=lam, lr=opt.param_groups[0]["lr"])

        sched.step()

        if args.eval:
            sids_v, pred_v, gt_v = predict(model, dl_valid, device, use_amp)
            per_dim, avg = avg_pearson(gt_v, pred_v)
            print("\n[VALID] Pearson per dim:")
            for k in EMO_COLS:
                print(f"  {k:15s}: {per_dim[k]:.6f}")
            print(f"[VALID] Average Pearson: {avg:.6f}\n")

            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            ckpt_path = os.path.join(args.output_dir, f"topk_epoch{ep:03d}_{avg:.6f}.pt")
            torch.save(state, ckpt_path)
            topk.append((avg, ckpt_path))
            topk = sorted(topk, key=lambda x: x[0], reverse=True)[: max(1, args.topk_checkpoints)]
            keep = {p for _, p in topk}
            for fn in os.listdir(args.output_dir):
                if fn.startswith("topk_epoch") and fn.endswith(".pt"):
                    full = os.path.join(args.output_dir, fn)
                    if full not in keep:
                        try:
                            os.remove(full)
                        except Exception:
                            pass

            if avg > best + 1e-4:
                best = avg
                bad = 0
                torch.save(state, best_path)
                print(f"[SAVE] {best_path} (avg pearson={best:.6f})")
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"[EARLY STOP] no improvement for {args.patience} evals.")
                    break

    if args.eval and os.path.isfile(best_path):
        load_ckpt_to_model(model, best_path, device)
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

        if len(topk) >= 2:
            sorted_topk = sorted(topk, key=lambda x: x[0], reverse=True)
            model_paths = [p for _, p in sorted_topk]
            model_scores = np.array([s for s, _ in sorted_topk], dtype=np.float32)
            temp = max(1e-6, float(np.std(model_scores)) + 1e-6)
            weights = np.exp((model_scores - model_scores.max()) / temp)
            weights = weights / np.sum(weights)
            models = []
            for p in model_paths:
                m = EMI_MMTA_V5_LateFusion(
                    vit0.shape[-1], w20.shape[-1], wlm0.shape[-1],
                    hidden=args.hidden, dropout=args.dropout, fusion_mode=args.fusion_mode,
                ).to(device)
                m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
                m.eval()
                models.append(m)

            all_sids, all_pred, all_gt = [], [], []
            all_pred_w = []
            with torch.no_grad():
                for sids, vit, w2v, wavlm, y in tqdm(dl_valid, desc="Infer(ens)", ncols=100):
                    vit = vit.to(device, non_blocking=True)
                    w2v = w2v.to(device, non_blocking=True)
                    wavlm = wavlm.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    pred_sum = 0.0
                    pred_weighted = 0.0
                    for i, m in enumerate(models):
                        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                            p = m(vit, w2v, wavlm)
                        pred_sum = pred_sum + p
                        pred_weighted = pred_weighted + p * float(weights[i])
                    pred = (pred_sum / float(len(models))).clamp(0.0, 1.0)
                    pred_w = pred_weighted.clamp(0.0, 1.0)
                    all_sids.extend(sids)
                    all_pred.append(pred.detach().cpu().numpy())
                    all_pred_w.append(pred_w.detach().cpu().numpy())
                    all_gt.append(y.detach().cpu().numpy())
            pred_e = np.concatenate(all_pred, axis=0)
            pred_e_w = np.concatenate(all_pred_w, axis=0)
            gt_e = np.concatenate(all_gt, axis=0)
            per_dim_e, avg_e = avg_pearson(gt_e, pred_e)
            per_dim_w, avg_w = avg_pearson(gt_e, pred_e_w)
            print("\n[FINAL VALID ENSEMBLE] Pearson per dim:")
            for k in EMO_COLS:
                print(f"  {k:15s}: {per_dim_e[k]:.6f}")
            print(f"[FINAL VALID ENSEMBLE] Average Pearson: {avg_e:.6f}")
            out_e = pd.DataFrame({"Filename": all_sids})
            for i, c in enumerate(EMO_COLS):
                out_e[c] = pred_e[:, i]
            out_e.to_csv(pred_ens_csv, index=False)
            print("Saved:", pred_ens_csv)

            print("\n[FINAL VALID ENSEMBLE WEIGHTED] Pearson per dim:")
            for k in EMO_COLS:
                print(f"  {k:15s}: {per_dim_w[k]:.6f}")
            print(f"[FINAL VALID ENSEMBLE WEIGHTED] Average Pearson: {avg_w:.6f}")
            out_w = pd.DataFrame({"Filename": all_sids})
            for i, c in enumerate(EMO_COLS):
                out_w[c] = pred_e_w[:, i]
            out_w.to_csv(pred_ens_w_csv, index=False)
            print("Saved:", pred_ens_w_csv)


if __name__ == "__main__":
    main()
