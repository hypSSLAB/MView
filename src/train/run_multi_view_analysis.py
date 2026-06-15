#!/usr/bin/env python3
"""Train Mamba+MixStyle multi-view classifier on a single (proto/dev/net) slot.

Uses the standalone `MS_Mamba` model (no BiMamba, no AttnPool).
Training: per-iface model, late-fusion via softmax averaging.
Hyperparameters: AdamW + OneCycleLR + Mixup(α=0.4) + label smoothing(0.1) + AMP.

Usage:
    python run_multi_view_analysis.py \
        --npz   /path/to/NPZ/Tor/CW/A34/WIFI \
        --ifaces tun0 wlan0 total lo \
        --train-file train.npz \
        --exp-name Tor_A34_WIFI_NoAug_MS_Mamba

    # With Aug7x:
    python run_multi_view_analysis.py \
        --npz   /path/to/NPZ/Tor/CW/A34/WIFI \
        --ifaces tun0 wlan0 total lo \
        --train-file train_aug.npz \
        --exp-name Tor_A34_WIFI_Aug7x_MS_Mamba
"""
import os, sys, json, time, random, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'model'))
from MS_Mamba import MSMamba


# ====================== Defaults (edit if needed) ======================

ROOT          = Path('./')
RESULTS_DIR   = ROOT / 'MV-Mamba' / 'results_all_experiments'
CKPT_BASE_DIR = ROOT / 'Dataset' / 'Model'

SEED       = 42
EPOCHS     = 100
LR         = 1e-3
BATCH      = 256
PATIENCE   = 30
MIXUP_A    = 0.4
LABEL_SMTH = 0.1


# ====================== Dataset ======================

class _NpzDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def _set_seed(s: int = SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ====================== Training ======================

def train_one_slot(npz_dir: Path,
                   ifaces: list[str],
                   train_file: str,
                   exp_name: str,
                   epochs: int = EPOCHS,
                   gpu: int = 0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _set_seed()

    print(f"\n{'='*60}\n  {exp_name}  npz_dir={npz_dir}\n  ifaces={ifaces}  train_file={train_file}\n{'='*60}", flush=True)

    # ---- Load NPZ per iface (with StandardScaler fit on train) ----
    lds = {}
    for iface in ifaces:
        nd = npz_dir / iface
        tr = np.load(nd / train_file, allow_pickle=True)
        va = np.load(nd / 'valid.npz',  allow_pickle=True)
        te = np.load(nd / 'test.npz',   allow_pickle=True)
        Xt, yt = tr['X'][:, :, :4], tr['y']
        Xv, yv = va['X'][:, :, :4], va['y']
        Xe, ye = te['X'][:, :, :4], te['y']
        n, t, f = Xt.shape

        sc = StandardScaler(); sc.fit(Xt.reshape(-1, f))
        Xtn = sc.transform(Xt.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
        Xvn = sc.transform(Xv.reshape(-1, f)).reshape(Xv.shape).astype(np.float32)
        Xen = sc.transform(Xe.reshape(-1, f)).reshape(Xe.shape).astype(np.float32)

        lds[iface] = {
            'train': DataLoader(_NpzDS(Xtn, yt), batch_size=BATCH, shuffle=True,
                                num_workers=2, pin_memory=True, drop_last=True),
            'val':   DataLoader(_NpzDS(Xvn, yv), batch_size=BATCH, shuffle=False,
                                num_workers=2, pin_memory=True),
            'test':  DataLoader(_NpzDS(Xen, ye), batch_size=BATCH, shuffle=False,
                                num_workers=2, pin_memory=True),
        }

    # ---- Build models (one per iface, identical arch) ----
    nc = int(np.load(npz_dir / ifaces[0] / train_file, allow_pickle=True)['y'].max() + 1)
    mds = {i: MSMamba(input_dim=4, num_classes=nc,
                            d_model=256, num_layers=4,
                            mixstyle_alpha=0.3, mixstyle_prob=0.5).to(device)
           for i in ifaces}
    n_params_M = sum(p.numel() for p in mds[ifaces[0]].parameters()) / 1e6
    print(f"  Params per iface: {n_params_M:.2f}M  n_classes={nc}", flush=True)

    # ---- Optimizers / schedulers / AMP ----
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMTH)
    opts, schs, amps = {}, {}, {}
    for n, m in mds.items():
        o = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.05)
        s = torch.optim.lr_scheduler.OneCycleLR(
            o, max_lr=LR, epochs=epochs,
            steps_per_epoch=max(1, len(lds[n]['train'])),
            pct_start=0.1, anneal_strategy='cos',
        )
        opts[n] = o
        schs[n] = s
        amps[n] = torch.amp.GradScaler('cuda')

    # ---- Fusion eval ----
    @torch.no_grad()
    def ev_fus(loaders, models, ifs):
        for m in models.values(): m.eval()
        ap, al = [], []
        its = {n: iter(loaders[n]) for n in ifs}
        n_batches = min(len(loaders[n]) for n in ifs)
        for _ in range(n_batches):
            lo, by = {}, None
            for n in ifs:
                x, y = next(its[n])
                lo[n] = models[n](x.to(device))
                if by is None: by = y
            mb = min(v.size(0) for v in lo.values())
            lo = {k: v[:mb] for k, v in lo.items()}
            by = by[:mb]
            avg = torch.stack([F.softmax(l, dim=1) for l in lo.values()]).mean(0)
            ap.extend(avg.argmax(1).cpu().tolist())
            al.extend(by.tolist())
        return ap, al

    # ---- Train loop with early stopping on val Fusion-F1 ----
    best_f1, best_state, pat_count = 0.0, {}, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        for m in mds.values(): m.train()
        its = {n: iter(lds[n]['train']) for n in ifaces}
        n_batches = min(len(lds[n]['train']) for n in ifaces)
        for _ in range(n_batches):
            for n in ifaces:
                x, y = next(its[n])
                x, y = x.to(device), y.to(device)
                opts[n].zero_grad()
                with torch.amp.autocast('cuda'):
                    if random.random() < 0.5:
                        xm, ya, yb, lam = _mixup(x, y, MIXUP_A)
                        ce = lam * crit(mds[n](xm), ya) + (1 - lam) * crit(mds[n](xm), yb)
                    else:
                        ce = crit(mds[n](x), y)
                amps[n].scale(ce).backward()
                amps[n].unscale_(opts[n])
                torch.nn.utils.clip_grad_norm_(mds[n].parameters(), 1.0)
                amps[n].step(opts[n]); amps[n].update()
                schs[n].step()

        # Eval on val
        p_fus, l_fus = ev_fus({n: lds[n]['val'] for n in ifaces}, mds, ifaces)
        fus_f1 = round(f1_score(l_fus, p_fus, average='macro', zero_division=0), 4)
        if fus_f1 > best_f1:
            best_f1, pat_count = fus_f1, 0
            best_state = {n: {k: v.cpu().clone() for k, v in m.state_dict().items()}
                          for n, m in mds.items()}
        else:
            pat_count += 1
        if ep % 10 == 0 or ep == 1 or pat_count == 0:
            print(f"  ep={ep:3d}  val_fus_f1={fus_f1:.4f}  best={best_f1:.4f}  pat={pat_count}", flush=True)
        if pat_count >= PATIENCE:
            print(f"  Early-stop @ ep={ep}", flush=True)
            break

    if best_state:
        for n, sd in best_state.items():
            mds[n].load_state_dict(sd)
    elapsed = time.time() - t0

    # ---- Per-iface test metrics ----
    individual = {}
    for i in ifaces:
        mds[i].eval(); pp, ll = [], []
        with torch.no_grad():
            for x, y in lds[i]['test']:
                pp.extend(mds[i](x.to(device)).argmax(1).cpu().tolist())
                ll.extend(y.tolist())
        pa, la = np.array(pp), np.array(ll)
        individual[i] = dict(
            acc =round(accuracy_score(la, pa), 4),
            prec=round(precision_score(la, pa, average='macro', zero_division=0), 4),
            rec =round(recall_score(la, pa, average='macro', zero_division=0), 4),
            f1  =round(f1_score(la, pa, average='macro', zero_division=0), 4),
        )

    # ---- Fusion test metrics (all ifaces) ----
    p_all, l_all = ev_fus({i: lds[i]['test'] for i in ifaces}, mds, ifaces)
    p_a, l_a = np.array(p_all), np.array(l_all)
    fusion = dict(
        acc =round(accuracy_score(l_a, p_a), 4),
        prec=round(precision_score(l_a, p_a, average='macro', zero_division=0), 4),
        rec =round(recall_score(l_a, p_a, average='macro', zero_division=0), 4),
        f1  =round(f1_score(l_a, p_a, average='macro', zero_division=0), 4),
    )

    print(f"\n  === {exp_name} ===", flush=True)
    for i in ifaces:
        m = individual[i]
        print(f"    {i:6s}: acc={m['acc']}  P={m['prec']}  R={m['rec']}  F1={m['f1']}", flush=True)
    print(f"    Fusion : acc={fusion['acc']}  P={fusion['prec']}  R={fusion['rec']}  F1={fusion['f1']}  [{elapsed/60:.1f}min]", flush=True)

    # ---- Save checkpoints + JSON ----
    ckpt_dir = CKPT_BASE_DIR / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for i, m in mds.items():
        torch.save(m.state_dict(), ckpt_dir / f'{i}.pt')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(dict(
        name=exp_name, model='MS_Mamba',
        npz_dir=str(npz_dir), train_file=train_file,
        ifaces=ifaces, individual=individual, fusion=fusion,
        best_val_fusion_f1=best_f1, params_M=n_params_M,
        time_min=round(elapsed / 60, 1), checkpoint_dir=str(ckpt_dir),
    ), open(RESULTS_DIR / f'{exp_name}.json', 'w'), indent=2)
    print(f"  ckpt: {ckpt_dir}\n  json: {RESULTS_DIR / f'{exp_name}.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', type=str, required=True,
                    help='NPZ root containing <iface>/{train,valid,test}.npz')
    ap.add_argument('--ifaces', nargs='+', required=True,
                    help='interface list, e.g. tun0 wlan0 total lo')
    ap.add_argument('--train-file', type=str, default='train.npz',
                    help='train.npz | train_aug.npz')
    ap.add_argument('--exp-name', type=str, required=True,
                    help='experiment name (used for ckpt dir + results JSON)')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    train_one_slot(
        npz_dir=Path(args.npz),
        ifaces=args.ifaces,
        train_file=args.train_file,
        exp_name=args.exp_name,
        epochs=args.epochs,
        gpu=args.gpu,
    )


if __name__ == '__main__':
    main()