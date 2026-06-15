#!/usr/bin/env python3
"""Run 4 baselines on combined NPZ + evaluate per-protocol test split.

Combined train: NV+TB or WG+OVPN merged via concat.
Test eval: split combined test into per-protocol subsets, report F1 each.

Args:
  --combined-npz: dir of combined train/valid/test for the iface
  --proto-tests: "proto1:test_npz_path1,proto2:test_npz_path2"
  --exp-name, --gpu, --train-file
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path('.//STATS_WF/results_per_proto')
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED, EPOCHS, BS, LR = 42, 100, 128, 1e-3


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, n_filters=32, bottleneck_size=32, kernels=(5, 11, 23)):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck_size, 1, bias=False) if in_ch > 1 else nn.Identity()
        self.bn0 = nn.BatchNorm1d(bottleneck_size if in_ch > 1 else in_ch)
        bn_in = bottleneck_size if in_ch > 1 else in_ch
        self.convs = nn.ModuleList([nn.Conv1d(bn_in, n_filters, k, padding=k // 2, bias=False) for k in kernels])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.maxconv = nn.Conv1d(in_ch, n_filters, 1, bias=False)
        self.bn = nn.BatchNorm1d(n_filters * (len(kernels) + 1))
    def forward(self, x):
        x_in = x; x = self.bottleneck(x); x = F.relu(self.bn0(x))
        outs = [c(x) for c in self.convs]
        outs.append(self.maxconv(self.maxpool(x_in)))
        return F.relu(self.bn(torch.cat(outs, dim=1)))


class InceptionTime(nn.Module):
    def __init__(self, in_ch, n_classes, n_blocks=2):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(n_blocks):
            layers.append(InceptionBlock(ch)); ch = 32 * 4
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, n_classes))
    def forward(self, x): return self.head(self.blocks(x))


# 1Hz × 60bin NPZ slice. channels: [up_bytes, dn_bytes, up_pkts, dn_pkts]
# Mischief: 10s × 4ch rate (raw bin)
def feat_mischief(X, n_bins=10):  return X[:, :n_bins, :].transpose(0, 2, 1)


# WiSec'16: 10s × 2ch cumsum bytes (tcp snd/rcv)
def feat_wisec(X, n_bins=10):
    cs_up = np.cumsum(X[:, :, 0], axis=1)
    cs_dn = np.cumsum(X[:, :, 1], axis=1)
    return np.stack([cs_up, cs_dn], axis=1)[:, :, :n_bins]


# ScanDroid: 10s × 1ch cumsum tx bytes (TotalTxBytes)
def feat_scandroid(X, n_bins=10):
    return np.cumsum(X[:, :, 0], axis=1)[:, :n_bins]


# ProCharvester: 6s × 1ch cumsum rx packet counter (uses 6 bins from 1Hz, paper is 6.5s)
def feat_procharvest(X, n_bins=6):
    return np.cumsum(X[:, :, 3], axis=1)[:, :n_bins]


def metrics(y, p):
    return dict(
        acc=round(accuracy_score(y, p), 4),
        prec=round(precision_score(y, p, average='macro', zero_division=0), 4),
        rec=round(recall_score(y, p, average='macro', zero_division=0), 4),
        f1=round(f1_score(y, p, average='macro', zero_division=0), 4),
    )


def train_inception(Xtr, ytr, Xva, yva, n_classes, gpu, name, epochs=EPOCHS):
    torch.manual_seed(SEED); np.random.seed(SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionTime(Xtr.shape[1], n_classes).to(device)
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xtr.astype(np.float32)),
                                     torch.from_numpy(ytr.astype(np.int64))),
                       batch_size=BS, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()
    best_f1, best_state, pat = 0.0, None, 0
    Xva_t = torch.from_numpy(Xva.astype(np.float32))
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xva_t), BS):
                preds.append(model(Xva_t[i:i+BS].to(device)).argmax(1).cpu().numpy())
        f1 = round(f1_score(yva, np.concatenate(preds), average='macro', zero_division=0), 4)
        if f1 > best_f1:
            best_f1, pat = f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: pat += 1
        if ep % 10 == 0 or pat == 0 or ep == 1:
            print(f"    [{name}] ep={ep}  val_f1={f1:.4f}  best={best_f1:.4f}  pat={pat}", flush=True)
        if pat >= 30: break
    if best_state: model.load_state_dict(best_state)
    return model, time.time() - t0


def predict_model(model, X, device, bs=BS):
    model.eval()
    Xt = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xt), bs):
            preds.append(model(Xt[i:i+bs].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--combined-npz', required=True)
    ap.add_argument('--proto-tests', required=True, help='"proto1:path1,proto2:path2"')
    ap.add_argument('--exp-name', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--train-file', default='train.npz')
    args = ap.parse_args()

    P = Path(args.combined_npz)
    tr = np.load(P / args.train_file, allow_pickle=True)
    va = np.load(P / 'valid.npz', allow_pickle=True)
    Xtr, ytr = tr['X'][:, :, :4].astype(np.float32), tr['y']
    Xva, yva = va['X'][:, :, :4].astype(np.float32), va['y']
    n_classes = int(max(ytr.max(), yva.max())) + 1

    # Parse proto tests
    proto_tests = {}
    for kv in args.proto_tests.split(','):
        proto, path = kv.split(':')
        d = np.load(Path(path), allow_pickle=True)
        proto_tests[proto] = (d['X'][:, :, :4].astype(np.float32), d['y'].astype(np.int64))

    print(f"\n{'='*60}\n  {args.exp_name}\n  combined={P}  train={args.train_file}\n  Xtr={Xtr.shape}  classes={n_classes}\n  protos: {[(p, x[0].shape) for p,x in proto_tests.items()]}\n{'='*60}", flush=True)

    results = {'name': args.exp_name, 'combined_npz': str(P), 'train_file': args.train_file,
               'n_train': int(len(ytr)), 'n_classes': n_classes, 'baselines': {}}

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mischief
    print("\n--- Mischief ---")
    Xtm, Xvm = feat_mischief(Xtr, 10), feat_mischief(Xva, 10)
    m_model, m_time = train_inception(Xtm, ytr, Xvm, yva, n_classes, args.gpu, 'mischief')
    results['baselines']['mischief'] = {'time_min': round(m_time/60, 2), 'per_proto': {}}
    for proto, (Xte, yte) in proto_tests.items():
        Xtm_te = feat_mischief(Xte, 10)
        yp = predict_model(m_model, Xtm_te, device)
        results['baselines']['mischief']['per_proto'][proto] = metrics(yte, yp)
        print(f"    [mischief] {proto:12s}  F1={results['baselines']['mischief']['per_proto'][proto]['f1']}", flush=True)

    # WiSec'16: 10s × 2ch cumsum bytes
    print("\n--- WiSec'16 ---")
    Xtw, Xvw = feat_wisec(Xtr, 10), feat_wisec(Xva, 10)
    w_model, w_time = train_inception(Xtw, ytr, Xvw, yva, n_classes, args.gpu, 'wisec16')
    results['baselines']['wisec16'] = {'time_min': round(w_time/60, 2), 'per_proto': {}}
    for proto, (Xte, yte) in proto_tests.items():
        Xtw_te = feat_wisec(Xte, 10)
        yp = predict_model(w_model, Xtw_te, device)
        results['baselines']['wisec16']['per_proto'][proto] = metrics(yte, yp)
        print(f"    [wisec16] {proto:12s}  F1={results['baselines']['wisec16']['per_proto'][proto]['f1']}", flush=True)

    # ProCharvester: 6s × cumsum(rx_pkts), kNN k=5
    print("\n--- ProCharvester (kNN k=5) ---")
    Xtp_proch = feat_procharvest(Xtr, 6)
    sc_p = StandardScaler().fit(Xtp_proch)
    knn5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1).fit(sc_p.transform(Xtp_proch), ytr)
    results['baselines']['procharvest'] = {'per_proto': {}}
    for proto, (Xte, yte) in proto_tests.items():
        Xte_s = sc_p.transform(feat_procharvest(Xte, 6))
        yp5 = knn5.predict(Xte_s)
        results['baselines']['procharvest']['per_proto'][proto] = metrics(yte, yp5)
        print(f"    [procharvest] {proto:12s}  F1={results['baselines']['procharvest']['per_proto'][proto]['f1']}", flush=True)

    # ScanDroid: 10s × cumsum(tx_bytes), 1-NN
    print("\n--- ScanDroid (1-NN) ---")
    Xtp_scan = feat_scandroid(Xtr, 10)
    sc_s = StandardScaler().fit(Xtp_scan)
    knn1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1).fit(sc_s.transform(Xtp_scan), ytr)
    results['baselines']['scandroid'] = {'per_proto': {}}
    for proto, (Xte, yte) in proto_tests.items():
        Xte_s = sc_s.transform(feat_scandroid(Xte, 10))
        yp1 = knn1.predict(Xte_s)
        results['baselines']['scandroid']['per_proto'][proto] = metrics(yte, yp1)
        print(f"    [scandroid] {proto:12s}  F1={results['baselines']['scandroid']['per_proto'][proto]['f1']}", flush=True)

    out = OUT_DIR / f'{args.exp_name}.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\n  saved → {out}")


if __name__ == '__main__':
    main()
