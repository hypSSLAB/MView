#!/usr/bin/env python3
"""Build OW unmonitored NPZ (5000 sessions, paired across N ifaces, 8:1:1 split).

General-purpose builder for Wireguard/Tor OW datasets.
Each NPZ has:
  X: (N, 60, 4)   - 1Hz × 60-bin × {up_bytes, dn_bytes, up_pkts, dn_pkts}
  y: (N,) = 100   - OW unmonitored label
  sites: (N,)     - session UID strings 'dev::site::base'
All ifaces share the SAME session order (alignment preserved) → paired NPZ.

Usage:
  # 1. Wireguard OW WiFi  (3 ifaces: wlan0+tun0+total)
  python build_ow_unmonitored_npz.py \\
      --preset wireguard_wifi \\
      --out ./data/raw/wireguard_OW/wifi

  # 2. Wireguard OW Cellular  (3 ifaces: rmnet+tun0+total)
  python build_ow_unmonitored_npz.py \\
      --preset wireguard_cellular \\
      --out ./data/raw/wireguard_OW/cellular

  # 3. Tor PRE_OW 4G / 5G / WIFI  (4 ifaces with lo)
  python build_ow_unmonitored_npz.py --preset tor_4g    --out .//Dataset/Orig/Tor/PRE_OW/4g
  python build_ow_unmonitored_npz.py --preset tor_5g    --out .//Dataset/Orig/Tor/PRE_OW/5g
  python build_ow_unmonitored_npz.py --preset tor_wifi  --out .//Dataset/Orig/Tor/PRE_OW/wifi

  # Custom preset (override target count / seed)
  python build_ow_unmonitored_npz.py --preset wireguard_wifi --target 5000 --seed 42

Preset semantics:
  Each preset defines:
    - sources: list of (src_root, device_tag, network_dir)
    - output_ifaces: list of output iface keys (e.g., wlan0, tun0, total, lo)
    - iface_src_map: dict[device_tag → dict[out_ifc → src_ifc]] — per-device source fallback
      e.g. RM13/WIFI uses _dev variants because standard ifaces are empty
      e.g. A34 cellular uses 'mobile' instead of 'rmnet' (empty)

If actual valid paired sessions > target: random downsample (seed-fixed).
If actual < target: random replicate with replacement (same indices across ifaces).
"""
from __future__ import annotations
import argparse, csv
from io import StringIO
from pathlib import Path
import numpy as np


NUM_BINS_1S = 60
BIN_SEC = 1.0
OW_LABEL = 100
RAW_DIFF_COLS = ("upload_bytes_diff", "download_bytes_diff",
                 "upload_packets_diff", "download_packets_diff")


# ============================================================
# Presets — each defines all per-(device,network) iface mapping
# ============================================================
PRESETS = {
    # -------- Wireguard --------
    'wireguard_wifi': {
        'sources': [
            ('./data/raw/wireguard_OW/Xiaomi_1/Xiaomi_1', 'Xiaomi_1', 'wifi'),
            ('./data/raw/wireguard_OW/A34_1/A34_1',       'A34_1',    'wifi'),
        ],
        'output_ifaces': ['wlan0', 'tun0', 'total'],
        'iface_src_map': {
            'Xiaomi_1': {'wlan0': 'wlan0', 'tun0': 'tun0', 'total': 'total'},
            'A34_1':    {'wlan0': 'wlan0', 'tun0': 'tun0', 'total': 'total'},
        },
    },
    'wireguard_cellular': {
        'sources': [
            ('./data/raw/wireguard_OW/Pixel8_2/Pixel8_2', 'Pixel8_2', 'cellular'),
            ('./data/raw/wireguard_OW/A34_2/A34_2',       'A34_2',    '5g'),
        ],
        'output_ifaces': ['rmnet', 'tun0', 'total'],
        'iface_src_map': {
            # A34_2/5g: rmnet is empty → fallback to 'mobile' as phy
            'Pixel8_2': {'rmnet': 'rmnet',  'tun0': 'tun0', 'total': 'total'},
            'A34_2':    {'rmnet': 'mobile', 'tun0': 'tun0', 'total': 'total'},
        },
    },

    # -------- NonVPN OW Cellular 4G (rmnet + total) --------
    # A34_2/4g: rmnet has data, but per-session fallback to mobile if rmnet empty (use src_iface fallback list)
    'nonvpn_4g': {
        'sources': [
            ('./data/raw/NonVPN/OW/Cellular/extracted/A34_2', 'A34_2', '4g'),
        ],
        'output_ifaces': ['rmnet', 'total'],
        'iface_src_map': {
            # Per-session fallback: try rmnet first; if invalid, fall back to mobile.
            'A34_2': {'rmnet': ['rmnet', 'mobile'], 'total': 'total'},
        },
    },

    # -------- NonVPN OW Cellular (4G + 5G combined) --------
    # A34_2/4g  : rmnet has data (mobile fallback)
    # Xiaomi_1/5g: rmnet is empty everywhere → use 'mobile' as source for 'rmnet' output key
    'nonvpn_cellular': {
        'sources': [
            ('./data/raw/NonVPN/OW/Cellular/extracted/A34_2',           'A34_2',    '4g'),
            ('./data/raw/NonVPN/OW/Cellular/extracted_5g/Xiaomi_1',     'Xiaomi_1', 'NonVPN_5g'),
        ],
        'output_ifaces': ['rmnet', 'total'],
        'iface_src_map': {
            'A34_2':    {'rmnet': ['rmnet', 'mobile'], 'total': 'total'},
            'Xiaomi_1': {'rmnet': ['mobile', 'rmnet'], 'total': 'total'},  # mobile primary, rmnet just in case
        },
    },

    # -------- OpenVPN OW WiFi (wlan0 + tun0 + total) --------
    # A34_1/wifi: all 3 ifaces have data; _dev variants richer when standard sparse
    # Xiaomi_1/wifi: wlan0 + wlan0_dev BOTH empty → only tun0+total available, wlan0 sessions skip
    'openvpn_wifi': {
        'sources': [
            ('./data/raw/OpenVPN/OW/Wi-Fi/extracted/A34_1',    'A34_1',    'wifi'),
            ('./data/raw/OpenVPN/OW/Wi-Fi/extracted/Xiaomi_1', 'Xiaomi_1', 'wifi'),
        ],
        'output_ifaces': ['wlan0', 'tun0', 'total'],
        'iface_src_map': {
            'A34_1':    {'wlan0': ['wlan0', 'wlan0_dev'], 'tun0': ['tun0', 'tun0_dev'], 'total': ['total', 'total_dev']},
            'Xiaomi_1': {'wlan0': ['wlan0', 'wlan0_dev'], 'tun0': ['tun0', 'tun0_dev'], 'total': ['total', 'total_dev']},
        },
    },

    # -------- OpenVPN OW Cellular 4G (rmnet + tun0 + total) --------
    # Source: A34_2_OW_4G/5g/ inside open_4g.zip (zip name says 4g but inner dir labelled 5g; treat as 4G OW).
    # rmnet empty → fallback priority rmnet → rmnet_dev → mobile (rmnet_dev has most data ~5× mobile)
    'openvpn_4g': {
        'sources': [
            ('./data/raw/OpenVPN/OW/Cellular/extracted/A34_2_OW_4G', 'A34_2_OW_4G', '5g'),
        ],
        'output_ifaces': ['rmnet', 'tun0', 'total'],
        'iface_src_map': {
            'A34_2_OW_4G': {'rmnet': ['rmnet', 'rmnet_dev', 'mobile'], 'tun0': 'tun0', 'total': 'total'},
        },
    },

    # -------- TorBrowser OW Cellular (rmnet + total) --------
    'torbrowser_cellular': {
        'sources': [
            ('./data/raw/TorBrowser/OW/Cellular/extracted/Pixel8_1', 'Pixel8_1', '4g'),
            ('./data/raw/TorBrowser/OW/Cellular/extracted/Xiaomi_1', 'Xiaomi_1', '4g'),
        ],
        'output_ifaces': ['rmnet', 'total'],
        'iface_src_map': {
            'Pixel8_1': {'rmnet': ['rmnet', 'mobile'], 'total': 'total'},
            'Xiaomi_1': {'rmnet': ['rmnet', 'mobile'], 'total': 'total'},
        },
    },

    # -------- TorBrowser OW WiFi (total + wlan0 only) --------
    'torbrowser_wifi': {
        'sources': [
            ('./data/raw/TorBrowser/OW/Wi-Fi/extracted/A34_1',    'A34_1',    'wifi'),
            ('./data/raw/TorBrowser/OW/Wi-Fi/extracted/Pixel8_2', 'Pixel8_2', 'wifi'),
        ],
        'output_ifaces': ['wlan0', 'total'],
        'iface_src_map': {
            'A34_1':    {'wlan0': 'wlan0', 'total': 'total'},
            'Pixel8_2': {'wlan0': 'wlan0', 'total': 'total'},
        },
    },

    # -------- Tor PRE_OW (with lo iface) --------
    'tor_4g': {
        'sources': [
            ('.//Dataset/Orig/Tor/PRE_OW/A34',    'A34',    '4G'),
            ('.//Dataset/Orig/Tor/PRE_OW/PIXEL8', 'PIXEL8', '4G'),
            ('.//Dataset/Orig/Tor/PRE_OW/RM13',   'RM13',   '4G'),
        ],
        'output_ifaces': ['rmnet', 'tun0', 'total', 'lo'],
        'iface_src_map': {
            # A34/4G: rmnet empty → use mobile
            'A34':    {'rmnet': 'mobile', 'tun0': 'tun0', 'total': 'total', 'lo': 'lo'},
            'PIXEL8': {'rmnet': 'rmnet',  'tun0': 'tun0', 'total': 'total', 'lo': 'lo'},
            'RM13':   {'rmnet': 'rmnet',  'tun0': 'tun0', 'total': 'total', 'lo': 'lo'},
        },
    },
    'tor_5g': {
        'sources': [
            ('.//Dataset/Orig/Tor/PRE_OW/A34',    'A34',    '5G'),
            ('.//Dataset/Orig/Tor/PRE_OW/PIXEL8', 'PIXEL8', '5G'),
            ('.//Dataset/Orig/Tor/PRE_OW/RM13',   'RM13',   '5G'),
        ],
        'output_ifaces': ['rmnet', 'tun0', 'total', 'lo'],
        'iface_src_map': {
            'A34':    {'rmnet': 'mobile', 'tun0': 'tun0', 'total': 'total', 'lo': 'lo'},
            'PIXEL8': {'rmnet': 'rmnet',  'tun0': 'tun0', 'total': 'total', 'lo': 'lo'},
            'RM13':   {'rmnet': 'rmnet',  'tun0': 'tun0', 'total': 'total', 'lo': 'lo'},
        },
    },
    'tor_wifi': {
        'sources': [
            ('.//Dataset/Orig/Tor/PRE_OW/A34',    'A34',    'WIFI'),
            ('.//Dataset/Orig/Tor/PRE_OW/PIXEL8', 'PIXEL8', 'WIFI'),
            ('.//Dataset/Orig/Tor/PRE_OW/RM13',   'RM13',   'WIFI'),
        ],
        'output_ifaces': ['wlan0', 'tun0', 'total', 'lo'],
        'iface_src_map': {
            'A34':    {'wlan0': 'wlan0',     'tun0': 'tun0',     'total': 'total',     'lo': 'lo'},
            'PIXEL8': {'wlan0': 'wlan0',     'tun0': 'tun0',     'total': 'total',     'lo': 'lo'},
            # RM13/WIFI: all standard ifaces are empty → use _dev variants
            'RM13':   {'wlan0': 'wlan0_dev', 'tun0': 'tun0_dev', 'total': 'total_dev', 'lo': 'lo_dev'},
        },
    },
}


# ============================================================
# CSV → (60,4) 1Hz binning (canonical pipeline)
# ============================================================
def load_raw_csv_1s(path):
    """Read CSV and floor-bin to (60, 4) at 1Hz. Returns None if invalid."""
    rows = []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fp:
            content = fp.read().replace("\x00", "")
        reader = csv.DictReader(StringIO(content))
        cols = set(reader.fieldnames or [])
        if "timestamp_s" not in cols or not all(c in cols for c in RAW_DIFF_COLS):
            return None
        for r in reader:
            try:
                ts = float(r["timestamp_s"])
                d = tuple(float(r[c]) for c in RAW_DIFF_COLS)
            except (TypeError, ValueError):
                continue
            rows.append((ts, *d))
    except (OSError, csv.Error):
        return None
    if len(rows) < 2: return None
    arr = np.asarray(rows, dtype=np.float64)
    nz = arr[:, 1:].sum(axis=1) > 0
    if not nz.any(): return None
    arr = arr[int(np.argmax(nz)):]
    if len(arr) < 2: return None
    rel_t = np.clip(arr[:, 0] - arr[0, 0], 0.0, None)
    bin_idx = np.floor(rel_t / BIN_SEC).astype(np.int64)
    keep = bin_idx < NUM_BINS_1S
    arr = arr[keep]; bin_idx = bin_idx[keep]
    if len(arr) == 0: return None
    feats = np.zeros((NUM_BINS_1S, 4), dtype=np.float32)
    vals = arr[:, 1:].astype(np.float32)
    for ch in range(4):
        feats[:, ch] = np.bincount(bin_idx, weights=vals[:, ch], minlength=NUM_BINS_1S)[:NUM_BINS_1S]
    if int((feats.sum(axis=1) > 0).sum()) < 3: return None
    return feats


def session_id(filename, iface):
    suf = f"_{iface}.csv"
    return filename[:-len(suf)] if filename.endswith(suf) else filename


# ============================================================
# Main pipeline
# ============================================================
def build(preset_name: str, out_base: Path, target_n: int, seed: int):
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Choices: {list(PRESETS.keys())}")
    cfg = PRESETS[preset_name]
    ifaces = cfg['output_ifaces']
    src_map_by_dev = cfg['iface_src_map']
    sources = cfg['sources']

    print(f"=== Preset: {preset_name} ===")
    print(f"  Output ifaces: {ifaces}")
    print(f"  Target: {target_n} sessions, 8:1:1 split")
    print(f"  Seed: {seed}")
    print(f"  Output: {out_base}\n")

    np.random.seed(seed)

    # 1) Collect candidate sessions (where all ifaces have matching CSV)
    print("Scanning sources for paired sessions...")
    sess = []
    for src_root, dev_tag, net_dir in sources:
        src_path = Path(src_root) / net_dir
        if not src_path.exists():
            print(f"  ⚠ skip missing {src_path}"); continue
        src_map = src_map_by_dev[dev_tag]
        # Flatten: each value can be str or list; need union of bases across all candidate ifaces
        for site_dir in sorted(src_path.iterdir()):
            if not site_dir.is_dir(): continue
            site = site_dir.name
            per_ifc_union = {}; ok = True
            for out_ifc, src in src_map.items():
                src_options = [src] if isinstance(src, str) else list(src)
                union = set()
                for ifc in src_options:
                    d = site_dir / ifc
                    if d.is_dir():
                        for f in d.iterdir():
                            if f.suffix == '.csv':
                                union.add(session_id(f.name, ifc))
                if not union:
                    ok = False; break
                per_ifc_union[out_ifc] = union
            if not ok: continue
            common = set.intersection(*per_ifc_union.values())
            for base in sorted(common):
                sess.append((f"{dev_tag}::{site}::{base}", dev_tag, src_path, site, base))
        src_repr = ' + '.join(
            f"{o}←{s if isinstance(s,str) else '|'.join(s)}"
            for o, s in src_map.items()
        )
        print(f"  {dev_tag}/{net_dir} ({src_repr}): cumulative paired = {len(sess)}")

    print(f"\nTotal paired sessions: {len(sess)}\n")

    # 2) Bin all CSVs (skip if any iface invalid)
    print("Loading & binning CSVs...")
    Xs = {ifc: [] for ifc in ifaces}
    sids = []; skipped = 0
    for i, (uid, dev_tag, src_path, site, base) in enumerate(sess):
        src_map = src_map_by_dev[dev_tag]
        feats = {}; ok = True
        for out_ifc, src in src_map.items():
            src_options = [src] if isinstance(src, str) else list(src)
            loaded = None
            for src_ifc in src_options:
                p = src_path / site / src_ifc / f"{base}_{src_ifc}.csv"
                if not p.exists(): continue
                loaded = load_raw_csv_1s(p)
                if loaded is not None: break
            if loaded is None: ok = False; break
            feats[out_ifc] = loaded
        if not ok:
            skipped += 1; continue
        for ifc in ifaces:
            Xs[ifc].append(feats[ifc])
        sids.append(uid)
        if (i+1) % 1000 == 0:
            print(f"  ... {i+1}/{len(sess)}  valid={len(sids)} skip={skipped}")
    n_valid = len(sids)
    print(f"\nValid paired: {n_valid} / {len(sess)} (skipped {skipped})\n")

    if n_valid == 0:
        raise RuntimeError("No valid paired sessions found.")

    # 3) Resize to target_n (same indices across ifaces)
    rng = np.random.RandomState(seed)
    if n_valid >= target_n:
        idx = rng.permutation(n_valid)[:target_n]
        print(f"Downsampling: {n_valid} → {target_n}")
    else:
        extra = target_n - n_valid
        rep_idx = rng.choice(n_valid, size=extra, replace=True)
        idx = np.concatenate([np.arange(n_valid), rep_idx])
        rng.shuffle(idx)
        print(f"Upsampling: {n_valid} → {target_n} (replicated {extra})")

    X_final = {ifc: np.stack([Xs[ifc][i] for i in idx], axis=0).astype(np.float32) for ifc in ifaces}
    sids_final = [sids[i] for i in idx]

    # 4) 8:1:1 split via single permutation
    perm = rng.permutation(target_n)
    n_tr = int(target_n * 0.8); n_va = int(target_n * 0.1); n_te = target_n - n_tr - n_va
    idx_tr, idx_va, idx_te = perm[:n_tr], perm[n_tr:n_tr+n_va], perm[n_tr+n_va:]
    print(f"Split: train={n_tr}  valid={n_va}  test={n_te}\n")

    y = np.full(target_n, OW_LABEL, dtype=np.int64)

    for ifc in ifaces:
        out_dir = out_base / ifc
        out_dir.mkdir(parents=True, exist_ok=True)
        X = X_final[ifc]
        for split, sel in [('train', idx_tr), ('valid', idx_va), ('test', idx_te)]:
            sites_arr = np.array([sids_final[i] for i in sel], dtype=object)
            np.savez_compressed(out_dir / f'{split}.npz', X=X[sel], y=y[sel], sites=sites_arr)
            print(f"  [{ifc}] {split}.npz  X={X[sel].shape}")
    print(f"\nDone → {out_base}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', required=True, choices=list(PRESETS.keys()))
    ap.add_argument('--out', required=True, help='output base dir (subdirs per iface)')
    ap.add_argument('--target', type=int, default=5000, help='target session count (default 5000)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    build(args.preset, Path(args.out), args.target, args.seed)


if __name__ == '__main__':
    main()
