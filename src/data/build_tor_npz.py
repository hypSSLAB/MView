#!/usr/bin/env python3
"""
Build Tor 4G/5G NPZ for `run_all_experiments.py:train_dataset()`.

Four input modes — all produce the same output layout:
    OUT_DIR/<iface>/{train,valid,test}.npz   (X: (N, 60, 4) float32, y: int64)

  --from-raw-csv  PIXEL8_4G_CW / PIXEL_5G_CW-style *raw* CSV root:
                  <root>/<site>/<iface>/*.csv, header =
                      timestamp_s, time_diff_s, upload_bytes_diff,
                      download_bytes_diff, upload_packets_diff,
                      download_packets_diff, flag, cumulative_*
                  Binned directly to 1s × 60 via timestamp floor (same logic as
                  run_all_experiments.load_csv_to_features). **Preferred path**
                  because no 0.5s→1s downsample is needed.

  --from-bins     PIXEL_5G_CW_BINS-style pre-binned CSV root:
                  <root>/<site>/<iface>/*.csv  with 120 rows of 0.5s bins.
                  Header columns must include:
                      upload_bytes, download_bytes, upload_packets, download_packets
                  Rows are summed adjacent-pairs to 1s × 60.

  --from-npz         Re-package an existing 4-view NPZ store.
                     Layout: <root>/<iface>/{train,test}.npz with X: (N, 120, 4).
                     Carves stratified valid split out of train + downsamples.

  --from-npz-single  Single-view NPZ store with train.npz/test.npz directly inside.

Usage:
    # Pixel8 4G — raw CSV (has tun0/rmnet/total/lo across 100 sites)
    python build_tor_npz.py --from-raw-csv \\
        ./data/raw/PIXEL8_4G_CW \\
        --out .//Dataset/NPZ/tor/Pixel8_4G_CW \\
        --ifaces tun0 rmnet total lo

    # Pixel8 5G — raw CSV (uniform 76 sessions/site across all 4 ifaces)
    python build_tor_npz.py --from-raw-csv \\
        ./data/raw/PIXEL_5G_CW \\
        --out .//Dataset/NPZ/tor/Pixel8_5G_CW \\
        --ifaces tun0 rmnet total lo
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW_COLS = ("upload_bytes", "download_bytes", "upload_packets", "download_packets")
# Raw CSV uses *_diff column names + a timestamp_s column
RAW_DIFF_COLS = ("upload_bytes_diff", "download_bytes_diff",
                 "upload_packets_diff", "download_packets_diff")
SEED = 42
NUM_BINS_1S = 60
BIN_SEC = 1.0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def downsample_05s_to_1s(X: np.ndarray) -> np.ndarray:
    """Sum adjacent pairs along the time axis: (N, 2T, F) -> (N, T, F)."""
    n, t, f = X.shape
    if t % 2 != 0:
        # Drop the trailing odd bin
        X = X[:, : t - 1, :]
        t -= 1
    return X.reshape(n, t // 2, 2, f).sum(axis=2).astype(np.float32)


def stratified_split(y: np.ndarray, valid_frac: float, rng: random.Random):
    """Per-class split: returns (train_idx, valid_idx)."""
    by_cls = defaultdict(list)
    for i, c in enumerate(y.tolist()):
        by_cls[c].append(i)
    train_idx, valid_idx = [], []
    for c, idxs in by_cls.items():
        rng.shuffle(idxs)
        nv = max(1, int(round(len(idxs) * valid_frac)))
        if nv >= len(idxs):
            nv = max(1, len(idxs) // 10) if len(idxs) > 1 else 0
        valid_idx.extend(idxs[:nv])
        train_idx.extend(idxs[nv:])
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


def write_split(out_dir: Path, X: np.ndarray, y: np.ndarray, sites: np.ndarray, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz",
                        X=X.astype(np.float32),
                        y=y.astype(np.int64),
                        sites=sites)
    print(f"      {name}: X={X.shape} y={y.shape}")


# ---------------------------------------------------------------------------
# Mode 1: from BINS CSV root
# ---------------------------------------------------------------------------

def load_csv_4ch(path: Path, rows_required: int) -> np.ndarray | None:
    try:
        with open(path, newline="") as fp:
            reader = csv.DictReader(fp)
            cols = reader.fieldnames or []
            if not all(c in cols for c in RAW_COLS):
                return None
            rows = []
            for r in reader:
                try:
                    rows.append([float(r[c]) for c in RAW_COLS])
                except ValueError:
                    return None
    except OSError:
        return None
    if len(rows) < rows_required:
        return None
    return np.asarray(rows[:rows_required], dtype=np.float32)


def load_raw_csv_1s(path: Path, num_bins: int = NUM_BINS_1S,
                    bin_sec: float = BIN_SEC) -> np.ndarray | None:
    """Bin a raw per-packet CSV (timestamp_s + *_diff cols) into (num_bins, 4).

    Mirrors `run_all_experiments.load_csv_to_features` but in pure numpy/csv so
    build_tor_npz.py stays dependency-light.
    """
    rows: list[tuple[float, float, float, float, float]] = []
    try:
        with open(path, newline="") as fp:
            reader = csv.DictReader(fp)
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
    except OSError:
        return None
    if len(rows) < 2:
        return None

    arr = np.asarray(rows, dtype=np.float64)      # (n, 5): ts + 4 diff cols
    feats_all = arr[:, 1:]                         # (n, 4)
    # Skip leading zero-rows (trace pre-roll)
    nz_mask = feats_all.sum(axis=1) > 0
    if not nz_mask.any():
        return None
    start = int(np.argmax(nz_mask))
    arr = arr[start:]
    if len(arr) < 2:
        return None

    rel_t = np.clip(arr[:, 0] - arr[0, 0], 0.0, None)
    bin_idx = np.floor(rel_t / bin_sec).astype(np.int64)
    keep = bin_idx < num_bins
    arr = arr[keep]
    bin_idx = bin_idx[keep]
    if len(arr) == 0:
        return None

    feats = np.zeros((num_bins, 4), dtype=np.float32)
    vals = arr[:, 1:].astype(np.float32)
    # Accumulate via bincount per channel (fast & vectorized)
    for ch in range(4):
        feats[:, ch] = np.bincount(bin_idx, weights=vals[:, ch], minlength=num_bins)[:num_bins]

    # Sanity: require ≥3 non-empty bins
    if int((feats.sum(axis=1) > 0).sum()) < 3:
        return None
    return feats


def build_from_raw_csv(raw_root: Path, out_root: Path, ifaces: list[str],
                       valid_frac: float, test_frac: float):
    """Build NPZ from per-packet raw CSV (timestamp_s + *_diff cols).

    No downsample — the loader bins directly to 1s × 60.
    Split is per-site stratified (session-level shuffle within each site).

    Sites that lack ANY of the requested interfaces are dropped so the
    resulting NPZ has one consistent class set across all views — otherwise
    `train_dataset` pulls a different n_classes per interface.
    """
    all_sites = sorted([d.name for d in raw_root.iterdir() if d.is_dir()])
    if not all_sites:
        raise SystemExit(f"no site dirs under {raw_root}")

    sites = []
    dropped = []
    need = set(ifaces)
    for s in all_sites:
        have = {d.name for d in (raw_root / s).iterdir() if d.is_dir()}
        if need.issubset(have):
            sites.append(s)
        else:
            dropped.append((s, sorted(need - have)))
    if dropped:
        print(f"[from-raw-csv] dropping {len(dropped)} sites missing required ifaces:")
        for s, miss in dropped[:10]:
            print(f"    {s}: missing {miss}")
        if len(dropped) > 10:
            print(f"    ... and {len(dropped)-10} more")

    print(f"[from-raw-csv] candidate sites={len(sites)} ifaces={ifaces}")

    # ---- Pass 1: load everything into memory, per (iface, site) ----
    # Also lets us filter out sites whose CSVs are all empty/invalid for any iface.
    per_iface_site_X: dict[str, dict[str, list[np.ndarray]]] = {
        iface: defaultdict(list) for iface in ifaces
    }
    skipped_csv: dict[str, int] = {iface: 0 for iface in ifaces}
    for iface in ifaces:
        for site in sites:
            iface_dir = raw_root / site / iface
            if not iface_dir.is_dir():
                continue
            for csv_path in sorted(iface_dir.glob("*.csv")):
                feat = load_raw_csv_1s(csv_path)
                if feat is None:
                    skipped_csv[iface] += 1
                    continue
                per_iface_site_X[iface][site].append(feat)

    # Keep only sites that have ≥1 valid sample across ALL requested ifaces.
    kept_sites = [
        s for s in sites
        if all(len(per_iface_site_X[iface].get(s, [])) > 0 for iface in ifaces)
    ]
    dropped_empty = [s for s in sites if s not in kept_sites]
    if dropped_empty:
        print(f"[from-raw-csv] post-parse drop {len(dropped_empty)} sites "
              f"with 0 valid samples in at least one iface:")
        for s in dropped_empty[:10]:
            per = {iface: len(per_iface_site_X[iface].get(s, [])) for iface in ifaces}
            print(f"    {s}: {per}")
        if len(dropped_empty) > 10:
            print(f"    ... and {len(dropped_empty)-10} more")

    site2id = {s: i for i, s in enumerate(kept_sites)}
    sites_arr = np.array(kept_sites, dtype=object)
    print(f"[from-raw-csv] final sites={len(kept_sites)}  (classes={len(kept_sites)})")

    # ---- Pass 2: stratified split + write NPZs ----
    rng = random.Random(SEED)
    for iface in ifaces:
        print(f"  [{iface}]  (skipped CSVs: {skipped_csv[iface]})")
        Xtr, ytr, Xva, yva, Xte, yte = [], [], [], [], [], []
        for site in kept_sites:
            feats = list(per_iface_site_X[iface][site])
            cls = site2id[site]
            rng.shuffle(feats)
            n = len(feats)
            n_test = max(1, int(round(n * test_frac)))
            n_val = max(1, int(round(n * valid_frac)))
            if n_test + n_val >= n:
                n_val = max(1, n // 10)
                n_test = max(1, n // 10)
            n_train = max(1, n - n_val - n_test)
            if n_train + n_val + n_test > n:
                # Fall back: at least one in train, rest split
                n_train = max(1, n - 2)
                n_val = min(1, n - n_train)
                n_test = n - n_train - n_val
            for f in feats[:n_train]:
                Xtr.append(f); ytr.append(cls)
            for f in feats[n_train:n_train + n_val]:
                Xva.append(f); yva.append(cls)
            for f in feats[n_train + n_val:]:
                Xte.append(f); yte.append(cls)

        Xtr_np = np.stack(Xtr).astype(np.float32)
        Xva_np = np.stack(Xva).astype(np.float32) if Xva else np.zeros((0, NUM_BINS_1S, 4), np.float32)
        Xte_np = np.stack(Xte).astype(np.float32) if Xte else np.zeros((0, NUM_BINS_1S, 4), np.float32)

        out_dir = out_root / iface
        write_split(out_dir, Xtr_np, np.asarray(ytr), sites_arr, "train")
        write_split(out_dir, Xva_np, np.asarray(yva), sites_arr, "valid")
        write_split(out_dir, Xte_np, np.asarray(yte), sites_arr, "test")
        print(f"    written → {out_dir}")


def build_from_bins(bins_root: Path, out_root: Path, ifaces: list[str],
                    rows_per_window: int, valid_frac: float, test_frac: float):
    sites = sorted([d.name for d in bins_root.iterdir() if d.is_dir()])
    if not sites:
        raise SystemExit(f"no site dirs under {bins_root}")
    site2id = {s: i for i, s in enumerate(sites)}
    sites_arr = np.array(sites, dtype=object)
    print(f"[from-bins] sites={len(sites)} ifaces={ifaces} rows={rows_per_window}")

    rng = random.Random(SEED)

    for iface in ifaces:
        print(f"  [{iface}]")
        per_site_X = defaultdict(list)
        skipped = 0
        for site in sites:
            iface_dir = bins_root / site / iface
            if not iface_dir.is_dir():
                continue
            for csv_path in sorted(iface_dir.glob("*.csv")):
                feat = load_csv_4ch(csv_path, rows_per_window)
                if feat is None:
                    skipped += 1
                    continue
                per_site_X[site].append(feat)

        if not per_site_X:
            print(f"    SKIP {iface}: no usable CSVs (skipped={skipped})")
            continue

        Xtr, ytr, Xva, yva, Xte, yte = [], [], [], [], [], []
        for site, feats in per_site_X.items():
            cls = site2id[site]
            rng.shuffle(feats)
            n = len(feats)
            n_test = max(1, int(round(n * test_frac)))
            n_val = max(1, int(round(n * valid_frac)))
            if n_test + n_val >= n:
                n_val = max(1, n // 10)
                n_test = max(1, n // 10)
            n_train = n - n_val - n_test
            for f in feats[:n_train]:
                Xtr.append(f); ytr.append(cls)
            for f in feats[n_train:n_train + n_val]:
                Xva.append(f); yva.append(cls)
            for f in feats[n_train + n_val:]:
                Xte.append(f); yte.append(cls)

        Xtr = downsample_05s_to_1s(np.stack(Xtr))
        Xva = downsample_05s_to_1s(np.stack(Xva))
        Xte = downsample_05s_to_1s(np.stack(Xte))

        out_dir = out_root / iface
        write_split(out_dir, Xtr, np.asarray(ytr), sites_arr, "train")
        write_split(out_dir, Xva, np.asarray(yva), sites_arr, "valid")
        write_split(out_dir, Xte, np.asarray(yte), sites_arr, "test")
        print(f"    written → {out_dir}  (skipped CSVs: {skipped})")


# ---------------------------------------------------------------------------
# Mode 2: from existing 4-view NPZ store (train.npz/test.npz, no valid)
# ---------------------------------------------------------------------------

def build_from_npz(src_root: Path, out_root: Path, ifaces: list[str], valid_frac: float):
    rng = random.Random(SEED)
    print(f"[from-npz] src={src_root} ifaces={ifaces}")
    for iface in ifaces:
        src_iface = src_root / iface
        if not (src_iface / "train.npz").exists():
            print(f"  SKIP {iface}: train.npz missing")
            continue
        tr = np.load(src_iface / "train.npz", allow_pickle=True)
        te = np.load(src_iface / "test.npz", allow_pickle=True)
        Xtr_full = tr["X"][:, :, :4].astype(np.float32)
        ytr_full = tr["y"].astype(np.int64)
        Xte = te["X"][:, :, :4].astype(np.float32)
        yte = te["y"].astype(np.int64)

        # Recover sites array if present, else fabricate from site2id
        if "sites" in tr.files:
            sites_arr = tr["sites"]
        elif "site2id" in tr.files:
            s2i = tr["site2id"]
            sites_arr = np.array([s2i[k][0] if hasattr(s2i[k], "__getitem__") else k
                                  for k in range(len(s2i))], dtype=object)
        else:
            sites_arr = np.array([f"site_{i}" for i in range(int(ytr_full.max()) + 1)],
                                 dtype=object)

        tr_idx, va_idx = stratified_split(ytr_full, valid_frac, rng)
        Xtr = downsample_05s_to_1s(Xtr_full[tr_idx])
        Xva = downsample_05s_to_1s(Xtr_full[va_idx])
        Xte_ds = downsample_05s_to_1s(Xte)
        ytr = ytr_full[tr_idx]
        yva = ytr_full[va_idx]

        out_dir = out_root / iface
        print(f"  [{iface}]")
        write_split(out_dir, Xtr, ytr, sites_arr, "train")
        write_split(out_dir, Xva, yva, sites_arr, "valid")
        write_split(out_dir, Xte_ds, yte, sites_arr, "test")
        print(f"    written → {out_dir}")


# ---------------------------------------------------------------------------
# Mode 3: from single-view NPZ (e.g. CLOSEDWORLD/A15_4G/{train,test}.npz)
# ---------------------------------------------------------------------------

def build_from_npz_single(src_root: Path, out_root: Path, iface_name: str,
                          valid_frac: float):
    rng = random.Random(SEED)
    tr = np.load(src_root / "train.npz", allow_pickle=True)
    te = np.load(src_root / "test.npz", allow_pickle=True)
    Xtr_full = tr["X"][:, :, :4].astype(np.float32)
    ytr_full = tr["y"].astype(np.int64)
    Xte = te["X"][:, :, :4].astype(np.float32)
    yte = te["y"].astype(np.int64)
    sites_arr = tr["sites"] if "sites" in tr.files else np.array(
        [f"site_{i}" for i in range(int(ytr_full.max()) + 1)], dtype=object)

    tr_idx, va_idx = stratified_split(ytr_full, valid_frac, rng)
    Xtr = downsample_05s_to_1s(Xtr_full[tr_idx])
    Xva = downsample_05s_to_1s(Xtr_full[va_idx])
    Xte_ds = downsample_05s_to_1s(Xte)
    ytr = ytr_full[tr_idx]
    yva = ytr_full[va_idx]

    out_dir = out_root / iface_name
    print(f"[from-npz-single] iface={iface_name}")
    write_split(out_dir, Xtr, ytr, sites_arr, "train")
    write_split(out_dir, Xva, yva, sites_arr, "valid")
    write_split(out_dir, Xte_ds, yte, sites_arr, "test")
    print(f"  written → {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-raw-csv", type=Path, metavar="DIR",
                   help="root of <site>/<iface>/*.csv per-packet raw CSVs "
                        "(timestamp_s + *_diff cols); binned to 1s×60 directly")
    g.add_argument("--from-bins", type=Path, metavar="DIR",
                   help="root of <site>/<iface>/*.csv binned data (0.5s × 120)")
    g.add_argument("--from-npz", type=Path, metavar="DIR",
                   help="root of <iface>/{train,test}.npz multi-view store")
    g.add_argument("--from-npz-single", type=Path, metavar="DIR",
                   help="single-view store with {train,test}.npz directly inside")

    ap.add_argument("--out", type=Path, required=True, metavar="DIR")
    ap.add_argument("--ifaces", nargs="+", default=["tun0", "rmnet", "total", "lo"],
                    help="(bins/npz modes) interface names to materialize")
    ap.add_argument("--iface-name", default="tun0",
                    help="(npz-single mode) iface label to use under OUT")
    ap.add_argument("--valid-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10,
                    help="(bins mode only) per-site test fraction")
    ap.add_argument("--rows-per-window", type=int, default=120,
                    help="(bins mode) CSV rows kept per session before downsample")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.from_raw_csv:
        build_from_raw_csv(args.from_raw_csv, args.out, args.ifaces,
                           args.valid_frac, args.test_frac)
    elif args.from_bins:
        build_from_bins(args.from_bins, args.out, args.ifaces,
                        args.rows_per_window, args.valid_frac, args.test_frac)
    elif args.from_npz:
        build_from_npz(args.from_npz, args.out, args.ifaces, args.valid_frac)
    else:
        build_from_npz_single(args.from_npz_single, args.out,
                              args.iface_name, args.valid_frac)

    print(f"\n✓ done → {args.out}")


if __name__ == "__main__":
    main()
