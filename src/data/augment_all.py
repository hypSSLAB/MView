#!/usr/bin/env python3
"""augment_all.py — single-file data augmentation runner for MWF.

Self-contained augmentation pipeline. Reads a `train.npz` (X: N×T×4 raw) and
writes a `train_aug.npz` (X: N·(1+aug_copies) × T × 7 augmented + cumulative
features).

Augmentation primitives (applied together in ONE pass per augmented copy)
------------------------------------------------------------------------
    1. Jitter             — per-element Gaussian noise proportional to
                              per-feature standard deviation.
    2. Magnitude-warping  — smooth time-varying scaling curve (cubic-spline
                              with random knots) that modulates all features.
    3. Time-warping       — non-linear time-axis re-mapping driven by another
                              cubic-spline curve.

After augmentation, `transform_type4` appends 3 cumulative-byte channels
(cum_up_bytes, cum_dn_bytes, cum_total_bytes), producing the final 7-channel
representation.

Two strategies
--------------
    uniform  — every site receives the same number of augmented copies.
    adaptive — per-site copies scale inversely with sample count (WireGuard:
               small sites get more copies).

Usage
-----
    # Augment every slot under Dataset/NPZ/{Tor, Wireguard, OpenVPN, TorBrowser}/CW
    python augment_all.py

    # Single slot
    python augment_all.py --proto Tor --device PIXEL8 --network WIFI

    # Custom config override
    python augment_all.py --proto Wireguard --aug-copies 8 --seed 7

    # Run on an arbitrary NPZ slot
    python augment_all.py --slot /path/to/NPZ/slot --config wireguard --strategy uniform

    # Overwrite existing train_aug.npz
    python augment_all.py --force
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


# =================================================================
# Paths
# =================================================================
DATASET_ROOT = Path(".//Dataset")
NPZ_ROOT = DATASET_ROOT / "NPZ"


# =================================================================
# Per-protocol augmentation configs
# (aug_copies counts augmented samples directly — each copy already
# applies jitter + magnitude-warping + time-warping together.)
# =================================================================
PROTOCOL_CONFIGS = {
    "normal": {
        "strategy": "uniform",
        "aug_copies": 6,             # 1 + 6 = 7×
        "jitter_sigma": 0.02,        # direct HTTP(S), no tunnel → very light noise
        "mag_sigma": 0.12,           # site-level magnitude variance is small
        "warp_sigma": 0.10,          # native timing → mild warp
    },
    "torbrowser": {
        "strategy": "uniform",
        "aug_copies": 4,             # 1 + 4 = 5×
        "jitter_sigma": 0.01,        # SOCKS-direct deterministic → minimal jitter
        "mag_sigma": 0.10,
        "warp_sigma": 0.05,          # active bins stable → minimal time warp
    },
    "openvpn": {
        "strategy": "uniform",
        "aug_copies": 6,             # 1 + 6 = 7×
        "jitter_sigma": 0.04,        # TLS overhead → slightly stronger noise
        "mag_sigma": 0.18,           # TLS record layer adds magnitude variation
        "warp_sigma": 0.15,
    },
    "wireguard": {
        "strategy": "adaptive",
        "max_copies": 12,            # small sites: up to 1+12 = 13×
        "min_copies": 2,             # large sites: minimum 1+2 = 3×
        "jitter_sigma": 0.03,        # 1-hop UDP raw → light noise
        "mag_sigma": 0.15,
        "warp_sigma": 0.15,
    },
    "tor": {
        "strategy": "uniform",
        "aug_copies": 6,             # 1 + 6 = 7×
        "jitter_sigma": 0.05,        # multi-hop + cell padding → higher noise tolerance
        "mag_sigma": 0.20,
        "warp_sigma": 0.20,          # 3-hop timing noisy → strong warp
    },
}


# =================================================================
# Augmentation primitive: smooth random curve (cubic spline)
# =================================================================
def _random_curve(seq_len: int, rng: np.random.RandomState,
                  n_knots: int = 4, sigma: float = 0.2) -> np.ndarray:
    """Smooth random curve via cubic spline with random knots (mean 1.0)."""
    knot_xs = np.linspace(0, seq_len - 1, n_knots + 2)
    knot_ys = rng.randn(n_knots + 2).astype(np.float64) * sigma + 1.0
    cs = CubicSpline(knot_xs, knot_ys)
    return cs(np.arange(seq_len)).astype(np.float32)


# =================================================================
# Combined augmentation: jitter + magnitude-warping + time-warping
# =================================================================
def augment_combined(sample: np.ndarray, rng: np.random.RandomState,
                     jitter_sigma: float = 0.05,
                     mag_sigma: float = 0.2,
                     warp_sigma: float = 0.2,
                     n_knots: int = 4) -> np.ndarray:
    """Apply jitter, magnitude-warping, and time-warping in a single pass.

    Args:
        sample:        (seq_len, 4) array (up_bytes, dn_bytes, up_pkts, dn_pkts).
        rng:           Seeded RandomState (cross-interface synchronisation).
        jitter_sigma:  Gaussian noise std as a fraction of per-feature std.
        mag_sigma:     Magnitude-warp curve amplitude.
        warp_sigma:    Time-warp curve amplitude.
        n_knots:       Spline knot count for both curves.

    Returns:
        Augmented (seq_len, 4) array, clipped to ≥ 0.
    """
    seq_len, n_feat = sample.shape

    # --- Time-warping: non-linear time-axis remapping ---
    warp_curve = np.clip(_random_curve(seq_len, rng, n_knots, warp_sigma), 0.1, None)
    warped_steps = np.cumsum(warp_curve)
    warped_steps = warped_steps / warped_steps[-1] * (seq_len - 1)
    orig_idx = np.arange(seq_len, dtype=np.float32)
    aug = np.zeros_like(sample)
    for f in range(n_feat):
        aug[:, f] = np.interp(orig_idx, warped_steps, sample[:, f])

    # --- Magnitude-warping: smooth per-timestep scaling ---
    mag_curve = _random_curve(seq_len, rng, n_knots, mag_sigma)
    aug = aug * mag_curve[:, None]

    # --- Jitter: per-element Gaussian noise proportional to feature std ---
    feat_std = np.clip(np.std(sample, axis=0, keepdims=True), 1e-6, None)
    noise = rng.randn(*sample.shape).astype(np.float32) * feat_std * jitter_sigma
    aug = aug + noise

    return np.clip(aug, 0, None)


# =================================================================
# Feature transform: 4ch → 7ch with byte-based cumulatives
# =================================================================
def transform_type4(X: np.ndarray) -> np.ndarray:
    """Append [cum_up_bytes, cum_dn_bytes, cum_total_bytes] to 4ch raw.

    Input:  (N, T, 4) — [up_bytes, dn_bytes, up_pkts, dn_pkts]
    Output: (N, T, 7) — [raw4, cum_up_bytes, cum_dn_bytes, cum_total_bytes]
    """
    n, seq_len, _ = X.shape
    X_new = np.zeros((n, seq_len, 7), dtype=np.float32)
    X_new[:, :, :4] = X[:, :, :4]
    X_new[:, :, 4] = np.cumsum(X[:, :, 0], axis=1)
    X_new[:, :, 5] = np.cumsum(X[:, :, 1], axis=1)
    X_new[:, :, 6] = np.cumsum(X[:, :, 0] + X[:, :, 1], axis=1)
    return X_new


# =================================================================
# Strategy: uniform (every sample gets `aug_copies` augmented copies)
# =================================================================
def augment_uniform(X_raw: np.ndarray, y_raw: np.ndarray,
                    aug_copies: int, seed: int,
                    jitter_sigma: float, mag_sigma: float, warp_sigma: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Output shape: (N · (1 + aug_copies), T, 7)."""
    N, T, F = X_raw.shape
    assert F >= 4, f"Expected ≥4 features, got {F}"
    X4 = X_raw[:, :, :4].copy()

    aug_X_list = [X4]
    aug_y_list = [y_raw]

    for copy_idx in range(aug_copies):
        aug_batch = np.zeros((N, T, 4), dtype=np.float32)
        for i in range(N):
            rng = np.random.RandomState(seed + copy_idx * N + i)
            aug_batch[i] = augment_combined(
                X4[i], rng,
                jitter_sigma=jitter_sigma,
                mag_sigma=mag_sigma,
                warp_sigma=warp_sigma,
            )
        aug_X_list.append(aug_batch)
        aug_y_list.append(y_raw.copy())

    X_aug_4ch = np.concatenate(aug_X_list, axis=0).astype(np.float32)
    y_aug = np.concatenate(aug_y_list, axis=0)

    perm = np.random.RandomState(seed).permutation(len(X_aug_4ch))
    X_aug_4ch = X_aug_4ch[perm]
    y_aug = y_aug[perm]

    return transform_type4(X_aug_4ch), y_aug


# =================================================================
# Strategy: adaptive (per-site copies inverse to sample count)
# =================================================================
def augment_adaptive(X_raw: np.ndarray, y_raw: np.ndarray,
                     max_copies: int, min_copies: int, seed: int,
                     jitter_sigma: float, mag_sigma: float, warp_sigma: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-site copies scale inversely with sample count."""
    X4 = X_raw[:, :, :4].copy()
    classes = np.unique(y_raw)
    class_counts = {c: int((y_raw == c).sum()) for c in classes}
    median_count = float(np.median(list(class_counts.values())))

    aug_X_list = [X4]
    aug_y_list = [y_raw]

    for c in classes:
        mask = y_raw == c
        X_c = X4[mask]
        n_c = len(X_c)

        ratio = median_count / n_c if n_c > 0 else max_copies
        copies = int(np.clip(np.round(ratio), min_copies, max_copies))

        for copy_idx in range(copies):
            aug_batch = np.zeros_like(X_c)
            for i in range(n_c):
                rng = np.random.RandomState(seed + c * 1000 + copy_idx * n_c + i)
                aug_batch[i] = augment_combined(
                    X_c[i], rng,
                    jitter_sigma=jitter_sigma,
                    mag_sigma=mag_sigma,
                    warp_sigma=warp_sigma,
                )
            aug_X_list.append(aug_batch)
            aug_y_list.append(np.full(n_c, c, dtype=y_raw.dtype))

    X_aug_4ch = np.concatenate(aug_X_list, axis=0).astype(np.float32)
    y_aug = np.concatenate(aug_y_list, axis=0)

    perm = np.random.RandomState(seed).permutation(len(X_aug_4ch))
    X_aug_4ch = X_aug_4ch[perm]
    y_aug = y_aug[perm]

    return transform_type4(X_aug_4ch), y_aug


# =================================================================
# Slot processor
# =================================================================
def detect_ifaces(npz_slot: Path) -> list[str]:
    return sorted([
        d.name for d in npz_slot.iterdir()
        if d.is_dir() and (d / "train.npz").exists()
    ])


def process_slot(npz_slot: Path, cfg: dict, copies_override: int | None,
                 seed: int, force: bool) -> bool:
    ifaces = detect_ifaces(npz_slot)
    if not ifaces:
        print(f"  ❌ {npz_slot} — no ifaces with train.npz")
        return False

    strategy = cfg["strategy"]
    for iface in ifaces:
        iface_dir = npz_slot / iface
        out_path = iface_dir / "train_aug.npz"
        if not force and out_path.exists():
            print(f"    ✓ {iface}/train_aug.npz — exists, skip")
            continue

        train_path = iface_dir / "train.npz"
        data = np.load(train_path, allow_pickle=True)
        X_tr, y_tr = data["X"], data["y"]

        print(f"    [{iface}] X={X_tr.shape} → ", end="", flush=True)
        t0 = time.time()

        if strategy == "adaptive":
            X_aug, y_aug = augment_adaptive(
                X_tr, y_tr,
                max_copies=cfg.get("max_copies", 12),
                min_copies=cfg.get("min_copies", 2),
                seed=seed,
                jitter_sigma=cfg["jitter_sigma"],
                mag_sigma=cfg["mag_sigma"],
                warp_sigma=cfg["warp_sigma"],
            )
        else:
            copies = copies_override or cfg.get("aug_copies", 6)
            X_aug, y_aug = augment_uniform(
                X_tr, y_tr, copies, seed,
                jitter_sigma=cfg["jitter_sigma"],
                mag_sigma=cfg["mag_sigma"],
                warp_sigma=cfg["warp_sigma"],
            )

        dur = time.time() - t0
        print(f"X_aug={X_aug.shape} ({dur:.1f}s)")

        save_dict = {"X": X_aug, "y": y_aug}
        for key in data.files:
            if key not in ("X", "y"):
                save_dict[key] = data[key]
        np.savez_compressed(out_path, **save_dict)

    return True


# =================================================================
# CLI
# =================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--slot", type=Path,
                    help="Specific NPZ slot dir (skips proto/device/network walk)")
    ap.add_argument("--proto", choices=["NonVPN", "Tor", "Wireguard", "OpenVPN", "TorBrowser"])
    ap.add_argument("--device", help="PIXEL8/RM13/A34")
    ap.add_argument("--network", help="WIFI/4G/5G/Cellular")
    ap.add_argument("--aug-copies", type=int, default=None,
                    help="Override copies (uniform strategy only)")
    ap.add_argument("--strategy", choices=["uniform", "adaptive"], default=None,
                    help="Force strategy when using --slot")
    ap.add_argument("--config", choices=list(PROTOCOL_CONFIGS.keys()), default=None,
                    help="Force protocol config when using --slot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing train_aug.npz")
    args = ap.parse_args()

    # --- Single-slot mode ---
    if args.slot:
        cfg_key = args.config or (args.proto.lower() if args.proto else "tor")
        cfg = dict(PROTOCOL_CONFIGS[cfg_key])
        if args.strategy:
            cfg["strategy"] = args.strategy
        print(f"\n{'='*60}\n  SLOT: {args.slot}\n"
              f"  config={cfg_key}, strategy={cfg['strategy']}\n{'='*60}")
        process_slot(args.slot, cfg, args.aug_copies, args.seed, args.force)
        return

    # --- Walk Dataset/NPZ/<proto>/CW ---
    protocols = [args.proto] if args.proto else \
                ["NonVPN", "Tor", "Wireguard", "OpenVPN", "TorBrowser"]
    ok, fail = 0, 0
    # Map filesystem protocol dir → PROTOCOL_CONFIGS key
    proto_alias = {"nonvpn": "normal"}
    for proto in protocols:
        cfg_key = proto_alias.get(proto.lower(), proto.lower())
        if cfg_key not in PROTOCOL_CONFIGS:
            print(f"  ⚠ {proto}: no config, using 'normal' fallback")
            cfg_key = "normal"
        cfg = PROTOCOL_CONFIGS[cfg_key]

        cw_root = NPZ_ROOT / proto / "CW"
        if not cw_root.exists():
            print(f"  ⚠ {cw_root} — not found, skip {proto}")
            continue

        for dev_dir in sorted(cw_root.iterdir()):
            if not dev_dir.is_dir(): continue
            if args.device and dev_dir.name != args.device: continue

            for net_dir in sorted(dev_dir.iterdir()):
                if not net_dir.is_dir(): continue
                if args.network and net_dir.name != args.network: continue

                if not detect_ifaces(net_dir): continue
                copies_desc = args.aug_copies or cfg.get("aug_copies", "(adaptive)")
                print(f"\n{'='*60}")
                print(f"  {proto}/CW/{dev_dir.name}/{net_dir.name}")
                print(f"  config={cfg_key}, strategy={cfg['strategy']}, "
                      f"copies={copies_desc}")
                print(f"  jitter={cfg['jitter_sigma']}, "
                      f"mag={cfg['mag_sigma']}, warp={cfg['warp_sigma']}")
                print(f"{'='*60}")

                if process_slot(net_dir, cfg, args.aug_copies, args.seed, args.force):
                    ok += 1
                else:
                    fail += 1

    print(f"\nDone. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
