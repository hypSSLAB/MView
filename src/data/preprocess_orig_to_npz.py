#!/usr/bin/env python3
"""
preprocess_orig_to_npz.py — Orig raw CSV → NPZ (1s × 60 bins, 4ch raw).

Walks `Dataset/Orig/{Tor,Wireguard}/CW/<DEV>/<NET>/` and builds
`Dataset/NPZ/{Tor,Wireguard}/CW/<DEV>/<NET>/<iface>/{train,valid,test}.npz`
using the canonical 1s floor-binning pipeline (same as build_tor_npz.py).

Features (4ch): upload_bytes_diff, download_bytes_diff,
                upload_packets_diff, download_packets_diff

Usage:
    # Process everything under Orig/Tor/CW + Orig/Wireguard/CW
    python preprocess_orig_to_npz.py

    # Specific slot only
    python preprocess_orig_to_npz.py --proto Tor --device PIXEL8 --network 4G

    # Custom ifaces
    python preprocess_orig_to_npz.py --ifaces tun0 rmnet total lo

    # Force rebuild (overwrite existing NPZ)
    python preprocess_orig_to_npz.py --force
"""
import argparse
import subprocess
import sys
from pathlib import Path

DATASET_ROOT = Path(".//Dataset")
ORIG_ROOT = DATASET_ROOT / "Orig"
NPZ_ROOT = DATASET_ROOT / "NPZ"
BUILD_SCRIPT = Path(".//Train_script/build_tor_npz.py")

# Default iface sets per network type
WIFI_IFACES = ["tun0", "wlan0", "total", "lo"]
CELLULAR_IFACES = ["tun0", "rmnet", "total", "lo"]


def detect_ifaces(orig_slot: Path) -> list[str]:
    """Auto-detect available ifaces from a sample site."""
    sites = [d for d in orig_slot.iterdir() if d.is_dir()]
    if not sites:
        return []
    sample = sites[0]
    available = {d.name for d in sample.iterdir() if d.is_dir()}
    # Pick standard 4-iface set based on what's available
    if "wlan0" in available:
        return [i for i in WIFI_IFACES if i in available]
    elif "rmnet" in available:
        return [i for i in CELLULAR_IFACES if i in available]
    elif "rmnet_dev" in available:
        return [i for i in ["tun0", "rmnet_dev", "total", "lo"] if i in available]
    else:
        return [i for i in CELLULAR_IFACES if i in available]


def has_valid_npz(npz_slot: Path, ifaces: list[str]) -> bool:
    return all(
        (npz_slot / i / "train.npz").exists() and
        (npz_slot / i / "valid.npz").exists() and
        (npz_slot / i / "test.npz").exists()
        for i in ifaces
    )


def process_slot(orig_slot: Path, npz_slot: Path, ifaces: list[str], force: bool):
    if not force and has_valid_npz(npz_slot, ifaces):
        print(f"  ✓ {npz_slot.relative_to(NPZ_ROOT)} — already built, skip")
        return True

    sites = [d for d in orig_slot.iterdir() if d.is_dir()]
    if not sites:
        print(f"  ❌ {orig_slot.relative_to(ORIG_ROOT)} — no sites, skip")
        return False

    print(f"  → building {npz_slot.relative_to(NPZ_ROOT)} ({len(sites)} sites, ifaces={ifaces})")
    cmd = [
        sys.executable, "-u", str(BUILD_SCRIPT),
        "--from-raw-csv", str(orig_slot),
        "--out", str(npz_slot),
        "--ifaces", *ifaces,
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"    [!] build failed (rc={rc})")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proto", choices=["Tor", "Wireguard"],
                    help="process only this protocol")
    ap.add_argument("--device", help="process only this device (PIXEL8/RM13/A34)")
    ap.add_argument("--network", help="process only this network (WIFI/4G/5G)")
    ap.add_argument("--ifaces", nargs="+", help="override iface list")
    ap.add_argument("--force", action="store_true", help="rebuild even if NPZ exists")
    args = ap.parse_args()

    protocols = [args.proto] if args.proto else ["Tor", "Wireguard"]
    ok, skip, fail = 0, 0, 0

    for proto in protocols:
        cw_root = ORIG_ROOT / proto / "CW"
        if not cw_root.exists():
            continue
        for dev_dir in sorted(cw_root.iterdir()):
            if not dev_dir.is_dir():
                continue
            if args.device and dev_dir.name != args.device:
                continue
            for net_dir in sorted(dev_dir.iterdir()):
                if not net_dir.is_dir():
                    continue
                if args.network and net_dir.name != args.network:
                    continue

                orig_slot = net_dir
                npz_slot = NPZ_ROOT / proto / "CW" / dev_dir.name / net_dir.name

                ifaces = args.ifaces or detect_ifaces(orig_slot)
                if not ifaces:
                    print(f"  ❌ {orig_slot.relative_to(ORIG_ROOT)} — no ifaces detected, skip")
                    fail += 1
                    continue

                result = process_slot(orig_slot, npz_slot, ifaces, args.force)
                if result:
                    ok += 1
                else:
                    fail += 1

    print(f"\n[done] ok={ok}, fail={fail}")


if __name__ == "__main__":
    main()
