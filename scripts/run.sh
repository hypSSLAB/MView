#!/usr/bin/env bash
# One-shot pipeline: augment → train MView on Tor PIXEL8 WiFi slot
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SLOT="data/Tor/CW/PIXEL8/WIFI"

echo "=== [1/2] augment Tor sample slot ==="
python src/data/augment_all.py \
    --slot "$SLOT" \
    --config tor --strategy uniform --seed 42

echo "=== [2/2] train MS-Mamba 4 views + late fusion ==="
python src/train/run_multi_view_analysis.py \
    --npz   "$SLOT" \
    --ifaces tun0 wlan0 total lo \
    --train-file train_aug.npz \
    --exp-name tor_pixel8_wifi

echo "=== DONE ==="
