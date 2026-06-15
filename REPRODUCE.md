# Reproduction Guide

This document maps paper tables/figures to specific commands.

## 0. Environment

```bash
conda create -n mwf python=3.10
conda activate mwf
pip install -r requirements.txt
```

GPU recommended (any CUDA-capable card with ≥ 8 GB memory).

## 1. Sample slot quick test (≈ 15 min, 1 GPU)

The shipped sample slot is **Tor / PIXEL8 / WiFi** with 4 interfaces.

```bash
# (a) augment
python src/data/augment_all.py \
    --slot data/Tor/CW/PIXEL8/WIFI \
    --config tor --strategy uniform --seed 42

# (b) train MView MS-Mamba (per-iface + late fusion)
python src/train/run_multi_view_analysis.py \
    --npz data/Tor/CW/PIXEL8/WIFI \
    --ifaces tun0 wlan0 total lo \
    --train-file train_aug.npz \
    --exp-name tor_pixel8_wifi
```

Expected fusion macro-F1: ≥ 0.99 (test set, 100 classes).

## 2. Full closed-world results (paper Tab. 2)

The full dataset contains 3 devices × 4 networks × 5 protocols.
Each (proto, dev, net) slot follows the same recipe as the sample slot above.

```bash
# Pre-process all raw CSV → 1Hz × 60-bin × 4-ch NPZ
python src/data/preprocess_orig_to_npz.py \
    --proto Tor --device PIXEL8 --network WIFI   # repeat per slot

# Augment per protocol (jitter + magnitude-warp + time-warp, single pass)
python src/data/augment_all.py --proto Tor     # uses tor config
python src/data/augment_all.py --proto Wireguard
python src/data/augment_all.py --proto OpenVPN
python src/data/augment_all.py --proto TorBrowser
python src/data/augment_all.py --proto NonVPN

# Train per slot
for slot in Tor/PIXEL8/WIFI Tor/PIXEL8/4G Tor/PIXEL8/5G ... ; do
    python src/train/run_multi_view_analysis.py \
        --npz data/full/$slot --ifaces tun0 wlan0 total lo \
        --train-file train_aug.npz --exp-name "$slot"
done
```

## 3. Open-world results (paper Tab. 4)

```bash
# Build unmonitored OW NPZ (5000 unmonitored samples, paired across ifaces)
python src/data/build_ow_unmonitored_npz.py \
    --preset tor_wifi --out data/full/Tor/OW/WIFI

# Merge CW + OW (class 100 added) — single per-iface, train + valid + test
# (script provided in supplementary; see paper §5.3)

# Train with the same trainer; --num-classes is auto-detected from NPZ
python src/train/run_multi_view_analysis.py \
    --npz data/full/Tor/OW/WIFI \
    --ifaces tun0 wlan0 total lo \
    --train-file train_aug.npz --exp-name Tor_OW_WIFI
```

## 4. Baseline comparison (paper Tab. 5)

```bash
python src/eval/run_baselines_per_proto.py \
    --combined-npz data/full/Tor/CW/PIXEL8/WIFI/total \
    --proto-tests "Tor:data/full/Tor/CW/PIXEL8/WIFI/total/test.npz" \
    --exp-name baselines_Tor_pixel8_wifi
```

The runner internally trains and evaluates four baselines:
WiSec'16, ProCharvester, ScanDroid, Mischief — all at the **canonical
1 Hz × 60-bin** sampling that matches MView's input window.

## 5. On-device deployment (paper §6)

Per-interface ONNX export (FP32):

```python
import torch
from src.model.MS_Mamba import MSMamba_Default

model = MSMamba_Default(input_dim=4, num_classes=101)
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

torch.onnx.export(
    model, torch.randn(1, 60, 4), "ms_mamba.onnx",
    input_names=["x"], output_names=["logits"],
    dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17, do_constant_folding=True,
)
# Each per-view ONNX file ≈ 6.84 MB (FP32).
```

## Seeds

All scripts default to `seed=42` for the data permutation, model
initialisation, and augmentation curves. Results are deterministic given
identical PyTorch / CUDA versions.

---

## Hardware footprint

| Stage | Memory | Wall-clock (1× A6000) |
|---|---|---|
| Augmentation (single slot) | < 4 GB RAM | 1–3 min |
| Training (4 ifaces, 100 epochs, patience 30) | ~6 GB VRAM | 10–25 min |
| Baseline (Mischief + WiSec'16 + 2 kNNs) | 4 GB VRAM | 5–15 min |
