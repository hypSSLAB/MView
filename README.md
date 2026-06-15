# MS-Mamba WF — 

> **Note for reviewers:** This artifact is anonymized for double-blind
> review. Author names, affiliations, contact information, and remote
> hosting URLs have been removed. Git history is squashed to a single
> initial commit before publication.

Minimal reproducible artifact for a mobile-traffic Website Fingerprinting
paper. Includes the model code, augmentation pipeline, end-to-end data
preprocessing, training, and baseline evaluation, plus **one sample data
slot** (Tor / `PIXEL8` / WiFi, 4 interfaces) so reviewers can run the
full pipeline in a few minutes.

---

## Repo layout

```
MView/
├── src/
│   ├── model/MS_Mamba.py                 # MS-Mamba (Mamba + MixStyle) backbone
│   ├── data/
│   │   ├── preprocess_orig_to_npz.py     # raw CSV → 1Hz 60-bin 4-ch NPZ
│   │   ├── build_tor_npz.py              # canonical NPZ builder
│   │   ├── build_ow_unmonitored_npz.py   # OW unmonitored NPZ builder
│   │   └── augment_all.py                # data augmentation runner
│   ├── train/run_multi_view_analysis.py       # per-iface trainer + late fusion
│   └── eval/run_baselines_per_proto.py   # 4 mobile-WF baselines + per-proto eval
├── data/Tor/CW/PIXEL8/WIFI/         # sample NPZ slot (~37 MB)
│   ├── lo/{train,valid,test}.npz
│   ├── tun0/{train,valid,test}.npz
│   ├── total/{train,valid,test}.npz
│   └── wlan0/{train,valid,test}.npz
├── scripts/run.sh                   # one-shot pipeline
├── requirements.txt
├── LICENSE
├── REPRODUCE.md
└── README.md
```

---

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Verify the model is loadable
python src/model/MS_Mamba.py
# expected: "MS-Mamba  in_dim=4 num_classes=100  params=1.90M"

# 3. Augment the sample slot (Tor → 7× volume)
python src/data/augment_all.py \
    --slot data/Tor/CW/PIXEL8/WIFI \
    --config tor --strategy uniform

# 4. Train MS-Mamba on the slot (4 ifaces × ~10 min, single GPU)
python src/train/run_multi_view_analysis.py \
    --npz   data/Tor/CW/PIXEL8/WIFI \
    --ifaces tun0 wlan0 total lo \
    --train-file train_aug.npz \
    --exp-name tor_pixel8_wifi

# 5. (Optional) Run baseline comparison
python src/eval/run_baselines_per_proto.py \
    --combined-npz data/Tor/CW/PIXEL8/WIFI/total \
    --proto-tests "Tor:data/Tor/CW/PIXEL8/WIFI/total/test.npz" \
    --exp-name baselines
```

Or run the one-shot script:

```bash
bash scripts/run.sh
```

---

## Sample data

| Split | Per iface | Shape | Note |
|---|---|---|---|
| `train.npz` | ~7.4 MB | (8000, 60, 4) | 100 sites × 80 sessions, 1 Hz × 60 s × 4 ch |
| `valid.npz` | ~0.9 MB | (1000, 60, 4) | |
| `test.npz`  | ~0.9 MB | (1000, 60, 4) | |

4 channels: `[upload_bytes, download_bytes, upload_packets, download_packets]`.

Full dataset (all 3 devices, 4 networks, 5 protocols) totals ~3 GB and is
distributed separately — see `REPRODUCE.md` for the download / regeneration
procedure.

---

## What the model does

`MS-Mamba` (a Mamba selective-SSM backbone augmented with MixStyle and a
multi-scale 1D-CNN branch) is trained **independently per network
interface** (view). At inference, per-view softmax outputs are averaged
(*late fusion*) to produce the final website prediction. This view-wise
training pattern is the central design of the proposed system.

See the accompanying paper (anonymized) for the full method description.

---

## License

Released under the MIT License (see `LICENSE`). The sample data is
released for research-reproduction purposes only.
