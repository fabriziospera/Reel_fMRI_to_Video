"""STEP 01 - Stimulus features of the three streams, once per dataset (shared by all subjects).

  early    VGG19-BN per frame, 5 layers on a 7x7 grid       (N, 6, 72128)    fp16
  ventral  OpenCLIP ViT-bigG-14 projection, mean over frames (N, 1280)        fp32
  dorsal   V-JEPA 2 last_hidden_state                        (N, 2048, 1024)  fp16

For bmd the 6-frame stimulus clips are first built from the dataset frames (uint8 cache).

  python 01_extract_features.py --dataset cc2017 --stream dorsal --split train
"""
import argparse
import os

import numpy as np
import torch

import datasets
import paths
from brainenc.backbones import (EARLY_DIM, VENTRAL_DIM, VJEPA_DIM, VJEPA_TOKENS, VJEPA2,
                                OpenCLIPTower, VGGPerFrame)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=paths.DATASETS)
    ap.add_argument("--stream", required=True, choices=["early", "ventral", "dorsal"])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    paths.ensure_dirs(a.dataset)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dst = paths.features(a.dataset, a.stream, a.split)
    if dst.exists():
        print(f"already cached: {dst}")
        return
    if a.dataset == "bmd":
        datasets.build_bmd_stimuli()
    clips = datasets.stimuli(a.dataset, a.split)
    N = len(clips)
    tmp = str(dst) + ".tmp.npy"

    if a.stream == "early":
        model = VGGPerFrame(dev).to(dev)
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(N, 6, EARLY_DIM))
        step = a.batch
    elif a.stream == "ventral":
        model = OpenCLIPTower(dev)
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(N, VENTRAL_DIM))
        step = a.batch
    else:
        model = VJEPA2(dev)
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(N, VJEPA_TOKENS, VJEPA_DIM))
        step = 16

    for i in range(0, N, step):
        j = min(i + step, N)
        if a.stream == "dorsal" and a.dataset == "bmd":                 # uint8 clips: uint8 path
            out[i:j] = model.from_uint8(np.load(paths.bmd_stimuli_u8(a.split), mmap_mode="r")[i:j]).cpu().numpy()
            continue
        v = torch.stack([clips[k] for k in range(i, j)]).to(dev)          # (b, 6, 3, H, W) in [0, 1]
        b, t = v.shape[:2]
        if a.stream == "early":
            out[i:j] = model(v.reshape(b * t, *v.shape[2:])).reshape(b, t, EARLY_DIM).half().cpu().numpy()
        elif a.stream == "ventral":
            out[i:j] = model(v.reshape(b * t, *v.shape[2:])).reshape(b, t, -1).mean(1).float().cpu().numpy()
        else:
            out[i:j] = model(v).cpu().numpy()
        if (i // step) % 50 == 0:
            print(f"  {a.dataset} {a.stream} {a.split} {i}/{N}", flush=True)
    out.flush()
    del out
    os.replace(tmp, dst)
    print(f"written {dst}")


if __name__ == "__main__":
    main()
