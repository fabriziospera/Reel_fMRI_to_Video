"""STEP 02 - Learned pooling modules, per subject: early (TemporalAttnPool on the per-frame VGG
features) and dorsal (FactorizedAttnPool on the V-JEPA tokens). The ventral stream has none: its
features are a fixed mean over frames.

Each pooler is trained through a temporary linear head that predicts the z-scored fMRI of its ROI
(MSE); the head is discarded and only the pooler is kept, so the readout of step 03 is fitted on
fixed pooled features. AdamW lr 3e-4, wd 1e-2, cosine schedule over 120 epochs, gradient clip 1,
bf16 autocast, batch 32 (early) / 64 (dorsal), early stopping with patience 6 on a validation
set INSIDE the training data (cc2017: movie group 0; bmd: random 12% of the clips, see datasets.py).

  python 02_train_poolers.py --dataset cc2017 --subject 1 --stream early
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

import datasets
import paths
from brainenc.backbones import EARLY_DIM, VJEPA_DIM
from brainenc.pooling import FactorizedAttnPool, TemporalAttnPool
from brainenc.rois import load_roi_masks

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, PATIENCE = 120, 6
LR, WEIGHT_DECAY, CLIP = 3e-4, 1e-2, 1.0
BATCH = {"early": 32, "dorsal": 64}


def build(stream):
    """-> (pooler, output dim). The dorsal pooler works on the 8 x 16 x 16 V-JEPA token grid."""
    if stream == "dorsal":
        return FactorizedAttnPool(), VJEPA_DIM
    return TemporalAttnPool(), EARLY_DIM


def take(X, idx):
    return X[torch.as_tensor(np.asarray(idx), device=X.device, dtype=torch.long)].float()


def train(X, Y, stream, tr, va, batch_seed, seed=0):
    """Returns the trained pooler (best validation epoch). The head is thrown away."""
    bs = BATCH[stream]
    Y_tr, Y_va = Y[tr].to(DEV), Y[va].to(DEV)
    torch.manual_seed(seed)
    pool, d_out = build(stream)
    pool = pool.to(DEV)
    head = torch.nn.Linear(d_out, Y.shape[1]).to(DEV)
    params = list(pool.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(EPOCHS):
        pool.train()
        order = np.random.default_rng(batch_seed + ep).permutation(len(tr))
        for i in range(0, len(tr), bs):
            b = tr[order[i:i + bs]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.startswith("cuda")):
                loss = F.mse_loss(head(pool(take(X, b))), Y_tr[order[i:i + bs]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
        sched.step()

        pool.eval()
        with torch.no_grad():
            vl = float(np.mean([F.mse_loss(head(pool(take(X, va[i:i + 64]))), Y_va[i:i + 64]).item()
                                for i in range(0, len(va), 64)]))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in pool.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        print(f"  epoch {ep:>3}  val {vl:.5f}  best {best:.5f}", flush=True)

    pool.load_state_dict(best_state)
    print(f"  stopped at epoch {ep}, best validation loss {best:.5f}")
    return pool.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=paths.DATASETS)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--stream", required=True, choices=["early", "dorsal"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert a.subject in paths.SUBJECTS[a.dataset]
    paths.ensure_dirs(a.dataset)
    dst = paths.pooler(a.dataset, a.subject, a.stream)
    if dst.exists():
        print(f"already trained: {dst}")
        return

    roi = paths.roi_map(a.dataset, a.subject)
    if not roi.exists():
        sys.exit(f"ROI map not found: {roi}\nrun step 00 first")
    mask = load_roi_masks(roi)[a.stream]
    Y = torch.load(paths.fmri(a.dataset, a.subject, "train"), map_location="cpu").float().mean(1)[:, mask]
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)                    # average the repeats, z-score per voxel

    src = paths.features(a.dataset, a.stream, "train")
    if not src.exists():
        sys.exit(f"features not found: {src}\nrun 01_extract_features.py first")
    X = torch.from_numpy(np.load(src, mmap_mode="r")[:]).to(DEV)
    tr, va, batch_seed = datasets.pooler_split(a.dataset, len(X), a.seed)
    print(f"{a.dataset} subject {a.subject} | {a.stream} | {int(mask.sum())} voxels | features "
          f"{tuple(X.shape)} | validation {len(va)} samples")
    pool = train(X, Y, a.stream, tr, va, batch_seed, a.seed)
    torch.save(pool.state_dict(), dst)
    print(f"written {dst}")


if __name__ == "__main__":
    main()
