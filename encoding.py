"""The three encoders of a subject: stimulus features -> fMRI of the stream's ROI. Used by steps
03 (diagnostics) and 04 (reranking). Everything is fitted on training stimuli only.

  early    VGG19-BN per frame -> TemporalAttnPool (step 02) -> dual ridge -> V1-V4
  ventral  OpenCLIP bigG, mean over frames                  -> ridge      -> ventral ROI
  dorsal   V-JEPA 2 tokens -> FactorizedAttnPool (step 02)  -> ridge      -> dorsal ROI

Targets: the training response averaged over repeats, z-scored per voxel with the training mean
and SD. The measured test response (averaged over its 10 repeats) is put in the same
normalisation, so predicted and measured live in the same space.
"""
import sys

import numpy as np
import torch

import datasets
import paths
from brainenc.pooling import FactorizedAttnPool, TemporalAttnPool
from brainenc.rois import load_roi_masks
from brainenc.scoring import STREAMS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DUAL = {"early": True, "ventral": False, "dorsal": False}


def load_pooler(dataset, subject, stream):
    f = paths.pooler(dataset, subject, stream)
    if not f.exists():
        sys.exit(f"missing pooler {f}\nrun 02_train_poolers.py first")
    m = TemporalAttnPool() if stream == "early" else FactorizedAttnPool()
    m.load_state_dict(torch.load(f, map_location=DEV))
    return m.to(DEV).eval()


@torch.no_grad()
def pooled(dataset, subject, stream, split):
    """Stimulus features of one stream with the subject's pooler applied (ventral: as extracted)."""
    X = np.load(paths.features(dataset, stream, split), mmap_mode="r")
    if stream == "ventral":
        return torch.from_numpy(np.asarray(X)).float()
    pool = load_pooler(dataset, subject, stream)
    bs = 64 if stream == "early" else 32
    return torch.cat([pool(torch.from_numpy(np.asarray(X[i:i + bs])).float().to(DEV)).cpu()
                      for i in range(0, len(X), bs)])


def fit_encoders(dataset, subject):
    """-> encoders {stream: predict}, measured test responses {stream: (N_test, V)}, lambdas."""
    masks = load_roi_masks(paths.roi_map(dataset, subject))
    f_tr = torch.load(paths.fmri(dataset, subject, "train"), map_location="cpu").float().mean(1)
    f_te = torch.load(paths.fmri(dataset, subject, "test"), map_location="cpu").float().mean(1)
    encoders, measured, lam = {}, {}, {}
    for stream in STREAMS:
        X = pooled(dataset, subject, stream, "train")
        Y = f_tr[:, masks[stream]]
        mu, sd = Y.mean(0), Y.std(0) + 1e-6
        encoders[stream] = datasets.fit_readout(dataset, X, (Y - mu) / sd, DUAL[stream])
        measured[stream] = (f_te[:, masks[stream]] - mu) / sd
        lam[stream] = encoders[stream].alpha
        del X
        if DEV.startswith("cuda"):
            torch.cuda.empty_cache()
    print(f"{dataset} subject {subject}: lambda " + ", ".join(f"{s} {lam[s]:g}" for s in STREAMS))
    return encoders, measured, lam
