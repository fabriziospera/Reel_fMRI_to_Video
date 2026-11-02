"""What differs between the two datasets, in one place. The numbered scripts call only this.

                      cc2017                                 bmd (BOLD Moments)
  samples             4320 train TRs = 18 movies x 240       1000 train clips, 102 test clips
                      1200 test TRs                          (3 / 10 repeats, averaged)
  internal splits     by whole movies                        random, by clip
    pooler early stop movie group 0 (3 movies, 720 TRs)      random 12% of the clips
    ridge lambda      6-fold CV over the movie groups        random 15% of the clips
  stimuli             GT_{split}_3fps.pt                     dataset frames, step 01
  candidates          gif, right half                        mp4, 6 frames over the whole video

Movie groups: GROUPS = default_rng(0).permutation(18).reshape(6, 3) (group 0 = movies 13, 1, 5,
1-based).
"""
import glob
import os
import re

import numpy as np
import torch

import paths
from brainenc import clips
from brainenc.ridge import fit_ridge, fit_ridge_grouped

CC2017_GROUPS = np.random.default_rng(0).permutation(18).reshape(6, 3)
TRS_PER_MOVIE = 240
BMD_POOLER_VAL_FRAC = 0.12


def movie_of(n):
    assert n == 18 * TRS_PER_MOVIE, n
    return np.arange(n) // TRS_PER_MOVIE


def movie_fold(n):
    """Fold of every training TR: the index of the group its movie belongs to."""
    group_of_movie = np.empty(18, int)
    for g, movies in enumerate(CC2017_GROUPS):
        group_of_movie[movies] = g
    return group_of_movie[movie_of(n)]


def pooler_split(dataset, n, seed=0):
    """(train indices, validation indices, batch-order seed) for the pooler's early stopping.

    Per epoch the batch order is default_rng(batch_seed + epoch)."""
    if dataset == "cc2017":
        va_mask = np.isin(movie_of(n), CC2017_GROUPS[0])
        return np.flatnonzero(~va_mask), np.flatnonzero(va_mask), 1000
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(n * BMD_POOLER_VAL_FRAC)
    return perm[n_val:], perm[:n_val], seed * 1000


def fit_readout(dataset, X, Y, dual):
    """Ridge from features to the z-scored ROI response, lambda chosen inside the training set."""
    if dataset == "cc2017":
        return fit_ridge_grouped(X, Y, movie_fold(len(X)), dual=dual)
    return fit_ridge(X, Y, dual=dual)


# ---------------------------------------------------------------- stimuli ----------------
def stimuli(dataset, split):
    """Indexable of stimulus clips; stimuli(...)[i] -> (6, 3, H, W) float in [0, 1]."""
    if dataset == "cc2017":
        src = paths.cc2017_stimuli(split)
        if not src.exists():
            raise FileNotFoundError(f"{src}\nset CC2017_DATA")
        return _FloatClips(torch.load(src, map_location="cpu", mmap=True))
    src = paths.bmd_stimuli_u8(split)
    if not src.exists():
        raise FileNotFoundError(f"{src}\nrun 01_extract_features.py --dataset bmd first")
    return _U8Clips(np.load(src, mmap_mode="r"))


def build_bmd_stimuli():
    """Write the uint8 stimulus clips of bmd (train vid 1..1000, test vid 1001..1102), once."""
    first = {"train": 1, "test": 1001}
    for split, n in paths.N_STIMULI["bmd"].items():
        dst = paths.bmd_stimuli_u8(split)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack([clips.read_bmd_stimulus(paths.BMD_FRAMES, first[split] + i) for i in range(n)])
        np.save(dst, arr)
        print(f"written {dst} {arr.shape}")


class _FloatClips:
    def __init__(self, t):
        self.t = t

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i].float()


class _U8Clips:
    def __init__(self, a):
        self.a = a

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return clips.as_input(np.asarray(self.a[i]))


# ---------------------------------------------------------------- candidates -------------
def candidate_files(dataset, subject, idx):
    """{seed: path} of the candidate reconstructions of test stimulus idx."""
    pattern = r"seed_(\d+)\.gif$" if dataset == "cc2017" else r"seed_(\d+)\.mp4$"
    out = {}
    for f in glob.glob(str(paths.candidates(dataset, subject) / f"sample_{idx}" / "*")):
        m = re.search(pattern, os.path.basename(f))
        if m and os.path.getsize(f) > 0:
            out[int(m.group(1))] = f
    return out


def read_candidate(dataset, path):
    """-> (6, H, W, 3) uint8: the pixels the evaluation sees."""
    return clips.read_gif_reconstruction(path) if dataset == "cc2017" else clips.read_bmd_candidate(path)
