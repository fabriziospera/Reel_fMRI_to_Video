"""STEP 00b - bmd: stimulus-responsive voxel selection.

Input: GLMsingle betas per trial (version B, fsLR32k, first 59412 cortical grayordinates), 1000
train stimuli x 3 repeats. The train stimuli are split into K = 18 random blocks (seed 0); per
block and vertex, Pearson r across stimuli between every pair of repeats, Fisher z, averaged over
the 3 pairs -> Zmat (V x K). One-sample t-test across blocks; selected: t > 0 and
p < 0.05 / 59412. Vertices with NaN are excluded. Test stimuli are not used.

Outputs, per subject, in BMD_DATA, in the cc2017 format read by the rest of the pipeline:
    subjXX_train_fmri.pt   (1000, 3, Nsel)   float32, repeats not averaged
    subjXX_test_fmri.pt    (102, 10, Nsel)   float32
    subjXX_roi_map.npz     vertex_idx, glasser, parcel, hemi, match_corr (=1, same space)
    subjXX_selection.npz   Zmat, t, p, selected mask over the 59412 vertices

  python 00b_bmd_voxel_selection.py                # all 10 subjects
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pickle
import torch
from scipy import stats

import paths
from brainenc.rois import SCHEME_A, STREAMS, area_name

BMD = paths.BMD_GLM
ATLAS = paths.ATLAS
OUT = paths.BMD_DATA

N_VERTICES = 59412          # cortical grayordinates, fs_LR_32k (same as cc2017)
PAIRS = ((0, 1), (0, 2), (1, 2))


def load_betas(subject, split):
    f = BMD / f"sub-{subject:02d}" / "prepared_betas" / f"sub-{subject:02d}_organized_betas_task-{split}_normalized.pkl"
    betas, stim = pickle.load(open(f, "rb"))
    return betas[:, :, :N_VERTICES], stim


def glasser_lookup():
    """Per-vertex Glasser name; same lookup as 00a_cc2017_roi_map.py."""
    keys = np.asarray(nib.load(str(ATLAS / "CAB-NP_parcels_LR.dlabel.nii")).get_fdata()).ravel().astype(int)[:N_VERTICES]
    rows = list(csv.reader(open(ATLAS / "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"),
                           delimiter="\t"))
    ki, gi = rows[0].index("KEYVALUE"), rows[0].index("GLASSERLABELNAME")
    key2name = {int(r[ki]): r[gi] for r in rows[1:]}
    return keys, np.array([key2name.get(int(k), "???") for k in keys])


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def corr_across_stimuli(a, b):
    """Voxelwise Pearson r between a and b, both (n_stim, V)."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):   # zeroed NaN vertices -> r = nan, excluded later
        return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def reproducibility(train, n_blocks, seed):
    """Zmat (V x K): per block of stimuli, Fisher z of the repeat-to-repeat r, averaged over pairs."""
    order = np.random.default_rng(seed).permutation(train.shape[0])
    blocks = np.array_split(order, n_blocks)
    Z = np.zeros((train.shape[2], n_blocks))
    for k, idx in enumerate(blocks):
        x = train[idx]
        Z[:, k] = np.mean([fisher_z(corr_across_stimuli(x[:, i], x[:, j])) for i, j in PAIRS], axis=0)
    return Z, [len(b) for b in blocks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--blocks", type=int, default=18, help="K stimulus blocks (cc2017: 18 segments)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Bonferroni-corrected threshold")
    ap.add_argument("--seed", type=int, default=0, help="stimulus -> block assignment")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    parcels, names = glasser_lookup()
    areas = np.array([area_name(g) for g in names])
    summary = []

    for s in a.subjects:
        print(f"\n=== sub-{s:02d}")
        train, stim_train = load_betas(s, "train")
        test, stim_test = load_betas(s, "test")
        valid = np.isfinite(train).all((0, 1)) & np.isfinite(test).all((0, 1))
        print(f"  betas train {train.shape}, test {test.shape}; vertices with NaN excluded: {(~valid).sum()}")

        Z, sizes = reproducibility(np.where(valid, train, 0.0), a.blocks, a.seed)
        t, p = stats.ttest_1samp(Z, 0.0, axis=1)
        t[~valid], p[~valid] = np.nan, 1.0
        # Bonferroni over all cortical grayordinates, as in cc2017; only positive reproducibility
        sel = valid & (p < a.alpha / N_VERTICES) & (t > 0)
        vidx = np.flatnonzero(sel)
        print(f"  K={a.blocks} blocks of {min(sizes)}-{max(sizes)} stimuli; "
              f"selected {len(vidx)} / {N_VERTICES} vertices; median z of selected {np.median(Z[sel].mean(1)):.3f}")

        tag = f"subj{s:02d}"
        torch.save(torch.from_numpy(train[:, :, vidx].astype(np.float32)), out / f"{tag}_train_fmri.pt")
        torch.save(torch.from_numpy(test[:, :, vidx].astype(np.float32)), out / f"{tag}_test_fmri.pt")
        glasser = names[vidx]
        np.savez(out / f"{tag}_roi_map.npz", vertex_idx=vidx.astype(np.int32), glasser=glasser,
                 parcel=parcels[vidx], hemi=np.array([str(g)[0] for g in glasser]),
                 match_corr=np.ones(len(vidx), np.float32))
        np.savez(out / f"{tag}_selection.npz", Zmat=Z.astype(np.float32), t=t, p=p, selected=sel,
                 valid=valid, n_blocks=a.blocks, alpha=a.alpha, seed=a.seed,
                 stim_train=np.array(stim_train), stim_test=np.array(stim_test))

        row = {"subject": s, "selected": len(vidx)}
        for stream in STREAMS:
            row[stream] = int(np.isin(areas[vidx], SCHEME_A[stream]).sum())
        row["other"] = len(vidx) - sum(row[k] for k in STREAMS)
        summary.append(row)
        print("  streams: " + ", ".join(f"{k} {row[k]}" for k in (*STREAMS, "other")))
        print(f"  top areas: {Counter(areas[vidx]).most_common(12)}")

    with open(out / f"summary_K{a.blocks}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
