"""STEP 04 - Rerank the candidate reconstructions of every test stimulus, brain-only.

  extract   embed every candidate with the same backbones and poolers as the stimuli
  rank      fit the encoders (as step 03), score the candidates (brainenc/scoring.py), keep the
            argmax per stimulus -> selections/<dataset>/selection_subjXX.npz
  export    write the chosen reconstructions in the layout of the evaluation code
            (sample_{i}/rank1_seed{S}_corr0.0.gif, [reference | reconstruction])

  python 04_rerank.py extract --dataset cc2017 --subject 1 [--start 0 --end 1200]
  python 04_rerank.py rank    --dataset cc2017 --subject 1
  python 04_rerank.py export  --dataset cc2017 --subject 1 --out /path/to/eval_dir
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import datasets
import encoding
import paths
from brainenc.backbones import VJEPA2, OpenCLIPTower, VGGPerFrame
from brainenc.clips import as_input
from brainenc.scoring import STREAMS, score_candidates

DEV = encoding.DEV
CHUNK = 25                                    # candidates per forward pass (VGG, V-JEPA)


def _read(args):
    dataset, path = args
    try:
        return datasets.read_candidate(dataset, path)
    except Exception as e:                    # a corrupt file must not stop the run
        return e


@torch.no_grad()
def extract(dataset, subject, start, end):
    dst = paths.candidate_features(dataset, subject)
    dst.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(start, end) if not (dst / f"sample_{i}.npz").exists()]
    print(f"{dataset} subject {subject} [{start}:{end}]: {len(todo)} stimuli to do")
    if not todo:
        return
    vgg = VGGPerFrame(DEV).to(DEV).eval()
    clip = OpenCLIPTower(DEV)
    vjepa = VJEPA2(DEV)
    early_pool = encoding.load_pooler(dataset, subject, "early")
    dorsal_pool = encoding.load_pooler(dataset, subject, "dorsal")
    io = ThreadPoolExecutor(8)

    for n, idx in enumerate(todo):
        files = datasets.candidate_files(dataset, subject, idx)
        if not files:
            print(f"  [skip] sample_{idx}: no candidates")
            continue
        seeds = sorted(files)
        read = list(io.map(_read, [(dataset, files[s]) for s in seeds]))
        for s, c in zip(seeds, read):
            if isinstance(c, Exception):
                print(f"  [skip] sample_{idx} seed {s}: {type(c).__name__}")
        seeds = [s for s, c in zip(seeds, read) if not isinstance(c, Exception)]
        u8 = np.stack([c for c in read if not isinstance(c, Exception)])        # (K, 6, H, W, 3)
        x = torch.stack([as_input(c) for c in u8])                               # (K, 6, 3, H, W)

        early, ventral, dorsal = [], [], []
        for i in range(0, len(x), CHUNK):
            xb = x[i:i + CHUNK].to(DEV)
            k = xb.shape[0]
            per_frame = vgg(xb.reshape(k * 6, *xb.shape[2:])).reshape(k, 6, -1)
            early.append(early_pool(per_frame).half().cpu())
            dorsal.append(dorsal_pool(vjepa.from_uint8(u8[i:i + CHUNK]).float()).half().cpu())
        for c in x:                                                              # fixed mean over frames
            ventral.append(clip(c.to(DEV)).mean(0).half().cpu())

        tmp = dst / f"sample_{idx}.tmp.npz"
        np.savez(tmp, seeds=np.array(seeds), early=torch.cat(early).numpy(),
                 ventral=torch.stack(ventral).numpy(), dorsal=torch.cat(dorsal).numpy())
        os.replace(tmp, dst / f"sample_{idx}.npz")
        if n % 20 == 0:
            print(f"  sample_{idx}: {len(seeds)} candidates ({n + 1}/{len(todo)})", flush=True)
    print(f"done -> {dst}")


def rank(dataset, subject):
    src = paths.candidate_features(dataset, subject)
    n_test = paths.N_STIMULI[dataset]["test"]
    missing = [i for i in range(n_test) if not (src / f"sample_{i}.npz").exists()]
    if missing:
        sys.exit(f"{len(missing)} stimuli without candidate features in {src} (first: {missing[:5]})\n"
                 f"run: 04_rerank.py extract --dataset {dataset} --subject {subject}")

    enc, measured, lam = encoding.fit_encoders(dataset, subject)
    chosen = np.zeros(n_test, int)
    for i in range(n_test):
        z = np.load(src / f"sample_{i}.npz")
        score = score_candidates({s: z[s] for s in STREAMS}, enc, {s: measured[s][i] for s in STREAMS})
        chosen[i] = int(z["seeds"][int(np.argmax(score))])
    dst = paths.selection(dataset, subject)
    np.savez(dst, stimulus=np.arange(n_test), seed=chosen, **{f"lambda_{s}": lam[s] for s in STREAMS})
    print(f"selected one candidate for each of {n_test} stimuli -> {dst}")


def export(dataset, subject, out):
    """Chosen reconstruction per stimulus as [reference | reconstruction] gif.

    cc2017: the candidate gif already has this layout and is linked. bmd: the reference is the
    6-frame stimulus clip of step 01 and the reconstruction its 6-frame conversion."""
    import imageio.v3 as iio
    sel = np.load(paths.selection(dataset, subject))
    gt = np.load(paths.bmd_stimuli_u8("test"), mmap_mode="r") if dataset == "bmd" else None
    for i, s in zip(sel["stimulus"], sel["seed"]):
        d = os.path.join(out, f"sample_{i}")
        os.makedirs(d, exist_ok=True)
        g = os.path.join(d, f"rank1_seed{s}_corr0.0.gif")
        if os.path.lexists(g):
            continue
        src = datasets.candidate_files(dataset, subject, int(i))[int(s)]
        if dataset == "cc2017":
            os.symlink(os.path.abspath(src), g)
        else:
            pair = np.concatenate([np.asarray(gt[i]), datasets.read_candidate(dataset, src)], axis=2)
            iio.imwrite(g, pair, extension=".gif", loop=0)
    print(f"{len(sel['seed'])} reconstructions -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("extract", "rank", "export"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=paths.DATASETS)
        p.add_argument("--subject", type=int, required=True)
        if name == "extract":
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--end", type=int, default=None)
        if name == "export":
            p.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.subject in paths.SUBJECTS[a.dataset]
    paths.ensure_dirs(a.dataset)

    if a.mode == "extract":
        extract(a.dataset, a.subject, a.start, a.end or paths.N_STIMULI[a.dataset]["test"])
    elif a.mode == "rank":
        rank(a.dataset, a.subject)
    else:
        export(a.dataset, a.subject, a.out)


if __name__ == "__main__":
    main()
