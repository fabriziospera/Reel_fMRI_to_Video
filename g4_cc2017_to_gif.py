"""STEP g4 - cc2017 only: generated mp4 -> 6-frame gif [reference | reconstruction], the format
read by the reranker (step 04) and by the evaluation code.

  reconstruction  mp4 (49, 480, 720) -> one trilinear F.interpolate to (6, 260, 259),
                  align_corners=False
  reference       left half of the NeuroClips gif of the same stimulus (lowest seed)

The gif palette is computed on the whole [reference | reconstruction] frame, so the NeuroClips
reference is needed to reproduce the reconstruction pixels exactly.

  python g4_cc2017_to_gif.py --subject 1 [--start 0 --end 1200]
"""
import argparse
import glob
import re

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

import paths

N_FRAMES, H, W = 6, 260, 259


def mp4_to_frames(path):
    """(T, 480, 720, 3) uint8 -> (6, 260, 259, 3) uint8."""
    v = iio.imread(path, index=None)
    t = torch.from_numpy(np.ascontiguousarray(v[..., :3])).float().div_(255).permute(3, 0, 1, 2)[None]
    t = F.interpolate(t, size=(N_FRAMES, H, W), mode="trilinear", align_corners=False)
    return t[0].permute(1, 2, 3, 0).clamp_(0, 1).mul_(255).round_().numpy().astype(np.uint8)


def reference(subject, idx):
    gs = sorted(glob.glob(str(paths.NEUROCLIPS_GIFS / f"subj{subject}" / f"sample_{idx}" / f"video_{idx}_seed_*.gif")),
                key=lambda p: int(re.search(r"seed_(\d+)\.gif$", p).group(1)))
    return np.split(iio.imread(gs[0], index=None), 2, axis=2)[0] if gs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True, choices=paths.SUBJECTS["cc2017"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=paths.N_STIMULI["cc2017"]["test"])
    a = ap.parse_args()
    src_root, dst_root = paths.generated("cc2017", a.subject), paths.candidates("cc2017", a.subject)
    done = 0
    for idx in range(a.start, a.end):
        srcs = {}
        for f in glob.glob(str(src_root / f"sample_{idx}" / "*.mp4")):
            m = re.search(r"seed_(\d+)\.mp4$", f)
            if m and not f.endswith(".tmp.mp4"):
                srcs[int(m.group(1))] = f
        out = dst_root / f"sample_{idx}"
        missing = [s for s in sorted(srcs) if not (out / f"video_{idx}_seed_{s}.gif").exists()]
        if not missing:
            continue
        ref = reference(a.subject, idx)
        if ref is None:
            print(f"  [skip] sample_{idx}: no NeuroClips reference in {paths.NEUROCLIPS_GIFS}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        for s in missing:
            try:
                pred = mp4_to_frames(srcs[s])
            except Exception as e:
                print(f"  [skip] sample_{idx} seed {s}: {type(e).__name__}")
                continue
            iio.imwrite(out / f"video_{idx}_seed_{s}.gif", np.concatenate([ref, pred], axis=2),
                        extension=".gif", loop=0)
        done += 1
    print(f"{done} stimuli converted -> {dst_root}")


if __name__ == "__main__":
    main()
