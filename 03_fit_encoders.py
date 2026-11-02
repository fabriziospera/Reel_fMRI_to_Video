"""STEP 03 - Fit the three encoders of a subject and print how well they predict held-out stimuli.

The test numbers (per-voxel correlation) are diagnostics only: nothing here is used to choose
lambda, the pooler or anything else. Step 04 refits the same encoders (deterministically).

  python 03_fit_encoders.py --dataset cc2017 --subject 1
"""
import argparse

import encoding
import paths
from brainenc.metrics import r_vox
from brainenc.scoring import STREAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=paths.DATASETS)
    ap.add_argument("--subject", type=int, required=True)
    a = ap.parse_args()
    assert a.subject in paths.SUBJECTS[a.dataset]

    enc, measured, lam = encoding.fit_encoders(a.dataset, a.subject)
    print(f"\n{a.dataset} subject {a.subject} - encoders on {len(measured['early'])} held-out stimuli")
    print(f"{'stream':>9} {'voxels':>7} {'lambda':>8} | {'r_vox':>8}")
    print("-" * 38)
    for stream in STREAMS:
        pred = enc[stream](encoding.pooled(a.dataset, a.subject, stream, "test"))
        true = measured[stream]
        print(f"{stream:>9} {true.shape[1]:>7} {lam[stream]:>8.0e} | {r_vox(pred, true):>+8.4f}")


if __name__ == "__main__":
    main()
