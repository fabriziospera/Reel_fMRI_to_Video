"""STEP g2 - fMRI -> text-conditioning decoder, per subject, and the decoded test embeddings.

One himalaya RidgeCV per token (226): averaged training fMRI of all selected voxels -> the
token's 4096 values (alpha per target, 5-fold CV inside the training set; grids in
brainenc/generation.py). The test predictions are renormalised with training statistics only
and saved; the 226 models themselves (~50 GB for a cc2017 subject) are not kept.

  python g2_fit_decoders.py --dataset cc2017 --subject 1
"""
import argparse

import numpy as np
import torch

import paths
from brainenc.generation import DECODER, N_TOKENS, renormalise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=paths.DATASETS)
    ap.add_argument("--subject", type=int, required=True)
    a = ap.parse_args()
    assert a.subject in paths.SUBJECTS[a.dataset]
    dst = paths.predicted_embeddings(a.dataset, a.subject)
    if dst.exists():
        print(f"already done: {dst}")
        return
    from himalaya.backend import set_backend
    from himalaya.ridge import RidgeCV
    backend = set_backend("torch_cuda")

    x_tr = torch.load(paths.fmri(a.dataset, a.subject, "train"), map_location="cpu").float().mean(1)
    x_te = torch.load(paths.fmri(a.dataset, a.subject, "test"), map_location="cpu").float().mean(1)
    emb = torch.load(paths.caption_embeddings(a.dataset, "train"))           # (N, 226, 4096)
    cfg = DECODER[a.dataset]
    print(f"{a.dataset} subject {a.subject}: fMRI {tuple(x_tr.shape)} -> T5 {tuple(emb.shape)}, "
          f"alphas {cfg['alphas']}, intercept {cfg['fit_intercept']}")

    pred_te, mu_pred, sd_pred, best = [], [], [], []
    for i in range(N_TOKENS):
        m = RidgeCV(alphas=cfg["alphas"], fit_intercept=cfg["fit_intercept"]).fit(x_tr, emb[:, i].float())
        p_tr = torch.as_tensor(backend.to_numpy(m.predict(x_tr)))
        pred_te.append(torch.as_tensor(backend.to_numpy(m.predict(x_te))))
        mu_pred.append(p_tr.mean(0))
        sd_pred.append(p_tr.std(0))
        best.append(backend.to_numpy(m.best_alphas_))
        if i % 25 == 0:
            print(f"  token {i}/{N_TOKENS}", flush=True)
    pred_te, mu_pred, sd_pred = torch.stack(pred_te, 1), torch.stack(mu_pred), torch.stack(sd_pred)

    # target statistics: cc2017 on the embeddings as stored (bf16), bmd in float
    true = emb if a.dataset == "cc2017" else emb.float()
    out = renormalise(pred_te, mu_pred, sd_pred, true.mean(0), true.std(0))
    torch.save(out.bfloat16(), dst)                       # generation feeds bf16 to CogVideoX
    vals, counts = np.unique(np.concatenate(best), return_counts=True)
    print("chosen alphas: " + ", ".join(f"{v:g} {100 * c / counts.sum():.1f}%" for v, c in zip(vals, counts)))
    print(f"written {dst} {tuple(out.shape)}")


if __name__ == "__main__":
    main()
