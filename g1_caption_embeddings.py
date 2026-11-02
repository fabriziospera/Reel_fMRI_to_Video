"""STEP g1 - T5 embeddings of the clip captions (CogVideoX-5b text encoder), once per dataset.

  cc2017  GT_{split}_caption.pt, one automatic caption per clip      -> (N, 226, 4096) bf16
  bmd     annotations.json, 5 human captions per clip, token-wise mean -> (N, 226, 4096) fp16

Only the training embeddings are used downstream (decoder targets and renormalisation
statistics); the test ones are written for diagnostics.

  python g1_caption_embeddings.py --dataset cc2017
"""
import argparse
import json

import torch

import paths
from brainenc.generation import load_pipeline


@torch.no_grad()
def encode(pipe, captions, device, batch=32):
    out = []
    for i in range(0, len(captions), batch):
        e, _ = pipe.encode_prompt(list(captions[i:i + batch]), do_classifier_free_guidance=False,
                                  device=device)
        out.append(e.cpu())
    return torch.cat(out)


def captions(dataset, split):
    if dataset == "cc2017":
        return [str(c) for c in torch.load(paths.cc2017_captions(split), weights_only=False)]
    annot = json.load(open(paths.BMD_ANNOTATIONS))
    ids = range(1, 1001) if split == "train" else range(1001, 1103)     # rows of the fMRI files
    caps = [annot[f"{i:04d}"]["text_descriptions"] for i in ids]
    assert all(len(c) == 5 for c in caps)
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=paths.DATASETS)
    a = ap.parse_args()
    paths.ensure_dirs(a.dataset)
    dev = "cuda"
    pipe = None
    for split in ("train", "test"):
        dst = paths.caption_embeddings(a.dataset, split)
        if dst.exists():
            print(f"already done: {dst}")
            continue
        pipe = pipe or load_pipeline(dev, text_only=True)
        caps = captions(a.dataset, split)
        if a.dataset == "cc2017":
            emb = encode(pipe, caps, dev)                                    # bf16, as encoded
        else:
            flat = [c for five in caps for c in five]
            emb = encode(pipe, flat, dev).float().view(len(caps), 5, 226, 4096).mean(1).half()
        torch.save(emb, dst)
        print(f"written {dst} {tuple(emb.shape)} {emb.dtype}")


if __name__ == "__main__":
    main()
