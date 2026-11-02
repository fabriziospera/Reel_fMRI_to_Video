"""STEP g3 - Generate the candidate pool: CogVideoX-5b on the decoded embeddings, 100 seeds per
test stimulus (settings in brainenc/generation.py). One process per GPU, a shared queue of
(subject, stimulus); resumable: existing videos are skipped, and each video is written to a
temporary file first so an interrupted one does not count as done.

  python g3_generate.py --dataset cc2017 --subjects 1 2 3 --gpus 0 1 2 3
  python g3_generate.py --dataset bmd --subjects 1 --samples 0 1 2 --gpus 0
"""
import argparse
import multiprocessing as mp
import os

import torch

import paths
from brainenc import generation as G


def worker(gpu, queue, dataset, seed_list):
    from diffusers.utils import export_to_video
    dev = f"cuda:{gpu}"
    torch.cuda.set_device(dev)
    pipe = G.load_pipeline(dev)
    embeds = {}
    while True:
        task = queue.get()
        if task is None:
            break
        subject, i = task
        if subject not in embeds:
            embeds = {subject: torch.load(paths.predicted_embeddings(dataset, subject))}
        out = paths.generated(dataset, subject) / f"sample_{i}"
        out.mkdir(parents=True, exist_ok=True)
        todo = [s for s in seed_list if not (out / f"seed_{s}.mp4").exists()]
        for s in todo:
            try:
                frames = G.generate(pipe, embeds[subject][i], s, dev)
                tmp = out / f"seed_{s}.tmp.mp4"
                export_to_video(frames, str(tmp), fps=G.EXPORT_FPS)
                os.replace(tmp, out / f"seed_{s}.mp4")
            except Exception as e:                      # one failure must not stop the pool
                print(f"[gpu {gpu}] subj{subject:02d} sample_{i} seed {s}: {e!r}", flush=True)
                torch.cuda.empty_cache()
        print(f"[gpu {gpu}] subj{subject:02d} sample_{i}: {len(todo)} generated", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=paths.DATASETS)
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--samples", type=int, nargs="+", help="test indices (default: all)")
    ap.add_argument("--n_seeds", type=int, default=G.N_SEEDS)
    a = ap.parse_args()
    for s in a.subjects:
        assert s in paths.SUBJECTS[a.dataset]
        if not paths.predicted_embeddings(a.dataset, s).exists():
            raise SystemExit(f"missing {paths.predicted_embeddings(a.dataset, s)}: run g2 first")

    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()
    n_test = paths.N_STIMULI[a.dataset]["test"]
    for s in a.subjects:
        for i in (a.samples if a.samples is not None else range(n_test)):
            queue.put((s, i))
    for _ in a.gpus:
        queue.put(None)
    procs = [mp.Process(target=worker, args=(g, queue, a.dataset, G.seeds(a.n_seeds))) for g in a.gpus]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("done")


if __name__ == "__main__":
    main()
