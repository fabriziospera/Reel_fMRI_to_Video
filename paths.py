"""Every filesystem path, and nothing else. Two datasets: cc2017 and bmd (BOLD Moments).

Inputs are read from the directories below (set the environment variables); everything this
repository writes goes under CACHE, CKPT and OUT.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATASETS = ("cc2017", "bmd")


def _env(name, default):
    return Path(os.environ.get(name, default))


# ---- inputs ------------------------------------------------------------------------------
# cc2017: preprocessed release (subjXX_{train,test}_fmri.pt, GT_{train,test}_3fps.pt,
#         GT_{train,test}_caption.pt)
CC2017_DATA = _env("CC2017_DATA", REPO / "data" / "cc2017")
CC2017_RAW = _env("CC2017_RAW", REPO / "data" / "cc2017_raw")             # step 00a only
# reference half of the cc2017 gifs: NeuroClips gifs sample_{i}/video_{i}_seed_*.gif (step g4)
NEUROCLIPS_GIFS = _env("NEUROCLIPS_GIFS", REPO / "data" / "neuroclips_gifs")
# bmd: GLMsingle betas (step 00b), stimulus frames, human captions
BMD_GLM = _env("BMD_GLM", REPO / "data" / "bmd_glm")                       # .../versionB/fsLR32k/GLM
BMD_FRAMES = _env("BMD_FRAMES", REPO / "data" / "bmd_frames")              # stimulus_set/frames
BMD_ANNOTATIONS = _env("BMD_ANNOTATIONS", REPO / "data" / "bmd_annotations.json")
# Glasser labels on fs_LR_32k (CAB-NP release), shared by both datasets
ATLAS = _env("GLASSER_ATLAS", REPO / "data" / "atlas_glasser")

# ---- outputs -----------------------------------------------------------------------------
BMD_DATA = _env("BMD_DATA", REPO / "data" / "bmd")       # written by step 00b, cc2017 format
CACHE = _env("BRAINENC_CACHE", REPO / "cache")
CKPT = _env("BRAINENC_CKPT", REPO / "checkpoints")
OUT = _env("BRAINENC_OUT", REPO / "selections")
GENERATED = _env("BRAINENC_GENERATED", REPO / "generated")                 # steps g3-g4
# candidates the reranker reads (default: what steps g3-g4 write)
CC2017_CANDIDATES = _env("CC2017_CANDIDATES", GENERATED / "cc2017_gif")    # subj{S}/sample_{i}/*.gif
BMD_CANDIDATES = _env("BMD_CANDIDATES", GENERATED / "bmd")                 # subj{S}/sample_{i}/*.mp4

SUBJECTS = {"cc2017": (1, 2, 3), "bmd": tuple(range(1, 11))}
N_STIMULI = {"cc2017": {"train": 4320, "test": 1200}, "bmd": {"train": 1000, "test": 102}}
N_FRAMES = 6


def data_dir(dataset):
    return CC2017_DATA if dataset == "cc2017" else BMD_DATA


def fmri(dataset, subject, split):
    """(N, repeats, voxels): cc2017 train (4320, 2, V), test (1200, 10, V); bmd (1000, 3, V), (102, 10, V)."""
    return data_dir(dataset) / f"subj{subject:02d}_{split}_fmri.pt"


def roi_map(dataset, subject):
    return data_dir(dataset) / f"subj{subject:02d}_roi_map.npz"


def cc2017_stimuli(split):
    """(N, 6, 3, 224, 224) float in [0, 1], 6 frames at 3 fps."""
    return CC2017_DATA / f"GT_{split}_3fps.pt"


def bmd_stimuli_u8(split):
    """(N, 6, 268, 268, 3) uint8, written by step 01 from the dataset frames."""
    return CACHE / "bmd" / f"gt_{split}_u8.npy"


def features(dataset, stream, split):
    """early (N, 6, 72128) fp16 | ventral (N, 1280) | dorsal (N, 2048, 1024) fp16 V-JEPA tokens."""
    return CACHE / dataset / f"{stream}_{split}.npy"


def pooler(dataset, subject, stream):
    return CKPT / dataset / f"{stream}_pool_subj{subject:02d}.pt"


# ---- generation (steps g1-g4) ------------------------------------------------------------
def cc2017_captions(split):
    """(N,) one caption per clip, shipped with the preprocessed release."""
    return CC2017_DATA / f"GT_{split}_caption.pt"


def caption_embeddings(dataset, split):
    """(N, 226, 4096) T5 embeddings of the captions (CogVideoX-5b text encoder), step g1."""
    return CACHE / dataset / f"t5_captions_{split}.pt"


def predicted_embeddings(dataset, subject):
    """(N_test, 226, 4096) bf16 embeddings decoded from fMRI and renormalised, step g2."""
    return CACHE / dataset / f"t5_predicted_subj{subject:02d}.pt"


def generated(dataset, subject):
    """sample_{i}/seed_{S}.mp4, CogVideoX-5b, 49 frames at 8 fps, step g3."""
    return GENERATED / dataset / f"subj{subject:02d}"


# ---- reranking ---------------------------------------------------------------------------
def candidates(dataset, subject):
    """cc2017: sample_{i}/video_{i}_seed_{S}.gif, [reference | reconstruction] side by side (g4).
    bmd:    sample_{i}/seed_{S}.mp4, the generated video (49 frames, 480x720) (g3)."""
    if dataset == "cc2017":
        return CC2017_CANDIDATES / f"subj{subject:02d}"
    return BMD_CANDIDATES / f"subj{subject:02d}"


def candidate_features(dataset, subject):
    """sample_{i}.npz: seeds (K,), early (K, 72128), ventral (K, 1280), dorsal (K, 1024)."""
    return CACHE / dataset / f"candidates_subj{subject:02d}"


def selection(dataset, subject):
    return OUT / dataset / f"selection_subj{subject:02d}.npz"


def ensure_dirs(dataset):
    for d in (CACHE / dataset, CKPT / dataset, OUT / dataset):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"repo  {REPO}")
    for name in ("CC2017_DATA", "CC2017_RAW", "NEUROCLIPS_GIFS", "BMD_GLM", "BMD_FRAMES",
                 "BMD_ANNOTATIONS", "ATLAS", "BMD_DATA", "CACHE", "CKPT", "OUT", "GENERATED",
                 "CC2017_CANDIDATES", "BMD_CANDIDATES"):
        p = globals()[name]
        print(f"{name:<18}{p}   {'ok' if p.exists() else 'MISSING'}")
