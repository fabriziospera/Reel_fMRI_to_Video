# fMRI-to-video reconstruction with brain-only reranking

Two stages:

1. **Generation (steps g1–g4).** The measured fMRI is decoded into the text-conditioning space of
   CogVideoX-5b. The video model then generates 100 candidate reconstructions per test stimulus,
   one per seed.
2. **Reranking (steps 00–04).** One candidate per stimulus is selected from the measured fMRI
   only. Three encoders predict, from each candidate, the fMRI response of a visual ROI. The
   candidate whose predicted responses best match the measured ones is kept.

Datasets: **CC2017** (Wen et al., 2018; 3 subjects) and **BOLD Moments** (BMD, Lahner et al.,
2024; 10 subjects). Every script takes `--dataset {cc2017,bmd}`. The dataset-specific settings
are in `datasets.py`.

## Generation (`brainenc/generation.py`)

| step | |
|---|---|
| g1 | T5 embeddings (CogVideoX-5b text encoder, 226 × 4096) of the training captions. CC2017: `GT_*_caption.pt`, one caption per clip. BMD: the 5 captions of `annotations.json`, averaged token by token. |
| g2 | One himalaya `RidgeCV` per token, from the averaged training fMRI (all selected voxels) to the token's 4096 values. α per target by 5-fold CV. CC2017: α ∈ {1, 10, 1e2, 1e3, 1e4}, no intercept. BMD: α ∈ {1, 10, 1e3, 1e4, 5e4, 1e5}, with intercept. Then renormalisation, per token and dimension: the decoded test embeddings are standardised with the mean/SD of the decoded training embeddings and rescaled to the mean/SD of the true training embeddings. |
| g3 | CogVideoX-5b in bf16: 49 frames, 50 steps, guidance 3.5, seeds 42–141, exported at 8 fps. |
| g4 | CC2017 only: mp4 → gif `[reference \| reconstruction]`. The reconstruction is reduced to 6×260×259 with one trilinear interpolation; the reference is the left half of the NeuroClips gif of the same stimulus (`NEUROCLIPS_GIFS`). The gif palette depends on the whole frame, so this reference is needed to reproduce the reconstruction pixels exactly. |

## Reranking

| stream | features | ROI (Glasser areas, bilateral) | pooling |
|---|---|---|---|
| early | VGG19-BN `relu1_2/2_2/3_3/4_2/5_1`, 112×112 input, 7×7 grid per frame (72128) | V1, V2, V3, V4 | learned `TemporalAttnPool` (one query per cell and layer, softmax over the 6 frames) |
| ventral | OpenCLIP ViT-bigG-14, final projection (1280) | FFC, PIT, V8, VMV1–3, VVC, PHA1–3, TE2p | mean over the 6 frames |
| dorsal | V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`, encoder): 6 → 16 frames (trilinear), side 292, centre crop 256, ImageNet normalisation → 2048 × 1024 tokens | V3A, V3B, V6, V6A, V7, IPS1, LO1–3, FST, MT, MST, V3CD, V4t, PH, IP0 | learned `FactorizedAttnPool` (1 query, 4 heads, time/space positions on the keys) |

- **Poolers** (per subject): trained with a temporary linear head that predicts the z-scored ROI
  fMRI (MSE); the head is then discarded. AdamW lr 3e-4, wd 1e-2, cosine schedule, 120 epochs,
  patience 6, gradient clip 1, bf16 autocast, batch 32 (early) / 64 (dorsal), seed 0.
- **Readouts:** one ridge per stream on the pooled features (dual form for the early stream),
  λ ∈ {1e1, …, 1e6}, refit on the whole training set with the chosen λ.
- **Targets:** training fMRI averaged over repeats and z-scored per voxel with the training
  mean/SD. The test fMRI (averaged over its 10 repeats) is normalised with the same statistics.
- **Scoring** (`brainenc/scoring.py`), per test stimulus:

  ```
  encoder(candidate features)  -> predicted ROI response, per stream
    -> Pearson r with the measured response over the ROI voxels (all voxels weigh the same)
    -> z-score of r across the K candidates, per stream (population SD, +1e-9)
    -> sum over the three streams
    -> argmax (first index on ties)
  ```

### Dataset-specific settings (`datasets.py`)

| | CC2017 | BMD |
|---|---|---|
| samples | 4320 train TRs (18 movies × 240), 1200 test | 1000 train clips, 102 test |
| voxels | released selection; Glasser areas recovered by `00a` | selected by `00b` |
| pooler early stopping | movie group 0 (3 whole movies, 720 TRs) | random 12% of the training clips |
| ridge λ | 6-fold CV over movie groups, MSE summed over folds | random 15% of the training clips |
| stimulus clips | `GT_{split}_3fps.pt` (6 frames) | dataset frames at the centres of 6 equal intervals, nearest frame, 268×268 |
| candidate clips | gif, right half (6×260×259) | mp4 → 6×268×268, one trilinear interpolation |



## Running it

```bash
# inputs
export CC2017_DATA=/path/to/cc2017_video_fmri_dataset     # subjXX_*_fmri.pt, GT_*_3fps.pt, GT_*_caption.pt
export CC2017_RAW=/path/to/cc2017_raw/video_fmri_dataset   # raw CIFTI, step 00a only
export NEUROCLIPS_GIFS=/path/to/neuroclips_gifs            # subj{S}/sample_{i}/video_{i}_seed_*.gif, step g4
export BMD_GLM=/path/to/BOLDMoments/derivatives/versionB/fsLR32k/GLM   # step 00b only
export BMD_FRAMES=/path/to/BOLDMoments/stimulus_set/frames
export BMD_ANNOTATIONS=/path/to/BOLDMoments/derivatives/stimuli_metadata/annotations.json
export GLASSER_ATLAS=/path/to/atlas_glasser                # CAB-NP dlabel + LabelKey
# outputs (defaults: inside the repository)
export BMD_DATA=... BRAINENC_CACHE=... BRAINENC_CKPT=... BRAINENC_OUT=... BRAINENC_GENERATED=...
# to rerank an existing candidate pool instead of running g3/g4:
#   CC2017_CANDIDATES (subjXX/sample_{i}/*.gif), BMD_CANDIDATES (subjXX/sample_{i}/seed_*.mp4)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python paths.py                                            # check what is found
```

**CC2017**
```bash
# generation
python g1_caption_embeddings.py --dataset cc2017
python g2_fit_decoders.py --dataset cc2017 --subject 1                        # per subject
python g3_generate.py --dataset cc2017 --subjects 1 2 3 --gpus 0 1 2 3
python g4_cc2017_to_gif.py --subject 1                                        # per subject
# reranking
python 00a_cc2017_roi_map.py --subject 1                                      # per subject
for st in early ventral dorsal; do for sp in train test; do
  python 01_extract_features.py --dataset cc2017 --stream $st --split $sp; done; done
python 02_train_poolers.py --dataset cc2017 --subject 1 --stream early        # per subject
python 02_train_poolers.py --dataset cc2017 --subject 1 --stream dorsal
python 03_fit_encoders.py  --dataset cc2017 --subject 1                       # optional diagnostic
python 04_rerank.py extract --dataset cc2017 --subject 1
python 04_rerank.py rank    --dataset cc2017 --subject 1
python 04_rerank.py export  --dataset cc2017 --subject 1 --out eval/cc2017_subj1
```

**BMD**: the same commands with `--dataset bmd` and subjects 1–10, except:
- step 00 is `python 00b_bmd_voxel_selection.py` (all subjects), run before g2;
- there is no g4.

Outputs:
- `04_rerank.py rank` writes `selections/<dataset>/selection_subjXX.npz`, with the chosen seed
  per test stimulus and the λ of the three readouts.
- `export` writes `sample_{i}/rank1_seed{S}_corr0.0.gif` (`[reference | reconstruction]`), the
  input of the evaluation code (`eval_metrics.py` of NEURONS).
- Feature extraction, generation and candidate extraction are resumable.
  `04_rerank.py extract` accepts `--start/--end` to split the test stimuli across GPUs.

## Files

| file | role |
|---|---|
| `paths.py` | filesystem paths |
| `datasets.py` | dataset-specific splits, λ selection and clip readers |
| `encoding.py` | the three encoders of a subject (pooled features → ridge) |
| `g1_caption_embeddings.py` | caption T5 embeddings |
| `g2_fit_decoders.py` | fMRI → T5 decoders, decoded and renormalised test embeddings |
| `g3_generate.py` | CogVideoX-5b candidates, multi-GPU |
| `g4_cc2017_to_gif.py` | CC2017 mp4 → `[reference \| reconstruction]` gif |
| `00a_cc2017_roi_map.py` | CC2017 voxel → Glasser area map (time-series matching on the raw CIFTI) |
| `00b_bmd_voxel_selection.py` | BMD voxel selection and ROI map |
| `01_extract_features.py` | stimulus features |
| `02_train_poolers.py` | early and dorsal poolers |
| `03_fit_encoders.py` | readouts and held-out per-voxel correlation (printed only) |
| `04_rerank.py` | candidate features, selection, export |
| `brainenc/backbones.py` | VGG19-BN, OpenCLIP bigG, V-JEPA 2 |
| `brainenc/pooling.py` | the two pooling modules |
| `brainenc/ridge.py` | ridge with random hold-out or grouped CV for λ |
| `brainenc/scoring.py` | candidate scoring |
| `brainenc/clips.py` | clip readers |
| `brainenc/generation.py` | generation settings |
| `brainenc/rois.py` | Glasser areas → stream masks |
| `brainenc/metrics.py` | per-voxel correlation |


## Requirements

Python 3.10+, `torch`, `torchvision`, `transformers` with `VJEPA2Model` (tested with 4.57.6),
`open_clip_torch` (`OPENCLIP_WEIGHTS` for a local bigG checkpoint), `diffusers` (tested with
0.31.0), `himalaya` (tested with 0.4.9), `numpy`, `scipy`, `imageio` with an mp4 backend,
`pillow`, `nibabel` (step 00). GPU required. The CC2017 training V-JEPA tokens take about 18 GB
(fp16).
