"""Candidate generation: fMRI -> text-conditioning embedding -> CogVideoX-5b, many seeds.

  1. caption embeddings   CogVideoX-5b's T5 encoder on the captions of the training clips,
                          (226 tokens x 4096). cc2017: one caption per clip (GT_*_caption.pt).
                          bmd: the 5 human captions of each clip, averaged token by token.
  2. fMRI decoder         one himalaya RidgeCV per token (226), averaged fMRI of all selected
                          voxels -> that token's 4096 values, per-target alpha by 5-fold CV on
                          the training set.
  3. renormalisation      the decoded test embeddings are standardised with the mean/SD of the
                          decoded training embeddings and mapped to the mean/SD of the true
                          training embeddings (per token and dimension).
  4. generation           CogVideoX-5b, bf16, 49 frames, 50 steps, guidance 3.5, seeds 42..141,
                          exported at 8 fps.
"""
import numpy as np
import torch

COGVIDEOX = "THUDM/CogVideoX-5b"
N_TOKENS, EMB_DIM = 226, 4096
NUM_FRAMES, NUM_STEPS, GUIDANCE, EXPORT_FPS = 49, 50, 3.5, 8
BASE_SEED, N_SEEDS = 42, 100

# Per-dataset decoder settings.
DECODER = {
    "cc2017": {"alphas": [1, 10, 1e2, 1e3, 1e4], "fit_intercept": False},
    "bmd": {"alphas": [1, 10, 1e3, 1e4, 5e4, 1e5], "fit_intercept": True},
}


def renormalise(pred_test, pred_train_mean, pred_train_std, true_train_mean, true_train_std):
    z = (pred_test.float() - pred_train_mean) / pred_train_std
    return z * true_train_std + true_train_mean


def seeds(n_seeds=N_SEEDS, base=BASE_SEED):
    return list(range(base, base + n_seeds))


def load_pipeline(device, text_only=False):
    from diffusers import CogVideoXPipeline
    if text_only:                      # encode_prompt needs only the text encoder
        pipe = CogVideoXPipeline.from_pretrained(COGVIDEOX, transformer=None, vae=None,
                                                 torch_dtype=torch.bfloat16)
        pipe.text_encoder.to(device)
        return pipe
    return CogVideoXPipeline.from_pretrained(COGVIDEOX, torch_dtype=torch.bfloat16).to(device)


@torch.no_grad()
def generate(pipe, prompt_embeds, seed, device):
    """(226, 4096) embedding -> list of 49 PIL frames."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipe(prompt_embeds=prompt_embeds[None].bfloat16().to(device), num_videos_per_prompt=1,
                num_inference_steps=NUM_STEPS, num_frames=NUM_FRAMES, guidance_scale=GUIDANCE,
                generator=generator).frames[0]


def as_numpy(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
