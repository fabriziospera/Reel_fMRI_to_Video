"""Reading stimulus and candidate clips as 6 frames. Everything returns uint8 (6, H, W, 3) except
the cc2017 stimuli, which the release already ships as float (N, 6, 3, 224, 224) in [0, 1].

cc2017   stimuli    GT_{split}_3fps.pt, 6 frames
         candidates gif [reference | reconstruction], right half (6, 260, 259, 3)
bmd      stimuli    dataset frames at the centres of 6 equal intervals, nearest frame, 268x268
         candidates CogVideoX mp4 (49, 480, 720) -> one trilinear F.interpolate to (6, 268, 268),
                    align_corners=False
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

N_FRAMES, BMD_SIZE = 6, 268


def as_input(clip_u8):
    """(T, H, W, 3) uint8 -> (T, 3, H, W) float in [0, 1], the input of every backbone."""
    return torch.from_numpy(np.ascontiguousarray(clip_u8)).float().div(255).permute(0, 3, 1, 2).contiguous()


def read_gif_reconstruction(path):
    """cc2017 candidate: gif [reference | reconstruction] -> reconstruction (T, H, W, 3) uint8."""
    import imageio.v3 as iio
    a = iio.imread(path, index=None)
    a = a[..., :3] if a.shape[-1] == 4 else a
    assert a.shape[2] % 2 == 0, f"odd width in {path}: {a.shape}"
    return np.ascontiguousarray(np.split(a, 2, axis=2)[1])


def frame_index(T, n=N_FRAMES):
    """Centres of n equal intervals over T frames -> indices (nearest frame)."""
    return np.clip(np.round((np.arange(n) + 0.5) * T / n - 0.5).astype(int), 0, T - 1)


def _to_size(frames, size=BMD_SIZE):
    t = torch.from_numpy(np.ascontiguousarray(frames[..., :3])).float().permute(0, 3, 1, 2)
    if t.shape[-2:] != (size, size):
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.clamp_(0, 255).round_().permute(0, 2, 3, 1).numpy().astype(np.uint8)


def read_bmd_stimulus(frames_dir, video_id):
    """bmd stimulus from the dataset jpgs ({video}_{frame}_{total}.jpg) -> (6, 268, 268, 3) uint8."""
    from PIL import Image
    d = os.path.join(frames_dir, f"{video_id:04d}")
    files = sorted(os.listdir(d), key=lambda f: int(f.split("_")[1]))
    pick = frame_index(len(files))
    return _to_size(np.stack([np.asarray(Image.open(os.path.join(d, files[i])).convert("RGB")) for i in pick]))


def read_bmd_candidate(path):
    """bmd candidate mp4 -> (6, 268, 268, 3) uint8 with a single trilinear interpolation
    (linear in time, bilinear in space, align_corners=False: it samples the same interval
    centres as the stimulus frames)."""
    import imageio.v3 as iio
    v = iio.imread(path, index=None)
    t = torch.from_numpy(np.ascontiguousarray(v[..., :3])).float().div_(255)
    t = t.permute(3, 0, 1, 2)[None]                                   # (1, 3, T, H, W)
    t = F.interpolate(t, size=(N_FRAMES, BMD_SIZE, BMD_SIZE), mode="trilinear", align_corners=False)
    return t[0].permute(1, 2, 3, 0).clamp_(0, 1).mul_(255).round_().numpy().astype(np.uint8)
