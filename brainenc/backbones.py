"""The three frozen visual backbones, in one place.

Used for both the stimulus features (step 01) and the candidate features (step 04).

Every backbone takes frames as float in [0, 1].
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# --- early: VGG19-BN, ReLU layers relu1_2, relu2_2, relu3_3, relu4_2, relu5_1 ----------------
VGG_LAYERS = [5, 12, 22, 32, 42]
VGG_CHANNELS = [64, 128, 256, 512, 512]
GRID = 7
VGG_INPUT = 112
EARLY_DIM = sum(c * GRID * GRID for c in VGG_CHANNELS)      # 72128

# --- ventral: OpenCLIP ViT-bigG-14 ----------------------------------------------------------
VENTRAL_DIM = 1280

# --- dorsal: V-JEPA 2 ViT-L, self-supervised checkpoint (no labels) -------------------------
VJEPA_NAME = "facebook/vjepa2-vitl-fpc64-256"
VJEPA_TOKENS, VJEPA_DIM = 2048, 1024                         # 8 temporal x 16x16 spatial
VJEPA_T, VJEPA_RESIZE, VJEPA_CROP = 16, 292, 256

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class VGGPerFrame(torch.nn.Module):
    """Multi-layer VGG19-BN features on a 7x7 grid, ONE VECTOR PER FRAME."""

    def __init__(self, device, size=VGG_INPUT):
        super().__init__()
        weights = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1
        vgg = torchvision.models.vgg19_bn(weights=weights)
        self.feats = vgg.features[:max(VGG_LAYERS) + 1].eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.size = size
        self.mean, self.std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    @torch.no_grad()
    def forward(self, x):                                    # (B, 3, H, W) in [0, 1]
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        outs, h = [], x
        for i, layer in enumerate(self.feats):
            h = layer(h)
            if i in VGG_LAYERS:
                outs.append(F.adaptive_avg_pool2d(h, GRID).flatten(1))
        return torch.cat(outs, 1)                            # (B, EARLY_DIM)


class OpenCLIPTower(torch.nn.Module):
    """ViT-bigG-14 vision tower, final projection. Set OPENCLIP_WEIGHTS for a local checkpoint."""

    def __init__(self, device):
        super().__init__()
        import open_clip
        ckpt = os.environ.get("OPENCLIP_WEIGHTS", "laion2b_s39b_b160k")
        model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=ckpt)
        self.visual = model.visual.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.mean, self.std = CLIP_MEAN.to(device), CLIP_STD.to(device)

    @torch.no_grad()
    def forward(self, x):                                    # (B, 3, H, W) in [0, 1]
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.visual((x - self.mean) / self.std)       # (B, VENTRAL_DIM)


class VJEPA2:
    """V-JEPA 2 encoder, last_hidden_state.

    Input: 6 frames -> 16 by trilinear interpolation (time and space at once), side 292,
    centre crop 256, ImageNet normalisation, fp16.
    """

    def __init__(self, device):
        from transformers import VJEPA2Model
        self.dev = device
        self.m = VJEPA2Model.from_pretrained(VJEPA_NAME, dtype=torch.float16).to(device).eval()
        self.mean = IMAGENET_MEAN.view(1, 1, 3, 1, 1).to(device)
        self.std = IMAGENET_STD.view(1, 1, 3, 1, 1).to(device)

    @torch.no_grad()
    def from_uint8(self, clips_u8):
        """(B, T, H, W, 3) uint8 -> (B, 2048, 1024) fp16. Use this for every uint8 clip (bmd
        stimuli, all candidates): the permuted layout changes the interpolation numerics, and the
        features are reproduced exactly only through this path."""
        x = torch.as_tensor(np.ascontiguousarray(clips_u8)).to(self.dev).float().div_(255)
        return self(x.permute(0, 1, 4, 2, 3))

    @torch.no_grad()
    def __call__(self, x):
        """(B, T, 3, H, W) float in [0, 1] -> (B, 2048, 1024) fp16 on the device (cc2017 stimuli)."""
        v = x.to(self.dev).float().permute(0, 2, 1, 3, 4)                  # (B, 3, T, H, W)
        v = F.interpolate(v, size=(VJEPA_T, VJEPA_RESIZE, VJEPA_RESIZE), mode="trilinear",
                          align_corners=False)
        o = (VJEPA_RESIZE - VJEPA_CROP) // 2
        v = v[..., o:o + VJEPA_CROP, o:o + VJEPA_CROP].permute(0, 2, 1, 3, 4)
        v = ((v - self.mean) / self.std).half()
        h = self.m(pixel_values_videos=v, skip_predictor=True).last_hidden_state
        assert h.shape[1:] == (VJEPA_TOKENS, VJEPA_DIM), h.shape
        return h
