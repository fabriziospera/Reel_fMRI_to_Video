"""Encoding diagnostic on held-out stimuli. Printed only, never used to choose anything."""


def r_vox(pred, true):
    """Per-voxel Pearson across stimuli, averaged over voxels."""
    p = pred - pred.mean(0, keepdim=True)
    t = true - true.mean(0, keepdim=True)
    r = (p * t).sum(0) / (p.norm(dim=0) * t.norm(dim=0) + 1e-9)
    return float(r.mean())
