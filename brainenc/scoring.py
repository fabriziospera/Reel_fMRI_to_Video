"""How a candidate reconstruction is scored against the measured brain response.

Given the candidate reconstructions of one test stimulus, each is embedded, passed through the
encoder of each stream, and compared with the fMRI actually measured for that stimulus. No
ground-truth video is involved: the score is brain-only.

    encoder(candidate features)                 predicted response, (K, V) per stream
      -> Pearson correlation over the voxels of the stream's ROI with the measured response
         (all voxels weigh the same; nothing is computed from the test repeats)
      -> z-score WITHIN the stimulus, per stream, across its K candidates
      -> sum of the three streams
      -> argmax
"""
import numpy as np
import torch

STREAMS = ("early", "ventral", "dorsal")


def unit_rows(X):
    """Centre each row over the voxels and bring it to norm 1: Pearson = dot product."""
    X = X - X.mean(1, keepdim=True)
    return X / (X.norm(dim=1, keepdim=True) + 1e-9)


def stream_correlations(predicted, measured):
    """(K, V) predicted responses of K candidates vs (V,) measured -> (K,) Pearson r, float64."""
    return (unit_rows(predicted.float()) @ unit_rows(measured.float()[None])[0]).double().numpy()


def zscore_within(c):
    """Standardise the scores of one stimulus across its candidates (population SD, +1e-9)."""
    c = np.asarray(c, dtype=float)
    return (c - c.mean()) / (c.std() + 1e-9)


@torch.no_grad()
def score_candidates(features, encoders, measured):
    """One stimulus, end to end.

    features    {stream: (K, D) candidate features}
    encoders    {stream: predict(X) -> (K, V)}   the fitted ridges
    measured    {stream: (V,) measured response, in the encoder's normalisation}

    Returns the combined score (K,). The chosen candidate is np.argmax of it (first index on ties).
    """
    total = 0.0
    for stream in STREAMS:
        predicted = encoders[stream](torch.as_tensor(features[stream]).float())
        total = total + zscore_within(stream_correlations(predicted, measured[stream]))
    return total
