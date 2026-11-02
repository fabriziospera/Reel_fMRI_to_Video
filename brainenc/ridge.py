"""Ridge readouts. lambda is chosen inside the training set, never on test.

Two ways of choosing it, one per dataset:
  fit_ridge / fit_ridge_dual   random 15% hold-out of the training samples (bmd)
  fit_ridge_grouped            cross-validation over groups of samples (cc2017: groups of whole
                               movies)
In all cases the features are standardised with the training statistics and the model is refit
on the whole training set with the chosen lambda. Dual = same solution solved in sample space,
used when d >> n (early stream).
"""
import numpy as np
import torch

ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
VAL_FRAC = 0.15
SEED = 0


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _predictor(Xn, Y, mu, sd, alpha, dual, dev, chunk=2048):
    """Refit on all of (Xn, Y) with `alpha` and return predict(X) -> tensor on CPU."""
    mu_d, sd_d = mu.to(dev).double(), sd.to(dev).double()
    if dual:
        A = torch.linalg.solve(Xn @ Xn.t() + alpha * torch.eye(len(Xn), device=dev, dtype=torch.double), Y)

        def predict(X):
            return torch.cat([(((X[i:i + chunk].to(dev).double() - mu_d) / sd_d) @ Xn.t() @ A).float().cpu()
                              for i in range(0, len(X), chunk)])
    else:
        W = torch.linalg.solve(Xn.t() @ Xn + alpha * torch.eye(Xn.shape[1], device=dev, dtype=torch.double),
                               Xn.t() @ Y)

        def predict(X):
            return (((X.to(dev).double() - mu_d) / sd_d) @ W).float().cpu()
    predict.alpha = alpha
    return predict


def _val_mse(Xn, Y, tr, va, dual, alphas):
    """MSE on `va` of the ridge fitted on `tr`, for every alpha."""
    if dual:
        K, Kv = Xn[tr] @ Xn[tr].t(), Xn[va] @ Xn[tr].t()
        I = torch.eye(len(tr), device=Xn.device, dtype=torch.double)
        return [((Kv @ torch.linalg.solve(K + a * I, Y[tr]) - Y[va]) ** 2).mean().item() for a in alphas]
    A, B = Xn[tr].t() @ Xn[tr], Xn[tr].t() @ Y[tr]
    I = torch.eye(Xn.shape[1], device=Xn.device, dtype=torch.double)
    return [((Xn[va] @ torch.linalg.solve(A + a * I, B) - Y[va]) ** 2).mean().item() for a in alphas]


def _standardise(Xtr, Ytr, dev):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    return ((Xtr - mu) / sd).to(dev).double(), Ytr.to(dev).double(), mu, sd


def fit_ridge(Xtr, Ytr, alphas=ALPHAS, val_frac=VAL_FRAC, seed=SEED, dual=False):
    """lambda on a random hold-out of `val_frac` of the samples."""
    dev = _device()
    Xn, Y, mu, sd = _standardise(Xtr, Ytr, dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    n = Xn.shape[0]
    nv = int(n * val_frac)
    perm = torch.randperm(n, generator=g, device=dev)
    tr, va = perm[nv:], perm[:nv]
    mse = _val_mse(Xn, Y, tr, va, dual, alphas)
    return _predictor(Xn, Y, mu, sd, alphas[int(np.argmin(mse))], dual, dev)


def fit_ridge_dual(Xtr, Ytr, alphas=ALPHAS, val_frac=VAL_FRAC, seed=SEED):
    return fit_ridge(Xtr, Ytr, alphas, val_frac, seed, dual=True)


def fit_ridge_grouped(Xtr, Ytr, fold, dual=False, alphas=ALPHAS):
    """lambda by cross-validation: fold[i] is the fold of sample i; the MSE is summed over folds."""
    dev = _device()
    Xn, Y, mu, sd = _standardise(Xtr, Ytr, dev)
    fold = np.asarray(fold)
    mse = np.zeros(len(alphas))
    for f in np.unique(fold):
        va = torch.as_tensor(np.flatnonzero(fold == f), device=dev)
        tr = torch.as_tensor(np.flatnonzero(fold != f), device=dev)
        mse += np.array(_val_mse(Xn, Y, tr, va, dual, alphas))
    return _predictor(Xn, Y, mu, sd, alphas[int(np.argmin(mse))], dual, dev)
