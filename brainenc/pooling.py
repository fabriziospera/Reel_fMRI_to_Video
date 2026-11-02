"""The two learned pooling modules. Both use the raw tokens as values and put the positional
information only in the keys, so the pooled output stays in the backbone's feature space."""
import torch
import torch.nn as nn

VGG_CHANNELS = [64, 128, 256, 512, 512]     # the five ReLU layers used by the early stream
GRID = 7                                     # spatial grid kept per layer
N_FRAMES = 6


class FactorizedAttnPool(nn.Module):
    """Dorsal stream: one learned query, multi-head attention over all V-JEPA tokens
    (T x H x W), with factorised temporal + spatial position embeddings added to the keys."""

    def __init__(self, C=1024, num_heads=4, T=8, H=16, W=16):
        super().__init__()
        assert C % num_heads == 0
        self.nh, self.C = num_heads, C
        self.pos_t = nn.Parameter(torch.randn(T, C) * C ** -0.5)
        self.pos_s = nn.Parameter(torch.randn(H * W, C) * C ** -0.5)
        self.q = nn.Parameter(torch.randn(1, C) * C ** -0.5)
        self.k = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C, bias=False)
        self.scale = (C // num_heads) ** -0.5

    def forward(self, tokens):                              # (B, T*H*W, C)
        B, N, C = tokens.shape
        pos = (self.pos_t[:, None, :] + self.pos_s[None, :, :]).reshape(1, N, C)
        K, V = self.k(tokens + pos), self.v(tokens)        # positions only in the keys
        split = lambda z: z.view(B, N, self.nh, C // self.nh).transpose(1, 2)
        Kh, Vh = split(K), split(V)                         # (B, heads, N, C/heads)
        Qh = self.q.view(1, self.nh, C // self.nh).permute(1, 0, 2)[None]    # (1, heads, 1, C/heads)
        attn = (Qh @ Kh.transpose(-2, -1) * self.scale).softmax(-1)         # (B, heads, 1, N)
        return (attn @ Vh).transpose(1, 2).reshape(B, C)


class TemporalAttnPool(nn.Module):
    """Early stream: for every layer and every cell of the 7x7 grid, a learned query weighs the
    6 frames of that cell (softmax over time). No spatial mixing: each cell keeps its position."""

    def __init__(self, ch=VGG_CHANNELS, T=N_FRAMES, S=GRID * GRID):
        super().__init__()
        self.ch, self.T, self.S = list(ch), T, S
        self.sizes = [c * S for c in self.ch]
        self.q = nn.ParameterList([nn.Parameter(torch.randn(S, c) * c ** -0.5) for c in self.ch])
        self.k = nn.ModuleList([nn.Linear(c, c, bias=False) for c in self.ch])
        self.pos_t = nn.ParameterList([nn.Parameter(torch.randn(T, c) * c ** -0.5) for c in self.ch])
        self.scale = [c ** -0.5 for c in self.ch]

    def forward(self, x):                                   # (B, T, sum(C_l * 49))
        B, T, D = x.shape
        assert D == sum(self.sizes), (D, sum(self.sizes))
        out, off = [], 0
        for li, (c, sz) in enumerate(zip(self.ch, self.sizes)):
            tok = x[:, :, off:off + sz].reshape(B, T, c, self.S).permute(0, 3, 1, 2)   # (B, S, T, C)
            K = self.k[li](tok + self.pos_t[li][None, None])         # positions only in the keys
            w = (torch.einsum("bstc,sc->bst", K, self.q[li]) * self.scale[li]).softmax(-1)  # (B, S, T)
            agg = (tok * w.unsqueeze(-1)).sum(2)                     # (B, S, C)
            out.append(agg.transpose(1, 2).reshape(B, c * self.S))
            off += sz
        return torch.cat(out, 1)
