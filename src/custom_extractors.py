# src/custom_extractors.py  (NEW FILE)
import math
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ScalarTokenTransformer(BaseFeaturesExtractor):
    """
    Turns a flat observation of length N into N one-dim tokens, runs a
    tiny Transformer encoder and returns the [CLS] embedding.
    Learns by itself which dimensions are noisy or duplicated.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim=embed_dim)

        self.n_tokens = observation_space.shape[0]
        self.token_proj = nn.Linear(1, embed_dim)

        # sinusoidal encodings
        pe = th.zeros(self.n_tokens, embed_dim)
        pos = th.arange(self.n_tokens, dtype=th.float32).unsqueeze(1)
        div = th.exp(
            th.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = th.sin(pos * div)
        pe[:, 1::2] = th.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

        self.cls = nn.Parameter(th.zeros(1, 1, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    # (B, N) -> (B, E)
    def forward(self, obs: th.Tensor) -> th.Tensor:
        B, N = obs.shape
        x = obs.view(B, N, 1)                    # (B,N,1)
        x = self.token_proj(x) + self.pe         # (B,N,E)
        cls = self.cls.expand(B, -1, -1)         # (B,1,E)
        x = th.cat([cls, x], dim=1)              # prepend [CLS]
        x = self.encoder(x)                      # (B,1+N,E)
        return x[:, 0]                           # only [CLS]
