# src/custom_extractors.py
import math
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ScalarTokenAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: (B, N, D)
        out, _ = self.attn(x, x, x, need_weights=False)
        return out

class ScalarTokenTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, embed_dim=64, n_heads=4, n_layers=1, dropout=0.0):
        super().__init__(observation_space, features_dim=embed_dim)
        self.n_tokens = observation_space.shape[0]

        self.token_proj = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.layers = nn.Sequential(*[
            ScalarTokenAttention(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        print("Obs shape:", obs.shape)
        B, N = obs.shape
        x = obs.view(B, N, 1)                          # (B, N, 1)
        x = self.token_proj(x)                # (B, N, D)
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat((cls, x), dim=1)                     # (B, N+1, D)
        x = x + self.pos_embedding[:, :x.shape[1], :]  # (B, N+1, D)
        
        x = self.layers(x)                            # (B, N, D)

        return x[:, 0] # cls token