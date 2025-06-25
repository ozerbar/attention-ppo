import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor


class ScalarTokenAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Self-attention with residual connection + LayerNorm
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)

        # Feedforward network with residual + LayerNorm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
    
class FeatureAttention(nn.Module):
    """
    One token  = one observation coordinate.
    Self-attention is applied across the *feature* axis, not across time.
    """
    def __init__(self, obs_dim: int, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        # self.proj_in  = nn.Linear(1, d_model)          # scalar  → d_model
        self.d_model  = d_model
        self.pos      = nn.Parameter(torch.zeros(1, obs_dim, d_model))
        self.attn     = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj_out = nn.Linear(d_model, 1)          # d_model → scalar

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: (B, obs_dim)
        # h = self.proj_in(x_flat.unsqueeze(-1)) + self.pos      # (B, D, d_model)
        h = x_flat.unsqueeze(-1).expand(-1, -1, self.d_model) + self.pos
        h, _ = self.attn(h, h, h)                              # self-attention over D tokens
        return self.proj_out(h).squeeze(-1)                    # back to (B, obs_dim)


class ScalarTokenTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, embed_dim=64, n_heads=2, n_layers=2, dropout=0.1):
        super().__init__(observation_space, features_dim=embed_dim)
        self.n_tokens = observation_space.shape[0]

        # Project each scalar feature to an embedding
        self.token_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )

        # [CLS] token to summarize the sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Stacked attention layers (transformer encoder blocks)
        self.layers = nn.Sequential(*[
            ScalarTokenAttention(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B, N = obs.shape  # (B, 29)
        x = obs.view(B, N, 1)                      # (B, 29, 1)
        x = self.token_proj(x)                     # (B, 29, 64)

        cls = self.cls_token.expand(B, 1, -1)      # (B, 1, 64)
        x = torch.cat([cls, x], dim=1)                # (B, 30, 64)

        x = x + self.pos_embedding[:, :x.shape[1]]
        x = self.layers(x)

        return x[:, 0]  # Return [CLS] token output → shape (B, 64)
    
class AttnMlpExtractor(MlpExtractor):
    def __init__(self, feature_dim: int, *,
                 use_attn_pi: bool, use_attn_vf: bool,
                 **mlp_kwargs):
        super().__init__(feature_dim, **mlp_kwargs)

        self.attn_pi = FeatureAttention(feature_dim) if use_attn_pi else nn.Identity()
        self.attn_vf = FeatureAttention(feature_dim) if use_attn_vf else nn.Identity()

    def forward(self, features: torch.Tensor):
        pi_latent = self.policy_net(self.attn_pi(features))
        vf_latent = self.value_net(self.attn_vf(features))
        return pi_latent, vf_latent