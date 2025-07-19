import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from typing import Tuple
import torch.nn.functional as F


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
    

    
class AttentionLayerBlock(nn.Module):
    def __init__(self, obs_dim: int = 87, d_model: int = 64, n_heads: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)        # (B, 87, 1) → (B, 87, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.token_proj = nn.Linear(d_model, 1)        # (B, 87, d_model) → (B, 87, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)              # (B, 87) → (B, 87, 1)
        x = self.input_proj(x)           # → (B, 87, 64)
        x, _ = self.attn(x, x, x)        # → (B, 87, 64)
        x = self.relu(x)
        x = self.token_proj(x)           # → (B, 87, 1)
        x = x.squeeze(-1)                # → (B, 87)
        return x

class ScalarTokenTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, embed_dim=128, n_heads=2, n_layers=2, dropout=0.1):
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
    

class FeatureAttention(nn.Module):
    """
    One token  = one observation coordinate.
    Self-attention is applied across the *feature* axis, not across time.
    """
    def __init__(self, obs_dim: int, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.proj_in  = nn.Linear(1, d_model)          # scalar  → d_model
        # self.d_model  = d_model
        self.pos      = nn.Parameter(torch.zeros(1, obs_dim, d_model))
        self.attn     = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj_out = nn.Linear(d_model, 1)          # d_model → scalar

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: (B, obs_dim)
        # h = self.proj_in(x_flat.unsqueeze(-1)) + self.pos      # (B, D, d_model)
        h = self.proj_in(x_flat.unsqueeze(-1)) + self.pos 
        h, _ = self.attn(h, h, h)                              # self-attention over D tokens
        return self.proj_out(h).squeeze(-1)                    # back to (B, obs_dim)

class AttnMlpExtractor(MlpExtractor):
    def __init__(self, feature_dim: int, *,
                 use_attn_pi: bool, use_attn_vf: bool,use_attn_common: bool,
                 **mlp_kwargs):
        super().__init__(feature_dim, **mlp_kwargs)

        self.attn_pi = FeatureAttention(feature_dim) if use_attn_pi else nn.Identity()
        self.attn_vf = FeatureAttention(feature_dim) if use_attn_vf else nn.Identity()
        self.attn_common = FeatureAttention(feature_dim) if use_attn_common else nn.Identity()

    def forward(self, features: torch.Tensor):
        features = self.attn_common(features)
        pi_latent = self.policy_net(self.attn_pi(features))
        vf_latent = self.value_net(self.attn_vf(features))
        return pi_latent, vf_latent
    

class AttnExtractor(nn.Module):
    def __init__(self, feature_dim, net_arch, activation_fn, device='cpu',
                 use_attn_pi=True, use_attn_vf=True):
        super().__init__()
        self.device = device

        def make_branch(use_attn: bool):
            layers = []
            if use_attn:
                layers.append(AttentionLayerBlock(obs_dim=feature_dim, d_model=128, n_heads=2))
                in_dim = feature_dim  # attention outputs shape (B, 87)
            else:
                layers.append(nn.Identity())
                in_dim = feature_dim  # still 87

            layers.append(nn.Linear(in_dim, 256))
            layers.append(activation_fn())
            layers.append(nn.Linear(256, 256))
            layers.append(activation_fn())
            return nn.Sequential(*layers)

        self.policy_net = make_branch(use_attn_pi)
        self.value_net  = make_branch(use_attn_vf)

        self.latent_dim_pi = 256
        self.latent_dim_vf = 256

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    


# ======================================================================================


class AttentionBlock_Direct_Override(nn.Module):
    def __init__(self, seq_len, frame_stack, embed_dim=32, num_heads=4, out_dim=256):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim) # Every scalar value (shape (B,S,1)) is mapped to a learned 
                                            # E-dimensional embedding: (B,S,1) -> (B,S,E)
        self.frame_pos = nn.Parameter(                      # (frame_stack, embed_dim)
            torch.randn(frame_stack, embed_dim) * 0.02
        )
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                        batch_first=True)
        self.proj  = nn.Linear(seq_len * embed_dim, out_dim)
        self.frame_stack = frame_stack
        self.obs_dim = seq_len // frame_stack
        self.printed = False

    def forward(self, x):
        B, S = x.shape                                    # x: (B,S)
        tok = self.embed(x.unsqueeze(-1))                      # (B,S,E)
        frame_ids = torch.arange(S, device=x.device) // self.obs_dim
        pos = self.frame_pos[frame_ids]                     # (S,E)
        tok = tok + pos                                     # broadcast add

        if not self.printed:  # w: (B,H,S,S)
            y, w = self.attn(tok, tok, tok, need_weights=True, average_attn_weights=False)
            print(f"[Attention] weight matrix shape: {w.shape}")
            self.printed = True
        else:
            y, _ = self.attn(tok, tok, tok, need_weights=False)

        y = y.flatten(1)                                     # (B,S*E)
        return F.relu(self.proj(y))                          # (B,256)

class Attention_Direct_Override_Extractor(MlpExtractor):
    def __init__(self, feat_dim, attn_act, attn_val,
                 embed_dim=32, num_heads=4, frame_stack=1):
        super().__init__(feat_dim, net_arch=[], activation_fn=nn.ReLU)

        def _fallback():
            return nn.Sequential(
                nn.Linear(feat_dim, 256), nn.ReLU(),
                nn.Linear(256, 256),      nn.ReLU()
            )

        self.policy_net = AttentionBlock_Direct_Override(feat_dim, frame_stack, embed_dim, num_heads) \
                          if attn_act else _fallback()
        self.value_net  = AttentionBlock_Direct_Override(feat_dim, frame_stack, embed_dim, num_heads) \
                          if attn_val else _fallback()

        # report the real latent sizes to the parent class
        self.latent_dim_pi = 256
        self.latent_dim_vf = 256

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)



# ======================================================================================


class MediumFeatureAttention(nn.Module):
    """
    One token = one observation coordinate.
    Self-attention is applied across the *feature* axis, not across time.
    """
    def __init__(self, obs_dim: int, d_model: int = 128, n_heads: int = 2, attn_output_dim: int = 32):
        super().__init__()
        self.proj_in = nn.Linear(1, d_model)  # scalar → d_model
        self.pos = nn.Parameter(torch.zeros(1, obs_dim, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj_out = nn.Linear(d_model, attn_output_dim)  # d_model → scalar
        self.attn_output_dim = attn_output_dim
        self.printed = False

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: (B, obs_dim)
        if not self.printed:
            print(f"Input shape (x_flat): {x_flat.shape}")

        h = self.proj_in(x_flat.unsqueeze(-1)) + self.pos  # (B, obs_dim, d_model)
        if not self.printed:
            print(f"Projected input (h): {h.shape}")

        if not self.printed:
            h_out, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
            print(f"[Attention] weight matrix shape: {w.shape}")
        else:
            h_out, _ = self.attn(h, h, h)

        h_out = F.relu(h_out)  # ReLU activation

        # output = self.proj_out(h_out).squeeze(-1)  # (B, obs_dim)
        output = self.proj_out(h_out) # (B, obs_dim, attn_output_dim)

        if not self.printed:
            print(f"Output shape after projection: {output.shape}")
            self.printed = True
        
        return output

class MediumAttentionBlock(nn.Module):
    def __init__(self, feat_dim, embed_dim, num_heads, net_arch, activation_fn, attn_output_dim):
        super().__init__()
        self.attn = MediumFeatureAttention(feat_dim, d_model=embed_dim, n_heads=num_heads , attn_output_dim=attn_output_dim)
        # Build MLP using net_arch specifications
        # mlp = create_mlp(feat_dim, -1, net_arch, activation_fn)
        # self.mlp = nn.Sequential(*mlp)
    
    def forward(self, x):
        x = self.attn(x)
        return x
        # return self.mlp(x)
        
class MediumAttnMlpExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device="auto",
                 attn_act=True, attn_val=False, embed_dim=128, num_heads=2, attn_output_dim=32):
        super().__init__(feature_dim, net_arch, activation_fn, device)
        
        self.feature_dim = feature_dim
        self.attn_output_dim = attn_output_dim
        self.attn_act = attn_act
        self.attn_val = attn_val
        
        if isinstance(net_arch, dict):
            pi_arch = net_arch['pi']
            vf_arch = net_arch['vf']
        else:
            pi_arch = vf_arch = net_arch
        
        # Override policy_net and latent dimension if attention is enabled
        if attn_act:
            self.policy_net = MediumAttentionBlock(
                feature_dim, embed_dim, num_heads, pi_arch, activation_fn, attn_output_dim
            )
            self.latent_dim_pi = feature_dim * attn_output_dim  # Flattened size
        
        # Override value_net and latent dimension if attention is enabled
        if attn_val:
            self.value_net = MediumAttentionBlock(
                feature_dim, embed_dim, num_heads, vf_arch, activation_fn, attn_output_dim
            )
            self.latent_dim_vf = feature_dim * attn_output_dim  # Flattened size

    # Add flattening to forward passes
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        out = self.policy_net(features)
        if self.attn_act:
            # Flatten last two dimensions: [B, obs_dim, attn_out] => [B, obs_dim*attn_out]
            return out.view(out.shape[0], -1)
        return out

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        out = self.value_net(features)
        if self.attn_val:
            # Flatten last two dimensions: [B, obs_dim, attn_out] => [B, obs_dim*attn_out]
            return out.view(out.shape[0], -1)
        return out

# =================================================================================================


class FrameAttention(nn.Module):
    """
    One token = [framestack * 1 (single observation feature vector)]
    """
    def __init__(self, flattened_obs_dim: int, d_model: int = 64, n_heads: int = 2, attn_output_dim: int = 32, frame_stack: int = 4, mlp_output_dim: int = 64):
        super().__init__()
        single_obs_dim = flattened_obs_dim // frame_stack  # e.g. flattened_obs_dim=116 / frame_stack=4 = 29
        self.frame_stack = frame_stack
        self.single_obs_dim = single_obs_dim

        self.proj_in = nn.Linear(frame_stack, d_model)  # single_obs_dim /(e.g. 29) → d_model
        self.pos = nn.Parameter(torch.zeros(1, single_obs_dim, frame_stack))

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # self.proj_out = nn.Linear(d_model, attn_output_dim) - Not used anymore, we use MLP after attention
        self.attn_output_dim = attn_output_dim
        self.mlp_output_dim = mlp_output_dim
        self.printed = False
        self.mlp = nn.Sequential(
            nn.Linear(single_obs_dim * d_model, mlp_output_dim),
            nn.ReLU()
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        dynamic_batch = x_flat.size(0) 
        x_flat = x_flat.view(dynamic_batch,self.frame_stack,self.single_obs_dim) # dynamic_batch,4,29
        x_flat = x_flat.transpose(1, 2)  # Transpose to shape [batch_size, 29, 4]
        if not self.printed:
            print(f"Input shape (x_flat): {x_flat.shape}")

        x_flat = x_flat + self.pos
        h = self.proj_in(x_flat) # [B, FRAMESTACK, d_model]

        if not self.printed:
            print(f"Projected input (h): {h.shape}")
            h_out, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
            print(f"[Attention] weight matrix shape: {w.shape}")
        else:
            h_out, _ = self.attn(h, h, h)

        h_out = F.relu(h_out)  # ReLU activation

        # Apply MLP after attention output
        h_out_flat = h_out.reshape(h_out.size(0), -1) 
        output = self.mlp(h_out_flat)  # Apply MLP (B, 64)

        if not self.printed:
            print(f"Output shape after projection: {output.shape}")
            self.printed = True
        
        return output


class FrameAttentionBlock(nn.Module):
    def __init__(self, flattened_obs_dim, d_model, num_heads, net_arch, activation_fn, attn_output_dim, frame_stack, mlp_output_dim):
        super().__init__()
        self.attn = FrameAttention(flattened_obs_dim=flattened_obs_dim, d_model=d_model, n_heads=num_heads , attn_output_dim=attn_output_dim, frame_stack=frame_stack, mlp_output_dim=mlp_output_dim)
        # Build MLP using net_arch specifications
        # mlp = create_mlp(feat_dim, -1, net_arch, activation_fn)
        # self.mlp = nn.Sequential(*mlp)
    
    def forward(self, x):
        x = self.attn(x) # (B,29,64)
        return x
        # return self.mlp(x)

        
class FrameAttnMlpExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device="auto",
                 attn_act=True, attn_val=False, d_model=64, num_heads=2, attn_output_dim=32, frame_stack=4, mlp_output_dim=64):
        super().__init__(feature_dim, net_arch, activation_fn, device)
        
        self.feature_dim = feature_dim
        self.attn_output_dim = attn_output_dim
        self.attn_act = attn_act
        self.attn_val = attn_val
        self.frame_stack = frame_stack
        self.mlp_output_dim = mlp_output_dim
        
        if isinstance(net_arch, dict):
            pi_arch = net_arch['pi']
            vf_arch = net_arch['vf']
        else:
            pi_arch = vf_arch = net_arch
        
        # Override policy_net and latent dimension if attention is enabled
        if attn_act:
            self.policy_net = FrameAttentionBlock(
                feature_dim, d_model, num_heads, pi_arch, activation_fn, attn_output_dim, frame_stack, mlp_output_dim
            )
            # self.latent_dim_pi = feature_dim * attn_output_dim  # Flattened size
            self.latent_dim_pi = mlp_output_dim # Flattened size
        
        # Override value_net and latent dimension if attention is enabled
        if attn_val:
            self.value_net = FrameAttentionBlock(
                feature_dim, d_model, num_heads, vf_arch, activation_fn, attn_output_dim, frame_stack, mlp_output_dim
            )
            # self.latent_dim_vf = feature_dim * attn_output_dim  # Flattened size
            self.latent_dim_vf = mlp_output_dim

    # Add flattening to forward passes
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        out = self.policy_net(features)
        if self.attn_act:
            # Flatten last two dimensions: [B, obs_dim, attn_out] => [B, obs_dim*attn_out]
            return out.view(out.shape[0], -1)
        return out

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        out = self.value_net(features)
        if self.attn_val:
            # Flatten last two dimensions: [B, obs_dim, attn_out] => [B, obs_dim*attn_out]
            return out.view(out.shape[0], -1)
        return out