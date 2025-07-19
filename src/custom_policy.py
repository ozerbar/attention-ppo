from stable_baselines3.common.policies import ActorCriticPolicy
from src.custom_extractors import ScalarTokenTransformer, AttnMlpExtractor, AttnExtractor, Attention_Direct_Override_Extractor, MediumAttnMlpExtractor, FrameAttnMlpExtractor
from stable_baselines3.common.torch_layers import FlattenExtractor
import torch
import torch.nn as nn
import gymnasium as gym



class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        features_extractor_kwargs = kwargs.pop("features_extractor_kwargs", {
            "embed_dim": 64,
            "n_heads": 2,
            "n_layers": 1,
            "dropout": 0.0,
        })
        super().__init__(
            *args,
            features_extractor_class=ScalarTokenTransformer,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs,
        )
        
class SelectiveAttentionPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        print(f"kwargs received: {kwargs}")
        self.attn_act = kwargs.pop("attn_act", False)
        self.attn_val = kwargs.pop("attn_val", False)
        self.attn_common = kwargs.pop("attn_common", False) # Use common attention for both actor and critic
        kwargs.pop("features_extractor_kwargs", None)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AttnMlpExtractor(
            feature_dim   = self.features_dim,
            use_attn_pi   = self.attn_act,
            use_attn_vf   = self.attn_val,
            use_attn_common = self.attn_common,
            net_arch      = self.net_arch,
            activation_fn = self.activation_fn,
            device        = self.device,
        )
        
class AttentionPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.attn_act = kwargs.pop("attn_act", True)
        self.attn_val = kwargs.pop("attn_val", True)
        kwargs.pop("features_extractor_kwargs", None)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AttnExtractor(
            feature_dim   = self.features_dim,
            net_arch      = self.net_arch,
            activation_fn = self.activation_fn,
            device        = self.device,
            use_attn_pi   = self.attn_act,
            num_frames    = 4,
            d_model       = 64,
            use_attn_vf   = self.attn_val,
        )

class AttentionDirectOverridePolicy(ActorCriticPolicy):
    def __init__(self, *args,
                 attn_act=False, attn_val=False,
                 embed_dim=32, num_heads=4, frame_stack=1,
                 **kwargs):

        super().__init__(*args,
                         features_extractor_class=FlattenExtractor,
                         **kwargs)

        self.mlp_extractor = Attention_Direct_Override_Extractor(
            self.features_dim, attn_act, attn_val, embed_dim, num_heads, frame_stack
        )

        latent_pi = self.mlp_extractor.latent_dim_pi   # 256
        latent_vf = self.mlp_extractor.latent_dim_vf

        action_dim = (self.action_space.shape[0] if isinstance(self.action_space, gym.spaces.Box)
                      else self.action_space.n)

        self.action_net = nn.Linear(latent_pi, action_dim)
        self.value_net  = nn.Linear(latent_vf, 1)



class MediumAttentionPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.attn_act = kwargs.pop("attn_act", True)
        self.attn_val = kwargs.pop("attn_val", False)
        self.attn_output_dim = kwargs.pop("attn_output_dim", 32)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MediumAttnMlpExtractor(
            feature_dim   = self.features_dim,
            attn_act   = self.attn_act,
            attn_val   = self.attn_val,
            net_arch      = self.net_arch,
            activation_fn = self.activation_fn,
            device        = self.device,
            attn_output_dim    = self.attn_output_dim,
        )



class FrameAttentionPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.attn_act = kwargs.pop("attn_act", True)
        self.attn_val = kwargs.pop("attn_val", False)
        self.attn_output_dim = kwargs.pop("attn_output_dim", 32)
        self.frame_stack = kwargs.pop("frame_stack", 4)
        self.d_model = kwargs.pop("d_model", 64)
        self.mlp_output_dim = kwargs.pop("mlp_output_dim", 64)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = FrameAttnMlpExtractor(
            feature_dim   = self.features_dim,
            attn_act   = self.attn_act,
            attn_val   = self.attn_val,
            net_arch      = self.net_arch,
            activation_fn = self.activation_fn,
            device        = self.device,
            attn_output_dim    = self.attn_output_dim,
            d_model        = self.d_model,
            frame_stack   = self.frame_stack,
            mlp_output_dim = self.mlp_output_dim

        )
