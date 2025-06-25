from stable_baselines3.common.policies import ActorCriticPolicy
from src.custom_extractors import ScalarTokenTransformer, AttnMlpExtractor



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
        self.attn_act = kwargs.pop("attn_act", True)
        self.attn_val = kwargs.pop("attn_val", False)
        kwargs.pop("features_extractor_kwargs", None)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AttnMlpExtractor(
            feature_dim   = self.features_dim,
            use_attn_pi   = self.attn_act,
            use_attn_vf   = self.attn_val,
            net_arch      = self.net_arch,
            activation_fn = self.activation_fn,
            device        = self.device,
        )