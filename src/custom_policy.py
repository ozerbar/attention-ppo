from stable_baselines3.common.policies import ActorCriticPolicy
from src.custom_extractors import ScalarTokenTransformer


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