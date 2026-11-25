"""
The training methods, picked by method.name in the config.

    from src.algorithms import make_algorithm
    algo = make_algorithm(cfg)

To add a method: write a file here with a subclass of Algorithm, then add one line to REGISTRY.
train.py does not need to change.
"""
from .base import Algorithm, SourceOnly
from .bn_only import BatchNormOnly
from .cdan import Cdan, ConditionalDiscriminator, multilinear_map
from .coral import Coral, batch_covariance, coral_loss
from .dann import Dann, DomainDiscriminator, dann_lambda, grad_reverse

# baseline and oracle share the same loss; they differ only in which datasets they are given,
# which src/data/crops.py decides
REGISTRY = {
    "baseline": SourceOnly,
    "oracle": SourceOnly,
    "bn_only": BatchNormOnly,
    "coral": Coral,
    "dann": Dann,
    "cdan": Cdan,
}


def make_algorithm(cfg) -> Algorithm:
    name = cfg.method.name
    if name not in REGISTRY:
        raise ValueError(f"unknown method {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](cfg)


__all__ = [
    "Algorithm", "SourceOnly", "BatchNormOnly", "Coral", "Dann", "Cdan", "DomainDiscriminator", "ConditionalDiscriminator",
    "REGISTRY", "make_algorithm",
    "coral_loss", "batch_covariance", "multilinear_map", "dann_lambda", "grad_reverse",
]
