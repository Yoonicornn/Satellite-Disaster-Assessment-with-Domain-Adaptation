"""
What every training method has in common.

The network is always the same (src/models/siamese.py). A method only decides what extra loss is
added on top of the weighted cross-entropy, and whether it needs an unlabelled target batch.

    baseline / oracle : nothing extra
    coral             : match the covariance of z between the source and target batch
    dann              : a discriminator behind a gradient reversal layer

Adding a method later means writing one file here and one line in __init__.py; train.py does not
change.
"""
from __future__ import annotations

import torch


class Algorithm:
    """Base class. Subclasses override extra_loss, and build() if they need their own modules."""

    name = "base"
    uses_target = False          # does a training step need an unlabelled target batch?

    def __init__(self, cfg):
        self.cfg = cfg
        self.modules: dict[str, torch.nn.Module] = {}   # extra trainable parts, e.g. a discriminator

    # ------------------------------------------------------------------ setup
    def build(self, model):
        """Called once, after the network exists. Create any extra modules here."""
        return self.modules

    def parameters(self):
        """Extra parameters the optimiser must see, on top of the network's own."""
        return [p for m in self.modules.values() for p in m.parameters()]

    def to(self, device):
        for m in self.modules.values():
            m.to(device)
        return self

    def train(self):
        for m in self.modules.values():
            m.train()

    def eval(self):
        for m in self.modules.values():
            m.eval()

    # ------------------------------------------------------------------ checkpointing
    def state_dict(self):
        return {name: m.state_dict() for name, m in self.modules.items()}

    def load_state_dict(self, state):
        for name, m in self.modules.items():
            if state and name in state:
                m.load_state_dict(state[name])

    # ------------------------------------------------------------------ the loss
    def extra_loss(self, model, z_source, target_batch, device, *, epoch, progress):
        """The method's own loss term.

        model         the shared network, in case the method needs model.features()
        z_source      bottleneck features of the labelled source batch
        target_batch  the unlabelled target batch, or None when uses_target is False
        epoch         current epoch, 0-based
        progress      how far through the whole run, 0 to 1 (used by the DANN ramp)

        Returns (loss tensor, dict of numbers to log). The default adds nothing.
        """
        return torch.zeros((), device=device), {}


class SourceOnly(Algorithm):
    """No domain adaptation: train on the labelled data and nothing else.

    Used by two configs. `baseline` gives it the source only, which is the number the adaptation
    methods have to beat. `oracle` gives it source + labelled target, which is the ceiling.
    The loss is identical; only the datasets differ, and that is decided in src/data/crops.py.
    """

    name = "source_only"
    uses_target = False
