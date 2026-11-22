"""
The control run: target images pass through the network, but add no loss.

Why this exists
---------------
CORAL and DANN forward an unlabelled target batch on every training step, while the network is
in train mode. Every BatchNorm layer therefore updates its running mean and variance using
target statistics. The baseline never does this, because it never sees a target image.

That is free adaptation, and it has nothing to do with the CORAL or DANN loss. It is essentially
AdaBN (Li et al., 2016), which is a domain-adaptation method in its own right. So if CORAL beats
the baseline, part or all of the gain may be BatchNorm rather than covariance alignment.

A separate forward pass does NOT avoid this. It only stops the two domains being mixed inside
one batch statistic; the running averages still move. Measured on a ResNet-18 bn1 layer:

    train mode, no_grad : running_mean moved by 0.001108
    eval  mode, no_grad : running_mean moved by 0.000000

This method isolates the effect. It forwards the target batch exactly as CORAL and DANN do, so
the BatchNorm statistics move in the same way, and then returns a loss of zero.

    baseline -> bn_only      = what BatchNorm adaptation alone is worth
    bn_only  -> coral/dann   = what the alignment loss is actually worth

Without this run those two effects cannot be told apart.
"""
from __future__ import annotations

import torch

from .base import Algorithm


class BatchNormOnly(Algorithm):
    name = "bn_only"
    uses_target = True

    def extra_loss(self, model, z_source, target_batch, device, *, epoch, progress):
        if target_batch is None:
            return torch.zeros((), device=device), {}

        # No gradient needed: the point is the forward pass itself, which is what moves the
        # BatchNorm running statistics. They update in forward regardless of grad mode, as
        # long as the module is in train mode.
        with torch.no_grad():
            model.features(target_batch["pre"].to(device, non_blocking=True),
                           target_batch["post"].to(device, non_blocking=True))

        return torch.zeros((), device=device), {"extra_loss": 0.0}
