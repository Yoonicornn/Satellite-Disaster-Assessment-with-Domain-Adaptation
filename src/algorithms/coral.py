"""
Deep CORAL: align the second-order statistics of the features.

Sun and Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (arXiv 1607.01719).

    L_coral = || C_s - C_t ||_F^2 / (4 d^2)

where C_s and C_t are the covariance matrices of the bottleneck feature z over the source batch
and the target batch, and d is the feature size. The total loss is

    L = L_cls + lambda * L_coral

No new parameters: the only cost is one extra forward pass for the target batch.
"""
from __future__ import annotations

import torch

from .base import Algorithm


def batch_covariance(x):
    """Covariance of one batch of features. x is (n, d), result is (d, d).

    Equation 2 of the paper, C = (D^T D - (1^T D)^T (1^T D) / n) / (n - 1), written in the
    centred form, which is the same value and better behaved numerically.
    """
    n = x.shape[0]
    if n < 2:
        raise ValueError("covariance needs at least 2 samples; keep drop_last=True on the loader")
    centred = x - x.mean(dim=0, keepdim=True)
    return centred.t() @ centred / (n - 1)


def coral_loss(z_source, z_target):
    """Equation 1 of the paper. The 1/(4 d^2) keeps the value comparable across feature sizes,
    so lambda does not need retuning when model.bottleneck changes."""
    d = z_source.shape[1]
    diff = batch_covariance(z_source) - batch_covariance(z_target)
    return (diff * diff).sum() / (4.0 * d * d)


class Coral(Algorithm):
    name = "coral"
    uses_target = True

    def weight(self, epoch):
        """lambda for this epoch, ramped linearly over method.ramp_epochs.

        The paper picks lambda so that the classification and CORAL terms end up roughly equal by
        the end of training. Watch train/cls_loss against train/extra_loss in TensorBoard and
        adjust method.lambda_weight if one dominates.
        """
        full = self.cfg.method.lambda_weight
        ramp = self.cfg.method.ramp_epochs
        return full if ramp <= 0 else full * min(1.0, (epoch + 1) / ramp)

    def extra_loss(self, model, z_source, target_batch, device, *, epoch, progress):
        lam = self.weight(epoch)
        # a separate forward pass for the target, so BatchNorm keeps the two domains' statistics
        # apart instead of mixing them into a hidden form of adaptation
        z_target = model.features(target_batch["pre"].to(device, non_blocking=True),
                                  target_batch["post"].to(device, non_blocking=True))
        raw = coral_loss(z_source, z_target)
        return lam * raw, {"extra_loss": raw.detach().item(), "lambda": lam}
