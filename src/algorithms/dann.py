"""
DANN: make the features domain-invariant by training against a discriminator.

Ganin et al., "Domain-Adversarial Training of Neural Networks" (arXiv 1505.07818).

A small discriminator tries to say whether z came from the source or the target. Between it and
the network sits a gradient reversal layer: forward it does nothing, backward it multiplies the
gradient by -lambda. So one ordinary optimiser plays both sides of the game - the discriminator
gets better at telling the domains apart, while the encoder is pushed to make them look alike.

    L = L_cls - lambda * L_domain          (the minus is what the GRL implements)
    lambda_p = 2 / (1 + exp(-gamma * p)) - 1,   p = progress from 0 to 1

Starting lambda at 0 matters: early on the classifier knows nothing, and letting the
discriminator pull on the features from step one tends to wreck them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Algorithm


# ----------------------------------------------------------------------------- gradient reversal
class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    """Identity going forward, gradient multiplied by -lambda coming back."""
    return _GradientReversal.apply(x, lambd)


def dann_lambda(progress, gamma=10.0, lambda_max=1.0):
    """The ramp from section 5.2.2 of the paper. progress runs from 0 to 1 over the whole run."""
    p = min(max(float(progress), 0.0), 1.0)
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


# ----------------------------------------------------------------------------- discriminator
class DomainDiscriminator(nn.Module):
    """Says whether z came from the source (0) or the target (1)."""

    def __init__(self, in_dim=512, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, z, lambd=1.0):
        return self.net(grad_reverse(z, lambd))


# ----------------------------------------------------------------------------- algorithm
class Dann(Algorithm):
    name = "dann"
    uses_target = True

    def build(self, model):
        self.modules["discriminator"] = DomainDiscriminator(
            in_dim=self.cfg.model.bottleneck,
            hidden=self.cfg.method.hidden,
            dropout=self.cfg.model.dropout,
        )
        return self.modules

    def extra_loss(self, model, z_source, target_batch, device, *, epoch, progress):
        lam = dann_lambda(progress, self.cfg.method.gamma, self.cfg.method.lambda_max)
        # A separate forward pass for the target, so the two domains are not mixed inside one
        # batch statistic. NOTE: BatchNorm's RUNNING statistics are still updated by target
        # data here - see src/algorithms/bn_only.py, the control that measures that effect.
        z_target = model.features(target_batch["pre"].to(device, non_blocking=True),
                                  target_batch["post"].to(device, non_blocking=True))

        z = torch.cat([z_source, z_target], dim=0)
        labels = torch.cat([
            torch.zeros(len(z_source), dtype=torch.long, device=device),
            torch.ones(len(z_target), dtype=torch.long, device=device),
        ])
        logits = self.modules["discriminator"](z, lam)
        loss = F.cross_entropy(logits, labels)

        # domain accuracy is worth watching: near 0.5 means the features look the same to the
        # discriminator, near 1.0 means it is winning and adaptation is not happening
        acc = (logits.argmax(1) == labels).float().mean().item()
        return loss, {"extra_loss": loss.detach().item(), "lambda": lam, "domain_acc": acc}
