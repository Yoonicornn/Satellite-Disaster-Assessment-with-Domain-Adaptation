"""
CDAN: condition the domain discriminator on what the classifier predicts.

Long, Cao, Wang and Jordan, "Conditional Adversarial Domain Adaptation" (arXiv 1705.10667),
NeurIPS 2018.

DANN asks the discriminator to separate domains from the feature z alone, so it aligns the
MARGINAL feature distribution p(z). That assumes the two domains have a similar class mix. Ours
do not: minor-damage is 8.3 % of the source and 54.5 % of the caribbean test split.

CDAN gives the discriminator the outer product of the feature and the predicted class
distribution:

    T(z, g) = g (x) z            flattened, so the input is (C * d) wide
    L = L_cls - lambda * L_domain(D(T(z, g)))

Because the discriminator now sees "which class does this look like, and what do its features
look like", the encoder is pushed to align p(z | y) rather than p(z). That is the part of the
shift DANN cannot address.

CDAN+E (method.entropy: true) additionally weights each example by how confident the classifier
is about it:

    H(g) = -sum_c g_c log g_c          entropy of the prediction
    w    = 1 + exp(-H(g))              confident examples weigh about 2, unsure ones about 1

Weights are normalised inside each domain, so source and target keep equal pull.

A WARNING specific to this project: our classifier is confidently WRONG on caribbean - it calls
two thirds of the island destroyed when a sixth of it is. CDAN+E upweights exactly the examples
it is most sure about, so it can amplify that bias. Run plain CDAN first and only add +E if
plain CDAN helps.

What CDAN does NOT fix: the Zhao et al. (ICML 2019) lower bound is about differing marginal
LABEL distributions, and conditioning on a predicted label does not estimate the target prior.
Expect it to help with conditional shift, not with label shift.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Algorithm
from .dann import dann_lambda, grad_reverse

# d * C above this makes the discriminator's first layer unreasonably wide. The paper uses a
# randomised multilinear map there; we just stop and say so, because a smaller bottleneck is
# the simpler fix for this project.
MAX_MAP_DIM = 4096


def multilinear_map(z, g):
    """The outer product of the feature and the predicted class distribution, flattened.

    z is (B, d), g is (B, C) -> (B, C * d).

    g is expected to be detached by the caller. The conditioning is meant to tell the
    discriminator which class it is looking at; letting the domain loss flow back into the
    classifier through this path would corrupt the classifier itself.
    """
    return torch.bmm(g.unsqueeze(2), z.unsqueeze(1)).flatten(1)


def entropy_weights(g, eps=1e-6):
    """w = 1 + exp(-H(g)). Confident predictions get about 2, uniform ones about 1."""
    h = -(g * torch.log(g + eps)).sum(dim=1)
    return 1.0 + torch.exp(-h)


class ConditionalDiscriminator(nn.Module):
    """Says whether (z, prediction) came from the source (0) or the target (1)."""

    def __init__(self, in_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, lambd=1.0):
        return self.net(grad_reverse(x, lambd))


class Cdan(Algorithm):
    name = "cdan"
    uses_target = True

    def build(self, model):
        d = self.cfg.model.bottleneck
        c = 4                                        # damage classes
        in_dim = d * c
        if in_dim > MAX_MAP_DIM:
            raise ValueError(
                f"CDAN's multilinear map is bottleneck x classes = {d} x {c} = {in_dim}, which "
                f"is wider than {MAX_MAP_DIM}. The paper uses a randomised map at this size; "
                f"here, lower model.bottleneck instead.")
        self.modules["discriminator"] = ConditionalDiscriminator(
            in_dim=in_dim, hidden=self.cfg.method.hidden, dropout=self.cfg.model.dropout)
        return self.modules

    def extra_loss(self, model, z_source, target_batch, device, *, epoch, progress):
        lam = dann_lambda(progress, self.cfg.method.gamma, self.cfg.method.lambda_max)

        # a separate forward pass for the target, as in coral.py and dann.py
        z_target = model.features(target_batch["pre"].to(device, non_blocking=True),
                                  target_batch["post"].to(device, non_blocking=True))
        z = torch.cat([z_source, z_target], dim=0)

        # the classifier's own head, so the conditioning reflects the model being trained.
        # Detached: this path conditions the discriminator, it must not train the classifier.
        g = torch.softmax(model.classifier(z), dim=1).detach()

        logits = self.modules["discriminator"](multilinear_map(z, g), lam)
        labels = torch.cat([
            torch.zeros(len(z_source), dtype=torch.long, device=device),
            torch.ones(len(z_target), dtype=torch.long, device=device),
        ])

        if self.cfg.method.entropy:
            per_sample = F.cross_entropy(logits, labels, reduction="none")
            w = entropy_weights(g)
            n_s = len(z_source)
            # normalise within each domain, so a domain cannot dominate by being more confident
            w_s = w[:n_s] / w[:n_s].sum().clamp_min(1e-6)
            w_t = w[n_s:] / w[n_s:].sum().clamp_min(1e-6)
            w = torch.cat([w_s, w_t])
            loss = (w * per_sample).sum() / w.sum()
        else:
            loss = F.cross_entropy(logits, labels)

        acc = (logits.argmax(1) == labels).float().mean().item()
        return loss, {"extra_loss": loss.detach().item(), "lambda": lam, "domain_acc": acc}
