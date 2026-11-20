"""
The network: one ResNet shared by the pre and post crop, then a classifier.

    pre  crop --\
                 ResNet (shared weights) -> f_pre   (512)
    post crop --/                           f_post  (512)

    f = [f_pre, f_post, f_post - f_pre]   (1536)   <- model.fusion = concat_diff
      or [f_pre, f_post]                  (1024)   <- model.fusion = concat

    z = Dropout(ReLU(Linear(f, 512)))     bottleneck, the feature CORAL and DANN align
    logits = Linear(z, 4)

Shared weights, rather than stacking the two crops into one 6-channel image: the xView2 work
reports that plain 6-channel stacking and image subtraction both did poorly, while two branches
with shared weights worked. It also halves the parameters.

Every method uses exactly this network. Only the loss changes, so a difference between the
baseline, CORAL and DANN cannot come from the architecture. The method-specific parts
(DANN's gradient reversal layer and discriminator) live in src/algorithms/.

Quick check (from the project root):
    python -m src.models.siamese --config configs/baseline.yaml
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

N_CLASSES = 4

# name -> (constructor, ImageNet weights, feature size)
BACKBONES = {
    "resnet18": (tvm.resnet18, "ResNet18_Weights", 512),
    "resnet34": (tvm.resnet34, "ResNet34_Weights", 512),
    "resnet50": (tvm.resnet50, "ResNet50_Weights", 2048),
}


def _encoder(name: str, pretrained: bool):
    """A ResNet with the final classification layer removed. Returns (module, feature size)."""
    if name not in BACKBONES:
        raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, got {name!r}")
    build, weights_enum, feat_dim = BACKBONES[name]
    weights = getattr(tvm, weights_enum).DEFAULT if pretrained else None
    net = build(weights=weights)
    net.fc = nn.Identity()          # keep conv stack + avgpool, drop the 1000-class head
    return net, feat_dim


# ----------------------------------------------------------------------------- the network
class SiameseDamageNet(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, fusion="concat_diff",
                 bottleneck=512, dropout=0.3, n_classes=N_CLASSES):
        super().__init__()
        if fusion not in ("concat_diff", "concat"):
            raise ValueError(f"fusion must be concat_diff or concat, got {fusion!r}")

        self.fusion = fusion
        self.encoder, feat_dim = _encoder(backbone, pretrained)
        fused_dim = feat_dim * (3 if fusion == "concat_diff" else 2)

        self.bottleneck = nn.Sequential(
            nn.Linear(fused_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(bottleneck, n_classes)

        self.feat_dim = feat_dim
        self.fused_dim = fused_dim
        self.bottleneck_dim = bottleneck

    def features(self, pre, post):
        """The bottleneck feature z. This is what the domain-adaptation losses work on."""
        f_pre = self.encoder(pre)                       # shared weights: the same call twice
        f_post = self.encoder(post)
        parts = [f_pre, f_post] + ([f_post - f_pre] if self.fusion == "concat_diff" else [])
        return self.bottleneck(torch.cat(parts, dim=1))

    def forward(self, pre, post):
        """Returns (logits, z). The training step needs z for CORAL and DANN."""
        z = self.features(pre, post)
        return self.classifier(z), z

    def param_groups(self, lr, backbone_lr):
        """Pretrained layers get a smaller learning rate than the new ones."""
        new = list(self.bottleneck.parameters()) + list(self.classifier.parameters())
        return [
            {"params": list(self.encoder.parameters()), "lr": backbone_lr, "name": "encoder"},
            {"params": new, "lr": lr, "name": "head"},
        ]


# ----------------------------------------------------------------------------- factory
def build_model(cfg):
    """Build the shared network from the config.

    The extra parts a method needs (DANN's discriminator) are built by that method,
    in src/algorithms/.
    """
    return SiameseDamageNet(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        fusion=cfg.model.fusion,
        bottleneck=cfg.model.bottleneck,
        dropout=cfg.model.dropout,
    )


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- self-test
def _main():
    import argparse

    from ..config import add_config_args, load_config

    p = add_config_args(argparse.ArgumentParser(description="Sanity check of the model."))
    p.add_argument("--size", type=int, default=64, help="crop size to test with")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    model = build_model(cfg)
    model.eval()

    print(f"method   : {cfg.method.name}")
    print(f"backbone : {cfg.model.backbone}  pretrained={cfg.model.pretrained}")
    print(f"fusion   : {cfg.model.fusion}  ->  {model.fused_dim} -> z({model.bottleneck_dim})")
    print(f"params   : {count_parameters(model):,}")

    pre = torch.randn(args.batch_size, 3, args.size, args.size)
    post = torch.randn(args.batch_size, 3, args.size, args.size)
    with torch.no_grad():
        logits, z = model(pre, post)
    print(f"\nlogits {tuple(logits.shape)}   z {tuple(z.shape)}")

    groups = model.param_groups(cfg.optim.lr, cfg.optim.backbone_lr)
    print("\nparameter groups")
    for g in groups:
        print(f"  {g['name']:8} lr={g['lr']:<9g} tensors={len(g['params'])}")

    print("\nOK")


if __name__ == "__main__":
    _main()
