"""
The network: one encoder shared by the pre and post crop, then a classifier.

    pre  crop --\
                 encoder (shared weights) -> f_pre
    post crop --/                           f_post

    f = [f_pre, f_post, f_post - f_pre]   <- model.fusion = concat_diff
      or [f_pre, f_post]                  <- model.fusion = concat

    z = Dropout(ReLU(Linear(f, bottleneck)))     the feature CORAL and DANN align
    logits = Linear(z, 4)

Shared weights, rather than stacking the two crops into one 6-channel image: the xView2 work
reports that plain 6-channel stacking and image subtraction both did poorly, while two branches
with shared weights worked. It also halves the parameters.

Every method uses exactly this network. Only the loss changes, so a difference between the
baseline, CORAL and DANN cannot come from the architecture. The method-specific parts
(DANN's gradient reversal layer and discriminator) live in src/algorithms/.

Two families of encoder
-----------------------
  resnet18 / resnet34 / resnet50     torchvision, ImageNet supervised weights
  dinov2_vits14 / dinov2_vitb14      self-supervised ViT, LVD-142M

They are NOT interchangeable without thought:

  * DINOv2 has patch size 14, so the input side must be a multiple of 14. Our crops are 64x64,
    so set `model.input_size` (70, 126 or 224) and the encoder resizes for you. 224 is DINOv2's
    native regime (256 patch tokens) and costs about 21x a ResNet-34 pass at 64x64; 126 gives
    81 tokens at about 6x. There is no free option.
  * DINOv2 is a ViT, so it normalises with LayerNorm, which keeps NO running statistics. The
    bn_only control measures nothing on a ViT backbone, and the free BatchNorm adaptation that
    CORAL and DANN get on a ResNet does not happen there.
  * Switching the backbone invalidates the whole grid. Re-run baseline and every method, or the
    comparison is between two different models.

Quick check (from the project root):
    python -m src.models.siamese --config configs/baseline.yaml
    python -m src.models.siamese --config configs/baseline_dinov2.yaml
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

N_CLASSES = 4

# name -> (family, torchvision ctor or timm id, ImageNet weights enum, feature size)
BACKBONES = {
    "resnet18":      ("resnet", tvm.resnet18, "ResNet18_Weights", 512),
    "resnet34":      ("resnet", tvm.resnet34, "ResNet34_Weights", 512),
    "resnet50":      ("resnet", tvm.resnet50, "ResNet50_Weights", 2048),
    "dinov2_vits14": ("dinov2", "vit_small_patch14_dinov2.lvd142m", None, 384),
    "dinov2_vitb14": ("dinov2", "vit_base_patch14_dinov2.lvd142m", None, 768),
}

DINOV2_PATCH = 14


def backbone_family(name: str) -> str:
    if name not in BACKBONES:
        raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, got {name!r}")
    return BACKBONES[name][0]


# ----------------------------------------------------------------------------- encoders
def _build_resnet(name, pretrained):
    _, build, weights_enum, feat_dim = BACKBONES[name]
    weights = getattr(tvm, weights_enum).DEFAULT if pretrained else None
    net = build(weights=weights)
    net.fc = nn.Identity()          # keep conv stack + avgpool, drop the 1000-class head
    return net, feat_dim


def _build_dinov2(name, pretrained):
    """timm first (weights from HuggingFace), torch.hub as a fallback.

    timm is preferred: no GitHub clone, no xformers warning, and `dynamic_img_size=True`
    interpolates the position embeddings so any multiple of 14 works.
    """
    _, timm_id, _, embed_dim = BACKBONES[name]
    try:
        import timm
    except ImportError as e:
        timm, last = None, e

    if timm is not None:
        try:
            net = timm.create_model(timm_id, pretrained=pretrained, num_classes=0,
                                    dynamic_img_size=True)
            return net, embed_dim, "timm"
        except Exception as e:                       # offline, or no HuggingFace access
            last = e

    try:
        net = torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
        return net, embed_dim, "hub"
    except Exception as e:
        raise RuntimeError(
            f"could not load {name}.\n"
            f"  timm     : {last}\n"
            f"  torch.hub: {e}\n"
            f"Install timm (`pip install timm`) and make sure the machine can reach "
            f"huggingface.co, or pre-download the weights."
        ) from None


# ----------------------------------------------------------------------------- the network
class SiameseDamageNet(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, fusion="concat_diff",
                 bottleneck=256, dropout=0.3, n_classes=N_CLASSES,
                 input_size=None, vit_pool="cls+avg", trainable_blocks=-1):
        super().__init__()
        if fusion not in ("concat_diff", "concat"):
            raise ValueError(f"fusion must be concat_diff or concat, got {fusion!r}")

        self.fusion = fusion
        self.backbone_name = backbone
        self.kind = backbone_family(backbone)
        self.input_size = input_size
        self.vit_pool = vit_pool
        self.source = None

        if self.kind == "resnet":
            # state_dict keys are unchanged from the resnet-only version of this file, so
            # checkpoints from earlier runs still load
            self.encoder, feat_dim = _build_resnet(backbone, pretrained)
        else:
            if vit_pool not in ("cls", "cls+avg"):
                raise ValueError(f"vit_pool must be cls or cls+avg, got {vit_pool!r}")
            if input_size is None:
                raise ValueError(
                    f"{backbone} has patch size {DINOV2_PATCH}; set model.input_size to a "
                    f"multiple of {DINOV2_PATCH} (70, 126 or 224). Crops are 64x64 and are "
                    f"resized by the encoder.")
            if input_size % DINOV2_PATCH:
                raise ValueError(f"model.input_size must be a multiple of {DINOV2_PATCH} for "
                                 f"{backbone}, got {input_size}")
            self.encoder, embed_dim, self.source = _build_dinov2(backbone, pretrained)
            # "cls+avg" concatenates the class token with the mean of the patch tokens, the
            # usual linear-probe read-out for DINOv2. It keeps some patch-level information,
            # which is where a ViT's advantage over a 2x2 ResNet feature map lies.
            feat_dim = embed_dim * (2 if vit_pool == "cls+avg" else 1)

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

        self.trainable_blocks = trainable_blocks
        self.set_trainable_blocks(trainable_blocks)

    # ------------------------------------------------------------------ freezing
    def encoder_groups(self):
        """Encoder parameter groups, earliest (most general) first."""
        if self.kind == "resnet":
            e = self.encoder
            return [nn.ModuleList([e.conv1, e.bn1]), e.layer1, e.layer2, e.layer3, e.layer4]
        return list(self.encoder.blocks)

    def set_trainable_blocks(self, n):
        """-1 = train the whole encoder, 0 = freeze it completely, n = train the last n groups.

        The head (bottleneck + classifier) is always trainable. Freezing the early blocks is the
        usual way to fine-tune a large pretrained model on a small dataset without destroying
        the general features that made it worth using.
        """
        self.trainable_blocks = n
        if n < 0:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            return

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if n == 0:
            return

        for g in self.encoder_groups()[-n:]:
            for p in g.parameters():
                p.requires_grad_(True)
        # the final norm sits after the last block and is cheap, so train it with them
        for attr in ("norm", "fc_norm"):
            mod = getattr(self.encoder, attr, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad_(True)

    def frozen_encoder(self) -> bool:
        return not any(p.requires_grad for p in self.encoder.parameters())

    # ------------------------------------------------------------------ forward
    def _embed(self, x):
        """One crop through the encoder. Returns (B, feat_dim)."""
        if self.input_size is not None and x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        if self.kind == "resnet":
            return self.encoder(x)

        out = self.encoder.forward_features(x)
        if isinstance(out, dict):                       # torch.hub build
            cls, patch = out["x_norm_clstoken"], out["x_norm_patchtokens"]
        else:                                           # timm: (B, prefix + patches, D)
            n_prefix = getattr(self.encoder, "num_prefix_tokens", 1)
            cls, patch = out[:, 0], out[:, n_prefix:]
        return cls if self.vit_pool == "cls" else torch.cat([cls, patch.mean(dim=1)], dim=1)

    def features(self, pre, post):
        """The bottleneck feature z. This is what the domain-adaptation losses work on."""
        f_pre = self._embed(pre)                        # shared weights: the same call twice
        f_post = self._embed(post)
        parts = [f_pre, f_post] + ([f_post - f_pre] if self.fusion == "concat_diff" else [])
        return self.bottleneck(torch.cat(parts, dim=1))

    def forward(self, pre, post):
        """Returns (logits, z). The training step needs z for CORAL and DANN."""
        z = self.features(pre, post)
        return self.classifier(z), z

    def param_groups(self, lr, backbone_lr):
        """Pretrained layers get a smaller learning rate than the new ones.

        Frozen parameters are left out, so a fully frozen encoder gives a single head group and
        the optimiser never sees a parameter it cannot update.
        """
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        new = list(self.bottleneck.parameters()) + list(self.classifier.parameters())
        groups = []
        if enc:
            groups.append({"params": enc, "lr": backbone_lr, "name": "encoder"})
        groups.append({"params": new, "lr": lr, "name": "head"})
        return groups


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
        input_size=cfg.model.input_size,
        vit_pool=cfg.model.vit_pool,
        trainable_blocks=cfg.model.trainable_blocks,
    )


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def describe(model) -> str:
    """One line for the training log: which encoder, at what input size, how much of it trains."""
    total = sum(p.numel() for p in model.parameters())
    train = count_parameters(model)
    size = f"{model.input_size}x{model.input_size}" if model.input_size else "native 64x64"
    frozen = ("fully frozen" if model.trainable_blocks == 0 else
              "fully trainable" if model.trainable_blocks < 0 else
              f"last {model.trainable_blocks} block(s) trainable")
    pool = f", {model.vit_pool}" if model.kind == "dinov2" else ""
    src = f" ({model.source})" if model.source else ""
    return (f"{model.backbone_name}{src}, input {size}{pool}, encoder {frozen}, "
            f"{total:,} params ({train:,} trainable)")


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
    print(f"backbone : {describe(model)}")
    print(f"fusion   : {cfg.model.fusion}  ->  {model.fused_dim} -> z({model.bottleneck_dim})")

    pre = torch.randn(args.batch_size, 3, args.size, args.size)
    post = torch.randn(args.batch_size, 3, args.size, args.size)
    with torch.no_grad():
        logits, z = model(pre, post)
    print(f"\nlogits {tuple(logits.shape)}   z {tuple(z.shape)}")

    print("\nparameter groups")
    for g in model.param_groups(cfg.optim.lr, cfg.optim.backbone_lr):
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:8} lr={g['lr']:<9g} tensors={len(g['params']):<4} params={n:,}")

    print("\nOK")


if __name__ == "__main__":
    _main()
