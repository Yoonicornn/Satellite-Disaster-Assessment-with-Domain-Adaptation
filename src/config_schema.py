"""
The shape of a config, written once as pydantic models.

This file is the single place that says which keys are allowed, what type each one is and what
counts as a legal value. Every section sets extra="forbid", so a typo in a YAML file
(`bottlenck: 512`) stops the run instead of being silently ignored while the default is used.

configs/ stays pure YAML. This file is the code that checks it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

STRICT = ConfigDict(extra="forbid", validate_assignment=True)


class Section(BaseModel):
    """Base for every section: unknown keys are an error."""
    model_config = STRICT


# ----------------------------------------------------------------------------- sections
class RunCfg(Section):
    out_dir: str = Field(default="runs", description="root folder for every run")
    group: str | None = Field(default=None, description="subfolder of out_dir; null -> the method name")
    name: str | None = Field(default=None, description="run folder; null -> seed<seed>[_<timestamp>]")
    timestamp: bool = Field(default=True, description="add a timestamp so a run never overwrites another")
    seed: int = Field(default=0, ge=0, description="model init and data order, never the split")
    config_file: str | None = Field(default=None, description="filled in by load_config")
    auto_name: bool = Field(default=False, description="set automatically; true when name was generated")


class AugCfg(Section):
    hflip: bool = True
    rot90: bool = True
    rotate_deg: float = Field(default=10.0, ge=0, le=180, description="extra rotation, +/- degrees")
    scale: tuple[float, float] = Field(default=(0.9, 1.1), description="random zoom range")

    @model_validator(mode="after")
    def _check_scale(self):
        lo, hi = self.scale
        if not 0 < lo <= hi:
            raise ValueError(f"scale must be 0 < low <= high, got {self.scale}")
        return self


class DataCfg(Section):
    crops_dir: str = Field(description="folder written by scripts/extract_crops.py")
    max_black: float = Field(default=0.5, ge=0, le=1, description="drop crops blacker than this")
    batch_size: int = Field(default=64, gt=0, description="source batch; target batch matches it")
    workers: int = Field(default=2, ge=0)
    aug: AugCfg = Field(default_factory=AugCfg)


class ModelCfg(Section):
    backbone: Literal["resnet18", "resnet34", "resnet50",
                      "dinov2_vits14", "dinov2_vitb14"] = "resnet34"
    pretrained: bool = True
    fusion: Literal["concat_diff", "concat"] = "concat_diff"
    bottleneck: int = Field(default=256, gt=0, description="size of z, the feature DA aligns")
    dropout: float = Field(default=0.3, ge=0, lt=1)
    input_size: int | None = Field(
        default=None, gt=0,
        description="resize crops to this before the encoder; null = use them at 64x64. "
                    "Required and must be a multiple of 14 for the dinov2 backbones.")
    vit_pool: Literal["cls", "cls+avg"] = Field(
        default="cls+avg",
        description="dinov2 only: class token, or class token concatenated with the mean "
                    "patch token (the usual linear-probe read-out)")
    trainable_blocks: int = Field(
        default=-1, ge=-1,
        description="-1 trains the whole encoder, 0 freezes it, n trains the last n blocks "
                    "(resnet groups: stem, layer1..layer4; dinov2: transformer blocks)")

    @model_validator(mode="after")
    def _backbone_needs_input_size(self):
        if self.backbone.startswith("dinov2"):
            if self.input_size is None:
                raise ValueError(f"{self.backbone} has patch size 14, so model.input_size must "
                                 f"be set to a multiple of 14 (70, 126 or 224)")
            if self.input_size % 14:
                raise ValueError(f"model.input_size must be a multiple of 14 for "
                                 f"{self.backbone}, got {self.input_size}")
        return self


class LossCfg(Section):
    class_weights: Literal["balanced", "sqrt", "none"] = "balanced"


class OptimCfg(Section):
    name: Literal["adamw", "sgd"] = "adamw"
    lr: float = Field(default=2e-4, gt=0, description="new layers")
    backbone_lr: float = Field(default=2e-5, gt=0, description="pretrained layers")
    weight_decay: float = Field(default=1e-6, ge=0)
    epochs: int = Field(default=25, gt=0)
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    amp: bool = True
    grad_clip: float = Field(default=0, ge=0, description="0 = off")
    early_stop: int = Field(default=0, ge=0,
                            description="stop after N epochs with no improvement; 0 = off")
    min_epochs: int = Field(default=0, ge=0, description="never stop early before this epoch")


class EvalCfg(Section):
    select_on: str = Field(default="source_val", description="split used to pick the best epoch")
    select_metric: Literal["macro_f1", "accuracy"] = "macro_f1"
    splits: list[str] = Field(default=["source_val", "target_test", "test"])
    track_splits: list[str] = Field(
        default=[],
        description="evaluated every epoch for the curves only - NEVER used to pick the best epoch")
    save_predictions: bool = True

    @model_validator(mode="after")
    def _selection_split_is_evaluated(self):
        if self.select_on not in self.splits:
            raise ValueError(f"select_on ({self.select_on!r}) must be one of splits ({self.splits})")
        unknown = [s for s in self.track_splits if s not in self.splits]
        if unknown:
            raise ValueError(f"eval.track_splits {unknown} are not in eval.splits ({self.splits})")
        if "target" in self.select_on or self.select_on == "test":
            print(f"WARNING: eval.select_on is {self.select_on!r}. Choosing the best epoch with "
                  "target labels invalidates an unsupervised domain-adaptation result.")
        return self


class LogCfg(Section):
    tensorboard: bool = True
    progress: bool = Field(default=True, description="tqdm progress bar")
    every_n_steps: int = Field(default=50, gt=0)
    save_last_every_epoch: bool = True


# ----------------------------------------------------------------------------- methods
# One model per method, chosen by method.name. A key that belongs to another method
# (gamma in coral.yaml, say) is rejected, so the four configs cannot drift into each other.
class BaselineMethod(Section):
    name: Literal["baseline"]


class OracleMethod(Section):
    name: Literal["oracle"]


class BnOnlyMethod(Section):
    """Control: forwards target batches so BatchNorm adapts, but adds no loss. Separates
    BatchNorm adaptation from what CORAL and DANN actually contribute."""
    name: Literal["bn_only"]


class CoralMethod(Section):
    name: Literal["coral"]
    lambda_weight: float = Field(gt=0, description="weight of the CORAL term")
    ramp_epochs: int = Field(default=0, ge=0, description="raise lambda from 0 over N epochs")


class DannMethod(Section):
    name: Literal["dann"]
    lambda_max: float = Field(default=1.0, gt=0, description="value the ramp approaches")
    gamma: float = Field(default=10.0, gt=0, description="steepness of the ramp (paper value)")
    hidden: int = Field(default=256, gt=0, description="discriminator: z -> hidden -> 2")


class CdanMethod(Section):
    """CDAN: the discriminator sees the outer product of z and the predicted class
    distribution, so the encoder aligns p(z|y) rather than p(z)."""
    name: Literal["cdan"]
    lambda_max: float = Field(default=1.0, gt=0, description="value the ramp approaches")
    gamma: float = Field(default=10.0, gt=0, description="steepness of the ramp (paper value)")
    hidden: int = Field(default=256, gt=0, description="discriminator: (bottleneck*4) -> hidden -> hidden -> 2")
    entropy: bool = Field(default=False,
                          description="CDAN+E: weight examples by prediction confidence. "
                                      "Risky here - the model is confidently wrong on caribbean")


Method = Annotated[
    Union[BaselineMethod, BnOnlyMethod, CoralMethod, DannMethod, CdanMethod, OracleMethod],
    Field(discriminator="name"),
]


# ----------------------------------------------------------------------------- root
class ExperimentConfig(Section):
    run: RunCfg = Field(default_factory=RunCfg)
    data: DataCfg
    model: ModelCfg = Field(default_factory=ModelCfg)
    loss: LossCfg = Field(default_factory=LossCfg)
    optim: OptimCfg = Field(default_factory=OptimCfg)
    method: Method
    eval: EvalCfg = Field(default_factory=EvalCfg)
    log: LogCfg = Field(default_factory=LogCfg)

    @model_validator(mode="after")
    def _finish(self):
        if self.run.group is None:
            self.run.group = self.method.name
        if self.run.name is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run.name = f"seed{self.run.seed}" + (f"_{stamp}" if self.run.timestamp else "")
            self.run.auto_name = True
        if self.method.name == "oracle" and self.eval.select_on != "source_val":
            print("WARNING: the oracle is selected on a split other than source_val, "
                  "so it is no longer comparable with the other runs.")
        return self

    @property
    def uses_target_unlabelled(self) -> bool:
        """True for every method whose training step needs an unlabelled target batch."""
        from .data.crops import TARGET_METHODS          # one list, no second copy to drift
        return self.method.name in TARGET_METHODS
