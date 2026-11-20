"""
Load the YAML configs in configs/ and check them against src/config_schema.py.

One base file holds everything shared. One small file per method changes only what that method
needs. Anything can also be changed on the command line, which is how the seed is passed.

    from src.config import load_config
    cfg = load_config("configs/dann.yaml", ["run.seed=1", "optim.lr=1e-4"])
    print(cfg.optim.lr, cfg.method.gamma)

Rules:
  * a config file may name its parent with  _base_: base.yaml  (relative to that file)
  * command line overrides win over the file, the file wins over the base
  * the merged result is validated by ExperimentConfig, so an unknown key, a wrong type or an
    illegal value stops the run at load time instead of halfway through training
  * the resolved config is written into the run folder, so a result can always be traced back

Needs pyyaml and pydantic:  conda install -n ecse552 pyyaml pydantic
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml
from pydantic import ValidationError

from .config_schema import ExperimentConfig

BASE_KEY = "_base_"


# ----------------------------------------------------------------------------- loading
def _deep_merge(base: dict, extra: dict) -> dict:
    """extra wins. Nested dicts are merged key by key, everything else is replaced."""
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _read(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop(BASE_KEY, None)
    if parent is None:
        return data
    return _deep_merge(_read((path.parent / parent).resolve()), data)


def _apply_override(data: dict, item: str):
    """'optim.lr=1e-4' -> data['optim']['lr'] = '1e-4', which pydantic then turns into a float.

    The key must already exist in the merged config, so a typo fails here with a clear message
    rather than being passed on as an unexpected key.
    """
    if "=" not in item:
        raise ValueError(f"override must look like key.sub=value, got {item!r}")
    dotted, raw = item.split("=", 1)
    keys = dotted.strip().split(".")
    node = data
    for i, k in enumerate(keys[:-1]):
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"unknown config section '{'.'.join(keys[:i + 1])}' in override {item!r}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key '{dotted}'. Add it to the YAML and to "
                       f"src/config_schema.py first, so typos are caught.")
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(path, overrides=None) -> ExperimentConfig:
    path = Path(path).resolve()
    data = _read(path)
    for item in overrides or []:
        _apply_override(data, item)
    try:
        cfg = ExperimentConfig(**data)
    except ValidationError as e:
        raise SystemExit(f"\nThe config is not valid: {path}\n{e}\n") from None
    cfg.run.config_file = str(path)
    return cfg


# ----------------------------------------------------------------------------- saving
def to_yaml(cfg: ExperimentConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, default_flow_style=False)


def save_config(cfg: ExperimentConfig, path):
    """Write the resolved config next to the run output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(cfg))
    return path


def add_config_args(parser):
    """Standard --config and --set arguments for any script that trains or evaluates."""
    parser.add_argument("--config", default="configs/baseline.yaml", help="YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="key.sub=value",
                        help="override config values, e.g. --set run.seed=1 optim.lr=1e-4")
    return parser


if __name__ == "__main__":
    import argparse

    p = add_config_args(argparse.ArgumentParser(description="Print a resolved config."))
    a = p.parse_args()
    print(to_yaml(load_config(a.config, a.set)))
