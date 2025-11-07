from typing import Callable, Dict
from ..backup.baseline_fnet import BaselineFNet
# from .raft_tiny import TinyRAFT

MODELS: Dict[str, Callable] = {
    "baseline_fnet": BaselineFNet,
    # "raft_tiny": TinyRAFT,
}

def build_model(cfg):
    name = cfg["model"]["name"]
    if name not in MODELS:
        raise KeyError(f"Unknown model: {name}")
    return MODELS[name](**{k:v for k,v in cfg["model"].items() if k != "name"})