# Maps dataset name -> builder function
from typing import Callable, Dict, Union, List
from torch.utils.data import ConcatDataset

from .animerun import build_animerun
from .linktoanime import build_linktoanime
from .flyingthings3D import build_flyingthings3D

DATASETS: Dict[str, Callable] = {
    "animerun": build_animerun,
    "linktoanime": build_linktoanime,
    "flyingthings3d": build_flyingthings3D,
}


def get_dataset_builder(name: Union[str, List[str]]) -> Callable:
    if isinstance(name, str):
        name = name.lower()
        if name not in DATASETS:
            raise KeyError(f"Unknown dataset: {name}")
        return DATASETS[name]
    elif isinstance(name, list):
        builders = [get_dataset_builder(n) for n in name]
        def combined_builder(cfg, split: str):
            datasets = [b(cfg, split) for b in builders]
            return ConcatDataset(datasets)
        return combined_builder