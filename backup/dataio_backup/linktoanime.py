import csv
from .base_dataset import FlowPairDataset






def _read_csv(csv_path):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({"img1": r["img1"], "img2": r["img2"], "flow": r["flow"]})
    return rows

def build_linktoanime(cfg, split: str):
    csv_path = cfg["dataset"][f"{split}_csv"]
    rows = _read_csv(csv_path)
    ds = FlowPairDataset(rows,
                         resize=tuple(cfg["dataset"]["resize"]),
                         aug_cfg=cfg["dataset"].get("aug", {}),
                         root=cfg["dataset"].get("root"),
                         is_test = (split == "val"))
    return ds