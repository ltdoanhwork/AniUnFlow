# file: models/zoo.py
# Simple factory so trainer doesn't depend on a specific backbone structure.
import os
import sys
import torch



def build_model(args, cfg):
    name = cfg.get("name", "raft").lower()
    ckpt = cfg.get("ckpt")


    if name == "raft":
    # Expect a RAFT implementation under models/raft.py with RAFT(iters=...)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RAFT", "core"))
        from models.RAFT.core.raft import RAFT
        args.mixed_precision = cfg.get("mixed_precision", True)
        args.small = True
        model = RAFT(args)
    elif name == "gmflow":
        from models.gmflow import GMFlow
        model = GMFlow()
    else:
        raise ValueError(f"Unknown model {name}")


    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        if "model" in sd: sd = sd["model"]
        model.load_state_dict(sd, strict=False)
    return model