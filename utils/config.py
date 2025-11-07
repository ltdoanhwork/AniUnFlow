import yaml
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def override_by_cli(cfg, overrides):
    # overrides like: trainer.ckpt_path=/x/best.pth model.hidden_dim=256
    for k, v in overrides.items():
        keys = k.split('.')
        node = cfg
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        try:
            # cast int/float/bool if possible
            if v.lower() in ("true","false"):
                v2 = v.lower()=="true"
            elif v.isdigit():
                v2 = int(v)
            else:
                v2 = float(v) if ('.' in v and v.replace('.','',1).isdigit()) else v
        except:
            v2 = v
        node[keys[-1]] = v2
    return cfg