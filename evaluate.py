import argparse
from utils.config import load_config, override_by_cli
from engine.evaluator import Evaluator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('overrides', nargs='*', help='key=val pairs to override config')
    args = ap.parse_args()

    cfg = load_config(args.config)
    kv = {}
    for ov in args.overrides:
        if '=' in ov:
            k, v = ov.split('=', 1)
            kv[k] = v
    cfg = override_by_cli(cfg, kv)

    ev = Evaluator(cfg)
    ev.run()

if __name__ == '__main__':
    main()