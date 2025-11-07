.PHONY: train eval tb
train:
python train.py --config configs/animerun_baseline.yaml


eval:
python eval.py --config configs/animerun_baseline.yaml trainer.ckpt_path=runs/animerun_baseline/best.pth


tb:
tensorboard --logdir runs