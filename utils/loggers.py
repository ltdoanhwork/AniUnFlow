import os
from torch.utils.tensorboard import SummaryWriter

class TBLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.tb = SummaryWriter(log_dir)
    def log_scalars(self, scalars: dict, step: int, prefix: str="train"):
        for k, v in scalars.items():
            self.tb.add_scalar(f"{prefix}/{k}", float(v), step)
    def close(self):
        self.tb.close()