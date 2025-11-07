import os, random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Saver:
    def __init__(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
    def path(self, name):
        return os.path.join(self.work_dir, name)