import math

def cosine_lr(step, total, base_lr, warmup=0):
    if step < warmup:
        return base_lr * (step / max(1, warmup))
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))