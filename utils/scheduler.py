import math
from torch.optim.lr_scheduler import LambdaLR


def cos_scheduler_per_epoch(
    optimizer,
    *,
    epoch: int,
    num_epochs: int,
    steps_in_epoch: int,
    warmup_epochs: float = 0.0,
    num_cycles: float = 0.5,   
    min_lr_ratio: float = 0.1,   
):

    def lr_lambda(step_in_epoch: int):
        # global progress in [0, 1]
        p = (epoch + step_in_epoch / max(1, steps_in_epoch)) / max(1, num_epochs)

        # warmup phase
        if warmup_epochs > 0:
            p_w = warmup_epochs / max(1, num_epochs)
            if p < p_w:
                return p / p_w

        # cosine decay phase
        p_w = warmup_epochs / max(1, num_epochs)
        p_decay = (p - p_w) / max(1e-12, 1.0 - p_w)

        cos_val = 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * p_decay))
        cos_val = max(0.0, cos_val)

        return min_lr_ratio + (1.0 - min_lr_ratio) * cos_val

    return LambdaLR(optimizer, lr_lambda)

