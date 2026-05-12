import torch
import torch.optim as optim

def unwrap(m):
    return m.module if hasattr(m, "module") else m

def continue_training(checkpoint_path, model, ema_model, optimizer: optim.Optimizer) -> int:
    """load the latest checkpoints and optimizers"""
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    step = ckpt["step"]
    shard = ckpt["shard"]
    epoch = ckpt["epoch"]

    unwrap(model).load_state_dict(ckpt["model"], strict=True) 
    ema_model.load_state_dict(ckpt["ema_model"], strict=True) 
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f'resume model and optimizer from {epoch} epoch, {shard} shard')

    return step, shard + 1, epoch

