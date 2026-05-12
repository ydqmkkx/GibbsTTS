import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext

from tqdm import tqdm

from datas.dataset import BucketDataset, collate_fn, EmiliaShardSampler
from datas.sampler import DistributedDynamicSampler

from config import ModelConfig
from models.model import GibbsTTS_Model

from utils.scheduler import cos_scheduler_per_epoch
from utils.load import continue_training

from collections import OrderedDict
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.module.named_parameters())

    for name, param in model_params.items():
        if not param.requires_grad:
            continue
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
def train(rank, local_rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    configs = ModelConfig()

    if not os.path.exists(configs.model_save_path):
        print(f'Creating {configs.model_save_path}')
        os.makedirs(configs.model_save_path, exist_ok=True)
    
    model = GibbsTTS_Model(configs).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    ema_model = GibbsTTS_Model(configs).to(local_rank)
    ema_model.load_state_dict(model.module.state_dict())
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    if rank == 0:
        writer = SummaryWriter(configs.log_dir)

    optimizer = optim.AdamW(model.parameters(), lr=configs.learning_rate)

    # load checkpoints
    if configs.load_ckpt_path is not None:
        steps, current_shard, current_epoch = continue_training(configs.load_ckpt_path, model, ema_model, optimizer)
        if current_shard >= configs.num_shards:
            current_epoch += 1
            current_shard = 0
    else:
        steps, current_shard, current_epoch = 0, 0, 0
    
    def worker_init_fn(worker_id):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
    
    emilia_shard = EmiliaShardSampler(configs.dataset_path)
    train_dataset = BucketDataset(configs)
    train_sampler = DistributedDynamicSampler(train_dataset, configs, num_replicas=world_size, rank=rank)
    
    model.train()
    batch_accum = configs.batch_accum
    for epoch in range(current_epoch, configs.num_epochs):
        train_sampler.set_epoch(epoch)

        for shard in range(current_shard, configs.num_shards):
            ds = emilia_shard(epoch, shard)
            train_dataset._load_datas(ds)
            train_sampler.set_shard(shard, train_dataset.id_buckets)

            train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=configs.num_workers, worker_init_fn=worker_init_fn, pin_memory=True, collate_fn=collate_fn, persistent_workers=False)

            steps_in_epoch = len(train_dataloader)
            scheduler = cos_scheduler_per_epoch(
                optimizer,
                epoch = epoch * configs.num_shards + shard,
                num_epochs = configs.num_epochs * configs.num_shards,
                steps_in_epoch = steps_in_epoch,
                warmup_epochs = configs.warmup_shards,
                min_lr_ratio = configs.min_lr_ratio,
            )

            if rank == 0:
                dataloader = tqdm(train_dataloader)
            else:
                dataloader = train_dataloader
                
            for batch_idx, datas in enumerate(dataloader):
                will_step = ((steps+1) % batch_accum == 0)
                ctx = model.no_sync() if not will_step else nullcontext()
                with ctx:
                    data = [d.to(local_rank, non_blocking=True) for d in datas]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss_dict, value_dict = model(*data)
                        loss = sum(loss_dict.values()) / batch_accum
                    loss.backward()

                if will_step:
                    if configs.grad_clip_thresh:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), configs.grad_clip_thresh)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema_model, model)

                scheduler.step()
                steps += 1
                if rank == 0 and steps % configs.log_interval_step == 0:
                    for key in loss_dict.keys():
                        writer.add_scalar(f"training/{key}", loss_dict[key].item(), steps)
                    if value_dict is not None:
                        for key in value_dict.keys():
                            writer.add_scalar(f"value/{key}", value_dict[key].item(), steps)
                    writer.add_scalar("learning_rate/learning_rate", scheduler.get_last_lr()[0], steps)
            
            if rank == 0:
                torch.save({
                    'step': steps,
                    'shard': shard,
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(configs.model_save_path, f'checkpoint_{epoch}_{shard}.pt'))
            dist.barrier()
            print(f"Rank {rank}, Shard {shard}, Epoch {epoch}, Loss {sum(loss_dict.values()).item()}")
        current_shard = 0

if __name__ == "__main__":
    os.environ['RANK'] = os.environ.get('OMPI_COMM_WORLD_RANK')
    os.environ['LOCAL_RANK'] = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')
    os.environ['WORLD_SIZE'] = os.environ.get('OMPI_COMM_WORLD_SIZE')

    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))

    import numpy as np 
    import random
    worker_seed = 1234
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    train(rank, local_rank, world_size)