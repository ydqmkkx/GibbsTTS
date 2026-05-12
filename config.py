from dataclasses import dataclass
from typing import Optional
from text import symbols

@dataclass
class ModelConfig:

    # model and train configs
    n_vocab: int = len(symbols) + 1
    n_lang: int = 3
    hidden_size: int = 1024
    intermediate_size: int = hidden_size * 4
    n_heads: int = hidden_size // 64
    n_layers: int = 16
    dropout: float = 0.
    cfg_dropout: float = 0.15
    quantizers_num: int = 12

    codebook_size: int = 1024
    special_codebook_size: int = 1
    batch_size: tuple[int, ...] = 16
    length_bins: tuple[int, ...] = (100, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536)

    t_grid_size: int = 1024

    batch_accum: int = 1
    learning_rate: float = 2e-4
    min_lr_ratio: float = 0.1
    grad_clip_thresh: Optional[float] = 1.0
    num_epochs: int = 10
    num_shards: int = 3
    warmup_shards: int = 1.5
    log_interval_step: int = 10
    save_interval_shard: int = 1
    num_workers: int = 8

    train: str = False
    codebook_weights_path: str = '/work/gj36/e43018/DFM/amphion_utils/coarse_weights'
    model_save_path: str = './checkpoints/tmp'
    log_dir: str = './runs/tmp'
    load_ckpt_path: str = None 

    # infer configs
    infer_ckpt_dir: str = '/home/sarulab/dong_yang/yd/DFM/ckpt'
    infer_ckpt_path: str = '/home/sarulab/dong_yang/yd/DFM/ckpt/GibbsTTS_large_ema.pt'
    steps: int = 32
    rescale_cfg: float = 0.75
    cfg: float = 2.5
    temperature: float = 0.6
    top_p: float = 1.