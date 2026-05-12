import torch
import bisect
import math
import numpy as np

import os
import random 
from datasets import Dataset, concatenate_datasets
class EmiliaShardSampler:
    def __init__(self):
        self.path = "/work/gj18/e43018/corpus/Emilia-zip1"

    def __call__(self, epoch, shard):
        arrows_en = os.listdir(f"{self.path}/en")
        arrows_zh = os.listdir(f"{self.path}/zh")

        arrows_en.sort()
        arrows_zh.sort()
        rng = random.Random(epoch)
        rng.shuffle(arrows_en)
        rng.shuffle(arrows_zh)

        datas = []
        n = 20
        for i in range(n):
            data_en = Dataset.from_file(f"{self.path}/en/{arrows_en[shard * n + i]}")
            datas.append(data_en)
            
            data_zh = Dataset.from_file(f"{self.path}/zh/{arrows_zh[shard * n + i]}")
            datas.append(data_zh)
        return concatenate_datasets(datas)

language_dict = {"EN": 0, "ZH": 1, "mixed": 2}
class BucketDataset(torch.utils.data.Dataset):
    def __init__(self, configs):
        self.length_bins = configs.length_bins
        self.id_buckets = [[]]

    def _load_datas(self, datas):
        id_buckets = [[] for i in range(len(self.length_bins) - 1)]
        min_length = self.length_bins[0]
        max_length = self.length_bins[-1]

        token_lengths = datas['token_length']
        for idx, token_length in enumerate(token_lengths):
            if min_length <= token_length < max_length:
                bin_id = bisect.bisect_right(self.length_bins, token_length) - 1
                id_buckets[bin_id].append(idx)

        self.id_buckets = id_buckets
        self.arrow_table = datas.data
    
    def __len__(self):
        ls = [len(self.id_buckets[i]) for i in range(len(self.id_buckets))]
        return sum(ls)

    def __getitem__(self, idx):
        token = self.arrow_table[f"token"][idx].as_py()
        token = torch.tensor(token).long().unsqueeze(0)

        phone = self.arrow_table["phone"][idx].as_py()
        phone = torch.tensor(phone).long()

        lang = language_dict[self.arrow_table["language"][idx].as_py()]
        return token, phone, lang
    

def collate_fn(batch):
    tokens = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    langs = [item[2] for item in batch]

    text_lengths = torch.tensor([text.size(-1) for text in texts], dtype=torch.long)
    token_lengths = torch.tensor([token.size(1) for token in tokens], dtype=torch.long)

    langs = torch.tensor(langs, dtype=torch.long)
    
    return torch.nested.nested_tensor(texts), text_lengths, torch.nested.nested_tensor(tokens), token_lengths, langs