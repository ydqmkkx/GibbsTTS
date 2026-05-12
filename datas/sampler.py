import torch
import random

class DistributedDynamicSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(
        self,
        dataset,
        configs,
        num_replicas=None, # world_size
        rank=None,
        drop_last=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        
        self.drop_last = drop_last
        self.batches = None

        self._build_epoch = 0
        self.shard = 0
        self._build_shard = 0

        self.batch_size = configs.batch_size
        self.length_bins = configs.length_bins

    def set_shard(self, shard, id_buckets):
        self.shard = shard
        self.id_buckets = id_buckets

    def _build_batches(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.shard)
        rng = random.Random(self.epoch + self.shard) 

        batches = []
        id_buckets = self.id_buckets
        for bucket_id in range(len(id_buckets)):
            batch_size = int(self.batch_size * self.length_bins[-1] / self.length_bins[bucket_id+1])

            id_list = id_buckets[bucket_id]
            rng.shuffle(id_list)

            k = 0
            while k < len(id_list):
                if k+batch_size > len(id_list) and self.drop_last:
                    break
                batch = id_list[k:k+batch_size]
                batches.append(batch)

                k += batch_size

        rng.shuffle(batches)
        batches = batches[len(batches) % self.num_replicas : ]
        batches = batches[self.rank :: self.num_replicas]
        self.batches = batches
        self._build_epoch = self.epoch
        self._build_shard = self.shard
    
    def __iter__(self):
        if self.batches is None or self._build_epoch != self.epoch or self._build_shard != self.shard:
            self._build_batches()
        return iter(self.batches)

    def __len__(self):
        if self.batches is None or self._build_epoch != self.epoch or self._build_shard != self.shard:
            self._build_batches()
        return len(self.batches)