import torch
import torch.nn as nn
import torch.nn.functional as F

def interp_table(t: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    T = table.shape[0]

    t = t.clamp(0.0, 1.0)
    pos = t * (T - 1)                  
    idx0 = pos.floor().long()          
    idx1 = (idx0 + 1).clamp(max=T - 1) 
    w = pos - idx0.float()             

    y0 = table[idx0]
    y1 = table[idx1]
    return y0 * (1 - w) + y1 * w

def decode_latents(hidden, weights):
    hidden = F.normalize(hidden, dim=-1)
    weights = F.normalize(weights, dim=-1)
    diff = hidden.unsqueeze(3) - weights.unsqueeze(0).unsqueeze(0)  
    dist = (diff ** 2).sum(-1)
    indices = dist.argmin(-1)
    return indices

def random_mask(token_lengths, min_ratio=0., max_ratio=0.3):
    b = token_lengths.shape[0]
    r = torch.rand(b).to(token_lengths.device) * (max_ratio - min_ratio) + min_ratio
    mask_start_ids = (token_lengths * r).round().int()
    return mask_start_ids

def sequence_mask(length: torch.Tensor, max_length: int = None, left_padded: bool=False) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)

    if left_padded:
        return x.unsqueeze(0) >= (max_length - length).unsqueeze(1)
    else:
        return x.unsqueeze(0) < length.unsqueeze(1)

def pad_nested_tensor(nested_tensor, padding_value=0, left_padded=False):
    if left_padded:
        reversed_sequences = [seq.flip(dims=[0]) for seq in nested_tensor.unbind()]
        reversed_nested_tensor = torch.nested.nested_tensor(reversed_sequences)

        padded_tensor = torch.nested.to_padded_tensor(reversed_nested_tensor, padding=padding_value)
        return padded_tensor.flip(dims=[1])
    else:
        return torch.nested.to_padded_tensor(nested_tensor, padding=padding_value)
    
def logits_top_p(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -torch.log(-torch.log(noise + 1e-6) + 1e-6)

def gumbel_sample(t, dim=-1):
    return (t + gumbel_noise(t)).argmax(dim=dim)