import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """
    Modified from: https://github.com/huggingface/transformers/blob/8ebfd84fa7f4d6c59f5059a439fad10ada26b3ff/src/transformers/models/llama/modeling_llama.py#L173
    """
    def __init__(self, hidden_size, intermediate_size, p_dropout=0.):
        super().__init__()    
        self.up_gate_proj = nn.Linear(hidden_size, 2 * intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(p_dropout)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        up, gate = self.up_gate_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.dropout(self.act_fn(gate) * up))
    

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, p_dropout=0.):
        super().__init__()
        assert hidden_size % n_heads == 0

        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.head_dim = hidden_size // n_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        self.rotary_pe = RotaryPositionalEmbeddings(self.head_dim)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, attn_mask=None, position_ids=None):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1) # [b, l, h]

        b, l, h = q.shape
        # [b, l, h] -> [b, self.n_heads, l, self.head_dim]
        q = q.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rotary_pe(q, position_ids)
        k = self.rotary_pe(k, position_ids)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.p_dropout if self.training else 0)
        attn = attn.transpose(1, 2).contiguous().view(b, l, h)

        x = self.out_proj(attn)
        return x


class RMSNorm(nn.Module):
    """
    Modified from: https://docs.pytorch.org/torchtune/0.2/_modules/torchtune/modules/rms_norm.html#RMSNorm
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

# modified from https://github.com/sh-lee-prml/HierSpeechpp/blob/main/modules.py#L390    
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, intermediate_size, num_heads, p_dropout):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, p_dropout)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = FFN(hidden_size, intermediate_size, p_dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
            
    def forward(self, x, c, x_mask, attn_mask=None, position_ids=None):
        """
        Args:
            x : [b, l, h]
            c : [b, h]
            x_mask : [b, l, 1]
            attn_mask: [b, 1, l, l]
        return the same shape as x
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=-1) # shape: [b, 1, h]

        x = x * x_mask
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask, position_ids) * x_mask
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x * x_mask
    

class DiTFinalLayer(nn.Module):
    """
    Modified from: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L125
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return x
  

class RotaryPositionalEmbeddings(nn.Module):
    """
    Modified from: 
    https://colab.research.google.com/drive/11SKfzvMotuvvXNqY9qBpsD2RQX1PK7rP?usp=sharing#scrollTo=XNeygwV2gEWH
    https://github.com/huggingface/transformers/blob/8ebfd84fa7f4d6c59f5059a439fad10ada26b3ff/src/transformers/models/llama/modeling_llama.py#L73
    """

    def __init__(self, d: int, base: int = 10_000):
        r"""
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device):
        r"""
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(device)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=device).float().to(device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1) 
        # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None):
        # Cache $\cos$ and $\sin$ values
        x = x.permute(2, 0, 1, 3) # [b, n_heads, l, d] -> [l, b, n_heads, d]
        device = x.device

        if position_ids is None:
            l = x.shape[0]
            self._build_cache(l, device)
            cos = self.cos_cached[:l]
            sin = self.sin_cached[:l] # [l, 1, 1, d]
        else:
            max_pos = int(position_ids.max().item()) + 1
            self._build_cache(max_pos, device)

            # cos_cached: [max_len, 1, 1, d]
            cos = self.cos_cached[position_ids].squeeze(3).squeeze(2)   # [b, l, 1, 1, d] -> [b, l, d]
            sin = self.sin_cached[position_ids].squeeze(3).squeeze(2)
            cos = cos.permute(1, 0, 2)[:, :, None, :]   # [b, l, d] -> [l, b, 1, d]
            sin = sin.permute(1, 0, 2)[:, :, None, :] 

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        x_rope = x_rope * cos + neg_half_x * sin # [l, b, n_heads, d]

        return torch.cat((x_rope, x_pass), dim=-1).permute(1, 2, 0, 3) # [l, b, n_heads, d] -> [b, n_heads, l, d]
    