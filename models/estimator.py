import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.diffusion_transformer import DiTBlock, DiTFinalLayer

    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, intermediate_size):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_channels, intermediate_size),
            nn.SiLU(inplace=True),
            nn.Linear(intermediate_size, out_channels)
        )

    def forward(self, x):
        return self.layer(x)
    

# reference: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/decoder.py
class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        hidden_size = configs.hidden_size
        intermediate_size = configs.intermediate_size
        self.codebook_size = configs.codebook_size
        quantizers_num = configs.quantizers_num

        self.text_embed = nn.Embedding(configs.n_vocab, hidden_size)
        self.token_embed = nn.Parameter(
            torch.empty(quantizers_num, configs.codebook_size + configs.special_codebook_size, hidden_size)
            )
        self.input_proj = nn.Linear(quantizers_num * hidden_size, hidden_size)

        self.time_embeddings = SinusoidalPosEmb(hidden_size//2)
        self.time_mlp = TimestepEmbedding(hidden_size//2, hidden_size//2, intermediate_size//2)

        self.cfg_dropout = configs.cfg_dropout
        self.text_cfg_embed = nn.Parameter(torch.empty(hidden_size))
        self.token_cfg_embed = nn.Parameter(torch.empty(hidden_size))
        self.prompt_embed = nn.Parameter(torch.empty(hidden_size))
        self.lang_embed = nn.Embedding(configs.n_lang, hidden_size//2)
        
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, intermediate_size, configs.n_heads, configs.dropout) for _ in range(configs.n_layers)])
        self.final_layer = DiTFinalLayer(hidden_size)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)

        for m in self.time_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        nn.init.normal_(self.token_embed, std=0.02)
        nn.init.normal_(self.text_cfg_embed, std=0.02)
        nn.init.normal_(self.token_cfg_embed, std=0.02)
        nn.init.normal_(self.prompt_embed, std=0.02)

    def forward(self, t, x_t, texts, mask, pred_mask, lang):
        b, l, c = x_t.shape
        texts_embed = self.text_embed(texts) # [b, l, h]

        idx_flat = torch.arange(c, device=x_t.device).view(1,1,c).expand(b,l,c).reshape(-1) # [b*l*c]
        embed = self.token_embed[idx_flat, x_t.reshape(-1), :].reshape(b, l, -1) # [b, l, c*h]
        x = self.input_proj(embed) # [b, l, h]
        
        x = x + (~pred_mask[:, :, None]).to(x.dtype) * self.prompt_embed  

        cfg_mask = torch.rand(b, device=x.device) < self.cfg_dropout
        texts_embed = torch.where(cfg_mask[:, None, None], self.text_cfg_embed, texts_embed)
        x = torch.where(cfg_mask[:, None, None] & ~pred_mask[:, :, None], self.token_cfg_embed, x)

        x = torch.cat([texts_embed, x], dim=1)

        mask = mask.unsqueeze(-1).to(x.dtype) # [b, l, 1]
        t = t.to(x.dtype)
        t = self.time_mlp(self.time_embeddings(t))
        lang_embed = self.lang_embed(lang)
        cond = torch.cat([t, lang_embed], dim=-1)
        
        attn_mask = mask * mask.transpose(1, 2) # [b, l, l]
        attn_mask = torch.zeros_like(attn_mask).masked_fill(attn_mask == 0, -torch.finfo(x.dtype).max).unsqueeze(1)  # [b, 1, l, l]

        for block in self.blocks:
            x = block(x, cond, mask, attn_mask)

        x = x[:, texts.shape[-1]:, :]
        x = self.final_layer(x, cond)

        logits = torch.einsum("blh,ckh->blck", x, self.token_embed[:, :self.codebook_size, :])
        return logits
    
    def infer(self, t, x_t, prompt_l, texts, lang, mask, rescale_cfg, cfg):
        b, l, c = x_t.shape # [b, l, c]
        texts_embed = self.text_embed(texts) # [1, l, h]
        texts_embed = torch.cat([texts_embed, 
                                 self.text_cfg_embed[None, None, :].expand_as(texts_embed)],
                                 dim=0)

        idx_flat = torch.arange(c, device=x_t.device).view(1,1,c).expand(b,l,c).reshape(-1) # [b*l*c]
        embed = self.token_embed[idx_flat, x_t.reshape(-1), :].view(b, l, -1)
        x = self.input_proj(embed) # [b, l, h]

        x[:, :prompt_l, :] += self.prompt_embed

        t = self.time_mlp(self.time_embeddings(t))
        lang_embed = self.lang_embed(lang)
        cond = torch.cat([t, lang_embed], dim=-1)

        # cfg
        x_cfg = x.clone()
        x_cfg[:, :prompt_l, :] = self.token_cfg_embed
        x = torch.cat([x, x_cfg], dim=0) # [2b, text+l, h]
        x = torch.cat([texts_embed, x], dim=1)

        attn_mask = mask * mask.transpose(1, 2) # [b, l, l]
        attn_mask = torch.zeros_like(attn_mask).masked_fill(attn_mask == 0, -torch.finfo(x.dtype).max).unsqueeze(1)  # [b, 1, l, l]

        for block in self.blocks:
            x = block(x, cond, mask, attn_mask)
            
        x = x[:, texts.shape[-1]:, :]
        x = self.final_layer(x, cond)
        
        # cfg
        x, x_cfg = torch.split(x, [b, b], dim=0)
        x_std = x.std()
        x = x + cfg * (x - x_cfg)
        rescale_x = x * x_std / x.std()
        x = rescale_cfg * rescale_x + (1 - rescale_cfg) * x

        logits = torch.einsum("blh,ckh->blck", x, self.token_embed[:, :self.codebook_size, :])
        return logits