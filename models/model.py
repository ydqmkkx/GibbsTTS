import torch
import torch.nn as nn
import torch.nn.functional as F
from models.estimator import Decoder
from models.utils import interp_table, random_mask, sequence_mask, pad_nested_tensor, logits_top_p, gumbel_sample


class GibbsTTS_Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.n_vocab = configs.n_vocab
        self.special_codebook_size = configs.special_codebook_size
        self.codebook_size = configs.codebook_size
        self.quantizers_num = configs.quantizers_num
        self.estimator = Decoder(configs)

        self.register_buffer("dist_matrix", torch.empty(configs.quantizers_num, configs.codebook_size+configs.special_codebook_size, configs.codebook_size))
        self.register_buffer("beta", torch.empty(configs.t_grid_size))
        self.register_buffer("beta_dt", torch.empty(configs.t_grid_size))
    
    def forward(self, texts, text_lengths, tokens, token_lengths, lang):
        with torch.no_grad():
            texts = pad_nested_tensor(texts, padding_value=0, left_padded=True)
            device = texts.device
            start_token = torch.full((texts.shape[0], 1), self.n_vocab-1, dtype=texts.dtype, device=device)
            texts = torch.cat([texts, start_token], dim=-1)
            text_lengths = text_lengths + 1

            x_1 = pad_nested_tensor(tokens, padding_value=self.codebook_size).squeeze(1) # pad_id: self.codebook_size
            b, l, c = x_1.shape
            mask_start_ids = random_mask(token_lengths)

            t = torch.rand(b, device=device)
            beta = interp_table(t, self.beta)[:, None, None, None]

            temp_flat = torch.arange(c).view(1, 1, c).to(device)
            idx_flat = temp_flat.expand(b, l, c).reshape(-1)
            dist_matrix = getattr(self, f'dist_matrix')
            dist_flat = dist_matrix[idx_flat, x_1.reshape(-1)]
            dist = dist_flat.view(b, l, c, -1) # [b, l, c, k]

            logits = - dist * beta

            x_t = gumbel_sample(logits, dim=-1)
            pred_mask = torch.arange(l).to(device).unsqueeze(0).expand(b, -1) >= mask_start_ids.unsqueeze(1) # [b, l], bool
            x_t = torch.where(pred_mask.unsqueeze(-1), x_t, x_1)

            weights_c = 1 - torch.arange(c, device=device) / c  # [c]
            mask_left = sequence_mask(text_lengths, left_padded = True)
            mask_right = sequence_mask(token_lengths) # [b, l]
            mask = torch.cat([mask_left, mask_right], dim=1)
            weights = (pred_mask & mask_right).float()[:, :, None] * weights_c[None, None, :] # [b, l, c]

        logits = self.estimator(t, x_t, texts, mask, pred_mask, lang)   # [b, l, c, k]

        dfm_loss = F.cross_entropy(
            logits.float().reshape(-1, self.codebook_size), 
            x_1.reshape(-1), 
            ignore_index=self.codebook_size,
            reduction='none').reshape(b, l, c) # [b, l, c]
        dfm_loss = (dfm_loss * weights).sum() / weights.sum()
        
        return {f"dfm_loss": dfm_loss}, None


    # def solver(self, t, h, x_t, logits):  
    #     b, l, c = x_t.shape
    #     temp_flat = torch.arange(self.quantizers_num, device=x_t.device).view(1, 1, self.quantizers_num)
    #     idx_flat = temp_flat.expand(b, l, c).reshape(-1)
        
    #     x_1 = gumbel_sample(logits, dim=-1)

    #     beta = interp_table(t, self.beta)[:, None, None, None]
    #     beta_dt = interp_table(t, self.beta_dt)[:, None, None, None]

    #     dist_matrix = getattr(self, f"dist_matrix")
    #     dist_flat = dist_matrix[idx_flat, x_1.reshape(-1)]
    #     dist = dist_flat.view(b, l, c, -1) # [b, l, c, k]
    #     d = torch.gather(dist, -1, x_t.unsqueeze(-1)) - dist

    #     p_t = F.softmax(- dist * beta, dim=-1) # [b, l, c, k]
    #     u = p_t * beta_dt * d.clamp_min(0)
        
    #     intensity = u.sum(dim=-1) 
    #     jump_prob = 1. - torch.exp(-h * intensity)
    
    #     mask_jump = (torch.rand_like(x_t.to(u.dtype)) <= jump_prob) & (intensity > 0)
    #     if mask_jump.any():
    #         probs = u[mask_jump]
    #         x_t[mask_jump] = torch.multinomial(probs, 1).squeeze(-1)
    #     return x_t
    

    def solver(self, t, h, x_t, logits):
        b, l, c = x_t.shape
        device = x_t.device

        temp_flat = torch.arange(c, device=device).view(1, 1, c)
        idx_flat = temp_flat.expand(b, l, c).reshape(-1)

        x_1 = gumbel_sample(logits, dim=-1)

        beta = interp_table(t, self.beta)[:, None, None, None]
        beta_dt = interp_table(t, self.beta_dt)[:, None, None, None]
        beta_next = interp_table(t + h, self.beta)[:, None, None, None]

        dist_matrix = getattr(self, "dist_matrix")

        dist_flat = dist_matrix[idx_flat, x_1.reshape(-1)]
        dist = dist_flat.view(b, l, c, -1)

        dist_cur = dist.gather(-1, x_t.unsqueeze(-1)).squeeze(-1)
        delta = dist_cur.unsqueeze(-1) - dist

        p_t = F.softmax(-dist * beta, dim=-1)
        u = p_t * (beta_dt * delta).clamp_min(0)
        intensity = u.sum(dim=-1)

        p_next = F.softmax(-dist * beta_next, dim=-1)
        dist_target = (p_next * dist).sum(dim=-1)

        need = dist_cur - dist_target
        progress = (u * delta).sum(dim=-1) / intensity.clamp_min(1e-8)

        q_base = 1.0 - torch.exp(-h * intensity)
        q_match = need / progress

        feasible = torch.isfinite(q_match) & (q_match >= 0) & (q_match <= 1)
        jump_prob = torch.where(feasible, q_match, q_base)
        
        mask_jump = (torch.rand_like(jump_prob) <= jump_prob) & (intensity > 0)
        if mask_jump.any():
            probs = u[mask_jump]
            x_t[mask_jump] = torch.multinomial(probs, 1).squeeze(-1)
        return x_t

    def synthesize(self, texts, lang, length, prompt_token, n_timesteps, temperature, top_p, rescale_cfg, cfg):
        device=texts.device
        start_token = torch.full((texts.shape[0], 1), self.n_vocab-1, dtype=texts.dtype, device=device)
        texts = torch.cat([texts, start_token], dim=-1)

        b, prompt_l, c = prompt_token.shape
        l = prompt_l + length

        x_0 = torch.randint(size=(b, l, c), high=self.codebook_size, device=device)
        x_t = x_0.clone()
        x_t[:, :prompt_l, :] = prompt_token
        x_0 = x_0[:, prompt_l:, :]
        mask = sequence_mask(torch.tensor(2 * [texts.shape[-1] + l], device=device), left_padded=True).unsqueeze(-1).float()   

        ts = torch.linspace(0, 1, steps=n_timesteps+1, device=device)
        xs = []
        for step in range(n_timesteps):
            t = ts[step].unsqueeze(0)
            h = ts[step+1].unsqueeze(0) - ts[step].unsqueeze(0)
            logits = self.estimator.infer(t, x_t, prompt_l, texts, lang, mask, rescale_cfg=rescale_cfg, cfg=cfg)[:, prompt_l:, :, :]
            if step == n_timesteps - 1:
                x_t[:, prompt_l:, :] = logits.argmax(dim=-1)
                xs.append(x_t[:, prompt_l:, :].clone())
                break

            logits = logits_top_p(logits, top_p) / temperature
            x_t[:, prompt_l:, :] = self.solver(t, h, x_t[:, prompt_l:, :], logits)
            xs.append(x_t[:, prompt_l:, :].clone())

        return {
                "x": x_t[:, prompt_l:, :],
                "xs": xs
            }