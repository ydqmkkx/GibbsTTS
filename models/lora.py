import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer, r, alpha, dropout):
        super().__init__()

        assert isinstance(base_layer, nn.Linear)

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.dropout = nn.Dropout(dropout)

        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)

        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(
            self.lora_A(self.dropout(x))
        ) * self.scaling


class LanguageLoRAManager:
    def __init__(self):
        pass

    def apply_lora(self, model, r, alpha, dropout):
        for module in model.modules():
            # attention
            if hasattr(module, "attn"):
                attn = module.attn

                if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                    attn.qkv = LoRALinear(
                        attn.qkv,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                    )

                if hasattr(attn, "out_proj") and isinstance(attn.out_proj, nn.Linear):
                    attn.out_proj = LoRALinear(
                        attn.out_proj,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                    )

            # FFN
            if hasattr(module, "mlp"):
                mlp = module.mlp

                if hasattr(mlp, "up_gate_proj") and isinstance(mlp.up_gate_proj, nn.Linear):
                    mlp.up_gate_proj = LoRALinear(
                        mlp.up_gate_proj,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                    )

                if hasattr(mlp, "down_proj") and isinstance(mlp.down_proj, nn.Linear):
                    mlp.down_proj = LoRALinear(
                        mlp.down_proj,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                    )

        return model

    def preprocess(self, model):
        self.optim_params = []
        self.save_param_names = []

        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = True
                self.optim_params.append(p)
                self.save_param_names.append(name)
            elif "lang_embed" in name or "text_embed.weight" in name:
                p.requires_grad = True
                self.optim_params.append(p)
                self.save_param_names.append(name)
            else:
                p.requires_grad = False

    def get_lora_state_dict(self, model):
        names = set(self.save_param_names)
        model_to_save = model.module if hasattr(model, "module") else model
        sd = model_to_save.state_dict()

        return {
            k: v.cpu()
            for k, v in sd.items()
            if k in names
        }
    
    def load_merged_weights(self, configs, model, base_sd, lora_sd):
        merged_sd = dict(base_sd)
        scaling = configs.lora.alpha / configs.lora.r

        for k in lora_sd.keys():
            if not k.endswith(".lora_A.weight"):
                continue

            prefix = k[: -len(".lora_A.weight")]
            lora_A_key = k
            lora_B_key = prefix + ".lora_B.weight"
            base_key = prefix + ".weight"

            A = lora_sd[lora_A_key].float()
            B = lora_sd[lora_B_key].float()
            delta_w = (B @ A) * scaling

            base_w = merged_sd[base_key]
            w = base_w.float() + delta_w.to(base_w.device)
            merged_sd[base_key] = w.to(dtype=base_w.dtype)

        k = "estimator.text_embed.weight"
        merged_sd[k] = lora_sd[k]

        k = "estimator.lang_embed.weight"
        merged_sd[k] = torch.cat([merged_sd[k], lora_sd[k]])

        num_lang, lang_dim = merged_sd["estimator.lang_embed.weight"].shape
        model.estimator.lang_embed = nn.Embedding(num_lang, lang_dim)
        model.load_state_dict(merged_sd)
        return model