import torch
from safetensors.torch import load_model
import json
from importlib.resources import files
from types import SimpleNamespace

from .amphion_codec.codec import CodecEncoder, CodecDecoder

class amphion_codec:
    def __init__(self, configs, device=None):
        self.device = device

        cfg_path = files(__package__) / "maskgct.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

        self.codec_encoder, self.codec_decoder = self.build_acoustic_codec(cfg.model.acoustic_codec, device)
        load_model(self.codec_encoder, f"{configs.infer_ckpt_dir}/MaskGCT_codec_encoder.safetensors")
        load_model(self.codec_decoder, f"{configs.infer_ckpt_dir}/MaskGCT_codec_decoder.safetensors")

    def build_acoustic_codec(self, cfg, device):
        codec_encoder = CodecEncoder(cfg=cfg.encoder)
        codec_decoder = CodecDecoder(cfg=cfg.decoder)
        codec_encoder.eval()
        codec_decoder.eval()
        codec_encoder.to(device)
        codec_decoder.to(device)
        return codec_encoder, codec_decoder
    
    @torch.no_grad()
    def encode(self, speech_24k):
        vq_emb = self.codec_encoder(speech_24k)
        _, vq, _, _, _ = self.codec_decoder.quantizer(vq_emb)
        acoustic_code = vq.permute(1, 2, 0)
        return acoustic_code # [b, l, c]
    
    @torch.no_grad()
    def decode(self, tokens, n_quantizers=12):
        vq_emb = self.codec_decoder.vq2emb(tokens.permute(2, 0, 1), n_quantizers)
        recovered_audio = self.codec_decoder(vq_emb)
        recovered_audio = recovered_audio[0][0].cpu().numpy()
        return recovered_audio